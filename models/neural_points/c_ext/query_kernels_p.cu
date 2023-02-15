#define KN  8

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "utils.h"

namespace cuda {          

    static __device__ inline uint8_t atomicAdd(uint8_t *address, uint8_t val) {
        size_t offset = (size_t)address & 3;
        uint32_t *address_as_ui = (uint32_t *)(address - offset);
        uint32_t old = *address_as_ui;
        uint32_t shift = offset * 8;
        uint32_t old_byte;
        uint32_t newval;
        uint32_t assumed;

        do {
            assumed = old;
            old_byte = (old >> shift) & 0xff;
            // preserve size in initial cast. Casting directly to uint32_t pads
            // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
            newval = static_cast<uint8_t>(val + old_byte);
            newval = (old & ~(0x000000ff << shift)) | (newval << shift);
            old = atomicCAS(address_as_ui, assumed, newval);
        } while (assumed != old);
        return __byte_perm(old, 0, offset);   // need validate
    }

    static __device__ inline char atomicAdd(char* address, char val) {
        // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
        size_t long_address_modulo = (size_t) address & 3;
        // the 32-bit address that overlaps the same memory
        auto* base_address = (unsigned int*) ((char*) address - long_address_modulo);
        // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
        // The "4" signifies the position where the first byte of the second argument will end up in the output.
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
        unsigned int selector = selectors[long_address_modulo];
        unsigned int long_old, long_assumed, long_val, replacement;

        long_old = *base_address;

        do {
            long_assumed = long_old;
            // replace bits in long_old that pertain to the char address with those from val
            long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
            replacement = __byte_perm(long_old, long_val, selector);
            long_old = atomicCAS(base_address, long_assumed, replacement);
        } while (long_old != long_assumed);
        return __byte_perm(long_old, 0, long_address_modulo);
    }            

    static __device__ inline int8_t atomicAdd(int8_t *address, int8_t val) {
        return (int8_t)cuda::atomicAdd((char*)address, (char)val);
    }

    static __device__ inline short atomicAdd(short* address, short val)
    {

        unsigned int *base_address = (unsigned int *)((size_t)address & ~2);

        unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;

        unsigned int long_old = ::atomicAdd(base_address, long_val);

        if((size_t)address & 2) {
            return (short)(long_old >> 16);
        } else {

            unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

            if (overflow)

                atomicSub(base_address, overflow);

            return (short)(long_old & 0xffff);
        }
    }

    static __device__ float cas(double *addr, double compare, double val) {
        unsigned long long int *address_as_ull = (unsigned long long int *) addr;
        return __longlong_as_double(atomicCAS(address_as_ull,
                                        __double_as_longlong(compare),
                                        __double_as_longlong(val)));
    }

    static __device__ float cas(float *addr, float compare, float val) {
        unsigned int *address_as_uint = (unsigned int *) addr;
        return __uint_as_float(atomicCAS(address_as_uint,
                                __float_as_uint(compare),
                                __float_as_uint(val)));
    }



    static __device__ inline uint8_t atomicCAS(uint8_t * const address, uint8_t const compare, uint8_t const value)
    {
        uint8_t const longAddressModulo = reinterpret_cast< size_t >( address ) & 0x3;
        uint32_t *const baseAddress  = reinterpret_cast< uint32_t * >( address - longAddressModulo );
        uint32_t constexpr byteSelection[] = { 0x3214, 0x3240, 0x3410, 0x4210 }; // The byte position we work on is '4'.
        uint32_t const byteSelector = byteSelection[ longAddressModulo ];
        uint32_t const longCompare = compare;
        uint32_t const longValue = value;
        uint32_t longOldValue = * baseAddress;
        uint32_t longAssumed;
        uint8_t oldValue;
        do {
            // Select bytes from the old value and new value to construct a 32-bit value to use.
            uint32_t const replacement = __byte_perm( longOldValue, longValue,   byteSelector );
            uint32_t const comparison  = __byte_perm( longOldValue, longCompare, byteSelector );

            longAssumed  = longOldValue;
            // Use 32-bit atomicCAS() to try and set the 8-bits we care about.
            longOldValue = ::atomicCAS( baseAddress, comparison, replacement );
            // Grab the 8-bit portion we care about from the old value at address.
            oldValue     = ( longOldValue >> ( 8 * longAddressModulo )) & 0xFF;
        } while ( compare == oldValue and longAssumed != longOldValue ); // Repeat until other three 8-bit values stabilize.
        return oldValue;
    }
}

using namespace cuda;

__global__ void get_occ_vox_kernel(
    const float* in_data,   // B * N * 3
    const int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const float *d_coord_shift,     // 3
    const float *d_voxel_size,      // 3
    const int *d_grid_size,       // 3
    const int *kernel_size,       // 3
    const int pixel_size,
    const int grid_size_vol,
    uint8_t *coor_occ,  // B * 400 * 400 * 400
    int8_t *loc_coor_counter,  // B * 400 * 400 * 400
    int *near_depth_id_tensor,  // B * 400 * 400
    int *far_depth_id_tensor,  // B * 400 * 400 
    const int inverse
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_actual_numpoints[i_batch]) {
        int coor[3];
        const float *p_pt = in_data + index * 3;
        coor[0] = floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
        if (coor[0] < 0 || coor[0] >= d_grid_size[0]) { return; }
        coor[1] = floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
        if (coor[1] < 0 || coor[1] >= d_grid_size[1]) { return; }
        float z = p_pt[2];
        if (inverse > 0){ z = 1.0 / z;}
        coor[2] = floor((z - d_coord_shift[2]) / d_voxel_size[2]);
        if (coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
        
        int frust_id_b, coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        if (loc_coor_counter[coor_indx_b] < (int8_t)0 || cuda::atomicAdd(loc_coor_counter + coor_indx_b, (int8_t)-1) < (int8_t)0) { return; }

        for (int coor_x = max(0, coor[0] - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], coor[0] + (kernel_size[0] + 1) / 2); coor_x++)    {
            for (int coor_y = max(0, coor[1] - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], coor[1] + (kernel_size[1] + 1) / 2); coor_y++)   {
                for (int coor_z = max(0, coor[2] - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], coor[2] + (kernel_size[2] + 1) / 2); coor_z++) {
                    frust_id_b = i_batch * pixel_size + coor_x * d_grid_size[1] + coor_y;
                    coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                    if (coor_occ[coor_indx_b] > (uint8_t)0) { continue; }
                    cuda::atomicCAS(coor_occ + coor_indx_b, (uint8_t)0, (uint8_t)1);
                    atomicMin(near_depth_id_tensor + frust_id_b, coor_z);
                    atomicMax(far_depth_id_tensor + frust_id_b, coor_z);
                }
            }
        }   
    }
}

__global__ void near_vox_full_kernel(
    const int B,
    const int SR,
    const int *pixel_idx,
    const int R,
    const int *vscale,
    const int *d_grid_size,
    const int pixel_size,
    const int grid_size_vol,
    const int *kernel_size,      // 3
    uint8_t *pixel_map,
    int8_t *ray_mask,     // B * R
    const uint8_t *coor_occ,  // B * 400 * 400 * 400
    int8_t *loc_coor_counter,    // B * 400 * 400 * 400
    const int *near_depth_id_tensor,  // B * 400 * 400
    const int *far_depth_id_tensor,  // B * 400 * 400 
    short *voxel_to_coorz_idx  // B * 400 * 400 * SR 
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / R;  // index of batch
    if (i_batch >= B) { return; }
    int vx_id = pixel_idx[index*2] / vscale[0], vy_id = pixel_idx[index*2 + 1] / vscale[1];
    int i_xyvox_id = i_batch * pixel_size + vx_id * d_grid_size[1] + vy_id;
    int near_id = near_depth_id_tensor[i_xyvox_id], far_id = far_depth_id_tensor[i_xyvox_id];
    ray_mask[index] = far_id > 0 ? (int8_t)1 : (int8_t)0;
    if (pixel_map[i_xyvox_id] > (uint8_t)0 || cuda::atomicCAS(pixel_map + i_xyvox_id, (uint8_t)0, (uint8_t)1) > (uint8_t)0) { return; }
    int counter = 0;
    for (int depth_id = near_id; depth_id <= far_id; depth_id++) {
        if (coor_occ[i_xyvox_id * d_grid_size[2] + depth_id] > (uint8_t)0) {
            voxel_to_coorz_idx[i_xyvox_id * SR + counter] = (short)depth_id;
            // if (i_xyvox_id>81920){
            //    printf("   %d %d %d %d %d %d %d %d %d %d    ", pixel_idx[index*2], vscale[0], i_batch, vx_id, vy_id, i_xyvox_id * SR + counter, i_xyvox_id, SR, counter, d_grid_size[1]);
            // }
            for (int coor_x = max(0, vx_id - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], vx_id + (kernel_size[0] + 1) / 2); coor_x++)  {
                for (int coor_y = max(0, vy_id - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], vy_id + (kernel_size[1] + 1) / 2); coor_y++)   {
                    for (int coor_z = max(0, depth_id - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], depth_id + (kernel_size[2] + 1) / 2); coor_z++)    {
                        int coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                        // cuda::atomicCAS(loc_coor_counter + coor_indx_b, (int8_t)-1, (int8_t)1);
                        int8_t loc = loc_coor_counter[coor_indx_b];
                        if (loc < (int8_t)0) {
                            loc_coor_counter[coor_indx_b] = (int8_t)1;
                        }
                    }
                }
            }
            if (counter >= SR - 1) { return; }
            counter += 1;
        }
    }
}


__global__ void insert_vox_points_kernel(        
    float* in_data,   // B * N * 3
    int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const int P,
    const int max_o,
    const int pixel_size,
    const int grid_size_vol,
    const float *d_coord_shift,     // 3
    const int *d_grid_size,
    const float *d_voxel_size,      // 3
    const int8_t *loc_coor_counter,    // B * 400 * 400 * 400
    short *voxel_pnt_counter,      // B * 400 * 400 * max_o 
    int *voxel_to_pntidx,      // B * pixel_size * max_o * P
    unsigned long seconds,
    const int inverse
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    if (index - i_batch * N < in_actual_numpoints[i_batch]) {
        const float *p_pt = in_data + index * 3;
        int coor_x = (p_pt[0] - d_coord_shift[0]) / d_voxel_size[0];
        int coor_y = (p_pt[1] - d_coord_shift[1]) / d_voxel_size[1];
        float z = p_pt[2];
        if (inverse > 0){ z = 1.0 / z;}
        int coor_z = (z - d_coord_shift[2]) / d_voxel_size[2];
        int pixel_indx_b = i_batch * pixel_size  + coor_x * d_grid_size[1] + coor_y;
        int coor_indx_b = pixel_indx_b * d_grid_size[2] + coor_z;
        if (coor_x < 0 || coor_x >= d_grid_size[0] || coor_y < 0 || coor_y >= d_grid_size[1] || coor_z < 0 || coor_z >= d_grid_size[2] || loc_coor_counter[coor_indx_b] < (int8_t)0) { return; }
        int voxel_indx_b = pixel_indx_b * max_o + (int)loc_coor_counter[coor_indx_b];
        //printf("voxel_indx_b, %d  ||   ", voxel_indx_b);
        int voxel_pntid = (int) cuda::atomicAdd(voxel_pnt_counter + voxel_indx_b, (short)1);
        if (voxel_pntid < P) {
            voxel_to_pntidx[voxel_indx_b * P + voxel_pntid] = index;
        } else {
            curandState state;
            curand_init(index+seconds, 0, 0, &state);
            int insrtidx = ceilf(curand_uniform(&state) * (voxel_pntid+1)) - 1;
            if(insrtidx < P){
                voxel_to_pntidx[voxel_indx_b * P + insrtidx] = index;
            }
        }
    }
}                        


__global__ void query_rand_along_ray_kernel(
    const float* in_data,   // B * N * 3
    const int B,
    const int SR,               // num. samples along each ray e.g., 128
    const int R,               // e.g., 1024
    const int max_o,
    const int P,
    const int K,                // num.  neighbors
    const int pixel_size,                
    const int grid_size_vol,
    const float radius_limit2,
    const float depth_limit2,
    const float *d_coord_shift,     // 3
    const int *d_grid_size,
    const float *d_voxel_size,      // 3
    const float *d_ray_voxel_size,      // 3
    const int *vscale,      // 3
    const int *kernel_size,
    const int *pixel_idx,               // B * R * 2
    const int8_t *loc_coor_counter,    // B * 400 * 400 * 400
    const short *voxel_to_coorz_idx,            // B * 400 * 400 * SR 
    const short *voxel_pnt_counter,      // B * 400 * 400 * max_o 
    const int *voxel_to_pntidx,      // B * pixel_size * max_o * P
    int *sample_pidx,       // B * R * SR * K
    float *sample_loc,       // B * R * SR * K
    unsigned long seconds,
    const int NN,
    const int inverse
) {
    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * SR);  // index of batch
    int ray_idx_b = index / SR;  
    if (i_batch >= B || ray_idx_b >= B * R) { return; }

    int ray_sample_loc_idx = index - ray_idx_b * SR;
    int frustx = pixel_idx[ray_idx_b * 2] / vscale[0];
    int frusty = pixel_idx[ray_idx_b * 2 + 1] / vscale[1];
    int vxy_ind_b = i_batch * pixel_size + frustx * d_grid_size[1] + frusty;
    int frustz = (int) voxel_to_coorz_idx[vxy_ind_b * SR + ray_sample_loc_idx];
    float centerx = d_coord_shift[0] + frustx * d_voxel_size[0] + (pixel_idx[ray_idx_b * 2] % vscale[0] + 0.5) * d_ray_voxel_size[0];
    float centery = d_coord_shift[1] + frusty * d_voxel_size[1] + (pixel_idx[ray_idx_b * 2 + 1] % vscale[1] + 0.5) * d_ray_voxel_size[1];
    float centerz = d_coord_shift[2] + (frustz + 0.5) * d_voxel_size[2];
    if (inverse > 0){ centerz = 1.0 / centerz;}
    sample_loc[index * 3] = centerx;
    sample_loc[index * 3 + 1] = centery;
    sample_loc[index * 3 + 2] = centerz;
    if (frustz < 0) { return; }
    int coor_indx_b = vxy_ind_b * d_grid_size[2] + frustz;
    int raysample_startid = index * K;
    int kid = 0;
    curandState state;
    for (int coor_x = max(0, frustx - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], frustx + (kernel_size[0] + 1) / 2); coor_x++) {
        for (int coor_y = max(0, frusty - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], frusty + (kernel_size[1] + 1) / 2); coor_y++) {
            int pixel_indx_b = i_batch * pixel_size  + coor_x * d_grid_size[1] + coor_y;
            for (int coor_z = max(0, frustz - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], frustz + (kernel_size[2] + 1) / 2); coor_z++) {
                int shift_coor_indx_b = pixel_indx_b * d_grid_size[2] + coor_z;
                if(loc_coor_counter[shift_coor_indx_b] < (int8_t)0) {continue;}
                int voxel_indx_b = pixel_indx_b * max_o + (int)loc_coor_counter[shift_coor_indx_b];
                for (int g = 0; g < min(P, (int) voxel_pnt_counter[voxel_indx_b]); g++) {
                    int pidx = voxel_to_pntidx[voxel_indx_b * P + g];
                    if ((radius_limit2 == 0 || (in_data[pidx*3]-centerx) * (in_data[pidx*3]-centerx) + (in_data[pidx*3 + 1]-centery) * (in_data[pidx*3 + 1]-centery) <= radius_limit2) && (depth_limit2==0 || (in_data[pidx*3 + 2]-centerz) * (in_data[pidx*3 + 2]-centerz) <= depth_limit2)) { 
                        if (kid++ < K) {
                            sample_pidx[raysample_startid + kid - 1] = pidx;
                        }
                        else {
                            curand_init(index+seconds, 0, 0, &state);
                            int insrtidx = ceilf(curand_uniform(&state) * (kid)) - 1;
                            if (insrtidx < K) {
                                sample_pidx[raysample_startid + insrtidx] = pidx;
                            }
                        }
                    }
                }
            }
        }
    }
}


__global__ void query_neigh_along_ray_layered_h_kernel(
    const float* in_data,   // B * N * 3
    const int B,
    const int SR,               // num. samples along each ray e.g., 128
    const int R,               // e.g., 1024
    const int max_o,
    const int P,
    const int K,                // num.  neighbors
    const int pixel_size,                
    const int grid_size_vol,
    const float radius_limit2,
    const float depth_limit2,
    const float *d_coord_shift,     // 3
    const int *d_grid_size,
    const float *d_voxel_size,      // 3
    const float *d_ray_voxel_size,      // 3
    const int *vscale,      // 3
    const int *kernel_size,
    const int *pixel_idx,               // B * R * 2
    const int8_t *loc_coor_counter,    // B * 400 * 400 * 400
    const short *voxel_to_coorz_idx,            // B * 400 * 400 * SR 
    const short *voxel_pnt_counter,      // B * 400 * 400 * max_o 
    const int *voxel_to_pntidx,      // B * pixel_size * max_o * P
    int *sample_pidx,       // B * R * SR * K
    float *sample_loc,       // B * R * SR * K
    unsigned long seconds,
    const int NN,
    const int inverse
) {
    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * SR);  // index of batch
    int ray_idx_b = index / SR;  
    if (i_batch >= B || ray_idx_b >= B * R) { return; }

    int ray_sample_loc_idx = index - ray_idx_b * SR;
    int frustx = pixel_idx[ray_idx_b * 2] / vscale[0];
    int frusty = pixel_idx[ray_idx_b * 2 + 1] / vscale[1];
    int vxy_ind_b = i_batch * pixel_size + frustx * d_grid_size[1] + frusty;
    int frustz = (int) voxel_to_coorz_idx[vxy_ind_b * SR + ray_sample_loc_idx];
    float centerx = d_coord_shift[0] + frustx * d_voxel_size[0] + (pixel_idx[ray_idx_b * 2] % vscale[0] + 0.5) * d_ray_voxel_size[0];
    float centery = d_coord_shift[1] + frusty * d_voxel_size[1] + (pixel_idx[ray_idx_b * 2 + 1] % vscale[1] + 0.5) * d_ray_voxel_size[1];
    float centerz = d_coord_shift[2] + (frustz + 0.5) * d_voxel_size[2];
    if (inverse > 0){ centerz = 1.0 / centerz;}
    sample_loc[index * 3] = centerx;
    sample_loc[index * 3 + 1] = centery;
    sample_loc[index * 3 + 2] = centerz;
    if (frustz < 0) { return; }
    // int coor_indx_b = vxy_ind_b * d_grid_size[2] + frustz;
    int raysample_startid = index * K;
    // curandState state;
    
    int kid = 0, far_ind = 0, coor_z, coor_y, coor_x;
    float far2 = 0.0;
    float xyz2Buffer[KN];
    for (int layer = 0; layer < (kernel_size[0]+1)/2; layer++){
        int zlayer = min((kernel_size[2]+1)/2-1, layer);
        
        for (int x = max(-frustx, -layer); x < min(d_grid_size[0] - frustx, layer+1); x++) {
            for (int y = max(-frusty, -layer); y < min(d_grid_size[1] - frusty, layer+1); y++) {                              
                coor_y = frusty + y;
                coor_x = frustx + x;
                int pixel_indx_b = i_batch * pixel_size  + coor_x * d_grid_size[1] + coor_y;
                for (int z =  max(-frustz, -zlayer); z < min(d_grid_size[2] - frustz, zlayer + 1); z++) {
                    //  if (max(abs(x),abs(y)) != layer || abs(z) != zlayer) continue;
                    if (max(abs(x),abs(y)) != layer && ((zlayer == layer) ? (abs(z) != zlayer) : 1)) continue;
                    // if (max(abs(x),abs(y)) != layer) continue;
                    coor_z = z + frustz;
                    
                    int shift_coor_indx_b = pixel_indx_b * d_grid_size[2] + coor_z;
                    if(loc_coor_counter[shift_coor_indx_b] < (int8_t)0) {continue;}
                    int voxel_indx_b = pixel_indx_b * max_o + (int)loc_coor_counter[shift_coor_indx_b];                  
                    for (int g = 0; g < min(P, (int) voxel_pnt_counter[voxel_indx_b]); g++) {
                        int pidx = voxel_to_pntidx[voxel_indx_b * P + g];
                        float x_v = (NN < 2) ? (in_data[pidx*3]-centerx) : (in_data[pidx*3] * in_data[pidx*3+2]-centerx*centerz) ;
                        float y_v = (NN < 2) ? (in_data[pidx*3+1]-centery) : (in_data[pidx*3+1] * in_data[pidx*3+2]-centery*centerz) ;
                        float xy2 = x_v * x_v + y_v * y_v;
                        float z2 = (in_data[pidx*3 + 2]-centerz) * (in_data[pidx*3 + 2]-centerz);
                        float xyz2 = xy2 + z2;
                        if ((radius_limit2 == 0 || xy2 <= radius_limit2) && (depth_limit2==0 || z2 <= depth_limit2)){
                            if (kid++ < K) {
                                sample_pidx[raysample_startid + kid - 1] = pidx;
                                xyz2Buffer[kid-1] = xyz2;
                                if (xyz2 > far2){
                                    far2 = xyz2;
                                    far_ind = kid - 1;
                                }
                            } else {
                                if (xyz2 < far2) {
                                    sample_pidx[raysample_startid + far_ind] = pidx;
                                    xyz2Buffer[far_ind] = xyz2;
                                    far2 = xyz2;
                                    for (int i = 0; i < K; i++) {
                                        if (xyz2Buffer[i] > far2) {
                                            far2 = xyz2Buffer[i];
                                            far_ind = i;
                                        }
                                    }
                                } 
                            }
                        }
                    }
                }
            }
        }
    }
}


void get_occ_vox_wrapper(
    const float *in_data,           // B * N * 3
    const int *in_actual_numpoints, // B
    const int B,
    const int N,
    const float *d_coord_shift, // 3
    const float *d_voxel_size,  // 3
    const int *d_grid_size,     // 3
    const int *kernel_size,     // 3
    const int pixel_size,
    const int grid_size_vol,
    uint8_t *coor_occ,         // B * 400 * 400 * 400
    int8_t *loc_coor_counter,  // B * 400 * 400 * 400
    int *near_depth_id_tensor, // B * 400 * 400
    int *far_depth_id_tensor,  // B * 400 * 400
    const int inverse,
    int grid, int block
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    get_occ_vox_kernel<<<grid, block, 0, stream>>>(
        in_data, in_actual_numpoints,
        B, N,
        d_coord_shift, d_voxel_size, d_grid_size, kernel_size,
        pixel_size, grid_size_vol,
        coor_occ, loc_coor_counter,
        near_depth_id_tensor, far_depth_id_tensor, 
        inverse
    );
    CUDA_CHECK_ERRORS();
}

void near_vox_full_wrapper(
    const int B,
    const int SR,
    const int *pixel_idx,
    const int R,
    const int *vscale,
    const int *d_grid_size,
    const int pixel_size,
    const int grid_size_vol,
    const int *kernel_size, // 3
    uint8_t *pixel_map,
    int8_t *ray_mask,                // B * R
    const uint8_t *coor_occ,         // B * 400 * 400 * 400
    int8_t *loc_coor_counter,        // B * 400 * 400 * 400
    const int *near_depth_id_tensor, // B * 400 * 400
    const int *far_depth_id_tensor,  // B * 400 * 400
    short *voxel_to_coorz_idx,       // B * 400 * 400 * SR
    int grid, int block
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    near_vox_full_kernel<<<grid, block, 0, stream>>>(
        B, SR, pixel_idx, R, vscale, d_grid_size, pixel_size,
        grid_size_vol, kernel_size, pixel_map, 
        ray_mask,  
        coor_occ, loc_coor_counter, 
        near_depth_id_tensor, far_depth_id_tensor,
        voxel_to_coorz_idx
    );
    CUDA_CHECK_ERRORS();
}

void insert_vox_points_wrapper(
    float *in_data,           // B * N * 3
    int *in_actual_numpoints, // B
    const int B,
    const int N,
    const int P,
    const int max_o,
    const int pixel_size,
    const int grid_size_vol,
    const float *d_coord_shift, // 3
    const int *d_grid_size,
    const float *d_voxel_size,      // 3
    const int8_t *loc_coor_counter, // B * 400 * 400 * 400
    short *voxel_pnt_counter,       // B * 400 * 400 * max_o
    int *voxel_to_pntidx,           // B * pixel_size * max_o * P
    unsigned long seconds,
    const int inverse,
    int grid, int block
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    insert_vox_points_kernel<<<grid, block, 0, stream>>>(
        in_data, in_actual_numpoints, 
        B, N, P, max_o, pixel_size, grid_size_vol,
        d_coord_shift, d_grid_size, d_voxel_size,
        loc_coor_counter, voxel_pnt_counter, voxel_to_pntidx,
        seconds, inverse
    );
    CUDA_CHECK_ERRORS();
}

void query_rand_along_ray_wrapper(
    const float *in_data, // B * N * 3
    const int B,
    const int SR, // num. samples along each ray e.g., 128
    const int R,  // e.g., 1024
    const int max_o,
    const int P,
    const int K, // num.  neighbors
    const int pixel_size,
    const int grid_size_vol,
    const float radius_limit2,
    const float depth_limit2,
    const float *d_coord_shift, // 3
    const int *d_grid_size,
    const float *d_voxel_size,     // 3
    const float *d_ray_voxel_size, // 3
    const int *vscale,             // 3
    const int *kernel_size,
    const int *pixel_idx,            // B * R * 2
    const int8_t *loc_coor_counter,  // B * 400 * 400 * 400
    const short *voxel_to_coorz_idx, // B * 400 * 400 * SR
    const short *voxel_pnt_counter,  // B * 400 * 400 * max_o
    const int *voxel_to_pntidx,      // B * pixel_size * max_o * P
    int *sample_pidx,                // B * R * SR * K
    float *sample_loc,               // B * R * SR * K
    unsigned long seconds,
    const int NN,
    const int inverse,
    int grid, int block
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    query_rand_along_ray_kernel<<<grid, block, 0, stream>>>(
        in_data,
        B, SR, R, max_o, P, K, pixel_size,
        grid_size_vol, radius_limit2, depth_limit2,
        d_coord_shift, d_grid_size, d_voxel_size, d_ray_voxel_size,
        vscale, kernel_size, pixel_idx, 
        loc_coor_counter, voxel_to_coorz_idx, 
        voxel_pnt_counter, voxel_to_pntidx, 
        sample_pidx, sample_loc,
        seconds, NN, inverse
    );
    CUDA_CHECK_ERRORS();
}

void query_neigh_along_ray_layered_h_wrapper(
    const float *in_data, // B * N * 3
    const int B,
    const int SR, // num. samples along each ray e.g., 128
    const int R,  // e.g., 1024
    const int max_o,
    const int P,
    const int K, // num.  neighbors
    const int pixel_size,
    const int grid_size_vol,
    const float radius_limit2,
    const float depth_limit2,
    const float *d_coord_shift, // 3
    const int *d_grid_size,
    const float *d_voxel_size,     // 3
    const float *d_ray_voxel_size, // 3
    const int *vscale,             // 3
    const int *kernel_size,
    const int *pixel_idx,            // B * R * 2
    const int8_t *loc_coor_counter,  // B * 400 * 400 * 400
    const short *voxel_to_coorz_idx, // B * 400 * 400 * SR
    const short *voxel_pnt_counter,  // B * 400 * 400 * max_o
    const int *voxel_to_pntidx,      // B * pixel_size * max_o * P
    int *sample_pidx,                // B * R * SR * K
    float *sample_loc,               // B * R * SR * K
    unsigned long seconds,
    const int NN,
    const int inverse,
    int grid, int block
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    query_neigh_along_ray_layered_h_kernel<<<grid, block, 0, stream>>>(
        in_data, B, SR, R, max_o, P, K,
        pixel_size, grid_size_vol,
        radius_limit2, depth_limit2,
        d_coord_shift, d_grid_size, d_voxel_size, d_ray_voxel_size,
        vscale, kernel_size, pixel_idx,
        loc_coor_counter, voxel_to_coorz_idx,
        voxel_pnt_counter, voxel_to_pntidx, 
        sample_pidx, sample_loc,
        seconds, NN, inverse
    );
    CUDA_CHECK_ERRORS();
};



