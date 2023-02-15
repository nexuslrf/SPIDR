
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

__device__ float2 RayAABBIntersection(
  const float3 &ori,
  const float3 &inv_dir,
  const float3 &bnd_min,
  const float3 &bnd_max) {

  float f_low = 0;
  float f_high = 100000.;
  float f_dim_low, f_dim_high, temp, inv_rays_d, start, b_min, b_max;

  for (int d = 0; d < 3; ++d) {  
    switch (d) {
      case 0:
        inv_rays_d = inv_dir.x; start = ori.x; b_min = bnd_min.x; b_max = bnd_max.x; break;
      case 1:
        inv_rays_d = inv_dir.y; start = ori.y; b_min = bnd_min.y; b_max = bnd_max.y; break;
      case 2:
        inv_rays_d = inv_dir.z; start = ori.z; b_min = bnd_min.z; b_max = bnd_max.z; break;
    }
  
    f_dim_low  = (b_min - start) * inv_rays_d; // pay attention to half_voxel --> center -> corner.
    f_dim_high = (b_max - start) * inv_rays_d;
  
    // Make sure low is less than high
    if (f_dim_high < f_dim_low) {
      temp = f_dim_low;
      f_dim_low = f_dim_high;
      f_dim_high = temp;
    }

    // If this dimension's high is less than the low we got then we definitely missed.
    if (f_dim_high < f_low) {
      return make_float2(-1.0f, -1.0f);
    }
  
    // Likewise if the low is less than the high.
    if (f_dim_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
      
    // Add the clip from this dimension to the previous results 
    f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
    f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
    
    if (f_low >= f_high) { // The corner case: if f_low == f_high, also treated as miss.
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
}



__global__ void claim_occ_kernel(
    const float* in_data,   // B * N * 3
    const int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const float *d_coord_shift,     // 3
    const float *d_voxel_size,      // 3
    const int *d_grid_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, // B, all 0
    int *coor_2_occ,  // B * 400 * 400 * 400, all -1
    int *occ_2_coor,  // B * max_o * 3, all -1
    unsigned long seconds
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_actual_numpoints[i_batch]) {
        int coor[3];
        const float *p_pt = in_data + index * 3;
        coor[0] = (int) floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
        coor[1] = (int) floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
        coor[2] = (int) floor((p_pt[2] - d_coord_shift[2]) / d_voxel_size[2]);
        // printf("p_pt %f %f %f %f; ", p_pt[2], d_coord_shift[2], d_coord_shift[0], d_coord_shift[1]);
        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        
        int voxel_idx = coor_2_occ[coor_indx_b];
        if (voxel_idx == -1) {  // found an empty voxel
            int old_voxel_num = atomicCAS(
                    &coor_2_occ[coor_indx_b],
                    -1, 0
            );
            if (old_voxel_num == -1) {
                // CAS -> old val, if old val is -1
                // if we get -1, this thread is the one who obtain a new voxel
                // so only this thread should do the increase operator below
                int tmp = atomicAdd(occ_idx+i_batch, 1); // increase the counter, return old counter
                    // increase the counter, return old counter
                if (tmp < max_o) {
                    int coord_inds = (i_batch * max_o + tmp) * 3;
                    occ_2_coor[coord_inds] = coor[0];
                    occ_2_coor[coord_inds + 1] = coor[1];
                    occ_2_coor[coord_inds + 2] = coor[2];
                } else {
                    curandState state;
                    curand_init(index+2*seconds, 0, 0, &state);
                    int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                    if(insrtidx < max_o){
                        int coord_inds = (i_batch * max_o + insrtidx) * 3;
                        occ_2_coor[coord_inds] = coor[0];
                        occ_2_coor[coord_inds + 1] = coor[1];
                        occ_2_coor[coord_inds + 2] = coor[2];
                    }
                }
            }
        }
    }
}

__global__ void map_coor2occ_kernel(
    const int B,
    const int *d_grid_size,       // 3
    const int *kernel_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, // B, all -1
    int *coor_occ,  // B * 400 * 400 * 400
    int *coor_2_occ,  // B * 400 * 400 * 400
    int *occ_2_coor  // B * max_o * 3
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / max_o;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - max_o * i_batch;
    if (i_pt < occ_idx[i_batch] && i_pt < max_o) {
        int coor[3];
        coor[0] = occ_2_coor[index*3];
        if (coor[0] < 0) { return; }
        coor[1] = occ_2_coor[index*3+1];
        coor[2] = occ_2_coor[index*3+2];
        
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        coor_2_occ[coor_indx_b] = i_pt;
        // printf("kernel_size[0] %d", kernel_size[0]);
        for (int coor_x = std::max(0, coor[0] - kernel_size[0] / 2) ; coor_x < std::min(d_grid_size[0], coor[0] + (kernel_size[0] + 1) / 2); coor_x++)    {
            for (int coor_y = std::max(0, coor[1] - kernel_size[1] / 2) ; coor_y < std::min(d_grid_size[1], coor[1] + (kernel_size[1] + 1) / 2); coor_y++)   {
                for (int coor_z = std::max(0, coor[2] - kernel_size[2] / 2) ; coor_z < std::min(d_grid_size[2], coor[2] + (kernel_size[2] + 1) / 2); coor_z++) {
                    coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                    if (coor_occ[coor_indx_b] > 0) { continue; }
                    atomicCAS(coor_occ + coor_indx_b, 0, 1);
                }
            }
        }   
    }
}

__global__ void fill_occ2pnts_kernel(
    const float* in_data,   // B * N * 3
    const int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const int P,
    const float *d_coord_shift,     // 3
    const float *d_voxel_size,      // 3
    const int *d_grid_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int *coor_2_occ,  // B * 400 * 400 * 400, all -1
    int *occ_2_pnts,  // B * max_o * P, all -1
    int *occ_numpnts,  // B * max_o, all 0
    unsigned long seconds
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_actual_numpoints[i_batch]) {
        int coor[3];
        const float *p_pt = in_data + index * 3;
        coor[0] = (int) floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
        coor[1] = (int) floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
        coor[2] = (int) floor((p_pt[2] - d_coord_shift[2]) / d_voxel_size[2]);
        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        
        int voxel_idx = coor_2_occ[coor_indx_b];
        if (voxel_idx > 0) {  // found an claimed coor2occ
            int occ_indx_b = i_batch * max_o + voxel_idx;
            int tmp = atomicAdd(occ_numpnts + occ_indx_b, 1); // increase the counter, return old counter
            if (tmp < P) {
                occ_2_pnts[occ_indx_b * P + tmp] = i_pt;
            } 
            else {
                curandState state;
                curand_init(index+2*seconds, 0, 0, &state);
                int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                if(insrtidx < P){
                    occ_2_pnts[occ_indx_b * P + insrtidx] = i_pt;
                }
            }
        }
    }
}

            
__global__ void mask_raypos_kernel(
    float *raypos,    // [B, 2048, 400, 3]
    int *coor_occ,    // B * 400 * 400 * 400
    const int B,       // 3
    const int R,       // 3
    const int D,       // 3
    const int grid_size_vol,
    const float *d_coord_shift,     // 3
    const int *d_grid_size,       // 3
    const float *d_voxel_size,      // 3
    int *raypos_mask    // B, R, D
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * D);  // index of batch
    if (i_batch >= B) { return; }
    int coor[3];
    coor[0] = (int) floor((raypos[index*3] - d_coord_shift[0]) / d_voxel_size[0]);
    coor[1] = (int) floor((raypos[index*3+1] - d_coord_shift[1]) / d_voxel_size[1]);
    coor[2] = (int) floor((raypos[index*3+2] - d_coord_shift[2]) / d_voxel_size[2]);
    // printf(" %f %f %f;", raypos[index*3], raypos[index*3+1], raypos[index*3+2]);
    if ((coor[0] >= 0) && (coor[0] < d_grid_size[0]) && (coor[1] >= 0) && (coor[1] < d_grid_size[1]) && (coor[2] >= 0) && (coor[2] < d_grid_size[2])) { 
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        raypos_mask[index] = coor_occ[coor_indx_b];
    }
}


__global__ void get_shadingloc_kernel(
    const float *raypos,    // [B, 2048, 400, 3]
    const int *raypos_mask,    // B, R, D
    const int B,       // 3
    const int R,       // 3
    const int D,       // 3
    const int SR,       // 3
    float *sample_loc,       // B * R * SR * 3
    int *sample_loc_mask       // B * R * SR
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * D);  // index of batch
    if (i_batch >= B) { return; }
    int temp = raypos_mask[index];
    if (temp >= 0) {
        int r = (index - i_batch * R * D) / D;
        int loc_inds = i_batch * R * SR + r * SR + temp;
        sample_loc[loc_inds * 3] = raypos[index * 3];
        sample_loc[loc_inds * 3 + 1] = raypos[index * 3 + 1];
        sample_loc[loc_inds * 3 + 2] = raypos[index * 3 + 2];
        sample_loc_mask[loc_inds] = 1;
    }
}


__global__ void query_neigh_along_ray_layered_kernel(
    const float* in_data,   // B * N * 3
    const int B,
    const int SR,               // num. samples along each ray e.g., 128
    const int R,               // e.g., 1024
    const int max_o,
    const int P,
    const int K,                // num.  neighbors
    const int grid_size_vol,
    const float radius_limit2,
    const float *d_coord_shift,     // 3
    const int *d_grid_size,
    const float *d_voxel_size,      // 3
    const int *kernel_size,
    const int *occ_numpnts,    // B * max_o
    const int *occ_2_pnts,            // B * max_o * P
    const int *coor_2_occ,      // B * 400 * 400 * 400 
    const float *sample_loc,       // B * R * SR * 3
    const int *sample_loc_mask,       // B * R * SR
    int *sample_pidx,       // B * R * SR * K
    unsigned long seconds,
    const int NN
) {
    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * SR);  // index of batch
    if (i_batch >= B || sample_loc_mask[index] <= 0) { return; }
    float centerx = sample_loc[index * 3];
    float centery = sample_loc[index * 3 + 1];
    float centerz = sample_loc[index * 3 + 2];
    int frustx = (int) floor((centerx - d_coord_shift[0]) / d_voxel_size[0]);
    int frusty = (int) floor((centery - d_coord_shift[1]) / d_voxel_size[1]);
    int frustz = (int) floor((centerz - d_coord_shift[2]) / d_voxel_size[2]);
                        
    // centerx = sample_loc[index * 3];
    // centery = sample_loc[index * 3 + 1];
    // centerz = sample_loc[index * 3 + 2];
                        
    int kid = 0, far_ind = 0, coor_z, coor_y, coor_x;
    float far2 = 0.0;
    float *xyz2Buffer = (float*)malloc(sizeof(float) * K);
    for (int layer = 0; layer < (kernel_size[0]+1)/2; layer++){                        
        for (int x = std::max(-frustx, -layer); x < std::min(d_grid_size[0] - frustx, layer + 1); x++) {
            coor_x = frustx + x;
            for (int y = std::max(-frusty, -layer); y < std::min(d_grid_size[1] - frusty, layer + 1); y++) {
                coor_y = frusty + y;
                for (int z =  std::max(-frustz, -layer); z < std::min(d_grid_size[2] - frustz, layer + 1); z++) {
                    coor_z = z + frustz;
                    if (std::max(std::abs(z), std::max(std::abs(x), std::abs(y))) != layer) continue;
                    int coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                    int occ_indx = coor_2_occ[coor_indx_b] + i_batch * max_o;
                    if (occ_indx >= 0) {
                        for (int g = 0; g < std::min(P, occ_numpnts[occ_indx]); g++) {
                            int pidx = occ_2_pnts[occ_indx * P + g];
                            float x_v = (in_data[pidx*3]-centerx);
                            float y_v = (in_data[pidx*3 + 1]-centery);
                            float z_v = (in_data[pidx*3 + 2]-centerz);
                            float xyz2 = x_v * x_v + y_v * y_v + z_v * z_v;
                            if ((radius_limit2 == 0 || xyz2 <= radius_limit2)){
                                if (kid++ < K) {
                                    sample_pidx[index * K + kid - 1] = pidx;
                                    xyz2Buffer[kid-1] = xyz2;
                                    if (xyz2 > far2){
                                        far2 = xyz2;
                                        far_ind = kid - 1;
                                    }
                                } else {
                                    if (xyz2 < far2) {
                                        sample_pidx[index * K + far_ind] = pidx;
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
        if (kid >= K) break;
    }
    free(xyz2Buffer);
}

__global__ void ray_voxel_intersect_kernel(
    const float *rays_d,
    const float *rays_o,
    const int B,
    const int R,
    const int max_hit,
    const int grid_size_vol,
    const float *ranges,     // 6 // TODO make sure ranges are corners instead of centers.
    const int *d_grid_size,       // 3
    const float *d_voxel_size,      // 3
    const int *coor_occ,
    float *t_near,
    float *t_far
) {
    // https://github.com/cgyurgyik/fast-voxel-traversal-algorithm/blob/master/overview/FastVoxelTraversalOverview.md
    int blk_index = blockIdx.x;
    int offset = blk_index * blockDim.x; // blockDim: threads per blk
    int index = threadIdx.x + offset;
    int stride = blockDim.x * gridDim.x; // gridDim: #blk
    int i_batch = 0;
    float t_start, t_end, min_tMax;
    float f_high = 100000.;
    int cnt, coor_indx, occ;
    float3 pnt, inv_dir, tMax, tDelta;
    int3 coor, step;
    for (int j = index; j < B * R; j+=stride)
    {
        i_batch = j / R;
        const float *dir = rays_d + j * 3;
        const float *ori = rays_o + i_batch * 3;
        // int *r2v = ray_2_occ + j * max_hit;
        float *t_n = t_near + j * max_hit;
        float *t_f = t_far + j * max_hit;
        // AABB on the outer whole grid
        inv_dir = make_float3(__fdividef(1.0f, dir[0]), 
                              __fdividef(1.0f, dir[1]), 
                              __fdividef(1.0f, dir[2]));
        float2 t_val = RayAABBIntersection(
            make_float3(ori[0], ori[1], ori[2]),
            inv_dir,
            make_float3(ranges[0], ranges[1], ranges[2]),
            make_float3(ranges[3], ranges[4], ranges[5])
        );
        if (t_val.x == -1.0f) {continue;}

        t_start = t_val.x;

        step.x = dir[0] > 0 ? 1: (dir[0] < 0 ? -1 : 0);
        step.y = dir[1] > 0 ? 1: (dir[1] < 0 ? -1 : 0);
        step.z = dir[2] > 0 ? 1: (dir[2] < 0 ? -1 : 0);

        tDelta.x = step.x * d_voxel_size[0] * inv_dir.x;
        tDelta.y = step.y * d_voxel_size[1] * inv_dir.y;
        tDelta.z = step.z * d_voxel_size[2] * inv_dir.z;

        pnt.x = ori[0] + t_start * dir[0];
        pnt.y = ori[1] + t_start * dir[1];
        pnt.z = ori[2] + t_start * dir[2];
        // get voxel loc
        coor.x = min(d_grid_size[0]-1, max(0, (int) floor((pnt.x - ranges[0]) / d_voxel_size[0])));
        coor.y = min(d_grid_size[1]-1, max(0, (int) floor((pnt.y - ranges[1]) / d_voxel_size[1])));
        coor.z = min(d_grid_size[2]-1, max(0, (int) floor((pnt.z - ranges[2]) / d_voxel_size[2])));
        // initial tMax values
        tMax.x = isnan(inv_dir.x)? f_high : ((coor.x+(step.x > 0)) * d_voxel_size[0] + ranges[0] - ori[0]) * inv_dir.x;
        tMax.y = isnan(inv_dir.y)? f_high : ((coor.y+(step.y > 0)) * d_voxel_size[1] + ranges[1] - ori[1]) * inv_dir.y;
        tMax.z = isnan(inv_dir.z)? f_high : ((coor.z+(step.z > 0)) * d_voxel_size[2] + ranges[2] - ori[2]) * inv_dir.z;
        // 
        cnt = -1;
        do {
            min_tMax = min(tMax.x, min(tMax.y, tMax.z));
            coor_indx = i_batch * grid_size_vol + coor.x * (d_grid_size[1] * d_grid_size[2]) + coor.y * d_grid_size[2] + coor.z;
            if (coor_occ[coor_indx] > 0) {
                // r2v[cnt] = coor_2_occ[coor_indx];
                if (cnt < 0 || t_f[cnt]<t_start) {
                    cnt++;
                    t_n[cnt] = t_start;
                }
                t_f[cnt] = min_tMax;
            }
            t_start = min_tMax;
            if (tMax.x == min_tMax){
                coor.x += step.x;
                if (coor.x >= d_grid_size[0] || coor.x < 0) {break;}
                tMax.x += tDelta.x;
            }
            if (tMax.y == min_tMax){
                coor.y += step.y;
                if (coor.y >= d_grid_size[1] || coor.y < 0) {break;}
                tMax.y += tDelta.y;
            }
            if (tMax.z == min_tMax){
                coor.z += step.z;
                if (coor.z >= d_grid_size[2] || coor.z < 0) {break;}
                tMax.z += tDelta.z;
            }
        } while(cnt+1 < max_hit);
    }
}

__global__ void seg_point_sample_kernel(
    const float *rays_d,
    const float *rays_o,
    const int B,
    const int R,
    const int max_pnts_per_ray,
    const int max_hit,
    const float step_size,
    float *t_near,
    float *t_far,
    float *noises, // [-0.5,+0.5]
    float *t_sample,
    float *seg_len
) {
    int blk_index = blockIdx.x;
    int offset = blk_index * blockDim.x;
    int index = threadIdx.x + offset;
    int stride = blockDim.x * gridDim.x;
    int i_batch = 0;
    float t_new, t_l, t_r;
    for (int j = index; j < B * R; j+=stride)
    {
        i_batch = j / R;
        const float *dir = rays_d + j * 3;
        const float *ori = rays_o + i_batch * 3;
        float *t_n = t_near + j * max_hit;
        float *t_f = t_far + j * max_hit;
        float *t_s = t_sample + j * max_pnts_per_ray;
        float *len = seg_len + j * max_pnts_per_ray;
        float *noise = noises + j * max_pnts_per_ray;
        // starts with a simple stratified sampling
        int s_cnt = 0, t_cnt = 0;
        t_l = t_n[t_cnt];
        t_new = t_n[t_cnt];
        for (;s_cnt < max_pnts_per_ray && t_new!=0; s_cnt++) {
            t_r = t_new + step_size + step_size * noise[s_cnt];
            t_r = max(t_n[t_cnt], min(t_f[t_cnt], t_r));
            len[s_cnt] = t_r - t_l;
            t_s[s_cnt] = (t_r + t_l) / 2;
            t_l = t_r;
            // t_s[s_cnt] = max(t_n[t_cnt], min(t_f[t_cnt], t_new + noise[s_cnt]));
            t_new = t_new + step_size;
            if (t_new > t_f[t_cnt]) {
                t_cnt++;
                t_new = t_n[t_cnt];
                t_l = t_n[t_cnt];
            }
        }
    }
}

__global__ void seg_point_sample_nonlinear_kernel(
    const float *rays_d,
    const float *rays_o,
    const int B,
    const int R,
    const int max_pnts_per_ray,
    const int max_hit,
    float step_size,
    float min_scale,
    float *t_ranges,
    float *t_near,
    float *t_far,
    float *noises, // [-0.5,+0.5]
    float *t_sample,
    float *seg_len
) {
    int blk_index = blockIdx.x;
    int offset = blk_index * blockDim.x;
    int index = threadIdx.x + offset;
    int stride = blockDim.x * gridDim.x;
    int i_batch = 0;
    float t_new, t_l, t_r;
    for (int j = index; j < B * R; j+=stride)
    {
        i_batch = j / R;
        const float *dir = rays_d + j * 3;
        const float *ori = rays_o + i_batch * 3;
        float *t_n = t_near + j * max_hit;
        float *t_f = t_far + j * max_hit;
        float *t_s = t_sample + j * max_pnts_per_ray;
        float *len = seg_len + j * max_pnts_per_ray;
        float *noise = noises + j * max_pnts_per_ray;
        float t_range = t_ranges[j];
        if (max_pnts_per_ray * step_size < t_range) {
            step_size = t_range / max_pnts_per_ray;
        }
        // starts with a simple stratified sampling
        int s_cnt = 0, t_cnt = -1, n_sample=0;
        float cur_step_size;
        float d_step_size;
        // for (;s_cnt < max_pnts_per_ray && t_new!=0; s_cnt++)
        do {
            if (t_cnt < 0 || t_new > t_f[t_cnt]) {
                t_cnt++;
                t_new = t_n[t_cnt];
                t_l = t_n[t_cnt];
                t_range = t_f[t_cnt] - t_n[t_cnt];
                n_sample = max((int) (t_range / step_size), 2);
                cur_step_size = min_scale * step_size;
                d_step_size = 2 * cur_step_size * (1/min_scale - 1) / (n_sample - 1);
            }
            if (step_size >= t_range) {
                len[s_cnt] = t_range;
                t_s[s_cnt] = (t_f[t_cnt] + t_n[t_cnt]) / 2 + noise[s_cnt] * t_range;
                t_new = t_f[t_cnt];
            }
            else {
                t_r = t_new + cur_step_size + cur_step_size * noise[s_cnt];
                t_r = max(t_n[t_cnt], min(t_f[t_cnt], t_r));
                len[s_cnt] = t_r - t_l;
                t_s[s_cnt] = (t_r + t_l) / 2;
                t_l = t_r;
            }
            // t_s[s_cnt] = max(t_n[t_cnt], min(t_f[t_cnt], t_new + noise[s_cnt]));
            t_new = t_new + cur_step_size;
            cur_step_size = cur_step_size + d_step_size;
            s_cnt++;
        } while(s_cnt < max_pnts_per_ray && t_new!=0);
    }
}

/////////////////////////////////////////

void claim_occ_wrapper(
    const float* in_data,   
    const int* in_actual_numpoints, 
    const int B,
    const int N,
    const float *d_coord_shift,     
    const float *d_voxel_size,      
    const int *d_grid_size,       
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, 
    int *coor_2_occ,  
    int *occ_2_coor,  
    unsigned long seconds,
    int grid, int block
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    claim_occ_kernel<<<grid, block, 0, stream>>>(in_data, in_actual_numpoints, B, N, 
            d_coord_shift, d_voxel_size, d_grid_size, grid_size_vol, max_o, 
            occ_idx, coor_2_occ, occ_2_coor, seconds);
    CUDA_CHECK_ERRORS();
}


void map_coor2occ_wrapper(
    const int B,
    const int *d_grid_size,       
    const int *kernel_size,       
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, 
    int *coor_occ,  
    int *coor_2_occ,  
    int *occ_2_coor,  
    int grid, int block
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    map_coor2occ_kernel<<<grid, block, 0, stream>>>(
        B, d_grid_size, kernel_size, grid_size_vol, max_o, 
        occ_idx, coor_occ, coor_2_occ, occ_2_coor
    );
    CUDA_CHECK_ERRORS();
}

void fill_occ2pnts_wrapper(
    const float* in_data,   
    const int* in_actual_numpoints, 
    const int B,
    const int N,
    const int P,
    const float *d_coord_shift,     
    const float *d_voxel_size,      
    const int *d_grid_size,       
    const int grid_size_vol,
    const int max_o,
    int *coor_2_occ,  
    int *occ_2_pnts,  
    int *occ_numpnts,
    unsigned long seconds,
    int grid, int block
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fill_occ2pnts_kernel<<<grid, block, 0, stream>>>(
        in_data, in_actual_numpoints, B, N, P, 
        d_coord_shift, d_voxel_size, d_grid_size, 
        grid_size_vol, max_o, 
        coor_2_occ, occ_2_pnts, occ_numpnts, seconds
    );
    CUDA_CHECK_ERRORS();
}

void mask_raypos_wrapper(
    float *raypos,    
    int *coor_occ,    
    const int B,       
    const int R,       
    const int D,       
    const int grid_size_vol,
    const float *d_coord_shift,     
    const int *d_grid_size,       
    const float *d_voxel_size,      
    int *raypos_mask,   
    int grid, int block
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mask_raypos_kernel<<<grid, block, 0, stream>>>(
        raypos, coor_occ, B, R, D, grid_size_vol, 
        d_coord_shift, d_grid_size, d_voxel_size, raypos_mask
    );
    CUDA_CHECK_ERRORS();
}

void get_shadingloc_wrapper(
    const float *raypos,    
    const int *raypos_mask,    
    const int B,       
    const int R,       
    const int D,       
    const int SR,       
    float *sample_loc,       
    int *sample_loc_mask,   
    int grid, int block
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    get_shadingloc_kernel<<<grid, block, 0, stream>>>(
        raypos, raypos_mask, B, R, D, SR, 
        sample_loc, sample_loc_mask
    );
    CUDA_CHECK_ERRORS();
}

void query_neigh_along_ray_layered_wrapper(
    const float* in_data,   
    const int B,
    const int SR,               
    const int R,               
    const int max_o,
    const int P,
    const int K,                
    const int grid_size_vol,
    const float radius_limit2,
    const float *d_coord_shift,     
    const int *d_grid_size,
    const float *d_voxel_size,      
    const int *kernel_size,
    const int *occ_numpnts,    
    const int *occ_2_pnts,            
    const int *coor_2_occ,      
    const float *sample_loc,       
    const int *sample_loc_mask,       
    int *sample_pidx,       
    unsigned long seconds,
    const int NN,   
    int grid, int block
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    query_neigh_along_ray_layered_kernel<<<grid, block, 0, stream>>>(
        in_data, B, SR, R, max_o, P, K, 
        grid_size_vol, radius_limit2, 
        d_coord_shift, d_grid_size, d_voxel_size, kernel_size, 
        occ_numpnts, occ_2_pnts, coor_2_occ, 
        sample_loc, sample_loc_mask, sample_pidx, 
        seconds, NN
    );
    CUDA_CHECK_ERRORS();
}

void ray_voxel_intersect_wrapper(
    const float *rays_d,
    const float *rays_o,
    const int B,
    const int R,
    const int max_hit,
    const int grid_size_vol,
    const float *ranges,     // 6 // TODO make sure ranges are corners instead of centers.
    const int *d_grid_size,       // 3
    const float *d_voxel_size,      // 3
    const int *coor_occ,
    float *t_near,
    float *t_far,
    int grid, int rays_per_blk
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ray_voxel_intersect_kernel<<<grid, rays_per_blk, 0, stream>>>(
        rays_d, rays_o, B, R, max_hit, grid_size_vol, 
        ranges, d_grid_size, d_voxel_size, 
        coor_occ, t_near, t_far
    );
    CUDA_CHECK_ERRORS();
}

void seg_point_sample_wrapper(
    const float *rays_d,
    const float *rays_o,
    const int B,
    const int R,
    const int max_pnts_per_ray,
    const int max_hit,
    const float step_size,
    float *t_near,
    float *t_far,
    float *noises, // [-1,+1]
    float *t_sample,
    float *seg_len,
    int grid, int rays_per_blk
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    seg_point_sample_kernel<<<grid, rays_per_blk, 0, stream>>>(
        rays_d, rays_o, B, R, max_pnts_per_ray, max_hit, step_size, 
        t_near, t_far, noises, t_sample, seg_len
    );
    CUDA_CHECK_ERRORS();
}

void seg_point_sample_nonlinear_wrapper(
    const float *rays_d,
    const float *rays_o,
    const int B,
    const int R,
    const int max_pnts_per_ray,
    const int max_hit,
    float step_size,
    float min_scale,
    float *t_ranges,
    float *t_near,
    float *t_far,
    float *noises, // [-0.5,+0.5]
    float *t_sample,
    float *seg_len,
    int grid, int rays_per_blk
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    seg_point_sample_nonlinear_kernel<<<grid, rays_per_blk, 0, stream>>>(
        rays_d, rays_o, B, R, max_pnts_per_ray, max_hit, step_size, 
        min_scale, t_ranges, t_near, t_far, noises, t_sample, seg_len
    );
    CUDA_CHECK_ERRORS();
}