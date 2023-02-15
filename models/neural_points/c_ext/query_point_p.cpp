#include "query_point_p.h"
#include <utility>
#include <torch/script.h>

#define kMaxThreadsPerBlock 1024

void get_occ_vox(
    at::Tensor point_xyz_w_tensor,
    at::Tensor actual_numpoints_tensor,
    int B, int N,
    at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
    at::Tensor kernel_size,
    int pixel_size,
    int grid_size_vol,
    at::Tensor coor_occ_tensor,
    at::Tensor loc_coor_counter_tensor,
    at::Tensor near_depth_id_tensor,
    at::Tensor far_depth_id_tensor,
    int inverse)
{
    int grid = (B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    get_occ_vox_wrapper(
        point_xyz_w_tensor.data_ptr<float>(),
        actual_numpoints_tensor.data_ptr<int>(),
        B, N,
        d_coord_shift.data_ptr<float>(),
        scaled_vsize.data_ptr<float>(),
        scaled_vdim.data_ptr<int>(),
        kernel_size.data_ptr<int>(),
        pixel_size, grid_size_vol,
        coor_occ_tensor.data_ptr<uint8_t>(),
        loc_coor_counter_tensor.data_ptr<int8_t>(),
        near_depth_id_tensor.data_ptr<int>(),
        far_depth_id_tensor.data_ptr<int>(),
        inverse,
        grid, kMaxThreadsPerBlock);
    return;
};

void near_vox_full(
    int B, int SR,
    at::Tensor pixel_idx,
    int R,
    at::Tensor vscale,
    at::Tensor d_grid_size,
    int pixel_size,
    int grid_size_vol,
    at::Tensor kernel_size,
    at::Tensor pixel_map,
    at::Tensor ray_mask,
    at::Tensor coor_occ_tensor,
    at::Tensor loc_coor_counter_tensor,
    at::Tensor near_depth_id_tensor,
    at::Tensor far_depth_id_tensor,
    at::Tensor voxel_to_coorz_idx)
{
    int grid = (B * R + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    near_vox_full_wrapper(
        B, SR,
        pixel_idx.data_ptr<int>(),
        R,
        vscale.data_ptr<int>(),
        d_grid_size.data_ptr<int>(),
        pixel_size,
        grid_size_vol,
        kernel_size.data_ptr<int>(),
        pixel_map.data_ptr<uint8_t>(),
        ray_mask.data_ptr<int8_t>(),
        coor_occ_tensor.data_ptr<uint8_t>(),
        loc_coor_counter_tensor.data_ptr<int8_t>(),
        near_depth_id_tensor.data_ptr<int>(),
        far_depth_id_tensor.data_ptr<int>(),
        voxel_to_coorz_idx.data_ptr<short>(),
        grid, kMaxThreadsPerBlock);
    return;
};

void insert_vox_points(
    at::Tensor in_data_tensor,
    at::Tensor actual_numpoints_tensor,
    int B, int N, int P, int max_o,
    int pixel_size, int grid_size_vol,
    at::Tensor d_coord_shift,
    at::Tensor d_grid_size,
    at::Tensor d_voxel_size,
    at::Tensor loc_coor_counter,  // B * 400 * 400 * 400
    at::Tensor voxel_pnt_counter, // B * 400 * 400 * max_o
    at::Tensor voxel_to_pntidx,   // B * pixel_size * max_o * P
    unsigned long seconds,
    int inverse)
{
    int grid = (B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    insert_vox_points_wrapper(
        in_data_tensor.data_ptr<float>(),
        actual_numpoints_tensor.data_ptr<int>(),
        B, N, P, max_o, pixel_size, grid_size_vol,
        d_coord_shift.data_ptr<float>(),
        d_grid_size.data_ptr<int>(),
        d_voxel_size.data_ptr<float>(),
        loc_coor_counter.data_ptr<int8_t>(),
        voxel_pnt_counter.data_ptr<short>(),
        voxel_to_pntidx.data_ptr<int>(),
        seconds,
        inverse,
        grid, kMaxThreadsPerBlock);
    return;
};

void query_rand_along_ray(
    at::Tensor in_data, // B * N * 3
    int B,
    int SR, // num. samples along each ray e.g., 128
    int R,  // e.g., 1024
    int max_o,
    int P,
    int K, // num.  neighbors
    int pixel_size,
    int grid_size_vol,
    float radius_limit2,
    float depth_limit2,
    at::Tensor d_coord_shift, // 3
    at::Tensor d_grid_size,
    at::Tensor d_voxel_size,     // 3
    at::Tensor d_ray_voxel_size, // 3
    at::Tensor vscale,           // 3
    at::Tensor kernel_size,
    at::Tensor pixel_idx,          // B * R * 2
    at::Tensor loc_coor_counter,   // B * 400 * 400 * 400
    at::Tensor voxel_to_coorz_idx, // B * 400 * 400 * SR
    at::Tensor voxel_pnt_counter,  // B * 400 * 400 * max_o
    at::Tensor voxel_to_pntidx,    // B * pixel_size * max_o * P
    at::Tensor sample_pidx,        // B * R * SR * K
    at::Tensor sample_loc,         // B * R * SR * K
    unsigned long seconds,
    int NN,
    int inverse)
{
    int grid = (R * SR + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    query_rand_along_ray_wrapper(
        in_data.data_ptr<float>(),
        B, SR, R, max_o, P, K, pixel_size, grid_size_vol,
        radius_limit2, depth_limit2,
        d_coord_shift.data_ptr<float>(),
        d_grid_size.data_ptr<int>(),
        d_voxel_size.data_ptr<float>(),
        d_ray_voxel_size.data_ptr<float>(),
        vscale.data_ptr<int>(),
        kernel_size.data_ptr<int>(),
        pixel_idx.data_ptr<int>(),
        loc_coor_counter.data_ptr<int8_t>(),
        voxel_to_coorz_idx.data_ptr<short>(),
        voxel_pnt_counter.data_ptr<short>(),
        voxel_to_pntidx.data_ptr<int>(),
        sample_pidx.data_ptr<int>(),
        sample_loc.data_ptr<float>(),
        seconds, NN, inverse,
        grid, kMaxThreadsPerBlock);
    return;
};

void query_neigh_along_ray_layered_h(
    at::Tensor in_data, // B * N * 3
    int B,
    int SR, // num. samples along each ray e.g., 128
    int R,  // e.g., 1024
    int max_o,
    int P,
    int K, // num.  neighbors
    int pixel_size,
    int grid_size_vol,
    float radius_limit2,
    float depth_limit2,
    at::Tensor d_coord_shift, // 3
    at::Tensor d_grid_size,
    at::Tensor d_voxel_size,     // 3
    at::Tensor d_ray_voxel_size, // 3
    at::Tensor vscale,           // 3
    at::Tensor kernel_size,
    at::Tensor pixel_idx,          // B * R * 2
    at::Tensor loc_coor_counter,   // B * 400 * 400 * 400
    at::Tensor voxel_to_coorz_idx, // B * 400 * 400 * SR
    at::Tensor voxel_pnt_counter,  // B * 400 * 400 * max_o
    at::Tensor voxel_to_pntidx,    // B * pixel_size * max_o * P
    at::Tensor sample_pidx,        // B * R * SR * K
    at::Tensor sample_loc,         // B * R * SR * K
    unsigned long seconds,
    int NN,
    int inverse)
{
    int grid = (R * SR + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    query_neigh_along_ray_layered_h_wrapper(
        in_data.data_ptr<float>(),
        B, SR, R, max_o, P, K, pixel_size, grid_size_vol,
        radius_limit2, depth_limit2,
        d_coord_shift.data_ptr<float>(),
        d_grid_size.data_ptr<int>(),
        d_voxel_size.data_ptr<float>(),
        d_ray_voxel_size.data_ptr<float>(),
        vscale.data_ptr<int>(),
        kernel_size.data_ptr<int>(),
        pixel_idx.data_ptr<int>(),
        loc_coor_counter.data_ptr<int8_t>(),
        voxel_to_coorz_idx.data_ptr<short>(),
        voxel_pnt_counter.data_ptr<short>(),
        voxel_to_pntidx.data_ptr<int>(),
        sample_pidx.data_ptr<int>(),
        sample_loc.data_ptr<float>(),
        seconds, NN, inverse,
        grid, kMaxThreadsPerBlock);
    return;
};
