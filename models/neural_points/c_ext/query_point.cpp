#include "query_point.h"
#include <utility>
#include <torch/script.h>

#define kMaxThreadsPerBlock 1024

void claim_occ(at::Tensor point_xyz_w_tensor, at::Tensor actual_numpoints_tensor,
               int B, int N,
               at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
               int grid_size_vol, int max_o,
               at::Tensor occ_idx_tensor, at::Tensor coor_2_occ_tensor, at::Tensor occ_2_coor_tensor,
               unsigned long seconds)
{
    // auto device = at::device(point_xyz_w_tensor.device());
    int grid = (B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    claim_occ_wrapper(
        point_xyz_w_tensor.data_ptr<float>(), actual_numpoints_tensor.data_ptr<int>(),
        B, N, d_coord_shift.data_ptr<float>(), scaled_vsize.data_ptr<float>(),
        scaled_vdim.data_ptr<int>(), grid_size_vol, max_o, occ_idx_tensor.data_ptr<int>(), coor_2_occ_tensor.data_ptr<int>(),
        occ_2_coor_tensor.data_ptr<int>(), seconds, grid, kMaxThreadsPerBlock);
    return;
}

void map_coor2occ(
    int B, at::Tensor scaled_vdim, at::Tensor kernel_size,
    int grid_size_vol, int max_o,
    at::Tensor occ_idx_tensor, at::Tensor coor_occ_tensor, at::Tensor coor_2_occ_tensor, at::Tensor occ_2_coor_tensor)
{
    int grid = (B * max_o + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    map_coor2occ_wrapper(
        B, scaled_vdim.data_ptr<int>(), kernel_size.data_ptr<int>(), grid_size_vol, max_o,
        occ_idx_tensor.data_ptr<int>(), coor_occ_tensor.data_ptr<int>(),
        coor_2_occ_tensor.data_ptr<int>(), occ_2_coor_tensor.data_ptr<int>(),
        grid, kMaxThreadsPerBlock);
    return;
}

void fill_occ2pnts(
    at::Tensor point_xyz_w_tensor, at::Tensor actual_numpoints_tensor,
    int B, int N, int P,
    at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
    int grid_size_vol, int max_o,
    at::Tensor coor_2_occ_tensor, at::Tensor occ_2_pnts_tensor, at::Tensor occ_numpnts_tensor,
    unsigned long seconds)
{
    int grid = (B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    fill_occ2pnts_wrapper(
        point_xyz_w_tensor.data_ptr<float>(),
        actual_numpoints_tensor.data_ptr<int>(),
        B, N, P,
        d_coord_shift.data_ptr<float>(), scaled_vsize.data_ptr<float>(),
        scaled_vdim.data_ptr<int>(), grid_size_vol, max_o,
        coor_2_occ_tensor.data_ptr<int>(), occ_2_pnts_tensor.data_ptr<int>(), occ_numpnts_tensor.data_ptr<int>(),
        seconds,
        grid, kMaxThreadsPerBlock);
    return;
}

void mask_raypos(
    at::Tensor raypos_tensor,
    at::Tensor coor_occ_tensor,
    int B, int R, int D, int grid_size_vol,
    at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
    at::Tensor raypos_mask_tensor)
{
    int grid = (B * R * D + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    mask_raypos_wrapper(
        raypos_tensor.data_ptr<float>(),
        coor_occ_tensor.data_ptr<int>(),
        B, R, D, grid_size_vol,
        d_coord_shift.data_ptr<float>(),
        scaled_vsize.data_ptr<int>(),
        scaled_vdim.data_ptr<float>(),
        raypos_mask_tensor.data_ptr<int>(),
        grid, kMaxThreadsPerBlock);
    return;
}

void get_shadingloc(
    at::Tensor raypos_tensor,
    at::Tensor raypos_mask_tensor,
    int B, int R, int D, int SR,
    at::Tensor sample_loc_tensor,
    at::Tensor sample_loc_mask_tensor)
{
    int grid = (B * R * D + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    get_shadingloc_wrapper(
        raypos_tensor.data_ptr<float>(),
        raypos_mask_tensor.data_ptr<int>(),
        B, R, D, SR,
        sample_loc_tensor.data_ptr<float>(),
        sample_loc_mask_tensor.data_ptr<int>(),
        grid, kMaxThreadsPerBlock);
    return;
}

void query_neigh_along_ray_layered(
    at::Tensor point_xyz_w_tensor,
    int B, int SR, int R, int max_o, int P, int K,
    int grid_size_vol, float radius_limit2,
    at::Tensor d_coord_shift,
    at::Tensor scaled_vdim,
    at::Tensor scaled_vsize,
    at::Tensor kernel_size,
    at::Tensor occ_numpnts_tensor,
    at::Tensor occ_2_pnts_tensor,
    at::Tensor coor_2_occ_tensor,
    at::Tensor sample_loc_tensor,
    at::Tensor sample_loc_mask_tensor,
    at::Tensor sample_pidx_tensor,
    unsigned long seconds,
    int NN)
{
    int grid = (B * R * SR + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    query_neigh_along_ray_layered_wrapper(
        point_xyz_w_tensor.data_ptr<float>(),
        B, SR, R, max_o, P, K, grid_size_vol, radius_limit2,
        d_coord_shift.data_ptr<float>(),
        scaled_vdim.data_ptr<int>(),
        scaled_vsize.data_ptr<float>(),
        kernel_size.data_ptr<int>(),
        occ_numpnts_tensor.data_ptr<int>(),
        occ_2_pnts_tensor.data_ptr<int>(),
        coor_2_occ_tensor.data_ptr<int>(),
        sample_loc_tensor.data_ptr<float>(),
        sample_loc_mask_tensor.data_ptr<int>(),
        sample_pidx_tensor.data_ptr<int>(),
        seconds, NN,
        grid, kMaxThreadsPerBlock);
    return;
}

void ray_voxel_intersect(
    at::Tensor rays_d,
    at::Tensor rays_o,
    int B, int R, int max_hit,
    int grid_size_vol,
    at::Tensor ranges, // 6 // TODO make sure ranges are corners instead of centers. int *d_grid_size,       // 3 float *d_voxel_size,      // 3 int *coor_occ, int *coor_2_occ, int rays_per_blk,
    at::Tensor scaled_vdim,
    at::Tensor scaled_vsize,
    at::Tensor coor_occ,
    at::Tensor t_near,
    at::Tensor t_far)
{
    int rays_per_blk = 256;
    int n_blocks = (R - 1) / rays_per_blk + 1;
    ray_voxel_intersect_wrapper(
        rays_d.data_ptr<float>(),
        rays_o.data_ptr<float>(),
        B, R, max_hit, grid_size_vol,
        ranges.data_ptr<float>(),
        scaled_vdim.data_ptr<int>(),
        scaled_vsize.data_ptr<float>(),
        coor_occ.data_ptr<int>(),
        t_near.data_ptr<float>(),
        t_far.data_ptr<float>(),
        n_blocks, rays_per_blk);
    return;
}

void seg_point_sample(
    at::Tensor rays_d,
    at::Tensor rays_o,
    int B,
    int R,
    int max_pnts_per_ray,
    int max_hit,
    float step_size,
    at::Tensor t_near,
    at::Tensor t_far,
    at::Tensor noises, // [-1,+1]
    at::Tensor t_sample,
    at::Tensor seg_len)
{
    int rays_per_blk = 256;
    int n_blocks = (R - 1) / rays_per_blk + 1;
    seg_point_sample_wrapper(
        rays_d.data_ptr<float>(),
        rays_o.data_ptr<float>(),
        B, R, max_pnts_per_ray, max_hit, step_size,
        t_near.data_ptr<float>(),
        t_far.data_ptr<float>(),
        noises.data_ptr<float>(),
        t_sample.data_ptr<float>(),
        seg_len.data_ptr<float>(),
        n_blocks, rays_per_blk);
    return;
}

void seg_point_sample_nonlinear(
    at::Tensor rays_d,
    at::Tensor rays_o,
    int B,
    int R,
    int max_pnts_per_ray,
    int max_hit,
    float step_size,
    float min_scale,
    at::Tensor t_ranges,
    at::Tensor t_near,
    at::Tensor t_far,
    at::Tensor noises, // [-1,+1]
    at::Tensor t_sample,
    at::Tensor seg_len)
{
    int rays_per_blk = 256;
    int n_blocks = (R - 1) / rays_per_blk + 1;
    seg_point_sample_nonlinear_wrapper(
        rays_d.data_ptr<float>(),
        rays_o.data_ptr<float>(),
        B, R, max_pnts_per_ray, max_hit, step_size, min_scale,
        t_ranges.data_ptr<float>(),
        t_near.data_ptr<float>(),
        t_far.data_ptr<float>(),
        noises.data_ptr<float>(),
        t_sample.data_ptr<float>(),
        seg_len.data_ptr<float>(),
        n_blocks, rays_per_blk);
    return;
}