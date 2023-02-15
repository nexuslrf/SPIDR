#pragma once

#include <torch/extension.h>
#include <utility>


void claim_occ(at::Tensor point_xyz_w_tensor, at::Tensor actual_numpoints_tensor, 
        int B, int N, 
        at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
        int grid_size_vol, int max_o,
        at::Tensor occ_idx_tensor, at::Tensor coor_2_occ_tensor, at::Tensor occ_2_coor_tensor,
        unsigned long seconds
);

void map_coor2occ(
    int B, at::Tensor scaled_vdim, at::Tensor kernel_size, 
    int grid_size_vol, int max_o,
    at::Tensor occ_idx_tensor, at::Tensor coor_occ_tensor, at::Tensor coor_2_occ_tensor, at::Tensor occ_2_coor_tensor
);

void fill_occ2pnts(
    at::Tensor point_xyz_w_tensor, at::Tensor actual_numpoints_tensor,
    int B, int N, int P, 
    at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
    int grid_size_vol, int max_o, 
    at::Tensor coor_2_occ_tensor, at::Tensor occ_2_pnts_tensor, at::Tensor occ_numpnts_tensor, 
    unsigned long seconds
);

void mask_raypos(
    at::Tensor raypos_tensor,
    at::Tensor coor_occ_tensor,
    int B, int R, int D, int grid_size_vol,
    at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
    at::Tensor raypos_mask_tensor
);

void get_shadingloc(
    at::Tensor raypos_tensor,
    at::Tensor raypos_mask_tensor,
    int B, int R, int D, int SR, 
    at::Tensor sample_loc_tensor,
    at::Tensor sample_loc_mask_tensor
);

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
    int NN
);

void ray_voxel_intersect(
    at::Tensor rays_d,
    at::Tensor rays_o,
    int B, int R, int max_hit, 
    int grid_size_vol, 
    at::Tensor ranges,     // 6 // TODO make sure ranges are corners instead of centers. int *d_grid_size,       // 3 float *d_voxel_size,      // 3 int *coor_occ, int *coor_2_occ, int rays_per_blk,
    at::Tensor scaled_vdim,
    at::Tensor scaled_vsize,
    at::Tensor coor_occ,
    at::Tensor t_near,
    at::Tensor t_far
);

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
    at::Tensor seg_len
);

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
    at::Tensor seg_len
);

void claim_occ_wrapper(
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
    unsigned long seconds,
    int grid, int block
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);