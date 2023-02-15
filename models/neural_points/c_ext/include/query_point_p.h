#pragma once

#include <torch/extension.h>
#include <utility>

void get_occ_vox(
	at::Tensor point_xyz_w_tensor, at::Tensor actual_numpoints_tensor,
	int B, int N,
	at::Tensor d_coord_shift, at::Tensor scaled_vsize, at::Tensor scaled_vdim,
	at::Tensor kernel_size,
	int pixel_size,
	int grid_size_vol,
	at::Tensor coor_occ_tensor,
	at::Tensor loc_coor_counter_tensor,
	at::Tensor near_depth_id_tensor,
	at::Tensor far_depth_id_tensor,
	int inverse
);

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
	at::Tensor voxel_to_coorz_idx
);

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
	int inverse);

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
	int inverse);

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
	int inverse);

void get_occ_vox_wrapper(
	const float* in_data,           // B * N * 3
	const int* in_actual_numpoints, // B
	const int B,
	const int N,
	const float* d_coord_shift, // 3
	const float* d_voxel_size,  // 3
	const int* d_grid_size,     // 3
	const int* kernel_size,     // 3
	const int pixel_size,
	const int grid_size_vol,
	uint8_t* coor_occ,         // B * 400 * 400 * 400
	int8_t* loc_coor_counter,  // B * 400 * 400 * 400
	int* near_depth_id_tensor, // B * 400 * 400
	int* far_depth_id_tensor,  // B * 400 * 400
	const int inverse,
	int grid, int block);

void near_vox_full_wrapper(
	const int B,
	const int SR,
	const int* pixel_idx,
	const int R,
	const int* vscale,
	const int* d_grid_size,
	const int pixel_size,
	const int grid_size_vol,
	const int* kernel_size, // 3
	uint8_t* pixel_map,
	int8_t* ray_mask,                // B * R
	const uint8_t* coor_occ,         // B * 400 * 400 * 400
	int8_t* loc_coor_counter,        // B * 400 * 400 * 400
	const int* near_depth_id_tensor, // B * 400 * 400
	const int* far_depth_id_tensor,  // B * 400 * 400
	short* voxel_to_coorz_idx,       // B * 400 * 400 * SR
	int grid, int block);

void insert_vox_points_wrapper(
	float* in_data,           // B * N * 3
	int* in_actual_numpoints, // B
	const int B,
	const int N,
	const int P,
	const int max_o,
	const int pixel_size,
	const int grid_size_vol,
	const float* d_coord_shift, // 3
	const int* d_grid_size,
	const float* d_voxel_size,      // 3
	const int8_t* loc_coor_counter, // B * 400 * 400 * 400
	short* voxel_pnt_counter,       // B * 400 * 400 * max_o
	int* voxel_to_pntidx,           // B * pixel_size * max_o * P
	unsigned long seconds,
	const int inverse,
	int grid, int block);

void query_rand_along_ray_wrapper(
	const float* in_data, // B * N * 3
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
	const float* d_coord_shift, // 3
	const int* d_grid_size,
	const float* d_voxel_size,     // 3
	const float* d_ray_voxel_size, // 3
	const int* vscale,             // 3
	const int* kernel_size,
	const int* pixel_idx,            // B * R * 2
	const int8_t* loc_coor_counter,  // B * 400 * 400 * 400
	const short* voxel_to_coorz_idx, // B * 400 * 400 * SR
	const short* voxel_pnt_counter,  // B * 400 * 400 * max_o
	const int* voxel_to_pntidx,      // B * pixel_size * max_o * P
	int* sample_pidx,                // B * R * SR * K
	float* sample_loc,               // B * R * SR * K
	unsigned long seconds,
	const int NN,
	const int inverse,
	int grid, int block);

void query_neigh_along_ray_layered_h_wrapper(
	const float* in_data, // B * N * 3
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
	const float* d_coord_shift, // 3
	const int* d_grid_size,
	const float* d_voxel_size,     // 3
	const float* d_ray_voxel_size, // 3
	const int* vscale,             // 3
	const int* kernel_size,
	const int* pixel_idx,            // B * R * 2
	const int8_t* loc_coor_counter,  // B * 400 * 400 * 400
	const short* voxel_to_coorz_idx, // B * 400 * 400 * SR
	const short* voxel_pnt_counter,  // B * 400 * 400 * max_o
	const int* voxel_to_pntidx,      // B * pixel_size * max_o * P
	int* sample_pidx,                // B * R * SR * K
	float* sample_loc,               // B * R * SR * K
	unsigned long seconds,
	const int NN,
	const int inverse,
	int grid, int block);