import os
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import torch
import pickle
import time
# import cupy
# import open3d.ml.tf as ml3d
# import frnn

from models.neural_points.c_ext import _ext
from data.load_blender import load_blender_data

# X = torch.cuda.FloatTensor(8)

class lighting_fast_querier():

    def __init__(self, device, opt):

        print("querier device", device, device.index)
        self.gpu = device.index
        self.opt = opt
        self.inverse = self.opt.inverse

    def get_hyperparameters(self, h, w, intrinsic, near_depth, far_depth, recache=True):
        # print("h,w,focal,near,far", h.shape, w.shape, focal.shape, near_depth.shape, far_depth.shape)
        # x_r = w / 2 / focal
        # y_r = h / 2 / focal
        # ranges = np.array([-x_r, -y_r, near_depth, x_r, y_r, far_depth], dtype=np.float32)
        # vdim = np.array([h, w, self.opt.z_depth_dim], dtype=np.int32)
        # vsize = np.array([2 * x_r / vdim[0], 2 * y_r / vdim[1], z_r / vdim[2]], dtype=np.float32)
        if recache or not hasattr(self, 'hyperparams_cache'):
            x_rl, x_rh = -intrinsic[0, 2] / intrinsic[0, 0], (w - intrinsic[0, 2]) / intrinsic[0, 0]
            y_rl, y_rh = -intrinsic[1, 2] / intrinsic[1, 1], (h - intrinsic[1, 2]) / intrinsic[1, 1],
            z_r = (far_depth - near_depth) if self.inverse == 0 else (1.0 / near_depth - 1.0 / far_depth)
            #  [-0.22929783 -0.1841962   2.125       0.21325193  0.17096843  4.525     ]
            ranges = np.array([x_rl, y_rl, near_depth, x_rh, y_rh, far_depth], dtype=np.float32) if self.inverse == 0 else np.array([x_rl, y_rl, 1.0 / far_depth, x_rh, y_rh, 1.0 / near_depth], dtype=np.float32)
            vdim = np.array([w, h, self.opt.z_depth_dim], dtype=np.int32)

            vsize = np.array([(x_rh - x_rl) / vdim[0], (y_rh - y_rl) / vdim[1], z_r / vdim[2]], dtype=np.float32)

            vscale = np.array(self.opt.vscale, dtype=np.int32)
            scaled_vdim = np.ceil(vdim / vscale).astype(np.int32)
            scaled_vsize = (vsize * vscale).astype(np.float32)

            ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = \
                [
                    torch.from_numpy(array).cuda()
                    for array in 
                    [
                        ranges, scaled_vsize, scaled_vdim, vscale, 
                        np.asarray(self.opt.kernel_size, dtype=np.int32),
                        np.asarray(self.opt.query_size, dtype=np.int32)
                    ]
                ]

            radius_limit, depth_limit = self.opt.radius_limit_scale * max(vsize[0], vsize[1]), self.opt.depth_limit_scale * vsize[2]
            self.hyperparams_cache = [radius_limit.astype(np.float32), depth_limit.astype(np.float32), \
                ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, \
                ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu]


        return self.hyperparams_cache


    def query_points(self, pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, 
                actual_numpoints_tensor, h, w, intrinsic, near_depth, far_depth, 
                ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor, build_occ=True):

        # print("attr", hasattr(self, "h"), self.opt.feedforward)
        #
        # if not hasattr(self, "h") or self.opt.feedforward > 0 or self.vscale != self.opt.vscale or self.kernel_size != self.opt.kernel_size:
        #     radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = self.get_hyperparameters(h, w, intrinsic, near_depth, far_depth)
        #     if self.opt.feedforward==0:
        #         self.radius_limit, self.depth_limit, self.ranges, self.vsize, self.vdim, self.scaled_vsize, self.scaled_vdim, self.vscale, self.range_gpu, self.scaled_vsize_gpu, self.scaled_vdim_gpu, self.vscale_gpu, self.kernel_size_gpu, self.kernel_size, self.query_size_gpu = radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, self.opt.kernel_size, query_size_gpu
        #
        # else:
        #     radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = self.radius_limit, self.depth_limit, self.ranges, self.vsize, self.vdim, self.scaled_vsize, self.scaled_vdim, self.vscale, self.range_gpu, self.scaled_vsize_gpu, self.scaled_vdim_gpu, self.vscale_gpu, self.kernel_size_gpu, self.query_size_gpu

        radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale,\
            range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu \
                = self.get_hyperparameters(h, w, intrinsic, near_depth, far_depth)

        sample_pidx_tensor, sample_loc_tensor, pixel_idx_cur_tensor, ray_mask_tensor = \
            self.query_grid_point_index(
                h, w,pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, 
                actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, 
                self.opt.SR, self.opt.K, 
                ranges, scaled_vsize, scaled_vdim, vscale, 
                self.opt.max_o, self.opt.P, 
                radius_limit, depth_limit, 
                range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, 
                kMaxThreadsPerBlock=self.opt.gpu_maxthr)

        self.inverse = self.opt.inverse

        if self.opt.is_train:
            sample_loc_tensor = getattr(self, self.opt.shpnt_jitter, None)(sample_loc_tensor, vsize)

        sample_loc_w_tensor, sample_ray_dirs_tensor = self.pers2w(sample_loc_tensor, cam_rot_tensor, cam_pos_tensor)

        return sample_pidx_tensor, sample_loc_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, ray_mask_tensor, vsize, ranges

    def pers2w(self, point_xyz_pers, camrotc2w, campos):
        #     point_xyz_pers    B X M X 3

        x_pers = point_xyz_pers[..., 0] * point_xyz_pers[..., 2]
        y_pers = point_xyz_pers[..., 1] * point_xyz_pers[..., 2]
        z_pers = point_xyz_pers[..., 2]
        xyz_c = torch.stack([x_pers, y_pers, z_pers], dim=-1)
        xyz_w_shift = torch.sum(xyz_c[...,None,:] * camrotc2w, dim=-1)
        # print("point_xyz_pers[..., 0, 0]", point_xyz_pers[..., 0, 0].shape, point_xyz_pers[..., 0, 0])
        ray_dirs = xyz_w_shift / (torch.linalg.norm(xyz_w_shift, dim=-1, keepdims=True) + 1e-7)

        xyz_w = xyz_w_shift + campos[:, None, :]
        return xyz_w, ray_dirs

    def gaussian(self, input, vsize):
        B, R, SR, _ = input.shape
        jitters = torch.normal(mean=torch.zeros([B, R, SR], dtype=torch.float32, device=input.device), std=torch.full([B, R, SR], vsize[2] / 4, dtype=torch.float32, device=input.device))
        input[..., 2] = input[..., 2] + torch.clamp(jitters, min=-vsize[2]/2, max=vsize[2]/2)
        return input

    def uniform(self, input, vsize):
        B, R, SR, _ = input.shape
        jitters = torch.rand([B, R, SR], dtype=torch.float32, device=input.device) - 0.5
        input[..., 2] = input[..., 2] + jitters * vsize[2]
        return input

    def switch_pixel_id(self, pixel_idx_tensor, h):
        pixel_id = torch.cat([pixel_idx_tensor[..., 0:1], h - 1 - pixel_idx_tensor[..., 1:2]], dim=-1)
        # print("pixel_id", pixel_id.shape, torch.min(pixel_id, dim=-2)[0], torch.max(pixel_id, dim=-2)[0])
        return pixel_id


    def query_grid_point_index(self, h, w, pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, SR, K, ranges, scaled_vsize, scaled_vdim, vscale, max_o, P, radius_limit, depth_limit, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kMaxThreadsPerBlock = 1024):

        device = point_xyz_pers_tensor.device
        B, N = point_xyz_pers_tensor.shape[0], point_xyz_pers_tensor.shape[1]
        pixel_size = scaled_vdim[0] * scaled_vdim[1]
        grid_size_vol = pixel_size * scaled_vdim[2]
        d_coord_shift = range_gpu[:3]
        # ray_vsize_gpu = (vsize_gpu / vscale_gpu).astype(np.float32)

        pixel_idx_cur_tensor = pixel_idx_tensor.reshape(B, -1, 2).clone()
        R = pixel_idx_cur_tensor.shape[1]

        # print("kernel_size_gpu {}, SR {}, K {}, ranges {}, scaled_vsize {}, scaled_vdim {}, vscale {}, max_o {}, P {}, radius_limit {}, depth_limit {}, range_gpu {}, scaled_vsize_gpu {}, scaled_vdim_gpu {}, vscale_gpu {} ".format(kernel_size_gpu, SR, K, ranges, scaled_vsize, scaled_vdim, vscale, max_o, P, radius_limit, depth_limit, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, pixel_idx_cur_tensor.shape))

        # print("point_xyz_pers_tensor", ranges, scaled_vdim_gpu, torch.min(point_xyz_pers_tensor, dim=-2)[0], torch.max(point_xyz_pers_tensor, dim=-2)[0])


        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        coor_occ_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], scaled_vdim[2]], dtype=torch.uint8, device=device)
        loc_coor_counter_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], scaled_vdim[2]], dtype=torch.int8, device=device)
        near_depth_id_tensor = torch.full([B, scaled_vdim[0], scaled_vdim[1]], scaled_vdim[2], dtype=torch.int32, device=device)
        far_depth_id_tensor = torch.full([B, scaled_vdim[0], scaled_vdim[1]], -1, dtype=torch.int32, device=device)

        _ext.get_occ_vox(
            point_xyz_pers_tensor,
            actual_numpoints_tensor,
            np.int32(B),
            np.int32(N),
            d_coord_shift,
            scaled_vsize_gpu,
            scaled_vdim_gpu,
            query_size_gpu,
            np.int32(pixel_size),
            np.int32(grid_size_vol),
            coor_occ_tensor,
            loc_coor_counter_tensor,
            near_depth_id_tensor,
            far_depth_id_tensor,
            np.int32(self.inverse)
        )

        # torch.cuda.synchronize()
        # print("near_depth_id_tensor", torch.min(near_depth_id_tensor), torch.max(far_depth_id_tensor),torch.max(loc_coor_counter_tensor), torch.max(torch.sum(coor_occ_tensor, dim=-1)), B*scaled_vdim[0]* scaled_vdim[1]*SR, pixel_size, scaled_vdim, vscale, scaled_vdim_gpu)

        gridSize = int((B * R + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        voxel_to_coorz_idx_tensor = torch.full([B, scaled_vdim[0], scaled_vdim[1], SR], -1, dtype=torch.int16, device=device)
        pixel_map_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1]], dtype=torch.uint8, device=device)
        ray_mask_tensor = torch.zeros([B, R], dtype=torch.int8, device=device)
        _ext.near_vox_full(
            np.int32(B),
            np.int32(SR),
            # Holder(self.switch_pixel_id(pixel_idx_cur_tensor,h)),
            pixel_idx_cur_tensor,
            np.int32(R),
            vscale_gpu,
            scaled_vdim_gpu,
            np.int32(pixel_size),
            np.int32(grid_size_vol),
            query_size_gpu,
            pixel_map_tensor,
            ray_mask_tensor,
            coor_occ_tensor,
            loc_coor_counter_tensor,
            near_depth_id_tensor,
            far_depth_id_tensor,
            voxel_to_coorz_idx_tensor,
        )

        # torch.cuda.synchronize()
        # print("voxel_to_coorz_idx_tensor max", torch.max(torch.sum(voxel_to_coorz_idx_tensor > -1, dim=-1)))
        # print("scaled_vsize_gpu",scaled_vsize_gpu, scaled_vdim_gpu)
        # print("ray_mask_tensor",ray_mask_tensor.shape, torch.min(ray_mask_tensor), torch.max(ray_mask_tensor))
        # print("pixel_idx_cur_tensor",pixel_idx_cur_tensor.shape, torch.min(pixel_idx_cur_tensor), torch.max(pixel_idx_cur_tensor))

        pixel_id_num_tensor = torch.sum(ray_mask_tensor, dim=-1)
        pixel_idx_cur_tensor = torch.masked_select(pixel_idx_cur_tensor, (ray_mask_tensor > 0)[..., None].expand(-1, -1, 2)).reshape(1, -1, 2)
        del coor_occ_tensor, near_depth_id_tensor, far_depth_id_tensor, pixel_map_tensor

        R = torch.max(pixel_id_num_tensor).cpu().numpy()
        # print("loc_coor_counter_tensor",loc_coor_counter_tensor.shape)
        loc_coor_counter_tensor = (loc_coor_counter_tensor > 0).to(torch.int8)
        loc_coor_counter_tensor = loc_coor_counter_tensor * torch.cumsum(loc_coor_counter_tensor, dtype=torch.int8, dim=-1) - 1

        if max_o is None:
            max_o = torch.max(loc_coor_counter_tensor).cpu().numpy().astype(np.int32) + 1
        # print("max_o", max_o)

        voxel_pnt_counter_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], max_o], dtype=torch.int16, device=device)
        voxel_to_pntidx_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], max_o, P], dtype=torch.int32, device=device)
        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        ray_vsize_gpu = (scaled_vsize_gpu / vscale_gpu).float()

        seconds = time.time()
        _ext.insert_vox_points(
            point_xyz_pers_tensor,
            actual_numpoints_tensor,
            np.int32(B),
            np.int32(N),
            np.int32(P),
            np.int32(max_o),
            np.int32(pixel_size),
            np.int32(grid_size_vol),
            d_coord_shift,
            scaled_vdim_gpu,
            scaled_vsize_gpu,
            loc_coor_counter_tensor,
            voxel_pnt_counter_tensor,
            voxel_to_pntidx_tensor,
            np.uint64(seconds),
            np.int32(self.inverse),
        )

        # torch.cuda.synchronize()
        # print("loc_coor_counter_tensor",loc_coor_counter_tensor.shape, torch.min(loc_coor_counter_tensor), torch.max(loc_coor_counter_tensor))
        # print("voxel_pnt_counter_tensor",voxel_pnt_counter_tensor.shape, torch.min(voxel_pnt_counter_tensor), torch.max(voxel_pnt_counter_tensor))
        # print("voxel_to_pntidx_tensor",voxel_to_pntidx_tensor.shape, torch.min(voxel_to_pntidx_tensor), torch.max(voxel_to_pntidx_tensor))

        sample_pidx_tensor = torch.full([B, R, SR, K], -1, dtype=torch.int32, device=device)
        sample_loc_tensor = torch.full([B, R, SR, 3], 0.0, dtype=torch.float32, device=device)
        gridSize = int((R * SR + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        seconds = time.time()


        # print(point_xyz_pers_tensor.shape, B, SR, R ,max_o, P, K, pixel_size, grid_size_vol, radius_limit, depth_limit, d_coord_shift, scaled_vdim_gpu, scaled_vsize_gpu, ray_vsize_gpu, vscale_gpu, kernel_size_gpu, pixel_idx_cur_tensor.shape, loc_coor_counter_tensor.shape, voxel_to_coorz_idx_tensor.shape, voxel_pnt_counter_tensor.shape, voxel_to_pntidx_tensor.shape, sample_pidx_tensor.shape, sample_loc_tensor.shape, gridSize)
        if R > 0:
            query_along_ray = _ext.query_neigh_along_ray_layered_h if self.opt.NN > 0 else _ext.query_rand_along_ray
            query_along_ray(
                point_xyz_pers_tensor,
                np.int32(B),
                np.int32(SR),
                np.int32(R),
                np.int32(max_o),
                np.int32(P),
                np.int32(K),
                np.int32(pixel_size),
                np.int32(grid_size_vol),
                np.float32(radius_limit ** 2),
                np.float32(depth_limit ** 2),
                d_coord_shift,
                scaled_vdim_gpu,
                scaled_vsize_gpu,
                ray_vsize_gpu,
                vscale_gpu,
                kernel_size_gpu,
                # self.switch_pixel_id(pixel_idx_cur_tensor,h),
                pixel_idx_cur_tensor,
                loc_coor_counter_tensor,
                voxel_to_coorz_idx_tensor,
                voxel_pnt_counter_tensor,
                voxel_to_pntidx_tensor,
                sample_pidx_tensor,
                sample_loc_tensor,
                np.uint64(seconds),
                np.int32(self.opt.NN),
                np.int32(self.inverse),
            )


        # torch.cuda.synchronize()
        # print("max_o", max_o)
        # print("voxel_pnt_counter", torch.max(voxel_pnt_counter_tensor))
        # print("sample_pidx_tensor", torch.max(torch.sum(sample_pidx_tensor >= 0, dim=-1)))
        # print("sample_pidx_tensor min max", torch.min(sample_pidx_tensor), torch.max(sample_pidx_tensor))

        # print("sample_pidx_tensor", sample_pidx_tensor.shape, sample_pidx_tensor[0,80,3], sample_pidx_tensor[0,80,6], sample_pidx_tensor[0,80,9])


        # print("sample_pidx_tensor, sample_loc_tensor, pixel_idx_cur_tensor, ray_mask_tensor", sample_pidx_tensor.shape, sample_loc_tensor.shape, pixel_idx_cur_tensor.shape, ray_mask_tensor.shape)


        return sample_pidx_tensor, sample_loc_tensor, pixel_idx_cur_tensor, ray_mask_tensor


def load_pnts(point_path, point_num):
    with open(point_path, 'rb') as f:
        print("point_file_path################", point_path)
        all_infos = pickle.load(f)
        point_xyz = all_infos["point_xyz"]
    print(len(point_xyz), point_xyz.dtype, np.mean(point_xyz, axis=0), np.min(point_xyz, axis=0),
          np.max(point_xyz, axis=0))
    np.random.shuffle(point_xyz)
    return point_xyz[:min(len(point_xyz), point_num), :]


def np_to_gpuarray(*args):
    result = []
    for x in args:
        if isinstance(x, np.ndarray):
            result.append(pycuda.gpuarray.to_gpu(x))
        else:
            print("trans",x)
    return result


def try_build(point_file, point_dir, ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, split=["train"], imgidx=0, gpu=0):
    point_path = os.path.join(point_dir, point_file)
    point_xyz = load_pnts(point_path, 819200000)  # 81920   233872
    imgs, poses, _, hwf, _ = load_blender_data(
        os.path.expandvars("${nrDataRoot}") + "/nerf/nerf_synthetic/{}".format(obj), split, half_res=False, testskip=1)
    H, W, focal = hwf
    plt.figure()
    plt.imshow(imgs[imgidx])
    point_xyz_pers = w2img(point_xyz, poses[imgidx], focal)
    point_xyz_tensor = torch.as_tensor(point_xyz, device="cuda:{}".format(gpu))[None, ...]
    # plt.show()
    point_xyz_pers_tensor = torch.as_tensor(point_xyz_pers, device="cuda:{}".format(gpu))[None, ...]
    actual_numpoints_tensor = torch.ones([1], device=point_xyz_tensor.device, dtype=torch.int32) * len(point_xyz)
    scaled_vsize = (vsize * vscale).astype(np.float32)
    scaled_vdim = np.ceil(vdim / vscale).astype(np.int32)
    print("vsize", vsize, "vdim", vdim, "scaled_vdim", scaled_vdim)
    range_gpu, vsize_gpu, vdim_gpu, vscale_gpu, kernel_size_gpu = np_to_gpuarray(ranges, scaled_vsize, scaled_vdim, vscale, kernel_size)
    pixel_idx_tensor = torch.as_tensor(pixel_idx, device="cuda:{}".format(gpu), dtype=torch.int32)[None, ...]
    sample_pidx_tensor, pixel_idx_cur_tensor = build_grid_point_index(pixel_idx_tensor, point_xyz_pers_tensor, actual_numpoints_tensor, kernel_size_gpu, SR, K, ranges, scaled_vsize, scaled_vdim, vscale, max_o, P, radius_limit, depth_limit, range_gpu, vsize_gpu, vdim_gpu, vscale_gpu, gpu=gpu)

    save_queried_points(point_xyz_tensor, point_xyz_pers_tensor, sample_pidx_tensor, pixel_idx_tensor,
                        pixel_idx_cur_tensor, vdim, vsize, ranges)


def w2img(point_xyz, transform_matrix, focal):
    camrot = transform_matrix[:3, :3]  # world 2 cam
    campos = transform_matrix[:3, 3]  #
    point_xyz_shift = point_xyz - campos[None, :]
    # xyz = np.sum(point_xyz_shift[:,None,:] * camrot.T, axis=-1)
    xyz = np.sum(camrot[None, ...] * point_xyz_shift[:, :, None], axis=-2)
    # print(xyz.shape, np.sum(camrot[None, None, ...] * point_xyz_shift[:,:,None], axis=-2).shape)
    xper = xyz[:, 0] / -xyz[:, 2]
    yper = xyz[:, 1] / xyz[:, 2]
    x_pixel = np.round(xper * focal + 400).astype(np.int32)
    y_pixel = np.round(yper * focal + 400).astype(np.int32)
    print("focal", focal, np.tan(.5 * 0.6911112070083618))
    print("pixel xmax xmin:", np.max(x_pixel), np.min(x_pixel), "pixel ymax ymin:", np.max(y_pixel), np.min(y_pixel))
    print("per xmax xmin:", np.max(xper), np.min(xper), "per ymax ymin:", np.max(yper), np.min(yper), "per zmax zmin:",
          np.max(xyz[:, 2]), np.min(xyz[:, 2]))
    print("min perx", -400 / focal, "max perx", 400 / focal)
    background = np.ones([800, 800, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .2

    plt.figure()
    plt.imshow(background)

    return np.stack([xper, yper, -xyz[:, 2]], axis=-1)





def render_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
    pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
    print(pixel_xy_inds.shape)
    y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .5
    plt.figure()
    plt.imshow(background)


def save_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
    pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
    print(pixel_xy_inds.shape)
    y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .5
    image_dir = os.path.join(self.opt.checkpoints_dir, opt.name, 'images')
    image_file = os.path.join(image_dir)


def render_pixel_mask(pixel_xy_inds, w, h):
    y_pixel, x_pixel = pixel_xy_inds[0, :, 1], pixel_xy_inds[0, :, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .0
    plt.figure()
    plt.imshow(background)


def save_queried_points(point_xyz_tensor, point_xyz_pers_tensor, sample_pidx_tensor, pixel_idx_tensor,
                        pixel_idx_cur_tensor, vdim, vsize, ranges):
    B, R, SR, K = sample_pidx_tensor.shape
    # pixel_inds = torch.as_tensor([3210, 3217,3218,3219,3220, 3221,3222,3223,3224,3225,3226,3227,3228,3229,3230, 3231,3232,3233,3234,3235, 3236,3237,3238,3239,3240], device=sample_pidx_tensor.device, dtype=torch.int64)
    point_inds = sample_pidx_tensor[0, :, :, :]
    # point_inds = sample_pidx_tensor[0, pixel_inds, :, :]
    mask = point_inds > -1
    point_inds = torch.masked_select(point_inds, mask).to(torch.int64)
    queried_point_xyz_tensor = point_xyz_tensor[0, point_inds, :]
    queried_point_xyz = queried_point_xyz_tensor.cpu().numpy()
    print("queried_point_xyz.shape", B, R, SR, K, point_inds.shape, queried_point_xyz_tensor.shape,
          queried_point_xyz.shape)
    print("pixel_idx_cur_tensor", pixel_idx_cur_tensor.shape)
    render_pixel_mask(pixel_idx_cur_tensor.cpu().numpy(), vdim[0], vdim[1])

    render_mask_pers_points(point_xyz_pers_tensor[0, point_inds, :].cpu().numpy(), vsize, ranges, vdim[0], vdim[1])

    plt.show()



if __name__ == "__main__":
    obj = "lego"
    point_file = "{}.pkl".format(obj)
    point_dir = os.path.expandvars("${nrDataRoot}/nerf/nerf_synthetic_points/")
    r = 0.36000002589322094
    ranges = np.array([-r, -r, 2., r, r, 6.], dtype=np.float32)
    vdim = np.array([800, 800, 400], dtype=np.int32)
    vsize = np.array([2 * r / vdim[0], 2 * r / vdim[1], 4. / vdim[2]], dtype=np.float32)
    vscale = np.array([2, 2, 1], dtype=np.int32)
    SR = 24
    P = 16
    kernel_size = np.array([5, 5, 1], dtype=np.int32)
    radius_limit = 0  # r / 400 * 5 #r / 400 * 5
    depth_limit = 0  # 4. / 400 * 1.5 # r / 400 * 2
    max_o = None
    K = 32

    xrange = np.arange(0, 800, 1, dtype=np.int32)
    yrange = np.arange(0, 800, 1, dtype=np.int32)
    xv, yv = np.meshgrid(xrange, yrange, sparse=False, indexing='ij')
    pixel_idx = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # 20000 * 2
    gpu = 0
    imgidx = 3
    split = ["train"]

    try_build(point_file, point_dir, ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, split=split, imgidx=imgidx, gpu=0)