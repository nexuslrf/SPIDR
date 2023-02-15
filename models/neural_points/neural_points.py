import os
import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from .query_point_indices import lighting_fast_querier as lighting_fast_querier_p
from .query_point_indices_worldcoords import lighting_fast_querier as lighting_fast_querier_w
from data.load_blender import load_blender_cloud
import numpy as np
from ..helpers.networks import init_seq, positional_encoding
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import Tensor
import open3d as o3d
import open3d.core as o3c
import torch.utils.dlpack

class NeuralPoints(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--load_points', type=int, default=0)

        parser.add_argument('--point_noise', type=str, default="", help='pointgaussian_0.1 | pointuniform_0.1')

        parser.add_argument('--num_point', type=int, default=8192)

        parser.add_argument('--construct_res', type=int, default=0)

        parser.add_argument('--grid_res', type=int, default=0)

        parser.add_argument('--cloud_path', type=str, default="")

        parser.add_argument('--shpnt_jitter', type=str, default="passfunc", help='passfunc | uniform | gaussian')

        parser.add_argument('--point_features_dim', type=int, default=64)
        parser.add_argument('--gpu_maxthr', type=int, default=1024)

        parser.add_argument('--z_depth_dim', type=int, default=400, help='number of coarse samples')

        parser.add_argument('--SR', type=int, default=24, help='max shading points number each ray')

        parser.add_argument('--K', type=int, default=32, help='max neural points per interpolation')

        parser.add_argument('--max_o', type=int, default=None, help='max nonempty voxels stored each frustum')

        parser.add_argument('--P', type=int, default=16, help='max neural points stored each block/voxel')

        parser.add_argument('--NN', type=int, default=0, help='0: radius search | 1: K-NN after radius search | 2: K-NN world coord after pers radius search')

        parser.add_argument('--radius_limit_scale', type=float, default=5.0, help='max neural points stored each block')
        parser.add_argument('--depth_limit_scale', type=float, default=1.3, help='max neural points stored each block')
        parser.add_argument('--default_conf', type=float, default=-1.0, help='max neural points stored each block')

        parser.add_argument('--vscale', type=int, nargs='+', default=(2, 2, 1), 
            help='vscale is the block size that store several voxels')

        parser.add_argument('--kernel_size', type=int, nargs='+', default=(7, 7, 1))
        parser.add_argument('--query_size', type=int, nargs='+', default=(0, 0, 0))

        parser.add_argument('--vsize', type=float, nargs='+', default=(0.005, 0.005, 0.005))
        parser.add_argument('--wcoord_query', type=int, default="0", help='0 for perspective voxels, and 1 for world coord')
        parser.add_argument('--ranges', type=float, nargs='+', default=(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0), 
            help='ranges of the volume, in the format of (x_min, y_min, z_min, x_max, y_max, z_max)')

        parser.add_argument('--xyz_grad',type=int,default=0)
        parser.add_argument('--feat_grad', type=int, default=1)
        parser.add_argument( '--conf_grad', type=int, default=1)
        parser.add_argument('--color_grad', type=int, default=1)
        parser.add_argument('--dir_grad', type=int, default=1)
        parser.add_argument('--specular_grad', type=int, default=1)
        parser.add_argument('--roughness_grad', type=int, default=1)
        parser.add_argument('--normal_grad', type=int, default=1)

        parser.add_argument('--feedforward', type=int, default=0)

        parser.add_argument('--inverse', type=int, default=0, help='1 for 1/n depth sweep')
        parser.add_argument('--point_conf_mode', type=str, default="0", help= '0 for only at features, 1 for multi at weight')
        parser.add_argument('--point_color_mode', type=str, default="0", help= '0 for only at features, 1 for multi at weight')
        parser.add_argument('--point_dir_mode', type=str, default="0", help= '0 for only at features, 1 for multi at weight')

        
        parser.add_argument('--deformed_bnd', action='store_true')
        parser.add_argument('--learn_point_specular', action='store_true')
        parser.add_argument('--learn_point_roughness', action='store_true')
        parser.add_argument('--learn_point_color', action='store_true')
        parser.add_argument('--den_n_normal', action='store_true')
        parser.add_argument('--marching_cube', action='store_true')
        parser.add_argument('--marching_cube_scale', type=float, default=1.0)
        parser.add_argument('--marching_cube_thresh', type=float, default=0.002,
                help='threshold for marching cube, can be larger if scene has thin objects')
        parser.add_argument('--marching_cube_smooth_iter', type=int, default=100, 
                help='number of smoothing iterations for marching cube, convolution propogation')
        parser.add_argument('--bake_light', action='store_true')

        parser.add_argument('--fine_pnt_sample_mode', type=str, choices=['linear', 'nonlinear', 'none'], default='none')
        parser.add_argument('--sample_stepsize', type=float, default=0.)
        parser.add_argument('--sample_stepsize_list', type=float, nargs='+', default=[])
        parser.add_argument('--SR_list', type=int, nargs='+', default=[])
        parser.add_argument('--sample_inc_iters', type=int, nargs='+', default=[])
        parser.add_argument('--use_sample_len', action='store_true')
        parser.add_argument('--sample_jitter_scale', type=float, default=0.8)
        parser.add_argument('--nonlinear_sample_min_scale', type=float, default=0.75)
        parser.add_argument('--sample_debug', action='store_true')

        parser.add_argument('--visible_prune_thresh', type=float, default=0)
        parser.add_argument('--visible_prune_thresh_max', type=float, default=0.05)
        parser.add_argument('--visible_prune_thresh_r', type=float, default=1.38) # 5**(1/5)
        parser.add_argument('--visible_prune_start_iter', type=float, default=30000)

    def __init__(self, num_channels, size, opt, device, checkpoint=None, feature_init_method='rand', reg_weight=0., feedforward=0):
        super().__init__()

        assert isinstance(size, int), 'size must be int'

        self.opt = opt
        self.grid_vox_sz = 0
        self.points_conf, self.points_dir, self.points_color, self.eulers, self.Rw2c = None, None, None, None, None
        self.Rdeform, self.ori_normal = None, None
        self.points_specular, self.points_roughness = None, None
        self.points_max_weight, self.weight_update_cnt = None, 0
        self.points_normal = None
        self.device=device
        if self.opt.load_points ==1:
            saved_features = None
            if checkpoint:
                saved_features = torch.load(checkpoint, map_location=device)
            if saved_features is not None and "neural_points.xyz" in saved_features:
                self.xyz = nn.Parameter(saved_features["neural_points.xyz"])
            else:
                point_xyz, _ = load_blender_cloud(self.opt.cloud_path, self.opt.num_point)
                point_xyz = torch.as_tensor(point_xyz, device=device, dtype=torch.float32)
                if len(opt.point_noise) > 0:
                    spl = opt.point_noise.split("_")
                    if float(spl[1]) > 0.0:
                        func = getattr(self, spl[0], None)
                        point_xyz = func(point_xyz, float(spl[1]))
                        print("point_xyz shape after jittering: ", point_xyz.shape)
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Loaded blender cloud ', self.opt.cloud_path, self.opt.num_point, point_xyz.shape)

                # filepath = "./aaaaaaaaaaaaa_cloud.txt"
                # np.savetxt(filepath, self.xyz.reshape(-1, 3).detach().cpu().numpy(), delimiter=";")

                if self.opt.construct_res > 0:
                    point_xyz, sparse_grid_idx, self.full_grid_idx = self.construct_grid_points(point_xyz)
                self.xyz = nn.Parameter(point_xyz)

                # filepath = "./grid_cloud.txt"
                # np.savetxt(filepath, point_xyz.reshape(-1, 3).detach().cpu().numpy(), delimiter=";")
                # print("max counts", torch.max(torch.unique(point_xyz, return_counts=True, dim=0)[1]))
                print("point_xyz", point_xyz.shape)

            self.xyz.requires_grad = opt.xyz_grad > 0
            shape = 1, self.xyz.shape[0], num_channels
            # filepath = "./aaaaaaaaaaaaa_cloud.txt"
            # np.savetxt(filepath, self.xyz.reshape(-1, 3).detach().cpu().numpy(), delimiter=";")

            if checkpoint:
                self.points_embeding = nn.Parameter(saved_features["neural_points.points_embeding"]) if "neural_points.points_embeding" in saved_features else None
                print("self.points_embeding", self.points_embeding.shape)
                # points_conf = saved_features["neural_points.points_conf"] if "neural_points.points_conf" in saved_features else None
                # if self.opt.default_conf > 0.0 and points_conf is not None:
                #     points_conf = torch.ones_like(points_conf) * self.opt.default_conf
                # self.points_conf = nn.Parameter(points_conf) if points_conf is not None else None

                self.points_conf = nn.Parameter(saved_features["neural_points.points_conf"]) if "neural_points.points_conf" in saved_features else None

                self.points_dir = nn.Parameter(saved_features["neural_points.points_dir"]) if "neural_points.points_dir" in saved_features else None
                self.points_color = nn.Parameter(saved_features["neural_points.points_color"]) if "neural_points.points_color" in saved_features else None
                self.eulers = nn.Parameter(saved_features["neural_points.eulers"]) if "neural_points.eulers" in saved_features else None
                self.Rw2c = nn.Parameter(saved_features["neural_points.Rw2c"]) if "neural_points.Rw2c" in saved_features else torch.eye(3, device=self.xyz.device, dtype=self.xyz.dtype)
                self.Rdeform = saved_features["neural_points.Rdeform"] if "neural_points.Rdeform" in saved_features else None
                self.ori_normal = saved_features["neural_points.ori_normal"] if "neural_points.ori_normal" in saved_features else None
                if self.opt.learn_point_specular: # The final [0,1] value should go through one sigmoid.
                    self.points_specular = nn.Parameter(saved_features["neural_points.points_specular"])\
                        if "neural_points.points_specular" in saved_features else nn.Parameter(torch.zeros_like(self.points_conf))
                if self.opt.learn_point_roughness:
                    self.points_roughness = nn.Parameter(saved_features["neural_points.points_roughness"])\
                        if "neural_points.points_roughness" in saved_features else nn.Parameter(torch.ones_like(self.points_conf) * self.opt.default_roughness)
                if self.opt.learn_point_normal and self.opt.which_agg_model=='viewmlp':
                    self.points_normal = nn.Parameter(saved_features["neural_points.points_normal"])\
                        if "neural_points.points_normal" in saved_features else nn.Parameter(torch.rand_like(self.points_dir))
            else:
                if feature_init_method == 'rand':
                    points_embeding = torch.rand(shape, device=device, dtype=torch.float32) - 0.5
                elif feature_init_method == 'zeros':
                    points_embeding = torch.zeros(shape, device=device, dtype=torch.float32)
                elif feature_init_method == 'ones':
                    points_embeding = torch.ones(shape, device=device, dtype=torch.float32)
                elif feature_init_method == 'pos':
                    if self.opt.point_features_dim > 3:
                        points_embeding = positional_encoding(point_xyz.reshape(shape[0], shape[1], 3), int(self.opt.point_features_dim / 6))
                        if int(self.opt.point_features_dim / 6) * 6 < self.opt.point_features_dim:
                            rand_embeding = torch.rand(shape[:-1] + (self.opt.point_features_dim - points_embeding.shape[-1],), device=device, dtype=torch.float32) - 0.5
                            print("points_embeding", points_embeding.shape, rand_embeding.shape)
                            points_embeding = torch.cat([points_embeding, rand_embeding], dim=-1)
                    else:
                        points_embeding = point_xyz.reshape(shape[0], shape[1], 3)
                elif feature_init_method.startswith("gau"):
                    std = float(feature_init_method.split("_")[1])
                    zeros = torch.zeros(shape, device=device, dtype=torch.float32)
                    points_embeding = torch.normal(mean=zeros, std=std)
                else:
                    raise ValueError(init_method)
                self.points_embeding = nn.Parameter(points_embeding)
                print("points_embeding init:", points_embeding.shape, torch.max(self.points_embeding), torch.min(self.points_embeding))
                self.points_conf=torch.ones_like(self.points_embeding[...,0:1])

            if self.points_embeding is not None:
                self.points_embeding.requires_grad = opt.feat_grad > 0
            if self.points_conf is not None:
                self.points_conf.requires_grad = self.opt.conf_grad > 0
            if self.points_dir is not None:
                self.points_dir.requires_grad = self.opt.dir_grad > 0
            if self.points_color is not None:
                self.points_color.requires_grad = self.opt.color_grad > 0
            if self.points_specular is not None:
                self.points_specular.requires_grad = self.opt.specular_grad > 0
            if self.points_roughness is not None:
                self.points_roughness.requires_grad = self.opt.roughness_grad > 0
            if self.points_normal is not None:
                self.points_normal.requires_grad = self.opt.normal_grad > 0
            if self.eulers is not None:
                self.eulers.requires_grad = False
            if self.Rw2c is not None:
                self.Rw2c.requires_grad = False

            if self.opt.visible_prune_thresh > 0:
                self.init_visible_pruning()

        self.reg_weight = reg_weight
        self.opt.query_size = self.opt.kernel_size if self.opt.query_size[0] == 0 else self.opt.query_size
        self.lighting_fast_querier = lighting_fast_querier_w if self.opt.wcoord_query > 0 else lighting_fast_querier_p
        self.querier = self.lighting_fast_querier(device, self.opt)
        self.build_occ = True

    def reset_querier(self):
        self.querier.clean_up()
        del self.querier
        self.querier = self.lighting_fast_querier(self.device, self.opt)


    def init_visible_pruning(self):
        self.points_max_weight = torch.zeros_like(self.points_conf.reshape(-1))
        self.weight_update_cnt = 0

    def prune(self, thresh, prune_mask=None):
        mask = None
        if thresh > 0:
            mask = self.points_conf[0,...,0] >= thresh
            if prune_mask is not None:
                mask = mask & prune_mask
        elif prune_mask is not None:
            mask = prune_mask
        if mask is None:
            return
        self.xyz = nn.Parameter(self.xyz[mask, :])
        self.xyz.requires_grad = self.opt.xyz_grad > 0
        self.build_occ = True
        if self.points_embeding is not None:
            self.points_embeding = nn.Parameter(self.points_embeding[:, mask, :])
            self.points_embeding.requires_grad = self.opt.feat_grad > 0
        if self.points_conf is not None:
            self.points_conf = nn.Parameter(self.points_conf[:, mask, :])
            self.points_conf.requires_grad = self.opt.conf_grad > 0
        if self.points_dir is not None:
            self.points_dir = nn.Parameter(self.points_dir[:, mask, :])
            self.points_dir.requires_grad = self.opt.dir_grad > 0
        if self.points_color is not None:
            self.points_color = nn.Parameter(self.points_color[:, mask, :])
            self.points_color.requires_grad = self.opt.color_grad > 0
        if self.eulers is not None and self.eulers.dim() > 1:
            self.eulers = nn.Parameter(self.eulers[mask, :])
            self.eulers.requires_grad = False
        if self.Rw2c is not None and self.Rw2c.dim() > 2:
            self.Rw2c = nn.Parameter(self.Rw2c[mask, :])
            self.Rw2c.requires_grad = False
        if self.points_normal is not None:
            self.points_normal = nn.Parameter(self.points_normal[:, mask, :])
            self.points_normal.requires_grad = self.opt.normal_grad > 0
        if self.points_roughness is not None:
            self.points_roughness = nn.Parameter(self.points_roughness[:, mask, :])
            self.points_roughness.requires_grad = self.opt.roughness_grad > 0
        
        print("@@@@@@@@@  pruned {}/{}".format(torch.sum(mask==0), mask.shape[0]))
        if self.opt.visible_prune_thresh > 0:
            self.init_visible_pruning()


    def grow_points(self, add_xyz, add_embedding, add_color, add_dir, add_conf, 
            add_eulers=None, add_Rw2c=None, add_roughness=None, add_specular=None, 
            add_normal=None):
        # print(self.xyz.shape, self.points_conf.shape, self.points_embeding.shape, self.points_dir.shape, self.points_color.shape)
        if self.opt.wcoord_query > 0:
            _, _, ranges_np, vsize_np, vdim_np, scaled_vsize_np, scaled_vdim_np, vscale_np, \
                range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, _, _ \
                    = self.querier.get_hyperparameters(self.opt.vsize, self.xyz[None,...], ranges=self.opt.ranges, recache=False)
            v_coord = ((add_xyz - range_gpu[:3]) / scaled_vsize_gpu).floor().long()
            mask_range = (v_coord < scaled_vdim_gpu[None,:]).all(-1) * (v_coord >= 0).all(-1)
            v_coord = v_coord[mask_range]
            occ_idx = self.querier.coor_2_occ_tensor[0, v_coord[...,0], v_coord[...,1], v_coord[...,2]]
            pnt_per_vox = self.querier.occ_numpnts_tensor[0,occ_idx.long()]
            mask_vox = (pnt_per_vox <= 1.2 * self.opt.P)
            mask_range[mask_range.clone()] = mask_vox
            mask_xyz = mask_range
            add_xyz, add_embedding, add_color, add_dir, add_conf, add_eulers, \
                add_Rw2c, add_roughness, add_specular, add_normal = \
                    [   k[mask_range] if k is not None else k 
                        for k in [add_xyz, add_embedding, add_color, add_dir, add_conf, \
                            add_eulers, add_Rw2c, add_roughness, add_specular,\
                            add_normal]
                    ]
                    
    
        self.xyz = nn.Parameter(torch.cat([self.xyz, add_xyz], dim=0))
        self.xyz.requires_grad = self.opt.xyz_grad > 0
        self.build_occ = True

        if self.points_embeding is not None:
            self.points_embeding = nn.Parameter(torch.cat([self.points_embeding, add_embedding[None, ...]], dim=1))
            self.points_embeding.requires_grad = self.opt.feat_grad > 0

        if self.points_conf is not None:
            self.points_conf = nn.Parameter(torch.cat([self.points_conf, add_conf[None, ...]], dim=1))
            self.points_conf.requires_grad = self.opt.conf_grad > 0
        if self.points_dir is not None:
            self.points_dir = nn.Parameter(torch.cat([self.points_dir, add_dir[None, ...]], dim=1))
            self.points_dir.requires_grad = self.opt.dir_grad > 0

        if self.points_color is not None:
            self.points_color = nn.Parameter(torch.cat([self.points_color, add_color[None, ...]], dim=1))
            self.points_color.requires_grad = self.opt.color_grad > 0

        if self.eulers is not None and self.eulers.dim() > 1:
            self.eulers = nn.Parameter(torch.cat([self.eulers, add_eulers[None,...]], dim=1))
            self.eulers.requires_grad = False
            
        if self.Rw2c is not None and self.Rw2c.dim() > 2:
            self.Rw2c = nn.Parameter(torch.cat([self.Rw2c, add_Rw2c[None,...]], dim=1))
            self.Rw2c.requires_grad = False

        if self.points_roughness is not None:
            self.points_roughness = nn.Parameter(torch.cat([self.points_roughness, add_roughness[None, ...]], dim=1))
            self.points_roughness.requires_grad = self.opt.roughness_grad > 0

        if self.points_specular is not None:
            self.points_specular = nn.Parameter(torch.cat([self.points_specular, add_specular[None, ...]], dim=1))
            self.points_specular.requires_grad = self.opt.specular_grad > 0

        if self.points_normal is not None:
            self.points_normal = nn.Parameter(torch.cat([self.points_normal, add_normal[None, ...]], dim=1))
            self.points_normal.requires_grad = self.opt.normal_grad > 0

        if self.opt.visible_prune_thresh > 0:
            self.init_visible_pruning()


    def set_points(self, points_xyz, points_embeding, points_color=None, points_dir=None, points_conf=None, parameter=False, Rw2c=None, eulers=None):
        if points_embeding.shape[-1] > self.opt.point_features_dim:
            points_embeding = points_embeding[..., :self.opt.point_features_dim]
        if self.opt.default_conf > 0.0 and self.opt.default_conf <= 1.0 and points_conf is not None:
            points_conf = torch.ones_like(points_conf) * self.opt.default_conf
        if parameter:
            self.xyz = nn.Parameter(points_xyz)
            self.xyz.requires_grad = self.opt.xyz_grad > 0

            if points_conf is not None:
                points_conf = nn.Parameter(points_conf)
                points_conf.requires_grad = self.opt.conf_grad > 0
                self.points_conf = points_conf

            if points_dir is not None:
                points_dir = nn.Parameter(points_dir)
                points_dir.requires_grad = self.opt.dir_grad > 0
                self.points_dir = points_dir

            if points_color is not None:
                points_color = nn.Parameter(points_color)
                points_color.requires_grad = self.opt.color_grad > 0
                self.points_color = points_color

            points_embeding = nn.Parameter(points_embeding)
            points_embeding.requires_grad = self.opt.feat_grad > 0
            self.points_embeding = points_embeding
                # print("self.points_embeding", self.points_embeding, self.points_color)

            # print("points_xyz", torch.min(points_xyz, dim=-2)[0], torch.max(points_xyz, dim=-2)[0])
        else:
            self.xyz = points_xyz

            if points_conf is not None:
                self.points_conf = points_conf

            if points_dir is not None:
                self.points_dir = points_dir

            if points_color is not None:
                self.points_color = points_color

            self.points_embeding = points_embeding

        if Rw2c is None:
            self.Rw2c = torch.eye(3, device=points_xyz.device, dtype=points_xyz.dtype)
        else:
            self.Rw2c = nn.Parameter(Rw2c)
            self.Rw2c.requires_grad = False


    def editing_set_points(self, points_xyz, points_embeding, points_color=None, points_dir=None, points_conf=None,
                   parameter=False, Rw2c=None, eulers=None):
        if self.opt.default_conf > 0.0 and self.opt.default_conf <= 1.0 and points_conf is not None:
            points_conf = torch.ones_like(points_conf) * self.opt.default_conf

        self.xyz = points_xyz
        self.points_embeding = points_embeding
        self.points_dir = points_dir
        self.points_conf = points_conf
        self.points_color = points_color

        if Rw2c is None:
            self.Rw2c = torch.eye(3, device=points_xyz.device, dtype=points_xyz.dtype)
        else:
            self.Rw2c = Rw2c



    def construct_grid_points(self, xyz):
        # --construct_res' '--grid_res',
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        self.space_edge = torch.max(xyz_max - xyz_min) * 1.1
        xyz_mid = (xyz_max + xyz_min) / 2
        self.space_min = xyz_mid - self.space_edge / 2
        self.space_max = xyz_mid + self.space_edge / 2
        self.construct_vox_sz = self.space_edge / self.opt.construct_res
        self.grid_vox_sz = self.space_edge / self.opt.grid_res

        xyz_shift = xyz - self.space_min[None, ...]
        construct_vox_idx = torch.unique(torch.floor(xyz_shift / self.construct_vox_sz[None, ...]).to(torch.int16), dim=0)
        # print("construct_grid_idx", construct_grid_idx.shape) torch.Size([7529, 3])

        cg_ratio = int(self.opt.grid_res / self.opt.construct_res)
        gx = torch.arange(0, cg_ratio+1, device=construct_vox_idx.device, dtype=construct_vox_idx.dtype)
        gy = torch.arange(0, cg_ratio+1, device=construct_vox_idx.device, dtype=construct_vox_idx.dtype)
        gz = torch.arange(0, cg_ratio+1, device=construct_vox_idx.device, dtype=construct_vox_idx.dtype)
        gx, gy, gz = torch.meshgrid(gx, gy, gz)
        gxyz = torch.stack([gx, gy, gz], dim=-1).view(1, -1, 3)
        sparse_grid_idx = construct_vox_idx[:, None, :] * cg_ratio + gxyz
        # sparse_grid_idx.shape: ([7529, 9*9*9, 3]) -> ([4376896, 3])
        sparse_grid_idx = torch.unique(sparse_grid_idx.view(-1, 3), dim=0).to(torch.int64)
        full_grid_idx = torch.full([self.opt.grid_res+1,self.opt.grid_res+1,self.opt.grid_res+1], -1, device=xyz.device, dtype=torch.int32)
        # full_grid_idx.shape:    ([401, 401, 401])
        full_grid_idx[sparse_grid_idx[...,0], sparse_grid_idx[...,1], sparse_grid_idx[...,2]] = torch.arange(0, sparse_grid_idx.shape[0], device=full_grid_idx.device, dtype=full_grid_idx.dtype)
        xyz = self.space_min[None, ...] + sparse_grid_idx * self.grid_vox_sz
        return xyz, sparse_grid_idx, full_grid_idx


    def null_grad(self):
        self.points_embeding.grad = None
        self.xyz.grad = None


    def reg_loss(self):
        return self.reg_weight * torch.mean(torch.pow(self.points_embeding, 2))


    def pers2img(self, point_xyz_pers_tensor, pixel_id, pixel_idx_cur, ray_mask, sample_pidx, ranges, h, w, inputs):
        xper = point_xyz_pers_tensor[..., 0].cpu().numpy()
        yper = point_xyz_pers_tensor[..., 1].cpu().numpy()

        x_pixel = np.clip(np.round((xper-ranges[0]) * (w-1) / (ranges[3]-ranges[0])).astype(np.int32), 0, w-1)[0]
        y_pixel = np.clip(np.round((yper-ranges[1]) * (h-1) / (ranges[4]-ranges[1])).astype(np.int32), 0, h-1)[0]

        print("pixel xmax xmin:", np.max(x_pixel), np.min(x_pixel), "pixel ymax ymin:", np.max(y_pixel),
              np.min(y_pixel), sample_pidx.shape,y_pixel.shape)
        background = np.zeros([h, w, 3], dtype=np.float32)
        background[y_pixel, x_pixel, :] = self.points_embeding.cpu().numpy()[0,...]

        background[pixel_idx_cur[0,...,1],pixel_idx_cur[0,...,0],0] = 1.0

        background[y_pixel[sample_pidx[-1]], x_pixel[sample_pidx[-1]], :] = self.points_embeding.cpu().numpy()[0,sample_pidx[-1]]

        gtbackground = np.ones([h, w, 3], dtype=np.float32)
        gtbackground[pixel_idx_cur[0 ,..., 1], pixel_idx_cur[0 , ..., 0],:] = inputs["gt_image"].cpu().numpy()[0][ray_mask[0]>0]

        print("diff sum",np.sum(inputs["gt_image"].cpu().numpy()[0][ray_mask[0]>0]-self.points_embeding.cpu().numpy()[0,sample_pidx[...,1,0][-1]]))

        plt.figure()
        plt.imshow(background)
        plt.figure()
        plt.imshow(gtbackground)
        plt.show()


    def get_point_indices(self, inputs, cam_rot_tensor, cam_pos_tensor, pixel_idx_tensor, 
                    near_plane, far_plane, h, w, intrinsic, vox_query=False):

        # point_xyz_pers_tensor = self.w2pers(self.xyz, cam_rot_tensor, cam_pos_tensor)
        actual_numpoints_tensor = torch.ones([cam_pos_tensor.shape[0]], device=self.xyz.device, dtype=torch.int32) * self.xyz.shape[0]
        # sample_pidx_tensor: B, R, SR, K
        ray_dirs_tensor = inputs["raydir"]
        # print("ray_dirs_tensor", ray_dirs_tensor.shape, self.xyz.shape)
        sample_pidx_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, ray_mask_tensor, vsize, ranges, sample_loc_len = \
            self.querier.query_points(pixel_idx_tensor, self.xyz[None,...], actual_numpoints_tensor, 
                h, w, intrinsic, near_plane, far_plane, ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor,
                build_occ=self.build_occ)
        if self.opt.wcoord_query > 0 and self.querier.occ_idx_tensor[0].item() <= self.opt.max_o:
            self.build_occ = False 

        # print("ray_mask_tensor",ray_mask_tensor.shape)
        # self.pers2img(point_xyz_pers_tensor, pixel_idx_tensor.cpu().numpy(), pixel_idx_cur_tensor.cpu().numpy(), ray_mask_tensor.cpu().numpy(), sample_pidx_tensor.cpu().numpy(), ranges, h, w, inputs)

        B, _, SR, K = sample_pidx_tensor.shape
        if vox_query:
            if sample_pidx_tensor.shape[1] > 0:
                sample_pidx_tensor = self.query_vox_grid(sample_loc_w_tensor, self.full_grid_idx, self.space_min, self.grid_vox_sz)
            else:
                sample_pidx_tensor = torch.zeros([B, 0, SR, 8], device=sample_pidx_tensor.device, dtype=sample_pidx_tensor.dtype)

        return sample_pidx_tensor, ray_mask_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, vsize, sample_loc_len


    def query_vox_grid(self, sample_loc_w_tensor, full_grid_idx, space_min, grid_vox_sz):
        # sample_pidx_tensor = torch.full(sample_loc_w_tensor.shape[:-1]+(8,), -1, device=sample_loc_w_tensor.device, dtype=torch.int64)
        B, R, SR, _ = sample_loc_w_tensor.shape
        vox_ind = torch.floor((sample_loc_w_tensor - space_min[None, None, None, :]) / grid_vox_sz).to(torch.int64) # B, R, SR, 3
        shift = torch.as_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.int64, device=full_grid_idx.device).reshape(1, 1, 1, 8, 3)
        vox_ind = vox_ind[..., None, :] + shift  # B, R, SR, 8, 3
        vox_mask = torch.any(torch.logical_or(vox_ind < 0, vox_ind > self.opt.grid_res).view(B, R, SR, -1), dim=3)
        vox_ind = torch.clamp(vox_ind, min=0, max=self.opt.grid_res).view(-1, 3)
        inds = full_grid_idx[vox_ind[..., 0], vox_ind[..., 1], vox_ind[..., 2]].view(B, R, SR, 8)
        inds[vox_mask, :] = -1
        # -1 for all 8 corners
        inds[torch.any(inds < 0, dim=-1), :] = -1
        return inds.to(torch.int64)


    def w2pers(self, point_xyz, camrotc2w, campos):
        point_xyz_shift = point_xyz[None, ...] - campos[:, None, :]
        xyz = torch.sum(camrotc2w[:, None, :, :] * point_xyz_shift[:, :, :, None], dim=-2)
        # print(xyz.shape, (point_xyz_shift[:, None, :] * camrot.T).shape)
        xper = xyz[:, :, 0] / xyz[:, :, 2]
        yper = xyz[:, :, 1] / xyz[:, :, 2]
        return torch.stack([xper, yper, xyz[:, :, 2]], dim=-1)


    def vect2euler(self, xyz):
        yz_norm = torch.norm(xyz[...,1:3], dim=-1)
        e_x = torch.atan2(-xyz[...,1], xyz[...,2])
        e_y = torch.atan2(xyz[...,0], yz_norm)
        e_z = torch.zeros_like(e_y)
        e_xyz = torch.stack([e_x, e_y, e_z], dim=-1)
        return e_xyz

    def euler2Rc2w(self, e_xyz):
        cosxyz = torch.cos(e_xyz)
        sinxyz = torch.sin(e_xyz)
        cxsz = cosxyz[...,0]*sinxyz[...,2]
        czsy = cosxyz[...,2]*sinxyz[...,1]
        sxsz = sinxyz[...,0]*sinxyz[...,2]
        r1 = torch.stack([cosxyz[...,1]*cosxyz[...,2], czsy*sinxyz[...,0] - cxsz, czsy*cosxyz[...,0] + sxsz], dim=-1)
        r2 = torch.stack([cosxyz[...,1]*sinxyz[...,2], cosxyz[...,0]*cosxyz[...,2] + sxsz*sinxyz[...,1], -cosxyz[...,2]*sinxyz[...,0] + cxsz * sinxyz[...,1]], dim=-1)
        r3 = torch.stack([-sinxyz[...,1], cosxyz[...,1]*sinxyz[...,0], cosxyz[...,0]*cosxyz[...,1]], dim=-1)

        Rzyx = torch.stack([r1, r2, r3], dim=-2)
        return Rzyx

    def euler2Rw2c(self, e_xyz):
        c = torch.cos(-e_xyz)
        s = torch.sin(-e_xyz)
        r1 = torch.stack([c[...,1] * c[...,2], -s[...,2], c[...,2]*s[...,1]], dim=-1)
        r2 = torch.stack([s[...,0]*s[...,1] + c[...,0]*c[...,1]*s[...,2], c[...,0]*c[...,2], -c[...,1]*s[...,0]+c[...,0]*s[...,1]*s[...,2]], dim=-1)
        r3 = torch.stack([-c[...,0]*s[...,1]+c[...,1]*s[...,0]*s[...,2], c[...,2]*s[...,0], c[...,0]*c[...,1]+s[...,0]*s[...,1]*s[...,2]], dim=-1)
        Rxyz = torch.stack([r1, r2, r3], dim=-2)
        return Rxyz


    def get_w2c(self, cam_xyz, Rw2c):
        t = -Rw2c @ cam_xyz[..., None] # N, 3
        M = torch.cat([Rw2c, t], dim=-1)
        ones = torch.as_tensor([[[0, 0, 0, 1]]], device=M.device, dtype=M.dtype).expand(len(M),-1, -1)
        return torch.cat([M, ones], dim=-2)

    def get_c2w(self, cam_xyz, Rc2w):
        M = torch.cat([Rc2w, cam_xyz[..., None]], dim=-1)
        ones = torch.as_tensor([[[0, 0, 0, 1]]], device=M.device, dtype=M.dtype).expand(len(M),-1, -1)
        return torch.cat([M, ones], dim=-2)


    def passfunc(self, input, vsize):
        return input

    def pointgaussian(self, input, std):
        M, C = input.shape
        input = torch.normal(mean=input, std=std)
        return input

    def pointuniform(self, input, std):
        M, C = input.shape
        jitters = torch.rand([M, C], dtype=torch.float32, device=input.device) - 0.5
        input = input + jitters * std * 2
        return input

    def pointuniformadd(self, input, std):
        addinput = self.pointuniform(input, std)
        return torch.cat([input,addinput], dim=0)

    def pointuniformdouble(self, input, std):
        input = self.pointuniform(torch.cat([input,input], dim=0), std)
        return input




    def forward(self, inputs):

        pixel_idx, camrotc2w, campos, near_plane, far_plane, h, w, intrinsic = inputs["pixel_idx"].to(torch.int32), inputs["camrotc2w"], inputs["campos"], inputs["near"], inputs["far"], inputs["h"], inputs["w"], inputs["intrinsic"]
        # 1, 294, 24, 32;   1, 294, 24;     1, 291, 2

        sample_pidx, ray_mask_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, vsize, sample_loc_len = \
            self.get_point_indices(inputs, camrotc2w, campos, pixel_idx, 
                torch.min(near_plane).cpu().numpy(), torch.max(far_plane).cpu().numpy(), 
                torch.max(h).cpu().numpy(), torch.max(w).cpu().numpy(), intrinsic.cpu().numpy()[0], vox_query=self.opt.NN<0)

        sample_pnt_mask = sample_pidx >= 0
        B, R, SR, K = sample_pidx.shape
        sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()
        sampled_xyz = torch.index_select(self.xyz[None, ...], 1, sample_pidx).view(B, R, SR, K, 3)
        sampled_embedding = torch.index_select(self.points_embeding, 1, sample_pidx).view(B, R, SR, K, self.points_embeding.shape[2])

        sampled_color = None if self.points_color is None else torch.index_select(self.points_color, 1, sample_pidx).view(B, R, SR, K, self.points_color.shape[2])

        sampled_dir = None if self.points_dir is None else torch.index_select(self.points_dir, 1, sample_pidx).view(B, R, SR, K, self.points_dir.shape[2])

        sampled_conf = None if self.points_conf is None else torch.index_select(self.points_conf, 1, sample_pidx).view(B, R, SR, K, self.points_conf.shape[2])

        sampled_Rw2c = self.Rw2c if self.Rw2c.dim() == 2 else torch.index_select(self.Rw2c, 0, sample_pidx).view(B, R, SR, K, self.Rw2c.shape[1], self.Rw2c.shape[2])

        sampled_roughness = None if self.points_roughness is None else torch.index_select(self.points_roughness, 1, sample_pidx).view(B, R, SR, K, self.points_roughness.shape[2])

        sampled_specular = None if self.points_specular is None else torch.index_select(self.points_specular, 1, sample_pidx).view(B, R, SR, K, self.points_specular.shape[2])

        sampled_normal = None if self.points_normal is None else torch.index_select(self.points_normal, 1, sample_pidx).view(B, R, SR, K, self.points_normal.shape[2])

        sampled_Rdeform = None if self.Rdeform is None else torch.index_select(self.Rdeform, 1, sample_pidx).view(B, R, SR, K, self.Rdeform.shape[2], self.Rdeform.shape[3])

        sampled_ori_normal = None if self.ori_normal is None else torch.index_select(self.ori_normal, 1, sample_pidx).view(B, R, SR, K, self.ori_normal.shape[2])
        
        # filepath = "./sampled_xyz_full.txt"
        # np.savetxt(filepath, self.xyz.reshape(-1, 3).detach().cpu().numpy(), delimiter=";")
        #
        # filepath = "./sampled_xyz_pers_full.txt"
        # np.savetxt(filepath, point_xyz_pers_tensor.reshape(-1, 3).detach().cpu().numpy(), delimiter=";")

        # if self.xyz.grad is not None:
        #     print("xyz grad:", self.xyz.requires_grad, torch.max(self.xyz.grad), torch.min(self.xyz.grad))
        # if self.points_embeding.grad is not None:
        #     print("points_embeding grad:", self.points_embeding.requires_grad, torch.max(self.points_embeding.grad))
        # print("points_embeding 3", torch.max(self.points_embeding), torch.min(self.points_embeding))
        
        sampled_points = {
            'color': sampled_color, 'Rw2c': sampled_Rw2c, 'dir': sampled_dir, 'conf': sampled_conf,
            'embedding': sampled_embedding, 'xyz': sampled_xyz, 'pnt_mask': sample_pnt_mask, 
            'pidx': sample_pidx.reshape(B, R, SR, K), 'roughness': sampled_roughness, 'specular': sampled_specular,
            'normal': sampled_normal, 'Rdeform': sampled_Rdeform, 'ori_normal': sampled_ori_normal,
        }
        
        return sampled_points, sample_loc_w_tensor, sample_ray_dirs_tensor,\
                ray_mask_tensor, vsize, self.grid_vox_sz, sample_loc_len

    def xyz_forward(self, queried_xyz: torch.Tensor, empty_check=False):

        # pixel_idx
        # 1, 294, 24, 32;   1, 294, 24;     1, 291, 2
        sample_pidx, sample_xyz, vsize = \
            self.querier.xyz_query(queried_xyz, self.xyz[None,...], self.opt.P, self.build_occ, empty_check)
        if self.opt.wcoord_query > 0 and self.build_occ and self.querier.occ_idx_tensor[0].item() <= self.opt.max_o: 
            self.build_occ = False
        sample_pnt_mask = sample_pidx >= 0
        R, K = sample_pidx.shape
        B, SR = 1, 1
        sample_pnt_mask = sample_pnt_mask.view(B, R, SR, K)
        sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()
        sampled_xyz = torch.index_select(self.xyz[None, ...], 1, sample_pidx).view(B, R, SR, K, 3)
        sampled_embedding = torch.index_select(self.points_embeding, 1, sample_pidx).view(B, R, SR, K, self.points_embeding.shape[2])

        sampled_color = None if self.points_color is None else torch.index_select(self.points_color, 1, sample_pidx).view(B, R, SR, K, self.points_color.shape[2])

        sampled_dir = None if self.points_dir is None else torch.index_select(self.points_dir, 1, sample_pidx).view(B, R, SR, K, self.points_dir.shape[2])

        sampled_conf = None if self.points_conf is None else torch.index_select(self.points_conf, 1, sample_pidx).view(B, R, SR, K, self.points_conf.shape[2])

        sampled_Rw2c = self.Rw2c if self.Rw2c.dim() == 2 else torch.index_select(self.Rw2c, 0, sample_pidx).view(B, R, SR, K, self.Rw2c.shape[1], self.Rw2c.shape[2])

        sampled_roughness = None if self.points_roughness is None else torch.index_select(self.points_roughness, 1, sample_pidx).view(B, R, SR, K, self.points_roughness.shape[2])

        sampled_specular = None if self.points_specular is None else torch.index_select(self.points_specular, 1, sample_pidx).view(B, R, SR, K, self.points_specular.shape[2])

        sampled_normal = None if self.points_normal is None else torch.index_select(self.points_normal, 1, sample_pidx).view(B, R, SR, K, self.points_normal.shape[2])

        sampled_points = {
            'color': sampled_color, 'Rw2c': sampled_Rw2c, 'dir': sampled_dir, 'conf': sampled_conf,
            'embedding': sampled_embedding, 'xyz': sampled_xyz, 'pnt_mask': sample_pnt_mask, 
            'pidx': sample_pidx, 'roughness': sampled_roughness, 'specular': sampled_specular,
            'normal': sampled_normal
            }
        return sampled_points, vsize, self.grid_vox_sz

    def save_pcd(self, feats=None, epoch='last'):
        state_dict = {} # = self.state_dict()
        state_dict['xyz'] = self.xyz
        state_dict['colors'] = self.points_color
        if feats is not None:
            for k, v in feats.items():
                state_dict[k] = v
        save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        save_filename = '{}_pcd.pth'.format(epoch)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(state_dict, save_path)

    @torch.no_grad()
    def interp_pnt_attr(self, add_xyz: Tensor, with_deform=False, chunk_size=65536):
        o3d_pcd = PointCloud(self.xyz)
        num_pts = add_xyz.shape[0]
        add_embedding, add_color, add_dir, add_conf = [], [], [], []
        add_roughness, add_specular, add_normal = [], [], []
        for i in tqdm(range(0, num_pts, chunk_size)):
            nn_idx, dist2 = o3d_pcd.knn_search(add_xyz[i:i+chunk_size], self.opt.K)
            radius_limit = 1.5 * self.opt.radius_limit_scale * max(self.opt.vsize[0], self.opt.vsize[1])
            nn_mask = dist2 <= radius_limit**2 # N, K
            nn_mask[..., 0] = True # ensure at least one point for interp.
            # inverse_distance interpolation
            weight = 1 / dist2.sqrt().clamp_min(1e-6)
            weight = weight * nn_mask / torch.sum(weight * nn_mask, dim=-1, keepdim=True).clamp_min(1e-6)
            valid_pnt_idx = nn_idx[nn_mask].long() # P
            valid_weight = weight[nn_mask][...,None] # P, 1
            nn_scatter_idx = nn_mask.nonzero()[..., 0] # P

            add_embedding.append(scatter_sum(self.points_embeding[0, valid_pnt_idx] * valid_weight, nn_scatter_idx, dim=0))
            if self.points_color is not None: add_color.append(scatter_sum(self.points_color[0, valid_pnt_idx] * valid_weight, nn_scatter_idx, dim=0))
            if self.points_dir is not None: add_dir.append(scatter_sum(self.points_dir[0, valid_pnt_idx] * valid_weight, nn_scatter_idx, dim=0))
            if self.points_conf is not None: add_conf.append(scatter_sum(self.points_conf[0, valid_pnt_idx] * valid_weight, nn_scatter_idx, dim=0))
            if self.points_roughness is not None: add_roughness.append(scatter_sum(self.points_roughness[0, valid_pnt_idx] * valid_weight, nn_scatter_idx, dim=0))
            if self.points_specular is not None: add_specular.append(scatter_sum(self.points_specular[0, valid_pnt_idx] * valid_weight, nn_scatter_idx, dim=0))
            if self.points_normal is not None: add_normal.append(scatter_sum(self.points_normal[0, valid_pnt_idx] * valid_weight, nn_scatter_idx, dim=0))
            
        add_embedding = torch.cat(add_embedding, dim=0)
        add_color = torch.cat(add_color, dim=0) if len(add_color) > 0 else None
        add_dir = torch.cat(add_dir, dim=0) if len(add_dir) > 0 else None
        add_conf = torch.cat(add_conf, dim=0) if len(add_conf) > 0 else None
        add_roughness = torch.cat(add_roughness, dim=0) if len(add_roughness) > 0 else None
        add_specular = torch.cat(add_specular, dim=0) if len(add_specular) > 0 else None
        add_normal = torch.cat(add_normal, dim=0) if len(add_normal) > 0 else None

        return add_xyz, add_embedding, add_color, add_dir, add_conf, \
            add_roughness, add_specular, add_normal



th_to_o3 = lambda x: o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(x))
o3_to_th = lambda x: torch.utils.dlpack.from_dlpack(x.to_dlpack())

class PointCloud(object): # Explicit3D
    def __init__(self, pcd):
        if isinstance(pcd, str):
            self.init_from_ply(pcd)
        else: # Tensor
            self.points = pcd # N, 3

        self.points_o3 = th_to_o3(self.points) # points and points_o3 share the same memory.
        self.nns = o3c.nns.NearestNeighborSearch(self.points_o3)
        self.nns.knn_index()

    def init_from_ply(self, pcd_file, device='cpu'):
        pcd = o3d.io.read_point_cloud(pcd_file)
        self.points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
        self.colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=device)

    def knn_search(self, q_pts, k):
        if isinstance(q_pts, Tensor):
            q_pts = th_to_o3(q_pts)
        indices, dists = self.nns.knn_search(q_pts, k) # note dists are squared
        return o3_to_th(indices), o3_to_th(dists)
