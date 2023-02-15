import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..helpers.networks import init_seq, positional_encoding
from utils.spherical import IDE, SphericalHarm_table, SphericalHarm
from ..helpers.geometrics import compute_world2local_dist, get_vector_rotation_matrices
from torch_scatter import scatter_max
from ..rendering.diff_render_func import linear2srgb


class PointAggregator(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--feature_init_method', type=str, default="rand", help='which agg model to use [feature_interp | graphconv | affine_mix]')
        parser.add_argument('--which_agg_model', type=str, default="viewmlp", help='which agg model to use [viewmlp | nsvfmlp]')
        parser.add_argument('--agg_distance_kernel', type=str, default="quadric", help='which agg model to use [quadric | linear | feat_intrp | harmonic_intrp]')

        parser.add_argument('--sh_degree',
            type=int,
            default=4,
            help='degree of harmonics')

        parser.add_argument('--sh_dist_func',
            type=str,
            default="sh_quadric",
            help='sh_quadric | sh_linear | passfunc')

        parser.add_argument('--sh_act',
            type=str,
            default="sigmoid",
            help='sigmoid | tanh | passfunc')

        parser.add_argument('--agg_axis_weight',
            type=float,
            nargs='+',
            default=None,
            help=
            '(1., 1., 1.)'
        )

        parser.add_argument('--agg_dist_pers', type=int, default=1, help='use pers dist')
        parser.add_argument('--apply_pnt_mask', type=int, default=1, help='use pers dist')
        parser.add_argument('--modulator_concat', type=int, default=0, help='use pers dist')
        parser.add_argument('--agg_intrp_order', type=int, default=0, 
                help='interpolate first and feature mlp 0 | feature mlp then interpolate 1 | feature mlp color then interpolate 2')
        parser.add_argument('--shading_feature_mlp_layer0', type=int, default=0, help='interp to agged features mlp num')
        parser.add_argument('--shading_feature_mlp_layer1', type=int, default=2, help='interp to agged features mlp num')
        parser.add_argument('--shading_feature_mlp_layer2', type=int, default=0, help='interp to agged features mlp num')
        parser.add_argument('--shading_feature_mlp_layer3', type=int, default=2, help='interp to agged features mlp num')
        parser.add_argument('--shading_feature_num', type=int, default=256, help='agged shading feature channel num')
        parser.add_argument('--point_hyper_dim', type=int, default=256, help='agged shading feature channel num')
        parser.add_argument('--shading_alpha_mlp_layer', type=int, default=1, help='agged features to alpha mlp num')
        parser.add_argument('--shading_alpha_channel_num', type=int, default=1, help='agged features to alpha mlp num')
        parser.add_argument('--shading_color_mlp_layer', type=int, default=1, help='agged features to alpha mlp num')

        parser.add_argument('--shading_color_channel_num',type=int, default=3,help='color channel num')
        parser.add_argument('--num_feat_freqs', type=int, default=0, help='color channel num')

        parser.add_argument('--num_hyperfeat_freqs', type=int, default=0, help='color channel num')
        parser.add_argument('--dist_xyz_freq', type=int, default=2, help='color channel num')

        parser.add_argument('--dist_xyz_deno', type=float, default=0, help='color channel num')

        parser.add_argument('--weight_xyz_freq', type=int, default=2, help='color channel num')

        parser.add_argument('--weight_feat_dim', type=int, default=8, help='color channel num')

        parser.add_argument('--agg_weight_norm', type=int, default=1, help='normalize weight, sum as 1')

        parser.add_argument('--view_ori', action='store_true', help='0 for pe+3 orignal channels')
        parser.add_argument('--normal_ori', action='store_true', help='0 for pe+3 orignal channels')

        parser.add_argument('--agg_feat_xyz_mode', type=str, default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument('--agg_alpha_xyz_mode', type=str, default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument('--agg_color_xyz_mode', type=str, default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument('--act_type', type=str, default="ReLU", # default="LeakyReLU",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument('--act_super', type=int, default=1, # default="LeakyReLU", 
                help='1 to use softplus and widden sigmoid for last activation')

        parser.add_argument('--color_act', type=str, choices=['linear', 'sigmoid'], default='sigmoid')
        parser.add_argument('--agg_grad_mode', type=str, default='loc')
        parser.add_argument('--sdf_mode', type=str, default='volsdf')
        parser.add_argument('--grad_detach', action='store_true')
        # parser.add_argument('--rgbnet_no_grad_in', action='store_true')
        parser.add_argument('--rgbnet_normal_mode', type=str, default='v')
                # choices=['v', 'v_n', 'v_r', 'r', 'v_n_dot'])
        parser.add_argument('--normal_anneal_iters', type=float, default=1)
        parser.add_argument('--normal_anneal_ratio', type=float, default=1.)
        parser.add_argument('--in_diffuse', action='store_true')
        parser.add_argument('--viewdir_norm', action='store_true')
        parser.add_argument('--depth_only', action='store_true')
        parser.add_argument('--residual_rgb', action='store_true')
        parser.add_argument('--learn_diffuse_color', action='store_true')
        parser.add_argument('--learn_point_normal', action='store_true')
        parser.add_argument('--optim_compute_normal', action='store_true')
        parser.add_argument('--weight_sum_one', action='store_true')
        parser.add_argument('--sum_one_denom_detach', action='store_true')
        parser.add_argument('--no_dist_in', action='store_true')
        parser.add_argument('--agg_weight_detach', action='store_true')
        parser.add_argument('--sdf_no_geo_init', action='store_true')
        parser.add_argument('--attr_weight_mode', type=str, default='agg', choices=['agg', 'dist'])
        parser.add_argument('--diffuse_branch_channel', type=int, default=0)
        parser.add_argument('--diffuse_branch_color_in', action='store_true')
        parser.add_argument('--monochro_specular', action='store_true')
        parser.add_argument('--shading_diffuse_mlp_layer', type=int, default=2, help='agged features to alpha mlp num')
        parser.add_argument('--shading_roughness_mlp_layer', type=int, default=0, help='agged features to rough mlp num')
        parser.add_argument('--shading_metallic_mlp_layer', type=int, default=0, help='agged features to metal mlp num')
        parser.add_argument('--shading_fresnel_mlp_layer', type=int, default=0, help='agged features to rough mlp num')
        parser.add_argument('--fresnel_branch_channel', type=int, default=3)
        # parser.add_argument('--srgb_residual', action='store_true')
        parser.add_argument('--sdf_beta', type=float, default=0.1)
        parser.add_argument('--sdf_beta_min', type=float, default=0.0001)
        parser.add_argument('--sdf_var', type=float, default=0.3)
        parser.add_argument('--sdf_scale', type=float, default=1.0)
        parser.add_argument('--cos_anneal_iters', type=float, default=5000)
        parser.add_argument('--cos_anneal_ratio', type=float, default=1.)
        parser.add_argument('--SH_color_branch', action='store_true')
        parser.add_argument('--SH_color_deg', type=int, default=3)
        parser.add_argument('--SH_dir_deg', type=int, default=0)
        parser.add_argument('--diffuse_branch_only', action='store_true')
        parser.add_argument('--diffuse_only_iters', type=int, default=0)
        parser.add_argument('--rot_v_dir', action='store_true')
        parser.add_argument('--ref_ide', action='store_true')
        parser.add_argument('--mlp_fast_forward', action='store_true')

    def __init__(self, opt):

        super(PointAggregator, self).__init__()
        self.act = getattr(nn, opt.act_type, None)
        print("opt.act_type!!!!!!!!!", opt.act_type)
        self.point_hyper_dim=opt.point_hyper_dim if opt.point_hyper_dim < opt.point_features_dim else opt.point_features_dim

        block_init_lst = []
        if opt.agg_distance_kernel == "feat_intrp":
            feat_weight_block = []
            in_channels = 2 * opt.weight_xyz_freq * 3 + opt.weight_feat_dim
            out_channels = int(in_channels / 2)
            for i in range(2):
                feat_weight_block.append(nn.Linear(in_channels, out_channels))
                feat_weight_block.append(self.act(inplace=True))
                in_channels = out_channels
            feat_weight_block.append(nn.Linear(in_channels, 1))
            feat_weight_block.append(nn.Sigmoid())
            self.feat_weight_mlp = nn.Sequential(*feat_weight_block)
            block_init_lst.append(self.feat_weight_mlp)
        elif opt.agg_distance_kernel == "sh_intrp":
            self.shcomp = SphericalHarm(opt.sh_degree)

        self.opt = opt
        self.dist_dim = (4 if self.opt.agg_dist_pers == 30 else 6) if self.opt.agg_dist_pers > 9 else 3
        # if self.opt.no_dist_in:
        #     self.dist_dim = 0
        self.dist_func = getattr(self, opt.agg_distance_kernel, None)
        assert self.dist_func is not None, "InterpAggregator doesn't have disance_kernel {} ".format(opt.agg_distance_kernel)

        self.axis_weight = None if opt.agg_axis_weight is None else torch.as_tensor(opt.agg_axis_weight, dtype=torch.float32, device="cuda")[None, None, None, None, :]

        self.num_freqs = opt.num_pos_freqs if opt.num_pos_freqs > 0 else 0
        self.num_viewdir_freqs = opt.num_viewdir_freqs if opt.num_viewdir_freqs > 0 else 0

        self.pnt_channels = (2 * self.num_freqs * 3) if self.num_freqs > 0 else 3
        self.viewdir_channels = (2 * self.num_viewdir_freqs * 3 + self.opt.view_ori * 3) if self.num_viewdir_freqs > 0 else 3
        if self.opt.SH_dir_deg > 0:
            self.viewdir_channels = (self.opt.SH_dir_deg + 1)**2
        self.which_agg_model = opt.which_agg_model.split("_")[0] if opt.which_agg_model.startswith("feathyper") else opt.which_agg_model
        getattr(self, self.which_agg_model+"_init", None)(opt, block_init_lst)

        self.density_super_act = torch.nn.Softplus()
        self.density_act = torch.nn.ReLU()
        if self.opt.color_act == 'linear':
            self.color_act = lambda x: (x+1) / 2
        else:
            self.color_act = torch.nn.Sigmoid()

    def raw2out_density(self, raw_density):
        if self.opt.act_super > 0: # may need to turn it off for sdf.
            # return self.density_act(raw_density - 1)  # according to mip nerf, to stablelize the training
            return self.density_super_act(raw_density - 1)  # according to mip nerf, to stablelize the training
        else:
            return self.density_act(raw_density)

    def raw2out_color(self, raw_color):
        color = self.color_act(raw_color)
        if self.opt.act_super > 0:
            color = color * (1 + 2 * 0.001) - 0.001 # according to mip nerf, to stablelize the training
        return color

    def passfunc(self, input):
        return input


    def trilinear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * 3
        # return B * R * SR * K
        dists = dists * pnt_mask[..., None]
        dists = dists / grid_vox_sz

        #  dist: [1, 797, 40, 8, 3];     pnt_mask: [1, 797, 40, 8]
        # dists = 1 + dists * torch.as_tensor([[1,1,1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=torch.float32, device=dists.device).view(1, 1, 1, 8, 3)

        dists = 1 - torch.abs(dists)

        weights = pnt_mask * dists[..., 0] * dists[..., 1] * dists[..., 2]
        norm_weights = weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)

        return norm_weights, embedding


    def avg(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        weights = pnt_mask * 1.0
        return weights, embedding


    def quadric(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 1] == 1 and axis_weight[..., 2] ==1):
            weights = 1./ torch.clamp(torch.sum(torch.square(dists[..., :3]), dim=-1), min= 1e-8)
        else:
            weights = 1. / torch.clamp(torch.sum(torch.square(dists)* axis_weight, dim=-1), min=1e-8)
        weights = pnt_mask * weights
        return weights, embedding


    def numquadric(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 1] == 1 and axis_weight[..., 2] ==1):
            weights = 1./ torch.clamp(torch.sum(torch.square(dists), dim=-1), min= 1e-8)
        else:
            weights = 1. / torch.clamp(torch.sum(torch.square(dists)* axis_weight, dim=-1), min=1e-8)
        weights = pnt_mask * weights
        return weights, embedding


    def linear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :
            weights = 1. / torch.clamp(torch.norm(dists[..., :3], dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        return weights, embedding


    def numlinear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :
            weights = 1. / torch.clamp(torch.norm(dists, dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        norm_weights = weights / torch.clamp(torch.sum(pnt_mask, dim=-1, keepdim=True), min=1)
        return norm_weights, embedding


    def sigmoid(self, input):
        return torch.sigmoid(input)

    def tanh(self, input):
        return torch.tanh(input)

    def sh_linear(self, dist_norm):
        return 1 / torch.clamp(dist_norm, min=1e-8)

    def sh_quadric(self, dist_norm):
        return 1 / torch.clamp(torch.square(dist_norm), min=1e-8)


    def sh_intrp(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        dist_norm = torch.linalg.norm(dists, dim=-1)
        dist_dirs = dists / torch.clamp(dist_norm[...,None], min=1e-8)
        shall = self.shcomp.sh_all(dist_dirs, filp_dir=False).view(dists.shape[:-1]+(self.shcomp.total_deg ** 2,))
        sh_coefs = embedding[..., :self.shcomp.total_deg ** 2]
        # shall: [1, 816, 24, 32, 16], sh_coefs: [1, 816, 24, 32, 16], pnt_mask: [1, 816, 24, 32]
        # debug: weights = pnt_mask * torch.sum(shall, dim=-1)
        # weights = pnt_mask * torch.sum(shall * getattr(self, self.opt.sh_act, None)(sh_coefs), dim=-1) * getattr(self, self.opt.sh_dist_func, None)(dist_norm)
        weights = pnt_mask * torch.sum(getattr(self, self.opt.sh_act, None)(shall * sh_coefs), dim=-1) * getattr(self, self.opt.sh_dist_func, None)(dist_norm) # changed
        return weights, embedding[..., self.shcomp.total_deg ** 2:]


    def gau_intrp(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # dist: [1, 752, 40, 32, 3]
        B, R, SR, K, _ = dists.shape
        scale = torch.abs(embedding[..., 0]) #
        radii = vsize[2] * 20 * torch.sigmoid(embedding[..., 1:4])
        rotations = torch.clamp(embedding[..., 4:7], max=np.pi / 4, min=-np.pi / 4)
        gau_dist = compute_world2local_dist(dists, radii, rotations)[..., 0]
        # print("gau_dist", gau_dist.shape)
        weights = pnt_mask * scale * torch.exp(-0.5 * torch.sum(torch.square(gau_dist), dim=-1))
        # print("gau_dist", gau_dist.shape, gau_dist[0, 0])
        # print("weights", weights.shape, weights[0, 0, 0])
        return weights, embedding[..., 7:]

    def print_point(self, dists, sample_loc_w, sampled_xyz, sample_loc, sampled_xyz_pers, sample_pnt_mask):

        # for i in range(dists.shape[0]):
        #     filepath = "./dists.txt"
        #     filepath1 = "./dists10.txt"
        #     filepath2 = "./dists20.txt"
        #     filepath3 = "./dists30.txt"
        #     filepath4 = "./dists40.txt"
        #     dists_cpu = dists.detach().cpu().numpy()
        #     np.savetxt(filepath1, dists_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath2, dists_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath3, dists_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath4, dists_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
        #     dists_cpu = dists[i,...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].detach().cpu().numpy()
        #     np.savetxt(filepath, dists_cpu.reshape(-1, 3), delimiter=";")

        for i in range(sample_loc_w.shape[0]):
            filepath = "./sample_loc_w.txt"
            filepath1 = "./sample_loc_w10.txt"
            filepath2 = "./sample_loc_w20.txt"
            filepath3 = "./sample_loc_w30.txt"
            filepath4 = "./sample_loc_w40.txt"
            sample_loc_w_cpu = sample_loc_w.detach().cpu().numpy()
            np.savetxt(filepath1, sample_loc_w_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sample_loc_w_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sample_loc_w_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sample_loc_w_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            sample_loc_w_cpu = sample_loc_w[i,...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].detach().cpu().numpy()
            np.savetxt(filepath, sample_loc_w_cpu.reshape(-1, 3), delimiter=";")


        for i in range(sampled_xyz.shape[0]):
            sampled_xyz_cpu = sampled_xyz.detach().cpu().numpy()
            filepath = "./sampled_xyz.txt"
            filepath1 = "./sampled_xyz10.txt"
            filepath2 = "./sampled_xyz20.txt"
            filepath3 = "./sampled_xyz30.txt"
            filepath4 = "./sampled_xyz40.txt"
            np.savetxt(filepath1, sampled_xyz_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sampled_xyz_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sampled_xyz_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sampled_xyz_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath, sampled_xyz_cpu[i, ...].reshape(-1, 3), delimiter=";")

        for i in range(sample_loc.shape[0]):
            filepath1 = "./sample_loc10.txt"
            filepath2 = "./sample_loc20.txt"
            filepath3 = "./sample_loc30.txt"
            filepath4 = "./sample_loc40.txt"
            filepath = "./sample_loc.txt"
            sample_loc_cpu =sample_loc.detach().cpu().numpy()

            np.savetxt(filepath1, sample_loc_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sample_loc_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sample_loc_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sample_loc_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath, sample_loc[i, ...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].reshape(-1, 3).detach().cpu().numpy(), delimiter=";")

        for i in range(sampled_xyz_pers.shape[0]):
            filepath1 = "./sampled_xyz_pers10.txt"
            filepath2 = "./sampled_xyz_pers20.txt"
            filepath3 = "./sampled_xyz_pers30.txt"
            filepath4 = "./sampled_xyz_pers40.txt"
            filepath = "./sampled_xyz_pers.txt"
            sampled_xyz_pers_cpu = sampled_xyz_pers.detach().cpu().numpy()

            np.savetxt(filepath1, sampled_xyz_pers_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sampled_xyz_pers_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sampled_xyz_pers_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sampled_xyz_pers_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")

            np.savetxt(filepath, sampled_xyz_pers_cpu[i, ...].reshape(-1, 3), delimiter=";")
        print("saved sampled points and shading points")
        exit()


    def gradient_clamp(self, sampled_conf, min=0.0001, max=1):
        diff = sampled_conf - torch.clamp(sampled_conf, min=min, max=max)
        return sampled_conf - diff.detach()


    def forward(self, s_pts, sample_loc, sample_loc_w, sample_ray_dirs, vsize, grid_vox_sz, ray_dist=None): 
        '''
        :param sampled_conf: B x valid R x SR x K x 1
        :param sampled_embedding: B x valid R x SR x K x F
        :param sampled_xyz_pers:  B x valid R x SR x K x 3
        :param sampled_xyz:       B x valid R x SR x K x 3
        :param sample_pnt_mask:   B x valid R x SR x K
        :param sample_loc:        B x valid R x SR x 3
        :param sample_loc_w:      B x valid R x SR x 3
        :param sample_ray_dirs:   B x valid R x SR x 3
        :param vsize:
        :return: B * R * SR * channel
        '''
        ray_valid = torch.any(s_pts['pnt_mask'], dim=-1).view(-1)
        total_len = len(ray_valid)
        in_shape = sample_loc_w.shape
        # import IPython; IPython.embed()
        if total_len == 0 or torch.sum(ray_valid).item() == 0:
            # print("skip since no valid ray, total_len:", total_len, torch.sum(ray_valid))
            return torch.zeros(in_shape[:-1] + (self.opt.shading_color_channel_num + 1,), device=ray_valid.device, dtype=torch.float32), ray_valid.view(in_shape[:-1]), None, None, None

        if self.opt.agg_dist_pers < 0:
            dists = sample_loc_w[..., None, :]
        elif self.opt.agg_dist_pers == 0:
            dists = s_pts['xyz'] - sample_loc_w[..., None, :]
        elif self.opt.agg_dist_pers == 1:
            dists = s_pts['xyz_pers'] - sample_loc[..., None, :]
        elif self.opt.agg_dist_pers == 2:
            if s_pts['xyz_pers'].shape[1] > 0:
                xdist = s_pts['xyz_pers'][..., 0] * s_pts['xyz_pers'][..., 2] - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]
                ydist = s_pts['xyz_pers'][..., 1] * s_pts['xyz_pers'][..., 2] - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]
                zdist = s_pts['xyz_pers'][..., 2] - sample_loc[:, :, :, None, 2]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)
            else:
                B, R, SR, K, _ = s_pts['xyz_pers'].shape
                dists = torch.zeros([B, R, SR, K, 3], device=s_pts['xyz_pers'].device, dtype=s_pts['xyz_pers'].dtype)

        elif self.opt.agg_dist_pers == 10:

            if s_pts['xyz_pers'].shape[1] > 0:
                dists = s_pts['xyz_pers'] - sample_loc[..., None, :]
                dists = torch.cat([s_pts['xyz'] - sample_loc_w[..., None, :], dists], dim=-1)
            else:
                B, R, SR, K, _ = s_pts['xyz_pers'].shape
                dists = torch.zeros([B, R, SR, K, 6], device=s_pts['xyz_pers'].device, dtype=s_pts['xyz_pers'].dtype)

        elif self.opt.agg_dist_pers == 20:

            if s_pts['xyz_pers'].shape[1] > 0:
                xdist = s_pts['xyz_pers'][..., 0] * s_pts['xyz_pers'][..., 2] - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]
                ydist = s_pts['xyz_pers'][..., 1] * s_pts['xyz_pers'][..., 2] - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]
                zdist = s_pts['xyz_pers'][..., 2] - sample_loc[:, :, :, None, 2]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)
                # dists = torch.cat([s_pts['xyz'] - sample_loc_w[..., None, :], dists], dim=-1)
                dists = torch.cat([s_pts['xyz'] - sample_loc_w[..., None, :], dists], dim=-1)
            else:
                B, R, SR, K, _ = s_pts['xyz_pers'].shape
                dists = torch.zeros([B, R, SR, K, 6], device=s_pts['xyz_pers'].device, dtype=s_pts['xyz_pers'].dtype)

        elif self.opt.agg_dist_pers == 30:

            if s_pts['xyz_pers'].shape[1] > 0:
                w_dists = s_pts['xyz'] - sample_loc_w[..., None, :]
                dists = torch.cat([torch.sum(w_dists*sample_ray_dirs[..., None, :], dim=-1, keepdim=True), dists], dim=-1)
            else:
                B, R, SR, K, _ = s_pts['xyz_pers'].shape
                dists = torch.zeros([B, R, SR, K, 4], device=s_pts['xyz_pers'].device, dtype=s_pts['xyz_pers'].dtype)
        else:
            print("illegal agg_dist_pers code: ", self.opt.agg_dist_pers)
            exit()
        # self.print_point(dists, sample_loc_w, s_pts['xyz'], sample_loc, s_pts['xyz_pers'], sample_pnt_mask)

        weight, s_pts['embedding'] = \
            self.dist_func(s_pts['embedding'], dists, s_pts['pnt_mask'], vsize, grid_vox_sz, axis_weight=self.axis_weight)

        conf_coefficient = 1
        if s_pts['conf'] is not None and self.opt.prune_thresh > 0:
            conf_coefficient = self.gradient_clamp(s_pts['conf'][..., 0], min=0.0001, max=1)

        if self.opt.agg_weight_detach:
            weight = weight.detach()

        if self.opt.agg_weight_norm > 0 and self.opt.agg_distance_kernel != "trilinear" and not self.opt.agg_distance_kernel.startswith("num"):
            weight = weight / torch.clamp(torch.sum(weight, dim=-1, keepdim=True), min=1e-8)
            
        if not self.opt.weight_sum_one:
            agg_weight = weight * conf_coefficient
        else:
            agg_weight = weight * conf_coefficient
            denom = torch.clamp(torch.sum(agg_weight, dim=-1, keepdim=True), min=1e-8)
            denom = denom.detach() if self.opt.sum_one_denom_detach else denom
            agg_weight = agg_weight / denom
       
        output, others = getattr(self, self.which_agg_model, None)(
                s_pts, sample_loc, sample_loc_w, sample_ray_dirs, vsize, agg_weight, 
                total_len, ray_valid, in_shape, dists, weight, ray_dist)
        if (self.opt.sparse_loss_weight <=0) and ("conf_coefficient" not in self.opt.zero_one_loss_items) and self.opt.prob == 0:
            weight, conf_coefficient = None, None
        with torch.set_grad_enabled(True):
            if output is None:
                return None, ray_valid.view(in_shape[:-1]), weight, conf_coefficient, others
            else:
                return output.view(in_shape[:-1] + (output.shape[-1], )), ray_valid.view(in_shape[:-1]), weight, conf_coefficient, others
