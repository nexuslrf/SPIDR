import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..helpers.networks import init_seq, positional_encoding
from utils.spherical import IDE, SphericalHarm_table, SphericalHarm
from ..helpers.geometrics import compute_world2local_dist, get_vector_rotation_matrices
from torch_scatter import scatter_max
from ..rendering.diff_render_func import linear2srgb
from .point_aggregators import PointAggregator

class SDFMLP(PointAggregator):
    def __init__(self, opt):
        super().__init__(opt)
    
    def sdfmlp_init(self, opt, block_init_lst, sdf_bias=0.6):
        dist_xyz_dim = self.dist_dim if opt.dist_xyz_freq == 0 else (2 * abs(opt.dist_xyz_freq)+1) * self.dist_dim
        in_channels = opt.point_features_dim
        extra_dim = (2 * opt.num_feat_freqs * in_channels if opt.num_feat_freqs > 0 else 0) + \
                + (dist_xyz_dim if opt.agg_intrp_order > 0 and not opt.no_dist_in else 0)  # it is necessary to add PE to feat??
        in_channels += extra_dim
        geometric_init=not opt.sdf_no_geo_init
        weight_norm=True if geometric_init else False
        ###### SDF Part ######
        def sdf_layer_init(layer, out_dim):
            if geometric_init: # https://arxiv.org/pdf/1911.10414.pdf
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                layer = nn.utils.weight_norm(layer)
            return layer

        sdf_act = lambda: nn.Softplus(beta=100)

        if opt.shading_feature_mlp_layer1 > 0:
            out_channels = opt.shading_feature_num
            block1 = []
            for i in range(opt.shading_feature_mlp_layer1):
                lin = nn.Linear(in_channels, out_channels)
                if geometric_init:
                    if i==0:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.constant_(lin.weight, 0.0)
                        torch.nn.init.normal_(lin.weight[:, :opt.point_features_dim], 0.0, np.sqrt(2) / np.sqrt(out_channels))
                        torch.nn.init.normal_(lin.weight[:, -(dist_xyz_dim+self.dist_dim):-dist_xyz_dim], 
                                                                                    0.0, np.sqrt(2) / np.sqrt(out_channels))
                        if weight_norm: lin = nn.utils.weight_norm(lin)
                    else:
                        lin = sdf_layer_init(lin, out_channels)
                block1.append(lin)
                block1.append(sdf_act())
                in_channels = out_channels
            self.block1 = nn.Sequential(*block1)
            if not geometric_init:
                block_init_lst.append(self.block1)
        else:
            self.block1 = self.passfunc

        if opt.shading_feature_mlp_layer2 > 0:
            in_channels = in_channels + (0 if opt.agg_feat_xyz_mode == "None" else self.pnt_channels) + (
                dist_xyz_dim if (opt.agg_intrp_order > 0 and opt.num_feat_freqs == 0) else 0)
            out_channels = opt.shading_feature_num
            block2 = []
            for i in range(opt.shading_feature_mlp_layer2):
                lin = nn.Linear(in_channels, out_channels)
                if geometric_init:
                    lin = sdf_layer_init(lin, out_channels)
                block2.append(lin)
                block2.append(sdf_act())
                in_channels = out_channels
            self.block2 = nn.Sequential(*block2)
            if not geometric_init:
                block_init_lst.append(self.block2)
        else:
            self.block2 = self.passfunc

        if opt.shading_feature_mlp_layer3 > 0:
            point_dir_channels = 0 if "1" not in list(opt.point_dir_mode) \
                                else (4 - (1 if "0" in list(opt.point_dir_mode) else 0))
            in_channels = in_channels + (3 if "1" in list(opt.point_color_mode) else 0) + point_dir_channels
            out_channels = opt.shading_feature_num
            block3 = []
            for i in range(opt.shading_feature_mlp_layer3):
                lin = nn.Linear(in_channels, out_channels)
                if geometric_init:
                    lin = sdf_layer_init(lin, out_channels)
                block3.append(lin)
                block3.append(sdf_act())
                in_channels = out_channels
            self.block3 = nn.Sequential(*block3)
            if not geometric_init:
                block_init_lst.append(self.block3)
        else:
            self.block3 = self.passfunc
        
        alpha_block = []
        in_channels = opt.shading_feature_num + (0 if opt.agg_alpha_xyz_mode == "None" else self.pnt_channels)
        out_channels = int(opt.shading_feature_num / 2)
        for i in range(opt.shading_alpha_mlp_layer - 1):
            lin = nn.Linear(in_channels, out_channels)
            if geometric_init:
                lin = sdf_layer_init(lin, out_channels)
            alpha_block.append(lin)
            alpha_block.append(sdf_act())
            in_channels = out_channels
        lin = nn.Linear(in_channels, opt.shading_alpha_channel_num)
        if geometric_init:
            torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_channels), std=0.0001)
            torch.nn.init.constant_(lin.bias, -sdf_bias)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
        alpha_block.append(lin)
        self.alpha_branch = nn.Sequential(*alpha_block)
        if not geometric_init:
            block_init_lst.append(self.alpha_branch)
        
        # TODO maybe add another parallel feat block...
        # beta param for Laplace Density
        if self.opt.sdf_mode == 'volsdf':
            self.sdf_beta = nn.Parameter(torch.tensor(self.opt.sdf_beta))
            self.sdf_act_val = self.opt.sdf_beta
            self.sdf_beta_min = self.opt.sdf_beta_min
        elif self.opt.sdf_mode == 'neus':
            self.sdf_var = nn.Parameter(torch.tensor(self.opt.sdf_var))
            self.sdf_act_val = self.opt.sdf_var

        if opt.diffuse_branch_channel > 0:
            diffuse_block = []      
            in_channels = opt.shading_feature_num
            out_channels = int(opt.shading_feature_num / 2)
            for i in range(opt.shading_diffuse_mlp_layer - 1):
                diffuse_block.append(nn.Linear(in_channels, out_channels))
                diffuse_block.append(self.act(inplace=False))
                in_channels = out_channels
            diffuse_block.append(nn.Linear(in_channels, opt.diffuse_branch_channel))
            self.diffuse_branch = nn.Sequential(*diffuse_block)
            block_init_lst.append(self.diffuse_branch)

        ###### RGB_layers ######

        color_block = []
        normal_channels = (2 * opt.num_normal_freqs * 3 + opt.normal_ori * 3) if opt.num_normal_freqs > 0 else 3
        if self.opt.SH_dir_deg > 0:
            if not self.opt.ref_ide:
                normal_channels = (self.opt.SH_dir_deg + 1)**2
                self.dir_sh_encoder = SphericalHarm_table(self.opt.SH_dir_deg+1) 
                assert self.opt.viewdir_norm
            else:
                normal_channels = sum([2**i+1 for i in range(self.opt.SH_dir_deg+1)])
                self.ide_encoder = IDE(self.opt.SH_dir_deg+1)
                assert self.opt.learn_point_roughness
        in_channels = opt.shading_feature_num + \
                    (0 if opt.agg_color_xyz_mode == "None" else self.pnt_channels) + \
                    (3 if opt.in_diffuse else 0)
        if not opt.SH_color_branch:
            in_channels +=((self.viewdir_channels if 'v' in opt.rgbnet_normal_mode else 0)+
                           (normal_channels if 'n' in opt.rgbnet_normal_mode else 0) + 
                           (normal_channels if 'r' in opt.rgbnet_normal_mode else 0) + 
                           (1 if 'dot' in opt.rgbnet_normal_mode else 0))
        color_channels = 3 if not opt.monochro_specular else 1
        if opt.SH_color_branch:
            color_channels = 3 * (opt.SH_color_deg + 1)**2
        out_channels = int(opt.shading_feature_num / 2)
        for i in range(opt.shading_color_mlp_layer - 1):
            lin = nn.Linear(in_channels, out_channels)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            color_block.append(lin)
            color_block.append(self.act(inplace=True))
            in_channels = out_channels
        lin = nn.Linear(in_channels, color_channels)
        if weight_norm: 
            lin = nn.utils.weight_norm(lin)
        color_block.append(lin)
        # TODO maybe adding sigmoid before final output?
        self.color_branch = nn.Sequential(*color_block)
        block_init_lst.append(self.color_branch)
        if opt.SH_color_branch:
            self.shcomputer = SphericalHarm_table(opt.SH_color_deg+1)

        if opt.use_microfacet_mlp:
            if opt.shading_roughness_mlp_layer > 0:
                roughness_block = []      
                in_channels = opt.shading_feature_num
                out_channels = int(opt.shading_feature_num / 2)
                for i in range(opt.shading_roughness_mlp_layer - 1):
                    roughness_block.append(nn.Linear(in_channels, out_channels))
                    roughness_block.append(self.act(inplace=False))
                    in_channels = out_channels
                roughness_block.append(nn.Linear(in_channels, 1))
                self.roughness_branch = nn.Sequential(*roughness_block)
                block_init_lst.append(self.roughness_branch)

            if opt.shading_metallic_mlp_layer > 0:
                metallic_block = []      
                in_channels = opt.shading_feature_num
                out_channels = int(opt.shading_feature_num / 2)
                for i in range(opt.shading_metallic_mlp_layer - 1):
                    metallic_block.append(nn.Linear(in_channels, out_channels))
                    metallic_block.append(self.act(inplace=False))
                    in_channels = out_channels
                metallic_block.append(nn.Linear(in_channels, 1))
                self.metallic_branch = nn.Sequential(*metallic_block)
                block_init_lst.append(self.metallic_branch)

            if opt.shading_fresnel_mlp_layer > 0:
                fresnel_block = []      
                in_channels = opt.shading_feature_num
                out_channels = int(opt.shading_feature_num / 2)
                for i in range(opt.shading_fresnel_mlp_layer - 1):
                    fresnel_block.append(nn.Linear(in_channels, out_channels))
                    fresnel_block.append(self.act(inplace=False))
                    in_channels = out_channels
                fresnel_block.append(nn.Linear(in_channels, opt.fresnel_branch_channel))
                self.fresnel_branch = nn.Sequential(*fresnel_block)
                block_init_lst.append(self.fresnel_branch)
            
        if opt.use_albedo_mlp:
            albedo_block = []      
            in_channels = opt.shading_feature_num
            out_channels = int(opt.shading_feature_num / 2)
            for i in range(opt.shading_diffuse_mlp_layer - 1):
                albedo_block.append(nn.Linear(in_channels, out_channels))
                albedo_block.append(self.act(inplace=False))
                in_channels = out_channels
            albedo_block.append(nn.Linear(in_channels, opt.diffuse_branch_channel))
            self.albedo_branch = nn.Sequential(*albedo_block)
            block_init_lst.append(self.albedo_branch)


        for m in block_init_lst:
            init_seq(m)
    
    def sdfmlp(self, s_pts, sample_loc, sample_loc_w, sample_ray_dirs, 
            vsize, weight, total_len, ray_valid, in_shape, dists, dist_weight, ray_dist=None):
        # print("sampled_Rw2c", sampled_Rw2c.shape, sampled_xyz.shape)
        # assert sampled_Rw2c.dim() == 2
        pnt_mask_flat = s_pts['pnt_mask'].view(-1)
        pts = sample_loc_w.view(-1, sample_loc_w.shape[-1])

        B, R, SR, K, _ = dists.shape
        sampled_Rw2c = s_pts['Rw2c'].transpose(-1, -2)
        uni_w2c = sampled_Rw2c.dim() == 2
        if not uni_w2c:
            sampled_Rw2c_ray = sampled_Rw2c[:,:,:,0,:,:].view(-1, 3, 3)
            sampled_Rw2c = sampled_Rw2c.reshape(-1, 3, 3)[pnt_mask_flat, :, :]
        pts_ray, pts_pnt = None, None
        if self.opt.agg_feat_xyz_mode != "None" or self.opt.agg_alpha_xyz_mode != "None" or self.opt.agg_color_xyz_mode != "None":
            if self.num_freqs > 0:
                pts = positional_encoding(pts, self.num_freqs)
            pts_ray = pts[ray_valid, :]
            if self.opt.agg_feat_xyz_mode != "None" and self.opt.agg_intrp_order > 0:
                pts_pnt = pts[..., None, :].repeat(1, K, 1).view(-1, pts.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    pts_pnt=pts_pnt[pnt_mask_flat, :]

        if sample_ray_dirs is not None:
            viewdirs_per_ray = sample_ray_dirs
            if self.opt.viewdir_norm:
                viewdirs_per_ray = F.normalize(viewdirs_per_ray, dim=-1)
            viewdirs = viewdirs_per_ray.expand(-1,-1,SR,-1).reshape(-1, viewdirs_per_ray.shape[-1])[ray_valid, :]
            viewdirs = viewdirs @ sampled_Rw2c if uni_w2c else (viewdirs[..., None, :] @ sampled_Rw2c_ray).squeeze(-2)

            # ori_viewdirs = viewdirs @ sampled_Rw2c if uni_w2c else (viewdirs[..., None, :] @ sampled_Rw2c_ray).squeeze(-2)
            # if self.num_viewdir_freqs > 0:
            #     viewdirs = positional_encoding(ori_viewdirs, self.num_viewdir_freqs, ori=self.opt.view_ori)
            ############# ATTENTION ###############
            # due to the awful impl. of pos enc. you need switch the following code for older checkpoints.
            # viewdirs = viewdirs @ sampled_Rw2c if uni_w2c else (viewdirs[..., None, :] @ sampled_Rw2c_ray).squeeze(-2)
            
        # if self.opt.agg_intrp_order == 0: NotImplemented # no consideration for this mode.
        Rdeform = s_pts['Rdeform'] if 'Rdeform' in s_pts and s_pts['Rdeform'] is not None else None

        dists_flat = dists.view(-1, dists.shape[-1])
        if self.opt.apply_pnt_mask > 0:
            dists_flat = dists_flat[pnt_mask_flat, :]
            Rdeform = Rdeform.view(-1, 3, 3)[pnt_mask_flat] if Rdeform is not None else None
        dists_flat /= (
            1.0 if self.opt.dist_xyz_deno == 0. else float(self.opt.dist_xyz_deno * np.linalg.norm(vsize)))

        weight = weight.view(B * R * SR, K, 1)
        weight_valid = weight[ray_valid]
        if self.opt.attr_weight_mode == 'agg':
            attr_weight_valid = weight_valid.detach()
        elif self.opt.attr_weight_mode == 'dist':
            attr_weight_valid = dist_weight.view(B * R * SR, K, 1)[ray_valid].detach()
        
        # TODO: pre-compute interpolated normals
        if Rdeform is None and 'ori_normal' in s_pts and s_pts['ori_normal'] is not None:
            n_ori = (s_pts['ori_normal'].reshape(-1, K, 3)[ray_valid] * attr_weight_valid).sum(-2)
            n_new = (s_pts['dir'].reshape(-1, K, 3)[ray_valid] * attr_weight_valid).sum(-2)
            n_ori = F.normalize(n_ori, dim=-1)
            n_new = F.normalize(n_new, dim=-1)
            # get the rotation matrix from n_ori to n_new
            R_agg = get_vector_rotation_matrices(n_ori, n_new) # (V, 3, 3)

            inter_mask = s_pts['pnt_mask'].reshape(-1, K)[ray_valid] # (V, K)
            r_idx_map = inter_mask.nonzero(as_tuple=False)[..., 0] # (P, )
            Rdeform = R_agg[r_idx_map]

        if Rdeform is not None:
            # dists_flat[..., :3] =
            if dists_flat.shape[-1] == 3:
                dists_flat = (Rdeform * dists_flat[..., None]).sum(-2)
            if dists_flat.shape[-1] > 3:
                dists_flat = torch.cat([(Rdeform * dists_flat[..., :3, None]).sum(-2),  (Rdeform * dists_flat[..., 3:, None]).sum(-2)])
        
        dists_flat[..., :3] = dists_flat[..., :3] @ sampled_Rw2c if uni_w2c else (dists_flat[..., None, :3] @ sampled_Rw2c).squeeze(-2)
        if self.opt.dist_xyz_freq != 0:
            # print(dists.dtype, (self.opt.dist_xyz_deno * np.linalg.norm(vsize)).dtype, dists_flat.dtype)
            dists_flat = positional_encoding(dists_flat, self.opt.dist_xyz_freq, ori=True) # we include dists itself!
        feat= s_pts['embedding'].view(-1, s_pts['embedding'].shape[-1])
        # print("feat", feat.shape)

        if self.opt.apply_pnt_mask > 0:
            feat = feat[pnt_mask_flat, :]

        if self.opt.num_feat_freqs > 0:
            feat = torch.cat([feat, positional_encoding(feat, self.opt.num_feat_freqs)], dim=-1)
        if not self.opt.no_dist_in:
            feat = torch.cat([feat, dists_flat], dim=-1)

        pts = pts_pnt

        # used_point_embedding = feat[..., : self.opt.point_features_dim]
        if self.opt.agg_feat_xyz_mode != "None":
            feat = torch.cat([feat, pts], dim=-1)
        # print("feat",feat.shape) # 501
        feat = self.block1(feat)

        if self.opt.shading_feature_mlp_layer2>0:
            if self.opt.agg_feat_xyz_mode != "None":
                feat = torch.cat([feat, pts], dim=-1)
            if self.opt.agg_intrp_order > 0:
                feat = torch.cat([feat, dists_flat], dim=-1)
            feat = self.block2(feat)

        if self.opt.shading_feature_mlp_layer3>0:
            if "1" in list(self.opt.point_color_mode) and s_pts['color'] is not None:
                sampled_color = s_pts['color'].view(-1, s_pts['color'].shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    sampled_color = sampled_color[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_color], dim=-1)
            if "1" in list(self.opt.point_dir_mode) and s_pts['dir'] is not None:
                sampled_dir = s_pts['dir'].view(-1, s_pts['dir'].shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    sampled_dir = sampled_dir[pnt_mask_flat, :]
                    sampled_dir = sampled_dir @ sampled_Rw2c if uni_w2c else (sampled_dir[..., None, :] @ sampled_Rw2c).squeeze(-2)
                if "0" in list(self.opt.point_dir_mode):
                    feat = torch.cat([feat, sampled_dir], dim=-1)
                elif "2" in list(self.opt.point_dir_mode):
                    sampled_dir = sampled_dir.detach()
                    dists_dir = F.normalize(dists.view(-1, dists.shape[-1])[pnt_mask_flat, :], dim=-1)
                    feat = torch.cat([feat, sampled_dir - dists_dir, torch.sum(sampled_dir*dists_dir, dim=-1, keepdim=True)], dim=-1)
                else:
                    in_viewdirs = viewdirs[..., None, :].repeat(1, K, 1).view(-1, viewdirs.shape[-1])
                    if self.opt.apply_pnt_mask > 0:
                        in_viewdirs = in_viewdirs[pnt_mask_flat.view(-1, K)[ray_valid].view(-1), :]
                    feat = torch.cat([feat, sampled_dir - in_viewdirs, torch.sum(sampled_dir*in_viewdirs, dim=-1, keepdim=True)], dim=-1)
            feat = self.block3(feat)

        if self.opt.agg_intrp_order == 1:
            # xyz_feat -weighted_sum-> loc_feat --> loc_alpha 
            NotImplemented

        elif self.opt.agg_intrp_order == 2: # xyz_feat --> xyz_alpha -weighted_sum-> loc_alpha
            alpha_in = feat
            if self.opt.agg_alpha_xyz_mode != "None":
                alpha_in = torch.cat([alpha_in, pts], dim=-1)
            alpha_out = self.alpha_branch(alpha_in)
            sdf_pts = alpha_out[...,:1]
            if self.opt.shading_alpha_channel_num > 1:
                feat = F.softplus(alpha_out[...,1:], beta=100)

            pts_gradient, agg_gradient = None, None
            pnt_sdf_var, pnt_feat_var, roughness_var = None, None, None
            # if weight > 0?
            pnt_mask_valid = s_pts['pnt_mask'].reshape(-1, K)[ray_valid]
            num_pnt_per_loc = pnt_mask_valid.sum(-1, keepdim=True)
            
            def compute_variance(pnt_attr, attr_mean):
                # pnt_attr_mean = \
                #         (pnt_attr * pnt_mask_valid[..., None]).sum(-2) / num_pnt_per_loc
                pnt_attr_var = \
                    (((pnt_attr - attr_mean[...,None,:])**2) * 
                        pnt_mask_valid[..., None]).sum(-2) / num_pnt_per_loc
                return pnt_attr_var

            if self.opt.agg_grad_mode == 'xyz':
                NotImplemented

            elif self.opt.agg_grad_mode == 'loc':
                # aggregate sdf first.
                if self.opt.apply_pnt_mask > 0:
                    sdf_holder = torch.zeros([B * R * SR * K, sdf_pts.shape[-1]], dtype=torch.float32, device=sdf_pts.device)
                    sdf_holder[pnt_mask_flat, :] = sdf_pts
                else:
                    sdf_holder = sdf_pts
                sdf_pts = sdf_holder.view(B * R * SR, K, sdf_holder.shape[-1])[ray_valid]
                sdf = torch.sum(sdf_pts * weight_valid, dim=-2)
                if self.opt.sdf_tv_weight > 0:
                    pnt_sdf_var = compute_variance(sdf_pts, sdf)

                # compute normal here
                grad_graph = (self.training or self.opt.return_normal_map) and (sample_ray_dirs is not None)
                if (not self.opt.depth_only and self.training and self.opt.learn_point_normal) \
                        or self.opt.return_agg_normal or self.opt.sdf_mode=='neus':
                    if sdf.requires_grad:
                        loc_gradient = torch.autograd.grad(sdf, sample_loc_w, torch.ones_like(sdf), 
                                                    retain_graph=grad_graph, create_graph=grad_graph)[0]
                        agg_gradient = loc_gradient.view([-1, 3])[ray_valid, :]
                        pts_gradient = agg_gradient
            
            if self.opt.sdf_mode == 'volsdf':
                beta = self.sdf_beta.abs() + self.sdf_beta_min
                self.sdf_act_val = beta.item()
                
                if self.opt.density_distill_weight > 0:
                    beta = beta.detach()

                sigma = 1 / beta * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta)) # Laplace CDF
                # sigma = sdf
                alpha = self.raw2out_density(sigma)
                # print(alpha_in.shape, alpha_in)
                sdf = sdf / self.opt.sdf_scale

            elif self.opt.sdf_mode == 'neus':
                if ray_dist is not None:
                    sec_dist = ray_dist.reshape(-1)[ray_valid][...,None]
                else:
                    sec_dist = self.opt.sample_stepsize
                if sample_ray_dirs is not None:
                    reg_grad = F.normalize(agg_gradient, dim=-1)
                    true_cos = (-viewdirs * reg_grad).sum(-1, keepdim=True)
                    iter_cos = -(F.relu(true_cos * 0.5 + 0.5) * (1.0 - self.opt.cos_anneal_ratio) +
                         F.relu(true_cos) * self.opt.cos_anneal_ratio)  # always non-positive
                else:
                    iter_cos = 0.9
                estimated_next_sdf = sdf + iter_cos * sec_dist * 0.5
                estimated_prev_sdf = sdf - iter_cos * sec_dist * 0.5

                inv_s = torch.exp(self.sdf_var * 10.).clip(1e-6, 1e6)
                self.sdf_act_val = self.sdf_var.item()

                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

                p = prev_cdf - next_cdf
                c = prev_cdf

                alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)


            alpha_placeholder = torch.zeros([total_len, 1], dtype=torch.float32, device=alpha.device)
            alpha_placeholder[ray_valid] = alpha
            
            if self.opt.depth_only:
                return alpha_placeholder, {'sdf': sdf, 'density': alpha}

            
            if self.opt.apply_pnt_mask > 0:
                feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
                feat_holder[pnt_mask_flat, :] = feat
            else:
                feat_holder = feat
            pnt_feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])[ray_valid]
            feat = torch.sum(pnt_feat * weight_valid, dim=-2)

            if self.opt.feat_tv_weight > 0:
                pnt_feat_var = compute_variance(pnt_feat, feat)
            
            # TODO add pred_gradient here, also add diffuse_color
            pred_gradient, diffuse_color, point_color = None, None, None
            normal = None
            specular, roughness = None, None
            pnt_diffuse_var, pnt_normal_var = None, None
            albedo, metallic, fresnel = None, None, None

            with torch.set_grad_enabled(self.training or self.opt.brdf_training):
                if self.opt.diffuse_branch_channel > 0:
                    diffuse_in = feat
                    diffuse = self.raw2out_color(self.diffuse_branch(diffuse_in))

                if self.opt.learn_point_color or self.opt.pnt_diffuse_weight > 0:
                    color_valid = s_pts['color'].reshape(-1, K, 3)[ray_valid]
                    point_color = (color_valid * attr_weight_valid).sum(-2)
                    point_color = self.gradient_clamp(point_color)
                    if self.opt.diffuse_tv_weight  > 0:
                        pnt_diffuse_var = compute_variance(color_valid, point_color)

                if self.opt.learn_diffuse_color:
                    if self.opt.diffuse_branch_channel >= 3:
                        diffuse_color = diffuse[...,:3]
                    else:
                        diffuse_color = point_color

                if self.opt.learn_point_normal:
                    dir_valid = s_pts['dir'].reshape(-1, K, 3)[ray_valid]
                    pred_gradient = (dir_valid * attr_weight_valid).sum(-2)
                    if self.opt.normal_tv_weight > 0:
                        pnt_normal_var = compute_variance(dir_valid, pred_gradient)
                
                if not self.opt.mlp_fast_forward:
                    if self.opt.learn_point_specular:
                        specular = (s_pts['specular'].reshape(-1, K, 1)[ray_valid] * attr_weight_valid).sum(-2).sigmoid()

                    if self.opt.learn_point_roughness:
                        if self.opt.shading_roughness_mlp_layer > 0:
                            raw_roughness = self.roughness_branch(feat)
                            roughness = torch.sigmoid(raw_roughness)
                            roughness = (self.opt.max_roughness - self.opt.min_roughness) * roughness + self.opt.min_roughness
                        else:
                            roughness_valid = s_pts['roughness'].reshape(-1, K, 1)[ray_valid]
                            roughness = (roughness_valid * attr_weight_valid).sum(-2).sigmoid() #.clamp(min=0.05, max=self.opt.max_roughness) #.sigmoid()
                            # roughness = self.gradient_clamp(roughness, min=self.opt.min_roughness, max=self.opt.max_roughness)
                            roughness = (self.opt.max_roughness - self.opt.min_roughness) * roughness + self.opt.min_roughness
                            if self.opt.roughness_tv_weight > 0:
                                roughness_var = compute_variance(roughness_valid, roughness)

                    if self.opt.use_microfacet_mlp:
                        if self.opt.shading_metallic_mlp_layer > 0:
                            metallic = torch.sigmoid(self.metallic_branch(feat) - 3)
                        fresnel = self.fresnel_branch(feat).sigmoid()
                    if self.opt.use_albedo_mlp:
                        albedo = self.raw2out_color(self.albedo_branch(diffuse_in))
            
            others = {'agg_gradient': agg_gradient, 'density': alpha, 'sdf': sdf, 
                  'diffuse_color': diffuse_color, 'pred_gradient': pred_gradient,
                  'specular': specular, 'roughness': roughness,
                  'pnt_diffuse_var': pnt_diffuse_var, 'pnt_normal_var': pnt_normal_var,
                  'pnt_sdf_var': pnt_sdf_var, 'pnt_feat_var': pnt_feat_var, 
                  'pnt_roughness_var': roughness_var, 'point_color': point_color,
                  'metallic': metallic, 'fresnel': fresnel, 'albedo': albedo
                } # 'xyz_gradient': pts_gradient, 

            if not self.opt.mlp_fast_forward:
                if sample_ray_dirs is not None and self.opt.orientation_weight > 0:
                    others['viewdirs'] = viewdirs

                if self.opt.relsdf_loss_weight > 0:
                    # others['sdf_pts'] = sdf_pts[...,0]
                    # others['dists'] = dists.detach().view(-1, K, 3)[ray_valid]
                    ray_depth = sample_loc[..., 2]
                    ray_delta = ray_depth[...,1:] - ray_depth[...,:-1]
                    ray_dist_d = (ray_dist[...,1:] + ray_dist[...,:-1]) / 2
                    cont_mask = (ray_delta - ray_dist_d).abs() < 0.2 * min(self.opt.vsize)
                    sdf_placeholder = sdf.new_zeros([total_len, 1])
                    sdf_placeholder[ray_valid] = sdf
                    sdf_placeholder = sdf_placeholder.reshape_as(ray_depth)

                    sdf_d = sdf_placeholder[...,1:] - sdf_placeholder[...,:-1]
                    # viewdirs_per_ray
                    ray_valid_ = ray_valid.reshape_as(ray_depth)
                    valid_mask = (ray_valid_[...,1:] * ray_valid_[...,:-1])
                    relsdf_mask = valid_mask * cont_mask

                    others['sdf_d'] = sdf_d[relsdf_mask] * self.opt.sdf_scale
                    others['ray_dist_d'] = ray_dist_d[relsdf_mask] * self.opt.sdf_scale
                    others['relsdf_mask'] = relsdf_mask
                    if agg_gradient is not None and 'est_delta' in self.opt.relsdf_loss_items:
                        relsdf_normal = agg_gradient
                        if self.opt.relsdf_normal_type == 'pred' and self.opt.learn_point_normal:
                            if normal is not None:
                                relsdf_normal = normal
                            elif self.opt.normal_anneal_ratio < 1:
                                relsdf_normal = (1 - self.opt.normal_anneal_ratio) * agg_gradient + \
                                    self.opt.normal_anneal_ratio * pred_gradient
                            else:
                                relsdf_normal = pred_gradient
                        grad_placeholder = sdf.new_zeros([total_len, 3])
                        grad_placeholder[ray_valid] = F.normalize(relsdf_normal.detach(), dim=-1)
                        grad_placeholder = grad_placeholder.reshape(*ray_depth.shape, 3)[..., :-1, :]
                        cos = (grad_placeholder * viewdirs_per_ray).sum(-1)
                        iter_cos = -torch.relu(-cos * 0.5 + 0.5) * (1.0 - self.opt.cos_anneal_ratio) + \
                            self.opt.cos_anneal_ratio * cos
                        est_sdf_d = iter_cos * ray_dist_d
                        if self.opt.relsdf_delta_thres > 0:
                            est_sdf_d = est_sdf_d * (sdf_d < 0) # since est_delta only applied to weight peek
                        others['est_sdf_d'] = est_sdf_d[relsdf_mask] * self.opt.sdf_scale

            if sample_ray_dirs is None: # or self.opt.use_microfacet_mlp:
                others['gradient'] = pred_gradient if self.opt.learn_point_normal else pts_gradient
                output_placeholder = None

                if self.opt.brdf_rendering:
                    with torch.set_grad_enabled(self.training or self.opt.brdf_training):
                        output = torch.cat([alpha, diffuse_color], -1)
                        output_placeholder = torch.zeros([total_len, self.opt.shading_color_channel_num + 1], 
                                                        dtype=torch.float32, device=output.device)
                        output_placeholder[ray_valid] = output
                return output_placeholder, others

            with torch.set_grad_enabled(grad_graph):
                if not self.opt.diffuse_branch_only:
                    color_in = feat
                    if self.opt.agg_color_xyz_mode != "None":
                        color_in = torch.cat([color_in, pts], dim=-1)
                    
                    # color_in = torch.cat([color_in, viewdirs], dim=-1)
                    if self.opt.rgbnet_normal_mode != 'v':
                        normal = agg_gradient
                        if self.opt.learn_point_normal:
                            if self.opt.normal_anneal_ratio < 1 and agg_gradient is not None:
                                normal = (1 - self.opt.normal_anneal_ratio) * agg_gradient + \
                                    self.opt.normal_anneal_ratio * pred_gradient
                            else:
                                normal = pred_gradient
                        normal = normal.detach() if self.opt.grad_detach else normal
                        normal = F.normalize(normal, dim=-1)
                        cos_v_n = None

                        if self.opt.normal_incidence_f0:
                            viewdirs = -normal

                        if 'r' in self.opt.rgbnet_normal_mode:
                            assert self.opt.viewdir_norm
                            cos_v_n = (-viewdirs * normal).sum(-1, keepdim=True)
                            ref = 2*cos_v_n * normal + viewdirs
                        if 'dot' in self.opt.rgbnet_normal_mode:
                            vn_dot = (-viewdirs*normal).sum(-1,keepdim=True) if cos_v_n is None else cos_v_n
                    
                    if self.opt.rot_v_dir and s_pts['ori_normal'] is not None:
                        ori_normal = (s_pts['ori_normal'].reshape(-1, K, 3)[ray_valid] * attr_weight_valid).sum(-2)
                        ori_normal = F.normalize(ori_normal, dim=-1)
                        ref = 2*(-viewdirs * normal).sum(-1, keepdim=True) * normal + viewdirs
                        cos_v_n_rot = (ori_normal * ref).sum(-1, keepdim=True)
                        viewdirs = -(2*cos_v_n_rot * ori_normal - ref)

                    if not self.opt.SH_color_branch:
                        if 'v' in self.opt.rgbnet_normal_mode:
                            if self.opt.SH_dir_deg > 0:
                                viewdirs_feat = self.dir_sh_encoder.sh_all(-viewdirs)
                            elif self.num_viewdir_freqs > 0:
                                viewdirs_feat = positional_encoding(viewdirs, self.num_viewdir_freqs, ori=True)[..., 3:]
                            else:
                                viewdirs_feat = viewdirs
                            color_in = torch.cat([color_in, viewdirs_feat], dim=-1)
                        if 'n' in self.opt.rgbnet_normal_mode:
                            if self.opt.SH_dir_deg > 0:
                                normal_feat = self.dir_sh_encoder.sh_all(normal)
                            elif self.opt.num_normal_freqs > 0:
                                normal_feat = positional_encoding(normal, self.opt.num_normal_freqs, ori=True)[..., 3:]
                            else:
                                normal_feat = normal
                            color_in = torch.cat([color_in, normal_feat], dim=-1)
                        if 'r' in self.opt.rgbnet_normal_mode:
                            if self.opt.SH_dir_deg > 0:
                                if not self.opt.ref_ide:
                                    ref_feat = self.dir_sh_encoder.sh_all(ref)
                                else:
                                    ref_feat = self.ide_encoder.encode(ref, 1/roughness)
                            elif self.opt.num_normal_freqs > 0:
                                ref_feat = positional_encoding(ref, self.opt.num_normal_freqs, ori=True)[..., 3:]
                            else:
                                ref_feat = ref
                            color_in = torch.cat([color_in, ref_feat], dim=-1)
                        if 'dot' in self.opt.rgbnet_normal_mode:
                            color_in = torch.cat([color_in, vn_dot], dim=-1)
                    else:
                        assert self.opt.viewdir_norm
                        if self.opt.rgbnet_normal_mode == 'v':
                            sh_coef = self.shcomputer.sh_all(viewdirs)
                        elif self.opt.rgbnet_normal_mode == 'r':
                            sh_coef = self.shcomputer.sh_all(ref)
                        else:
                            raise NotImplementedError

                    if self.opt.in_diffuse:
                        color_in = torch.cat([color_in, diffuse_color], dim=-1)

                    raw_color_out = self.color_branch(color_in)
                    if self.opt.monochro_specular:
                        raw_color_out = raw_color_out.expand(*raw_color_out.shape[:-1], 3)
                    if self.opt.SH_color_branch:
                        raw_color_out = (raw_color_out.reshape(*raw_color_out.shape[:-1], 3, -1) * sh_coef[...,None,:]).sum(-1)
                    color_output = self.raw2out_color(raw_color_out)

                else:
                    color_output = torch.zeros_like(diffuse_color)
                color_placeholder = color_output.new_zeros([total_len, self.opt.shading_color_channel_num])
                color_placeholder[ray_valid] = color_output

                output_placeholder = torch.cat([alpha_placeholder, color_placeholder], dim=-1)
                
        return output_placeholder, others