from typing import Dict, Optional
from tqdm import tqdm
from models.rendering.bg_model import NeRFxx

from models.rendering.diff_ray_marching import ray_march_neus
from .base_rendering_model import *
from .neural_points.neural_points import NeuralPoints
from .aggregators import PointAggregator, ViewMLP, SDFMLP
import os
from torch_scatter import scatter_sum, scatter_max
from .rendering.brdf_models import BRDFRender, linear2srgb
            
class NeuralPointsVolumetricModel(BaseRenderingModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        BaseRenderingModel.modify_commandline_options(parser, is_train)
        NeuralPoints.modify_commandline_options(parser, is_train)
        PointAggregator.modify_commandline_options(parser, is_train)
        BRDFRender.modify_commandline_options(parser, is_train)
        NeRFxx.modify_commandline_options(parser, is_train)

        # encoder
        parser.add_argument("--compute_depth", action='store_true', 
            help="If compute detph or not. If false, depth is only computed when depth is required by losses")

        parser.add_argument('--save_point_freq', type=int, default=100000, help='frequency of showing training results on console')
        parser.add_argument('--alter_step', type=int, default=0, help='0 for no alter,')
        parser.add_argument('--prob', type=int, default=0, help='will be set as 0 for normal traing and 1 for prob, ')
        parser.add_argument('--brdf_rendering', action='store_true')
        parser.add_argument('--depth_agg_mode', type=str, default='highest_peak', 
                                choices=['sum', 'curve', 'cutoff', 'curve_cutoff', 'highest_peak'])
        parser.add_argument('--depth_cutoff_thres', type=float, default=0.6)
        parser.add_argument('--low_trans_as_miss', type=float,  default=0)
        parser.add_argument('--bg_trans_thresh', type=float,  default=0.95)
        parser.add_argument('--optim_vis_depth', action='store_true')
        parser.add_argument('--no_grad', action='store_true')


    def add_default_color_losses(self, opt):
        if "coarse_raycolor" not in opt.color_loss_items:
            opt.color_loss_items.append('coarse_raycolor')
        if opt.fine_sample_num > 0:
            opt.color_loss_items.append('fine_raycolor')

    def add_default_visual_items(self, opt):
        opt.visual_items = ['gt_image', 'coarse_raycolor', 'queried_shading']
        if opt.fine_sample_num > 0:
            opt.visual_items.append('fine_raycolor')

    def run_network_models(self, **kwargs):
        return self.fill_invalid(self.net_ray_marching(**self.input, **kwargs))

    def fill_invalid(self, output):  # TODO put fill_invalid in the net_ray_marching
        # ray_mask:             torch.Size([1, 1024])
        # coarse_is_background: torch.Size([1, 336, 1])  -> 1, 1024, 1
        # coarse_raycolor:      torch.Size([1, 336, 3])  -> 1, 1024, 3
        # coarse_point_opacity: torch.Size([1, 336, 24]) -> 1, 1024, 24
        ray_mask = output["ray_mask"]
        B, OR = ray_mask.shape
        ray_inds = torch.nonzero(ray_mask) # 336, 2
        coarse_is_background_tensor = torch.ones([B, OR, 1], dtype=output["coarse_is_background"].dtype, device=output["coarse_is_background"].device)

        coarse_is_background_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_is_background"]
        output["coarse_is_background"] = coarse_is_background_tensor
        output['coarse_mask'] = 1 - coarse_is_background_tensor
        
        def fill_tensor(ten: torch.Tensor, fill_value: float=0):
            if ten.shape[:2] == torch.Size([B, OR]):
                return ten
            k_tensor = ten.new_full([B, OR, ten.shape[-1]], fill_value)
            k_tensor[ray_inds[..., 0], ray_inds[..., 1],:] = ten
            return k_tensor

        for k in ['coarse_raycolor', 'diffuse_raycolor', 'point_raycolor', 'specular_raycolor', 
            'brdf_diffuse_raycolor', 'brdf_specular_raycolor', 'brdf_combine_raycolor']:

            if k in output and output[k] is not None:
                if "bg_ray" in self.input:
                    raycolor_tensor = coarse_is_background_tensor * self.input["bg_ray"]
                    raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] += output[k][0]
                elif "sphere_bg" in output:
                    raycolor_tensor =  output['sphere_bg'] * coarse_is_background_tensor
                    raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] += output[k][0]
                else:
                    raycolor_tensor = self.tonemap_func(
                        torch.ones([B, OR, 3], dtype=output[k].dtype, device=output[k].device) * self.input["bg_color"][None, ...])
                    raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output[k]
                output[k] = raycolor_tensor

        if "coarse_point_opacity" in output:
            output["coarse_point_opacity"] = fill_tensor(output["coarse_point_opacity"], 0)

        if "queried_shading" in output:
            output["queried_shading"] = fill_tensor(output["queried_shading"], 1)

        if not self.net_ray_marching.training:
            for k in ['normal_map', 'agg_normal', 'pred_normal']:
                if k in output and k in self.opt.visual_items:
                    output[k] = fill_tensor((output[k] + 1)/2, 1) # 1 # rescale [-1,1] -> [0,1]
                
            if 'coarse_depth' in output and 'coarse_depth' in self.opt.visual_items:
                output['coarse_depth'] = fill_tensor(output['coarse_depth'][...,None], 0)
            
            for k in ['roughness', 'metallic', 'fresnel', 'albedo']:
                if k in output and k in self.opt.visual_items:
                    output[k] = fill_tensor(output[k], 1)

            if 'shadow' in output and 'shadow' in self.opt.visual_items:
                output['shadow'] = fill_tensor(output['shadow'], 1)

            if 'vis_map' in output and 'vis_map' in self.opt.visual_items:
                output['vis_map'] = fill_tensor(output['vis_map'], 1)

            # just for debug...
            for k in ['blend_weight', 'ray_depth', 'raw_normal']:
                if k in self.opt.visual_items:
                    output[k] = fill_tensor(output[k], 0)

        if self.opt.prob == 1 and "ray_max_shading_opacity" in output:
            # print("ray_inds", ray_inds.shape, torch.sum(output["ray_mask"]))
            output = self.unmask(ray_inds, output, 
                ["ray_max_sample_loc_w", "ray_max_shading_opacity", "shading_avg_color", "shading_avg_dir", 
                 "shading_avg_conf", "shading_avg_embedding", "ray_max_far_dist", "shading_avg_normal", 
                 "shading_avg_roughness"], B, OR)
        return output

    def unmask(self, ray_inds, output, names, B, OR):
        for name in names:
            if name in output and output[name] is not None:
                name_tensor = torch.zeros([B, OR, *output[name].shape[2:]], dtype=output[name].dtype, device=output[name].device)
                name_tensor[ray_inds[..., 0], ray_inds[..., 1], ...] = output[name]
                output[name] = name_tensor
        return output

    def get_additional_network_params(self, opt):
        param = {}
        # additional parameters

        self.aggregator = self.check_getAggregator(opt)

        self.is_compute_depth = opt.compute_depth or opt.depth_loss_items
        checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, '{}_net_ray_marching.pth'.format(opt.resume_iter))
        checkpoint_path = checkpoint_path if os.path.isfile(checkpoint_path) else None
        if opt.num_point > 0:
            self.neural_points = NeuralPoints(opt.point_features_dim, opt.num_point, opt, self.device, checkpoint=checkpoint_path, feature_init_method=opt.feature_init_method, reg_weight=0., feedforward=opt.feedforward)
        else:
            self.neural_points = None

        add_property2dict(param, self, [
            'aggregator', 'is_compute_depth', "neural_points", "opt",
        ])
        add_property2dict(param, opt, [
            'num_pos_freqs', 'num_viewdir_freqs'
        ])

        return param

    def create_network_models(self, opt):

        params = self.get_additional_network_params(opt)
        # network
        self.net_ray_marching = NeuralPointsRayMarching(
            **params, **self.found_funcs)

        self.model_names = ['ray_marching'] if getattr(self, "model_names", None) is None else self.model_names + ['ray_marching']

        # parallel
        if self.opt.gpu_ids: # very likely not working...
            self.net_ray_marching.to(self.device)
            self.net_ray_marching = torch.nn.DataParallel(
                self.net_ray_marching, self.opt.gpu_ids)


    def check_getAggregator(self, opt, **kwargs):
        if opt.which_agg_model == "sdfmlp":
            aggregator = SDFMLP(opt)
        else:
            aggregator = ViewMLP(opt)
        return aggregator


    def optimize_parameters(self, backward=True, total_steps=0, **kwargs):
        with torch.set_grad_enabled(backward):
            self.forward(**kwargs)
            self.update_rank_ray_miss(total_steps)
        if backward and self.output['ray_mask'].any(): 
            self.backward(total_steps)

    def update_rank_ray_miss(self, total_steps):
        raise NotImplementedError

    def density_n_normal(self, chunk_size=2048):
        # get xyz:
        # queried xyz's neighbors
        # mlp forward
        # return density & normal
        num_pts = self.neural_points.xyz.shape[0]
        outputs = {}
        for i in tqdm(range(0, num_pts, chunk_size)):
            pts_chunk = self.neural_points.xyz[i:i+chunk_size]
            # MLP TODO
            output = self.net_ray_marching(queried_xyz=pts_chunk, xyz_forward=True)
            if output['sdf'].shape[0] != pts_chunk.shape[0]:
                print(f'WTF! {i}')
            for k, v in output.items():
                if v is None: continue
                if k not in outputs: outputs[k] = []
                outputs[k].append(v.detach())
            [optimizer.zero_grad() for optimizer in self.optimizers]
            torch.cuda.empty_cache()
        outputs = {k:torch.cat(v, 0) for k, v in outputs.items()}
        return outputs

def safe_sum_one_weights(weight: torch.Tensor):
    return weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)

# @torch.jit.script
def curved_weights(weight: torch.Tensor):
    s_curve = lambda x: torch.where(x<0.5, 2*x**2, 1-2*(1-x)**2)
    # s_curve = lambda x: torch.where(x<0.5, 4*x**3, 1-4*(1-x)**3)
    return safe_sum_one_weights(s_curve(weight))

# @torch.jit.script
def cutoff_weights(weight: torch.Tensor, thresh: float):
    accum_weight = weight.cumsum(-1)
    m = accum_weight < thresh
    m_shift = torch.cat([torch.ones_like(m[...,:1]), m], -1)
    m_f = (m | m_shift[...,:-1])
    return safe_sum_one_weights(weight * m_f)

def highest_peak_weights(weight: torch.Tensor, ray_dist=None, ray_depth=None, radius=5):
    max_w, max_idx = weight.max(dim=-1, keepdim=True)
    SR = weight.shape[-1]
    idx = torch.arange(SR, device=weight.device).reshape(1,1,SR)
    mask = (idx - max_idx).abs() <= radius
    if ray_dist is not None and ray_depth is not None:
        sum_range = torch.arange(0, radius)[None, None, :].to(weight.device)
        upper_idx = (max_idx + sum_range).clamp(0, SR-1)
        upper_bnd = torch.gather(ray_dist, 2, upper_idx).sum(-1, keepdim=True)
        lower_idx = (max_idx - sum_range).clamp(0, SR-1)
        lower_bnd = torch.gather(ray_dist, 2, lower_idx).sum(-1, keepdim=True)
        max_depth = torch.gather(ray_depth, 2, max_idx)
        upper_mask = (ray_depth - max_depth) <= upper_bnd
        lower_mask = (max_depth - ray_depth) <= lower_bnd
        mask = upper_mask * lower_mask * mask
    return safe_sum_one_weights(weight * mask)

class NeuralPointsRayMarching(nn.Module):
    def __init__(self,
             tonemap_func=None,
             render_func=None,
             blend_func=None,
             aggregator=None,
             is_compute_depth=False,
             neural_points=None,
             opt=None,
             num_pos_freqs=0,
             num_viewdir_freqs=0,
             **kwargs):
        super(NeuralPointsRayMarching, self).__init__()

        self.aggregator = aggregator

        self.num_pos_freqs = num_pos_freqs
        self.num_viewdir_freqs = num_viewdir_freqs
        # ray generation

        self.render_func = render_func
        self.blend_func = blend_func

        self.tone_map = tonemap_func
        self.return_color = True
        self.opt = opt
        self.neural_points = neural_points
        
        self.brdf_renderer = None
        if self.opt.brdf_rendering or 'vis_map' in self.opt.visual_items:
            self.brdf_renderer = BRDFRender(opt)

        if self.opt.bgmodel == 'sphere':
            self.bgmodel = NeRFxx(self.opt.bg_num_sample, self.opt.sphere_bound)
            if self.opt.brdf_rendering:
                self.bgmodel.fixed = True

    def forward(self,
                campos=None,
                raydir=None,
                gt_image=None,
                bg_color=None,
                camrotc2w=None,
                pixel_idx=None,
                near=None,
                far=None,
                focal=None,
                h=None,
                w=None,
                intrinsic=None,
                xyz_forward=False, queried_xyz=None, empty_check=False,
                **kargs):
        output = {}
        if not xyz_forward:
            # # B, channel, 292, 24, 32;      B, 3, 294, 24, 32;     B, 294, 24;     B, 291, 2
            s_pts, sample_loc_w, sample_ray_dirs, ray_mask_tensor, vsize, grid_vox_sz, sample_loc_len =\
              self.neural_points({"pixel_idx": pixel_idx, "camrotc2w": camrotc2w, "campos": campos, "near": near, "far": far, 
                        "focal": focal, "h": h, "w": w, "intrinsic": intrinsic,"gt_image":gt_image, "raydir":raydir})
            sample_viewdir = sample_ray_dirs
        else:
            # 
            mlp_fast_forward_ori = self.opt.mlp_fast_forward 
            self.opt.mlp_fast_forward = True
            s_pts, vsize, grid_vox_sz = self.neural_points.xyz_forward(queried_xyz, empty_check)
            B, R, SR, K, _ = s_pts['xyz'].shape
            sample_loc_w = queried_xyz.view(B, R, SR, 3)
            sample_viewdir = None
            

        if self.opt.fixed_weight_copy and 'model_fixed' in kargs and kargs['model_fixed'] is not None:
            with torch.no_grad():
                s_pts_fixed = {k: v for k, v in s_pts.items() if k!='embedding'}
                s_pts_fixed['embedding'] = torch.index_select(
                        kargs['model_fixed'].neural_points.points_embeding, 1, s_pts['pidx'].reshape(-1)
                    ).reshape_as(s_pts['embedding'])

        if xyz_forward or (self.opt.brdf_rendering and not self.opt.brdf_mlp):
            sample_ray_dirs = None

        if self.opt.return_normal_map or (self.opt.learn_point_normal and self.opt.which_agg_model=='viewmlp'):
            sample_loc_w.requires_grad_(True)
        if self.opt.which_agg_model == 'sdfmlp':
            if self.opt.agg_grad_mode == 'xyz':
                s_pts['xyz'].requires_grad_(True)
            elif self.opt.agg_grad_mode == 'loc':
                sample_loc_w.requires_grad_(True)
        if camrotc2w is not None:
            s_pts['xyz_pers'] = \
                self.neural_points.w2pers(s_pts['xyz'].reshape(-1,3), camrotc2w, campos).reshape_as(s_pts['xyz'])
            sample_loc_pers = self.neural_points.w2pers(sample_loc_w.reshape(-1,3), camrotc2w, campos).reshape_as(sample_loc_w)
        else:
            s_pts['xyz_pers'], sample_loc_pers = None, None

        ray_dist = None
        if not xyz_forward:
            with torch.no_grad():
                if sample_loc_len is None or not self.opt.use_sample_len:
                    ray_dist = torch.cummax(sample_loc_pers[..., 2], dim=-1)[0]
                    ray_dist = torch.cat([ray_dist[..., 1:] - ray_dist[..., :-1], 
                                    torch.full((ray_dist.shape[0], ray_dist.shape[1], 1), vsize[2], device=ray_dist.device)
                                ], dim=-1)
                    mask = ray_dist < 1e-8
                    mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
                    # mask = mask.to(torch.float32)
                    ray_dist = ray_dist * (~mask) + mask * vsize[2]
                else:
                    ray_dist = sample_loc_len
                
        if self.opt.which_agg_model == 'sdfmlp':
            beta = self.aggregator.sdf_beta.abs().detach() + self.aggregator.sdf_beta_min # using detached beta 

        with torch.set_grad_enabled((self.training and not self.opt.brdf_rendering and not self.opt.no_grad) 
            or self.opt.return_agg_normal or self.opt.return_normal_map 
            or self.opt.brdf_joint_optim or self.opt.sdf_mode=='neus'):
            # TODO early stop
            decoded_features, ray_valid, weight, conf_coefficient, others =\
                self.aggregator(s_pts, sample_loc_pers, sample_loc_w, 
                                    sample_ray_dirs, vsize, grid_vox_sz, ray_dist)

        if self.opt.fixed_weight_copy and 'model_fixed' in kargs and kargs['model_fixed'] is not None:
            with torch.no_grad():
                decoded_features_fixed, _, _, _, others_fixed =\
                    kargs['model_fixed'].aggregator(s_pts_fixed, sample_loc_pers, sample_loc_w, 
                                    sample_ray_dirs, vsize, grid_vox_sz, ray_dist)
                if self.training and self.opt.density_distill_weight > 0 and sample_loc_w.numel() > 0:
                    output['reg_density_fixed'] = others_fixed['density'] * beta
                    output['reg_density'] = others['density'] * beta
                
                                    
        with torch.set_grad_enabled(self.training):
            if xyz_forward:
                self.opt.mlp_fast_forward = mlp_fast_forward_ori
                if not empty_check:
                    # directly return sdf and normals
                    return others
                else:
                    return others, s_pts['pnt_mask'].any(-1).view(-1)
            
            # raydir: N x Rays x 3sampled_color
            # raypos: N x Rays x Samples x 3
            # ray_dist: N x Rays x Samples
            # ray_valid: N x Rays x Samples
            # ray_features: N x Rays x Samples x Features
            # Output
            # ray_color: N x Rays x 3
            # point_color: N x Rays x Samples x 3
            # opacity: N x Rays x Samples
            # acc_transmission: N x Rays x Samples
            # blend_weight: N x Rays x Samples x 1
            # background_transmission: N x Rays x 1

            # ray march
            output["queried_shading"] = torch.logical_not(torch.any(ray_valid, dim=-1, keepdims=True)).repeat(1, 1, 3).to(torch.float32)

            ray_dist *= ray_valid.float()

            if self.return_color and not self.opt.depth_only:
                if "bg_ray" in kargs:
                    bg_color = None
                if self.opt.bgmodel=='sphere':
                    bg_color = None
                    with torch.set_grad_enabled(self.training and not self.opt.brdf_rendering):
                        output['sphere_bg'] = self.bgmodel(campos, raydir[0])
                ray_march_fn = ray_march if self.opt.sdf_mode != 'neus' else ray_march_neus
                (
                    ray_color,
                    point_color,
                    opacity,
                    acc_transmission,
                    blend_weight,
                    background_transmission,
                    _,
                ) = ray_march_fn(ray_dist, ray_valid, decoded_features, self.render_func, self.blend_func)

            else:
                (
                    opacity,
                    acc_transmission,
                    blend_weight,
                    background_transmission,
                    _,
                ) = alpha_ray_march(ray_dist, ray_valid, decoded_features, self.blend_func)

            output["coarse_is_background"] = background_transmission
            contour_mask = background_transmission > self.opt.bg_trans_thresh
            ray_mask_tensor = ray_mask_tensor.bool()
            output["ray_mask"] = ray_mask_tensor
            ray_valid_nz = ray_valid.nonzero()
            ray_valid_idx, ray_valid_order = ray_valid_nz[...,1], ray_valid_nz[...,2]
            if self.opt.sdf_center_weight > 0: output['ray_valid_order'] = ray_valid_order
            if self.opt.specular_weighted_sparse:
                if not self.opt.diffuse_branch_only:
                    output["specularity"] = point_color.detach()[ray_valid].norm(dim=-1)
                else:
                    output["specularity"] = torch.zeros(1, dtype=blend_weight.dtype, device=blend_weight.device)

            if others is not None:
                for key, val in others.items():
                    output[key] = val 

            if self.opt.fixed_weight_copy and 'model_fixed' in kargs and kargs['model_fixed'] is not None:
                with torch.no_grad():
                    _, _, blend_weight_fixed, background_transmission_fixed, _ = \
                        alpha_ray_march(ray_dist, ray_valid, decoded_features_fixed, self.blend_func)
                
                contour_mask = background_transmission_fixed > self.opt.bg_trans_thresh
                
                if self.opt.fixed_blend_weight:
                    blend_weight = blend_weight_fixed
                    background_transmission = background_transmission_fixed
                    if kargs['model_fixed'].opt.depth_only:
                        ray_color = torch.sum(point_color * blend_weight, dim=-2, keepdim=False)
                
                if not kargs['model_fixed'].opt.depth_only:
                    with torch.no_grad():
                        if others_fixed is not None:
                            output['diffuse_color'] = others_fixed['diffuse_color']
                        point_color_fixed = decoded_features_fixed[...,1:4]
                        ray_color_fixed = torch.sum(point_color_fixed * blend_weight, dim=-2, keepdim=False) 
                        ray_color = ray_color_fixed

            if self.training and self.opt.sparse_loss_weight > 0 and sample_loc_w.numel() > 0:
                max_dist, _ = ray_dist.max(-1, keepdim=True)
                output['dist_weight'] = (ray_dist / max_dist.clamp_min(1e-5))[ray_valid]
                if self.opt.which_agg_model == 'sdfmlp': 
                    output['reg_density'] = (0.5 + 0.5 * output['sdf'].sign() *\
                        torch.expm1(-output['sdf'].abs() * self.opt.sdf_scale / beta))[...,0] # rescale to [0, 1]
                #     output['reg_density'] = (1/beta * (output['sdf'] < 0) + 
                #                     output['sdf'].sign() * output['density'])[...,0]
                # else:
                    # output['reg_density'] = beta * output['density'][...,0] # rescale back to [0, 1]
                # output.pop('density')

            if weight is not None:
                output["weight"] = weight.detach()
                output["blend_weight"] = blend_weight.detach()
                output["conf_coefficient"] = conf_coefficient

            if self.opt.compute_depth or self.opt.brdf_rendering:
                if sample_loc_w.numel() > 0:
                    if self.opt.depth_from_mesh:
                        focal = intrinsic[0,0,0] if focal is None else focal
                        with torch.no_grad():
                            _, depth = self.brdf_renderer.mesh_renderer.render_depth(campos[0], camrotc2w[0], 
                                focal.item(), h.item(), w.item(), near.item(), far.item())
                            pid = pixel_idx.reshape(1, -1, 2)[ray_mask_tensor].long()
                            output["coarse_depth"] = depth[pid[...,1], pid[...,0], :][None,...,0]
                    else:
                        with torch.set_grad_enabled(self.opt.optim_vis_depth):
                            if self.opt.fixed_weight_copy and 'model_fixed' in kargs and kargs['model_fixed'] is not None:
                                depth_weight = blend_weight_fixed.view(blend_weight.shape[:3])
                            else:
                                depth_weight = blend_weight.view(blend_weight.shape[:3])
                            ray_depth = sample_loc_pers[...,2].detach()
                            depth_weight = safe_sum_one_weights(depth_weight)

                            if self.opt.depth_agg_mode == 'sum':
                                depth_weight = torch.where(~contour_mask, depth_weight, depth_weight.new_zeros(1))
                            elif self.opt.depth_agg_mode == 'curve':
                                depth_weight = torch.where(~contour_mask, 
                                            curved_weights(depth_weight), depth_weight.new_zeros(1))
                            elif self.opt.depth_agg_mode == 'cutoff':
                                depth_weight = torch.where(~contour_mask, 
                                            cutoff_weights(depth_weight, self.opt.depth_cutoff_thres), depth_weight.new_zeros(1))
                            elif self.opt.depth_agg_mode == 'curve_cutoff':
                                depth_weight = torch.where(~contour_mask, cutoff_weights(
                                    curved_weights(depth_weight), self.opt.depth_cutoff_thres), depth_weight.new_zeros(1))
                            elif self.opt.depth_agg_mode == 'highest_peak':
                                depth_weight = torch.where(~contour_mask, 
                                            highest_peak_weights(depth_weight, ray_dist, ray_depth), depth_weight.new_zeros(1))
                            avg_depth = (depth_weight * ray_depth).sum(-1)
                            output["coarse_depth"] = avg_depth #.clamp(0, far.reshape(-1))
                else:
                    output["coarse_depth"] = torch.zeros_like(sample_loc_w[...,0,0])
                if self.opt.depth_only:
                    return output

            require_diffuse = ['diffuse_raycolor'] if self.opt.residual_rgb else []
            
            blend_diffuse_key = 'diffuse_raycolor'
            
            if self.opt.pnt_diffuse_weight > 0:
                require_diffuse.append('point_raycolor')

            for vk, ok in [('diffuse_raycolor', 'diffuse_color'), 
                            ('point_raycolor', 'point_color'), ('diffuse_shadowed', 'diffuse_shadowed')]:
                if vk in self.opt.visual_items or vk in require_diffuse:
                    if sample_loc_w.numel() > 0:
                        diffuse_placeholder = torch.zeros(s_pts['pnt_mask'].shape[1], 3, 
                                                                device=sample_loc_w.device)
                        # ray_valid = torch.any(s_pts['pnt_mask'], dim=-1)
                        output[vk] = scatter_sum(
                            blend_weight[ray_valid] * output[ok], 
                            ray_valid_idx, 0, diffuse_placeholder)[None,...]
                    else:
                        output[vk] = torch.ones_like(sample_loc_w[...,0,:])

            specular_color = None
            if self.opt.residual_rgb: #and not self.opt.use_microfacet_mlp:
                specular_color = ray_color
                ray_color = specular_color + output[blend_diffuse_key] # * self.opt.albedo_slope + self.opt.albedo_bias
                output['specular_raycolor'] = specular_color

            if bg_color is not None:
                output["coarse_raycolor"] = ray_color + bg_color.to(ray_color.device).float().view(
                    background_transmission.shape[0], 1, 3) * background_transmission
            else:
                output["coarse_raycolor"] = ray_color

            output["coarse_raycolor"] = self.tone_map(output["coarse_raycolor"])
            output["coarse_point_opacity"] = opacity

            # if self.opt.ref_normal_reg and 'agg_gradient' in output:
            if self.opt.ref_weight_detach:
                output["blend_weight"] = blend_weight[ray_valid].detach()
            else:
                output["blend_weight"] = blend_weight[ray_valid]
            if self.opt.relsdf_loss_weight > 0 and 'sdf' in output:
                output["relsdf_weight"] = blend_weight.detach()
                # output["relsdf_sdf"] = output['sdf'].new_zeros(ray_valid.shape).masked_scatter_(ray_valid, output['sdf'])[...,None]
            
            # just for debug...
            if 'blend_weight' in self.opt.visual_items:
                output["blend_weight"] = blend_weight[...,0].detach()
            if 'ray_depth' in self.opt.visual_items:
                output['ray_depth'] =  sample_loc_pers[...,2].detach()

            if self.opt.return_normal_map:
                with torch.set_grad_enabled(True):
                    if sample_loc_w.numel() > 0:
                        sigma = decoded_features[..., 0] * ray_valid.float()
                        gradients = -torch.autograd.grad(sigma, sample_loc_w, torch.ones_like(sigma), 
                                            retain_graph=self.training, create_graph=self.training)[0]
                        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-12)
                        normal_map = torch.sum(blend_weight * normals, -2)
                        # output['raw_normal'] = normals.detach()
                        output['normal_map'] = normal_map
                    else:
                        output['normal_map'] = torch.zeros_like(sample_loc_w[...,0,:])
            
            def scatter_normal(grad_key:str):
                if sample_loc_w.numel() > 0:
                    normal_placeholder = torch.zeros(s_pts['pnt_mask'].shape[1], 3, 
                                                            device=sample_loc_w.device)
                    # ray_valid = torch.any(s_pts['pnt_mask'], dim=-1)
                    grad = output[grad_key]
                    normals = F.normalize(grad, dim=-1)
                    scatter_sum(
                        blend_weight[ray_valid] * normals, ray_valid_idx, 0, normal_placeholder)
                    out_normal = normal_placeholder[None,...]
                else:
                    out_normal = torch.zeros_like(sample_loc_w[...,0,:])
                return out_normal

            if self.opt.return_agg_normal:
                output["agg_normal"] = scatter_normal('agg_gradient')
            if self.opt.return_pred_normal or self.opt.brdf_rendering:
                output["pred_normal"] = scatter_normal('pred_gradient')

            # import IPython; IPython.embed(); exit()
            
            if self.opt.brdf_rendering or 'vis_map' in self.opt.visual_items:
                if sample_loc_w.numel() > 0:
                    output["vis_map"], light_dir, surf_xyz = self.brdf_renderer.compute_visibility(
                        output["coarse_depth"], 
                        pixel_idx.reshape(*raydir.shape[:-1],2)[ray_mask_tensor][None,:], 
                        intrinsic[0], camrotc2w, campos, 
                        output['pred_normal'].detach() if 'pred_normal' in output else None) # [1, N_ray, N_light]
                    output['vis_map'] = torch.where(~contour_mask, output['vis_map'], output['vis_map'].new_ones(1))
                else:
                    output['vis_map'] = torch.ones(*sample_loc_w.shape[:2], self.brdf_renderer.num_lights, 
                                                device=sample_loc_w.device)
                    output['pixel_map'] = torch.ones(*sample_loc_w.shape[:2], 5, self.brdf_renderer.num_lights, 
                                                device=sample_loc_w.device)
                    light_dir = None

            if self.opt.brdf_rendering:
                '''
                '''
                B, OR = ray_mask_tensor.shape
                if self.opt.light_tv_weight > 0 or self.opt.light_achro_weight > 0 or \
                    (self.opt.light_reg_weight > 0 and self.opt.light_reg_mode == 'act'):
                    output['light_map'] = self.brdf_renderer.light_map
                if self.opt.light_reg_weight > 0 and self.opt.light_reg_mode == 'raw':
                    output['raw_light_map'] = self.brdf_renderer.raw_light_map
                if sample_loc_w.numel() > 0:
                    if self.opt.learn_point_roughness:
                        roughness_placeholder = torch.zeros(s_pts['pnt_mask'].shape[1], 1, 
                                                            device=sample_loc_w.device)
                        scatter_sum(
                                blend_weight[ray_valid] * output['roughness'], 
                                ray_valid_idx, 0, roughness_placeholder)
                        output['roughness'] = roughness_placeholder[None,...]

                        if self.opt.roughness_anneal_ratio < 1:
                            output['roughness'] = (1 - self.opt.roughness_anneal_ratio) * self.opt.default_roughness + \
                                    self.opt.roughness_anneal_ratio * output['roughness']
                        
                    else:
                        output['roughness'] = None

                    specular = specular_color
                    if self.opt.use_microfacet_mlp:
                        for k in ['metallic', 'fresnel']:
                            if output[k] is not None:
                                param_placeholder = torch.zeros(s_pts['pnt_mask'].shape[1], output[k].shape[-1], 
                                                                    device=sample_loc_w.device)
                                scatter_sum(blend_weight[ray_valid] * output[k], ray_valid_idx, 0, param_placeholder)
                                output[k] = param_placeholder[None,...]
                    
                    if self.opt.use_albedo_mlp and output['albedo'] is not None:
                        param_placeholder = torch.zeros(s_pts['pnt_mask'].shape[1], output['albedo'].shape[-1], 
                                                                    device=sample_loc_w.device)
                        scatter_sum(blend_weight[ray_valid] * output['albedo'], ray_valid_idx, 0, param_placeholder)
                        output['albedo'] = param_placeholder[None,...]
                        albedo = output['albedo']
                    else:
                        albedo = ray_color if not self.opt.residual_rgb else output['diffuse_raycolor']

                    in_normal = output['pred_normal'].detach() # by default detach()...!!!
                    
                    output['brdf_combine_raycolor'], others_brdf = self.brdf_renderer(
                        light_dir, sample_viewdir[...,0,:], in_normal, albedo,
                        output['roughness'], output['vis_map'], surf_xyz, background_transmission, bg_color,
                        specular, output)

                    for k in ['diffuse', 'specular']:
                        output[f'brdf_{k}_raycolor'] = others_brdf[f'{k}'] #.clamp(0,1)
                        # output[f'{k}_raycolor'] = output[f'{k}_raycolor'] #**(1/2.2)
                        if bg_color is not None:
                            output[f'brdf_{k}_raycolor'] += bg_color[0] * background_transmission
                        if self.opt.brdf_tonemap=='srgb':
                            output[f'brdf_{k}_raycolor'] = linear2srgb(output[f'brdf_{k}_raycolor'])

                    output['albedo'] = others_brdf['albedo']
                    if self.opt.brdf_tonemap=='srgb': output['albedo'] = linear2srgb(output['albedo']) 

                    if self.opt.mlp_diffuse_cor_weight > 0 and 'vis_raycolor' in others_brdf:
                        assert 'point_raycolor' in output
                        if self.opt.brdf_tonemap=='srgb':
                            output['re_diffuse_raycolor'] = linear2srgb(others_brdf['albedo'])
                            output['point_raycolor'] = output['point_raycolor'] / linear2srgb(others_brdf['vis_raycolor'] / np.pi).clamp_min(0.001)
                        else:
                            output['re_diffuse_raycolor'] = others_brdf['albedo']
                            output['point_raycolor'] = output['point_raycolor'] / others_brdf['vis_raycolor'].clamp_min(0.001) * np.pi

                else:
                    output['brdf_combine_raycolor'] = torch.ones_like(sample_loc_w[...,0,:])
                    output['brdf_diffuse_raycolor'] = torch.ones_like(sample_loc_w[...,0,:])
                    output['brdf_specular_raycolor'] = torch.ones_like(sample_loc_w[...,0,:])
                    output['albedo'] = torch.ones_like(sample_loc_w[...,0,:])
                    output['roughness'] = torch.zeros_like(sample_loc_w[...,0,:1])
                    if self.opt.use_microfacet_mlp:
                        output['metallic'] = torch.zeros_like(sample_loc_w[...,0,:1]) if self.opt.shading_metallic_mlp_layer > 0 else None
                        output['fresnel'] = torch.zeros_like(sample_loc_w[...,0,:self.opt.fresnel_branch_channel])
                    if self.opt.mlp_diffuse_cor_weight > 0 or 're_diffuse_raycolor' in self.opt.visual_items:
                        output['re_diffuse_raycolor'] = torch.ones_like(sample_loc_w[...,0,:])
            
            for k in ['diffuse_raycolor', 'specular_raycolor']:
                if k in output and output[k] is not None:
                    if (not self.training and k not in self.opt.visual_items):
                        del output[k]
                    else:
                        # output[f'{k}_raycolor'] = output[f'{k}_raycolor'] #**(1/2.2)
                        if bg_color is not None:
                            output[k] += bg_color[0] * background_transmission
                        output[k] = self.tone_map(output[k])

            if not self.training:
                if self.opt.return_agg_normal:
                    output["agg_normal"] +=  1 * background_transmission
                if self.opt.return_pred_normal:
                    output["pred_normal"] +=  1 * background_transmission
                if 'albedo' in output and output['albedo'] is not None:
                    output['albedo'] += 1 * background_transmission
                if 'roughness' in output and output['roughness'] is not None:
                    output['roughness'] += 1 * background_transmission

            if self.opt.visible_prune_thresh > 0:
                with torch.no_grad():
                    point_blend_weight = blend_weight.expand_as(s_pts['pnt_mask'])[s_pts['pnt_mask']]
                    point_idx = s_pts['pidx'][s_pts['pnt_mask']]
                    scatter_max(point_blend_weight, point_idx, dim=-1, out=self.neural_points.points_max_weight.reshape(-1))
                    self.neural_points.weight_update_cnt += 1

            if self.opt.prob == 1 and output["coarse_point_opacity"].shape[1] > 0 :
                B, OR, _, _ = s_pts['pnt_mask'].shape
                if weight is not None:
                    with torch.no_grad():
                        output["ray_max_shading_opacity"], opacity_ind = torch.max(output["coarse_point_opacity"], dim=-1, keepdim=True)
                        opacity_ind=opacity_ind[..., None] # 1, 1024, 1, 1, 1

                        output["ray_max_sample_loc_w"] = torch.gather(
                            sample_loc_w, 2, opacity_ind.expand(-1, -1, -1, sample_loc_w.shape[-1])).squeeze(2) # 1, 1024, 3
                        weight = torch.gather(
                            weight*conf_coefficient, 2, opacity_ind.expand(-1, -1, -1, weight.shape[-1])).squeeze(2).unsqueeze(-1) # 1, 1024, 8

                        opacity_ind = opacity_ind[...,None] # 1, 1024, 1, 1, 1
                        # Gather all fields at once using a for loop
                        fields = ["xyz", "color", "dir", "conf", "embedding", "roughness", "specular"]
                        for field in fields:
                            if s_pts[field] is not None:
                                s_pts[field] = torch.gather(
                                    s_pts[field], 2, 
                                    opacity_ind.expand(-1, -1, -1, *s_pts[field].shape[-2:])).squeeze(2) # 1, 1024, 3
                            else:
                                s_pts[field] = None

                        sampled_xyz_max_opacity = s_pts['xyz']
                        output["ray_max_far_dist"] = torch.min(
                                torch.norm(sampled_xyz_max_opacity - output["ray_max_sample_loc_w"][..., None,:], dim=-1), 
                            axis=-1, keepdim=True)[0]

                    fields = ["color", "dir", "conf", "embedding", "roughness", "specular", "normal"]
                    for field in fields:
                        if s_pts[field] is not None:
                            output[f"shading_avg_{field}"] = torch.sum(s_pts[field] * weight, dim=-2)
                        else:
                            None
                else:
                    output.update({
                        "ray_max_shading_opacity": torch.zeros([0, 0, 1, 1], device="cuda"),
                        "ray_max_sample_loc_w": torch.zeros([0, 0, 3], device="cuda"),
                        "ray_max_far_dist": torch.zeros([0, 0, 1], device="cuda"),
                        "shading_avg_color": torch.zeros([0, 0, 3], device="cuda"),
                        "shading_avg_dir": torch.zeros([0, 0, 3], device="cuda"),
                        "shading_avg_conf": torch.zeros([0, 0, 1], device="cuda"),
                        "shading_avg_embedding": torch.zeros([0, 0, s_pts['embedding'].shape[-1]], device="cuda"),
                        "shading_avg_roughness": torch.zeros([0, 0, 1], device="cuda"),
                        "shading_avg_specular": torch.zeros([0, 0, 1], device="cuda"),
                        "shading_avg_normal": torch.zeros([0, 0, 3], device="cuda")
                    })

            return output