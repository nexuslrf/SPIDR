import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..helpers.networks import init_seq, positional_encoding
from utils.lighting import compute_visibility, gen_light_xyz, load_light
from models.rendering.mesh_renderer import MeshRenderer
from utils.spherical import SphericalHarm_table, SphericalHarm

def safe_divide(a, b):
    # make sure b >= 0
    return torch.where(b > 1e-8, a/b.clamp_min(1e-8), b.new_zeros(1))
    # return torch.divide(a, b.clamp_min(1e-8))

def gradient_clamp(sampled_conf, min=0.0001, max=1):
    diff = sampled_conf - torch.clamp(sampled_conf, min=min, max=max)
    return sampled_conf - diff.detach()

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)
    
    def sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)

class ImplicitLightMap(nn.Module):
    def __init__(self, n_layers=3, dim_hidden=128, num_dir_freqs=2, sh_deg=0, mode='pe', out_dim=3, act_type='softplus'):
        super().__init__()
        layers = []
        act = lambda : nn.Softplus(beta=50)
        if act_type == 'LeakyReLU':
            act = nn.LeakyReLU
        self.mode = mode
        if mode=='pe':
            in_channels = 3 + 2 * 3 * num_dir_freqs
        elif mode == 'sh':
            if sh_deg > 0: # sh can overwrite PE
                in_channels = (sh_deg + 1)**2
                self.shcomputer = SphericalHarm_table(sh_deg+1)
        elif mode == 'sine':
            in_channels = 3
            act = Sine
        out_channels = dim_hidden
        self.num_dir_freqs = num_dir_freqs
        self.sh_deg = sh_deg
        for i in range(n_layers-1):
            lin = nn.Linear(in_channels, out_channels)
            # torch.nn.init.constant_(lin.bias, 0)
            layers.append(lin)
            layers.append(act())
            in_channels = out_channels
            out_channels = dim_hidden
        lin = nn.Linear(in_channels, out_dim)
        # torch.nn.init.constant_(lin.bias, 0)
        layers.append(lin)
        # layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        if mode == 'sine':
            self.layers.apply(Sine.sine_init)
            self.layers[0].apply(Sine.first_layer_sine_init)
        else:
            init_seq(self.layers)
    
    def forward(self, x):
        if self.mode=='sh' and self.sh_deg > 0:
            x = self.shcomputer.sh_all(x)
        elif self.mode=='pe' and self.num_dir_freqs > 0:
            x = positional_encoding(x, self.num_dir_freqs, ori=True)
        return self.layers(x)

class Microfacet(nn.Module):
    """
        pytorch implementation of
        Microfacet Models for Refraction through Rough Surfaces [EGSR '07]
    """

    def __init__(self, opt=None, default_rough=0.3, lambert_only=False, f0=0.04, 
        learn_f0=False, glossy_slope=0.5, use_nerf_specular=False, 
        normal_incidence_f0=False, roughness_tuned_f0=False) -> None:
        
        super().__init__()
        self.opt = opt
        self.default_rough = default_rough
        self.lambert_only = lambert_only
        self.f0 = f0
        self.learn_f0 = learn_f0
        self.glossy_slope = glossy_slope
        self.use_nerf_specular = use_nerf_specular
        self.normal_incidence_f0 = normal_incidence_f0
        self.roughness_tuned_f0 = roughness_tuned_f0

    def forward(self, light_dir, view_dir, normals, 
            albedo=None, roughness=None, normalized=False, light_area=None, 
            specular=None, metallic=None, **kargs):
        """
        light_dir: N x Light x 3, ray direction of points to light
        view_dir: N x 3, ray direction of points to camera
        normals: N x Rays x 3, point normals
        albedo: N x Rays x 3, albedo color
        roughness: N x Rays x 1, material roughness
        """
        # roughness = None
        if albedo is None:
            albedo = torch.ones(view_dir.shape[0], 3, device=normals.device)

        lambert = albedo / np.pi
        if self.lambert_only:
            return lambert[..., None,:], 0

        if roughness is None:
            roughness = self.default_rough * torch.ones(view_dir.shape[0], 1, device=normals.device)


        # make sure all directions are normalized 
        if not normalized:
            light_dir = F.normalize(light_dir, dim=-1)
            view_dir = F.normalize(view_dir, dim=-1)
            normals = F.normalize(normals, dim=-1)
        # print(light_dir.shape, view_dir.shape, normals.shape)

        # half-direction vector
        h = F.normalize(light_dir + view_dir[...,None,:], dim=-1)

        NoL = (light_dir * normals[...,None,:]).sum(-1).abs() + 1e-5
        NoV = (view_dir * normals).sum(-1).relu()[...,None]
        NoH = (h * normals[...,None,:]).sum(-1).relu()
        LoH = (light_dir * h).sum(-1).relu()

        alpha = torch.clamp(roughness, min=self.opt.min_roughness, max=self.opt.max_roughness)
        if not self.opt.use_linear_roughness:
            alpha = alpha ** 2
        else:
            roughness = torch.sqrt(alpha)

        distribution = self.__distribution(NoH, alpha)
        geometric = self.__geometric(NoV, NoL, alpha)

        # fresnel
        if not self.learn_f0:
            f0 = self.f0
        else:
            if self.use_nerf_specular:
                f_sub = torch.pow((1 - LoH), 5)
                b0 = geometric * distribution
                b1 = b0 * (1-f_sub)
                b2 = b0 * f_sub
            
                if not self.normal_incidence_f0:
                    with torch.no_grad():
                        cos_dl = NoL * light_area.reshape(1,1,-1)
                        B1 = (b1 * cos_dl).sum(-1, keepdim=True)
                        B2 = (b2 * cos_dl).sum(-1, keepdim=True)
                    f0 = safe_divide(F.softplus(specular - B2, beta=100), B1)
                else:
                    f0 = specular * specular

            if self.opt.use_microfacet_mlp:
                if metallic is not None:
                    f0_dielectric = specular.expand(*specular.shape[:-1], albedo.shape[-1])
                    metallic = metallic.expand(*metallic.shape[:-1], albedo.shape[-1])
                    f0 = f0_dielectric * (1-metallic) + albedo.clone() * metallic
                else:
                    f0 = specular
            
            f0 = f0[...,None,:]
        
                
        fresnel = self.__fresnel(LoH[...,None], f0, 
                roughness[...,None,:] if self.roughness_tuned_f0 else None)
        microfacet = fresnel * geometric[...,None] * distribution[...,None]

        kS = fresnel
        kD = 1.0 - kS

        glossy_term = microfacet

        diffuse_term = kD * torch.broadcast_to(lambert[...,None,:], glossy_term.shape)
        glossy_term = self.glossy_slope*glossy_term
        # print(glossy_term.shape)
        # brdf = glossy_term + diffuse_term
        return diffuse_term, glossy_term


    @staticmethod
    def __distribution(NoH, alpha=0.1):
        """Microfacet distribution (GXX) (Disney's version)"""
        a = NoH * alpha
        k = alpha / (1.0 - NoH * NoH + a * a)
        return k * k * (1./np.pi)

    @staticmethod
    def __geometric(NoV, NoL, alpha=0.1):
        """Geometric function (Smith's Visibility)"""
        a2 = alpha * alpha
        GGXV = NoL * torch.sqrt(NoV * NoV * (1-a2) + a2)
        GGXL = NoV * torch.sqrt(NoL * NoL * (1-a2) + a2)
        return 0.5 / (GGXV + GGXL)

    @staticmethod
    def __fresnel(LoH, f0, roughness=None):
        # copied from https://github.com/cgtuebingen/Neural-PIL/tree/main/nn_utils/preintegrated_rendering.py#L23
        f90 = 1 if roughness is None else torch.maximum(1 - roughness, f0)
        return f0 + (f90 - f0) * torch.pow((1 - LoH), 5)
        

class BRDFRender(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--fresnel_f0', type=float, default=0.04)
        parser.add_argument('--learn_f0', action='store_true')
        parser.add_argument('--roughness_tuned_f0', action='store_true')
        parser.add_argument('--default_roughness', type=float, default=0.3)
        parser.add_argument('--use_linear_roughness', action='store_true')
        parser.add_argument('--roughness_anneal_iters', type=int, default=0)
        parser.add_argument('--roughness_anneal_ratio', type=float, default=1.)
        parser.add_argument('--max_roughness', type=float, default=1.0)
        parser.add_argument('--min_roughness', type=float, default=0.05)
        parser.add_argument('--fixed_specular', action='store_true')
        parser.add_argument('--light_cache_path', type=str, default='default')
        parser.add_argument('--light_env_path', type=str, default=None)
        parser.add_argument('--light_radius', type=float, default=100)
        parser.add_argument('--light_env_h', type=int, default=16)
        parser.add_argument('--light_env_w', type=int, default=32)
        parser.add_argument('--light_channels', type=int, default=3)
        parser.add_argument('--light_inner_act', type=str, choices=['softplus', 'LeakyReLU'], default='LeakyReLU')
        parser.add_argument('--light_act', type=str, choices=['softplus', 'exp'], default='softplus')
        parser.add_argument('--light_intensity', type=float, default=1.0)
        parser.add_argument('--light_dir_perturb', type=float, default=0.0)
        parser.add_argument('--light_reg_thresh', type=float, default=0.05)
        parser.add_argument('--light_reg_mode', type=str, choices=['raw', 'act'], default='raw')
        parser.add_argument('--depth_diff_thresh', type=float, default=0.01)
        parser.add_argument('--brdf_linear2srgb', action='store_true')
        parser.add_argument('--brdf_tonemap', type=str, choices=['none', 'srgb'], default='srgb')
        parser.add_argument('--brdf_tone_range', type=float, default=1.0)
        parser.add_argument('--brdf_s_curve_phi', type=float, default=0.4)
        parser.add_argument('--brdf_s_curve_omega', type=float, default=4.0)
        parser.add_argument('--albedo_slope', type=float, default=1.0) # 0.77
        parser.add_argument('--albedo_bias', type=float, default=0.0) # 0.03
        parser.add_argument('--glossy_slope', type=float, default=0.5)
        parser.add_argument('--brdf_step', type=int, default=0)
        parser.add_argument('--diffuse_tonemap', type=str, choices=['none', 'gamma', 'srgb'], default='none')
        parser.add_argument('--diffuse_tonemap_order', type=str, choices=['pre', 'post'], default='pre')
        parser.add_argument('--diffuse_gamma', type=float, default=2.2)
        parser.add_argument('--lambert_only', action='store_true')
        parser.add_argument('--implicit_light_map', action='store_true')
        parser.add_argument('--implicit_light_mode', choices=['pe', 'sh', 'sine'], default='pe')
        parser.add_argument('--brdf_training', action='store_true')
        parser.add_argument('--brdf_mlp', action='store_true')
        parser.add_argument('--brdf_mlp_reinit', action='store_true')
        parser.add_argument('--brdf_joint_optim', action='store_true')
        parser.add_argument('--light_dir_freqs', type=int, default=2)
        parser.add_argument('--light_mlp_layers', type=int, default=3)
        parser.add_argument('--optimize_light_only', action='store_true')
        parser.add_argument('--recenter_vis_dir', action='store_true')
        parser.add_argument('--brdf_model', type=str, choices=['vdir', 'microfacet'], default='microfacet')
        parser.add_argument('--brdf_use_nerf_specular', action='store_true')
        parser.add_argument('--normal_incidence_f0', action='store_true')
        parser.add_argument('--depth_from_mesh', action='store_true')
        parser.add_argument('--mesh_path', type=str, default='')
        parser.add_argument('--refine_vis', action='store_true')
        parser.add_argument('--refine_gamma', type=float, default=1.4)
        parser.add_argument('--refine_slope', type=float, default=0.0008)
        parser.add_argument('--refine_base', type=float, default=0.4)
        parser.add_argument('--refine_normal', action='store_true')
        parser.add_argument('--SH_light_deg', type=int, default=0)
        parser.add_argument('--shadow_dot_bias', action='store_true')
        parser.add_argument('--fixed_weight_copy', action='store_true')
        parser.add_argument('--fixed_blend_weight', action='store_true')
        parser.add_argument('--use_microfacet_mlp', action='store_true')
        parser.add_argument('--use_albedo_mlp', action='store_true')

    def __init__(self, opt, 
            light_cache=None,
            trainable=True):
        super().__init__()
        self.opt = opt
        if opt.use_microfacet_mlp:
            assert \
                (opt.shading_fresnel_mlp_layer > 0 and opt.fresnel_branch_channel == 3) or \
                (opt.shading_fresnel_mlp_layer > 0 and opt.shading_metallic_mlp_layer > 0)
        self.implicit_light = opt.implicit_light_map and opt.light_env_path is None
        if opt.brdf_model == 'microfacet':
            self.brdf_model = Microfacet(opt=opt, f0=opt.fresnel_f0, lambert_only=opt.lambert_only, 
                    default_rough=opt.default_roughness, learn_f0=opt.learn_f0, 
                    glossy_slope=opt.glossy_slope, use_nerf_specular=opt.brdf_use_nerf_specular,
                    normal_incidence_f0=opt.normal_incidence_f0, roughness_tuned_f0=opt.roughness_tuned_f0)

        self.init_light_depth_maps(light_cache, opt.light_cache_path)
        self.init_light_env_maps(opt.light_env_path, trainable)
        self.albedo_slope = opt.albedo_slope
        self.albedo_bias = opt.albedo_bias
        self.num_lights = self.light_pos.shape[0]
        self.light_dir = None
        self.light_dir_perturb = opt.light_dir_perturb
        if opt.depth_from_mesh:
            self.mesh_renderer = MeshRenderer(opt.mesh_path)

    def init_light_depth_maps(self, light_cache = None, light_cache_path: Optional[str]=None):         
        if light_cache is None and light_cache_path is not None:
            if light_cache_path == 'default':
                light_cache_path = f'{self.opt.resume_iter}'
            if light_cache_path.isdigit():
                resume_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
                light_cache_path = f'{resume_dir}/light_{light_cache_path}/light_cache.pth'
            try:
                light_cache = torch.load(light_cache_path)
                print(f'Loaded light cache from {light_cache_path}')
            except:
                print(f'Cannot load light cache from {light_cache_path}')
                raise FileNotFoundError
        self.register_buffer('depth_maps', light_cache['depth_maps'], False)
        self.register_buffer('light_w2c', light_cache['light_poses'][...,:3,:3].transpose(-1,-2).float(), False)
        self.register_buffer('light_pos', light_cache['light_poses'][...,:3,-1].float(), False)
        self.register_buffer('light_intrinsic', torch.from_numpy(light_cache['intrinsic']).float(), False)

    def init_light_env_maps(self, light_env_path: Optional[str]=None, trainable=False):
        self.h , self.w = self.opt.light_env_h, self.opt.light_env_w
        lxyz, lareas = gen_light_xyz(self.h, self.w, self.opt.light_radius)
        self.register_buffer('light_areas', torch.from_numpy(lareas).float(), False)
        self.light_max, self.light_min = 0, 0
        if self.implicit_light:
            self.light_map_model = ImplicitLightMap(
                n_layers=self.opt.light_mlp_layers, num_dir_freqs=self.opt.light_dir_freqs, 
                sh_deg=self.opt.SH_light_deg, mode=self.opt.implicit_light_mode, 
                out_dim=self.opt.light_channels, act_type=self.opt.light_inner_act)
            if self.opt.SH_light_deg > 0:
                assert not self.opt.recenter_vis_dir
        else:
            if light_env_path is not None:
                if light_env_path.endswith('.pth'):
                    light_map = torch.load(light_env_path)
                else:
                    light_map = torch.from_numpy(load_light(light_env_path)).float()
                self.opt.light_act = ''
            else:
                light_init_max = 1.0
                light_map = torch.rand(self.h, self.w, 3) * light_init_max

            # self.register_buffer('light_map', light_map, False)
            self._light_map = nn.Parameter(light_map)
            self._light_map.requires_grad_(trainable)

    @property
    def light_map(self) -> torch.Tensor:
        return self.get_light_map()

    @property
    def raw_light_map(self) -> torch.Tensor:
        return self.get_light_map(with_act=False)

    def get_light_map(self, with_act=True) -> torch.Tensor:
        if self.implicit_light:
            if self.training and self.light_dir_perturb > 0:
                lxyz, _ = gen_light_xyz(self.h, self.w, self.opt.light_radius, self.light_dir_perturb)
                light_dir = F.normalize(torch.tensor(lxyz, dtype=torch.float32, device=self.light_pos.device), dim=-1)
                light_dir = light_dir.reshape(-1, 3) #* (1 + torch.randn_like(self.light_pos[...,:1]).clamp(-2, 2) * 0.05)
            else:
                if self.light_dir is None:
                    lxyz, _ = gen_light_xyz(self.h, self.w, self.opt.light_radius)
                    self.light_dir = F.normalize(
                            torch.tensor(lxyz, dtype=torch.float32, device=self.light_pos.device), dim=-1
                        ).reshape(-1, 3)
                light_dir = self.light_dir

            light_dir = 0.5 * light_dir + 0.5 if self.opt.recenter_vis_dir else light_dir
            
            raw_light_probes = self.light_map_model(light_dir).reshape(self.h, self.w, -1)
        else:
            raw_light_probes = self._light_map

        if with_act:
            # diff = raw_light_probes - raw_light_probes.clamp_min(0)
            # light_probes = raw_light_probes - diff.detach()
            if self.opt.light_act == 'softplus':
                light_probes = F.softplus(raw_light_probes, beta=100)
            elif self.opt.light_act == 'exp':
                light_probes = torch.exp(raw_light_probes.clamp(-10,10))
            else:
                light_probes = raw_light_probes
            self.light_max, self.light_min = light_probes.max().item(), light_probes.min().item()
        
        else:
            light_probes = raw_light_probes
        # import IPython; IPython.embed()
        # for debug
        # light_probes.data[:,:8] = 0.
        # light_probes.data[:,24:] = 0.
        light_probes = light_probes * self.opt.light_intensity
        return light_probes #.expand(*light_probes.shape[:2], 3)

    def compute_visibility(self, view_depth, uv, K_cam, camrotc2w, campos, normals=None):
        vis_map, light_dir, surf_xyz = compute_visibility(view_depth, self.depth_maps, 
                 uv, K_cam, self.light_intrinsic, camrotc2w, campos,
                 self.light_w2c, self.light_pos, depth_thres=self.opt.depth_diff_thresh,
                 dot_bias=self.opt.shadow_dot_bias, normals=normals
                )
        vis_map = vis_map.t()[None, :]
        light_dir = light_dir.transpose(0,1)[None, :]
        return vis_map, light_dir, surf_xyz # light_dir is l2surf

    def forward(self, light_dir, view_dir, normals, albedo, roughness, vis_map, 
            surf_xyz=None, bg_transmission=None, bg_color=None, specular=None, output=None):
        
        light_dir = F.normalize(-light_dir, dim=-1) # pay attention!
        view_dir = F.normalize(-view_dir, dim=-1)
        normals = F.normalize(normals, dim=-1)

        # refine vis_map
        # 1. every sampled points shold not have a pure black vis_map
        # 2. v_dir shold within the visible region. BTW, normal dir can be invisible
        cos_l_v = None
        if self.opt.refine_vis:
            total_vis = vis_map.sum(-1).float()
            # low_vis_mask = total_vis <= 8
            vis_mask = vis_map > 0.5

            cos_l_v = (light_dir * view_dir[...,None,:]).sum(-1)
            # cos_thresh = (0.5 + 0.001 * total_vis**2).clamp_max(0.98)[...,None]
            cos_thresh = (self.opt.refine_base + self.opt.refine_slope * total_vis**self.opt.refine_gamma).clamp_max(0.98)[...,None]
            cos_mask = cos_l_v > cos_thresh

            mask = (~vis_mask) * cos_mask
            vis_map = (~mask) * vis_map + mask * cos_l_v.clamp_min(0.8)

        # refine normal
        # 1. normal & v_dir should not have large negative cos value
        # UPDATE: this normal refinement is not usful.
        if self.opt.refine_normal:
            normals = normals.detach()
            cos_v_n = (view_dir * normals).sum(-1)
            cos_l_v = (light_dir * view_dir[...,None,:]).sum(-1) if cos_l_v is None else cos_l_v
            normal_mask = cos_v_n < -0.3
            est_normal_region = vis_map[normal_mask] * (cos_l_v[normal_mask] > 0)
            est_normals = (est_normal_region[...,None] * light_dir[normal_mask]).sum(-2)
            est_normals = F.normalize(est_normals, dim=-1)
            normals[normal_mask] = est_normals

        metallic = None
        if self.opt.use_microfacet_mlp and not self.brdf_model.lambert_only:
            specular = output['fresnel']
            metallic = output['metallic']

        tuned_albedo = albedo * self.albedo_slope + self.albedo_bias

        # if self.brdf_model.lambert_only and not self.opt.fixed_specular and specular is not None:
        #     tuned_albedo = tuned_albedo + specular

        diffuse, glossy = self.brdf_model(light_dir, view_dir, normals, tuned_albedo, roughness, 
                normalized=True, light_area=self.light_areas, specular=specular, metallic=metallic)
        rgb_diffuse = brdf_render(vis_map, diffuse, self.light_map, light_dir, normals, self.light_areas)

        if self.brdf_model.lambert_only and self.opt.fixed_specular and specular is not None:
            rgb_glossy = specular.detach()
        else:
            rgb_glossy = brdf_render(vis_map, glossy, self.light_map, light_dir, normals, self.light_areas)
        
        rgb_glossy = (bg_transmission < max(self.opt.bg_trans_thresh*0.8, 0.7)) * rgb_glossy
        others = {'albedo': tuned_albedo, 'diffuse': rgb_diffuse, 'specular': rgb_glossy}

        rgb = rgb_diffuse + rgb_glossy

        if self.opt.mlp_diffuse_cor_weight > 0:
            vis_raycolor = brdf_render(vis_map, 1, self.light_map, light_dir, normals, self.light_areas)
            others['vis_raycolor'] = vis_raycolor.detach()
            
        if bg_color is not None:
            rgb = rgb + bg_color[0] * bg_transmission
        if self.opt.brdf_tonemap=='srgb':
            rgb = linear2srgb(rgb)

        return rgb, others

    def vis_light(self, visualizer=None, gamma=4.):
        light_probe = self.light_map.detach() #.clamp(0,1)
        print(f"Light Probe: max: {light_probe.max()}; min: {light_probe.min()}")
        # gamma tone map
        light_probe = ((light_probe / light_probe.max()) ** (1 / gamma)).clamp(0,1)
        if light_probe.shape[-1] == 1:
            light_probe = light_probe.expand(*light_probe.shape[:-1], 3)
        if visualizer is not None:
            visualizer.display_current_results(
                {'light_probe': light_probe.cpu().numpy()}, opt=self.opt)
        return light_probe

def brdf_render(light_vis, brdf, light_map, ldir, normal, lareas, soft=False):
    cos = (ldir * normal[...,None,:]).sum(-1) # / ldir.norm(2,-1) / normal.norm(2,-1)[...,None]
    if soft:
        cos = F.leaky_relu(cos, 0.01)
        lvis = light_vis
    else:
        front_lit = cos > 0
        lvis = front_lit * light_vis
    # light = torch.einsum('...k,kc->...kc', lvis, light_map.reshape(-1,light_map.shape[-1]))
    light = lvis[...,None] * light_map.view(1, 1, -1, light_map.shape[-1])
    rgb = (brdf * light * cos[...,None] * lareas.reshape(-1,1)).sum(-2) #.clamp(0, 1) # TODO to see whether this clip impacts.
    return rgb


def linear2srgb(rgb):
    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_linear = rgb * srgb_linear_coeff
    tensor_pow = torch.pow(rgb.clamp_min(srgb_linear_thres), 1 / srgb_exponent)
    tensor_nonlinear = srgb_exponential_coeff * tensor_pow - \
                    (srgb_exponential_coeff - 1)
    
    is_linear = rgb <= srgb_linear_thres
    tensor_srgb = torch.where(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb

if __name__ == '__main__':
    pass