from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from models.helpers.networks import init_seq, init_weights, positional_encoding
from models.rendering.diff_ray_marching import ray_march

'''
Modules are basically follow the design of the original repo, but also with some modifications.
https://github.com/lioryariv/volsdf/blob/main/code/model/network.py
'''
class GeoNet(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            skip_in=(),
            sphere_scale=1.0,
    ):
        super().__init__()
        dims = [d_in] + dims # + [d_out + feature_vector_size] 
        self.num_layers = len(dims) - 2
        self.skip_in = skip_in
        
        self.lins = []
        self.act = nn.LeakyReLU(inplace=True)
        self.lins = nn.ModuleList()
        for l in range(0, self.num_layers):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            init_weights(lin, gain=nn.init.calculate_gain('leaky_relu', self.act.negative_slope))
            self.lins.append(lin)
        
        # last layer.
        self.den_layer = nn.Linear(dims[-1], d_out)
        self.feat_layer = nn.Linear(dims[-1], feature_vector_size)
        for m in [self.den_layer, self.feat_layer]:
            init_weights(m)
        

    def forward(self, input: Tensor):
        x = input
        for i, lin in enumerate(self.lins):
            if i in self.skip_in:
                x = torch.cat([x, input], -1) # / (2**0.5)
            x = lin(x)
            x = self.act(x)
        den = self.den_layer(x)
        return den, self.feat_layer(x)

class RGBNet(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=False,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.weight_norm = weight_norm
        self.num_layers = len(dims) - 1

        self.act = nn.LeakyReLU(inplace=True)
        self.lins = nn.ModuleList()
        for l in range(0, self.num_layers):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            init_weights(lin, gain=nn.init.calculate_gain('leaky_relu', self.act.negative_slope))
            self.lins.append(lin)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self, rendering_input: Tensor
    ):

        x = rendering_input

        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < self.num_layers - 1:
                x = self.act(x)

        x = self.sigmoid(x)
        return x


# @torch.jit.script
def uniform_sample(
    N_samples: int,
    N_rays: int=1,
    near: float=0., far: float=1.,
    perturb: float=0.,
    device: torch.device='cuda'
    ):
    '''
    Args:
        N_samples: int, number of points sampled per ray
        near, far: float, from camera model
        rays_o: Tensor, the orgin of rays. [N_rays, 3]
        rays_d: Tensor, the direction of rays. [N_rays, 3]
    Return:
        pts: Tensor, sampled points. [N_rays, N_samples, 3]
        z_vals: Tensor, [N_rays, N_samples] or [1, N_samples]
    '''
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)[None,:]
    init_z_vals = near * (1.-t_vals) + far * (t_vals) # (1, N_samples)

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (init_z_vals[..., 1:] + init_z_vals[..., :-1])
        upper = torch.cat([mids, init_z_vals[..., -1:]], -1)
        lower = torch.cat([init_z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand([N_rays, N_samples], device=device) * perturb
        z_vals = lower + (upper - lower) * t_rand # [N_rays, N_samples]
    else:
        z_vals = init_z_vals # [1, N_samples]
    
    return z_vals


# @torch.jit.script
def depth2pts_outside(ray_o: Tensor, ray_d: Tensor, depth: Tensor, bounding_sphere: float):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
    under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - bounding_sphere ** 2)
    d_sphere = torch.sqrt(under_sqrt) - o_dot_d
    p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
    p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin((p_mid_norm / bounding_sphere).clamp(-0.99, 0.99))
    theta = torch.asin((p_mid_norm * depth).clamp(-0.99, 0.99))  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                    torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                    rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)
    if pts.isnan().any().item():
        import IPython; IPython.embed()

    return pts

class NeRFxx(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--sphere_bound', type=float, default=2.)
        parser.add_argument('--bg_num_sample', type=int, default=24)

    def __init__(self, 
            N_sample=24, sphere_bound=2, coord_freq=8, dir_freq=-1,
            feature_size=256, W_bg_den=256, W_bg_rgb=128, 
            D_bg_den=6, D_bg_rgb=1, bg_den_skip_in=[3]):
        
        super().__init__()
        self.N_samples_inverse_sphere = N_sample
        self.scene_bounding_sphere = sphere_bound
        self.coord_freq = coord_freq
        self.dir_freq = dir_freq
        self.fixed = False
        bg_den_d_in = (2*coord_freq+1)*4 if coord_freq >= 0 else 4
        bg_rgb_d_in = (2*dir_freq+1)*3 if dir_freq >= 0 else 0
        self.bg_den_net=GeoNet(feature_size, bg_den_d_in, 1, 
                        [W_bg_den]*D_bg_den, skip_in=bg_den_skip_in)
        self.bg_rgb_net=RGBNet(feature_size, 'nerf', bg_rgb_d_in, 3, 
                        [W_bg_rgb]*D_bg_rgb)

    def point_sample(self, rays_o, rays_d): # normalized dir
        z_vals_inverse_sphere = uniform_sample(self.N_samples_inverse_sphere, rays_d.shape[0],
                                        perturb=0.9 if self.training or self.fixed else 0, device=rays_d.device)
        z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
        z_vals_bg = torch.flip(z_vals_inverse_sphere, dims=[-1,])
        if z_vals_bg.shape[0] == 1:
            z_vals_bg = z_vals_bg.expand(rays_d.shape[0], -1)
        pts_bg = depth2pts_outside(rays_o[...,None,:], rays_d[...,None,:], z_vals_bg, 
                        self.scene_bounding_sphere)
        return pts_bg, z_vals_bg

    def forward(self, rays_o, rays_d):
        rays_d = F.normalize(rays_d, dim=-1)
        pts_bg, z_vals_bg = self.point_sample(rays_o, rays_d)
        pts_embed_bg = positional_encoding(pts_bg, self.coord_freq, ori=True)
        den_bg, feats_bg = self.bg_den_net(pts_embed_bg)
        if self.dir_freq >= 0:
            view_embed = positional_encoding(rays_d, self.coord_freq)
            rgb_in = torch.cat([view_embed.expand(*pts_bg.shape[:-1], -1), feats_bg], dim=-1)
        else:
            rgb_in = feats_bg
        rgb_bg = self.bg_rgb_net(rgb_in)
        density_bg = F.softplus(den_bg-1, beta=5)
        dists = torch.cat([
            z_vals_bg[..., 1:] - z_vals_bg[..., :-1],
            1e10 * torch.ones_like(z_vals_bg[..., :1])
        ], -1)
        ray_color, _, _, _, _, _, _ = \
            ray_march(dists[None,:], 1, torch.cat([density_bg, rgb_bg], dim=-1)[None,:])
        return ray_color

