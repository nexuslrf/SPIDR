# codes are adapted from https://github.com/google/nerfactor .
import torch
import numpy as np
import utils.util as util
import utils.sphere_geo as sph
import torch.nn.functional as F
import copy
###########################################

def load_light(envmap_path, envmap_inten=1., envmap_h=None, vis_path=None):
    if envmap_path == 'white':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.ones((h, 2 * h, 3), dtype=float)

    elif envmap_path == 'point':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.zeros((h, 2 * h, 3), dtype=float)
        i = -envmap.shape[0] // 4
        j = -int(envmap.shape[1] * 7 / 8)
        d = 2
        envmap[(i - d):(i + d), (j - d):(j + d), :] = 1

    else:
        envmap = util.read_map(envmap_path)

    # Optionally resize
    if envmap_h is not None:
        envmap = util.resize(envmap, new_h=envmap_h)

    # Scale by intensity
    envmap = envmap_inten * envmap

    # visualize the environment map in effect
    if vis_path is not None:
        util.write_arr(envmap, vis_path, clip=True)

    return envmap

def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2, perturb=0, rot=0):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 1) # np.pi / (envmap_h + 2) # this is + 1 actually
    lng_step_size = 2 * np.pi / (envmap_w)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    if perturb > 0.:
        lng_scale = lng_step_size * perturb
        lngs = lngs + (np.random.randn(*lngs.shape) * lng_scale / 3).clip(-lng_scale, lng_scale)
        lat_scale = lat_step_size * perturb
        lats = lats + (np.random.randn(*lats.shape) * lat_scale / 3).clip(-lat_scale, lat_scale)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph.sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    if rot != 0: # rot is rot angle around z-axis, in degree
        rot = (rot % 360) / 180 * np.pi / lng_step_size
        rot = int(round(rot))
        xyz = np.roll(xyz, rot, 1)

    return xyz, areas

def compute_visibility(cam_depth, light_depth, uv, cam_K, light_K, 
        camrotc2w, cam_pos, lightrotw2c, light_pos, depth_thres=0.01, 
        soft_vis=True, dot_bias=False, normals=None):

    f_x, f_y = cam_K[0,0], cam_K[1,1]
    c_x, c_y = cam_K[0,2], cam_K[1,2]
    f_x_l, f_y_l = light_K[0,0], light_K[1,1]
    c_x_l, c_y_l = light_K[0,2], light_K[1,2]
    
    # pix_mask = (cam_depth>0).reshape(-1)# (light_depth_reproj[...,2]>8).reshape(-1)
    u, v = uv[...,0], uv[...,1]
    cam_depth_c = torch.stack([
        (u-c_x)/f_x * cam_depth, (v-c_y)/f_y * cam_depth, cam_depth
    ], -1)
    
    cam_depth_w = (cam_depth_c[...,None,:] * camrotc2w[:,None,...]).sum(-1) + cam_pos
    light_dir = cam_depth_w - light_pos[:,None]
    light_depth_reproj = (light_dir[...,None,:] * lightrotw2c[:,None,...]).sum(-1)
    
    depth_reproj = light_depth_reproj[...,2]
    uv_reproj = light_depth_reproj[...,:2] / depth_reproj[...,None]
    # rescale to [-1,1] for F.grid_sample
    uv_reproj[...,0] = (uv_reproj[...,0] * f_x_l) / c_x_l
    uv_reproj[...,1] = (uv_reproj[...,1] * f_y_l) / c_y_l
    sample_depth = F.grid_sample(
        light_depth[:,None,...,0], uv_reproj[:,None,...], 
        padding_mode="border", align_corners=True)[:,0,0,:] # align_corners=True is better.
    # shadow_bias
    soft_r = 1.
    if dot_bias:
        light_dir = F.normalize(light_dir, dim=-1)
        normals = F.normalize(normals, dim=-1)
        depth_thres = (depth_thres * (1 - (-light_dir * normals).sum(-1).clamp_min(0))).clamp_min(0.5 * depth_thres)
        # depth_thres = (depth_thres * (-light_dir * normals).sum(-1).acos().tan()).clamp(0.1*depth_thres, 2*depth_thres)
        # soft_r = 0.5
    if not soft_vis: # or dot_bias:
        visibility = ~((depth_reproj - sample_depth) > depth_thres)
    else:
        # visibility = 1 - (depth_reproj - sample_depth).clamp(0, depth_thres) / depth_thres
        if not dot_bias:
            visibility = 1 - (depth_reproj - sample_depth - depth_thres).clamp(0, depth_thres*soft_r)\
                    / (depth_thres*soft_r)
        else:
            depth_diff = (depth_reproj - sample_depth - depth_thres).clamp_min(0)
            visibility = 1- torch.where(depth_diff < depth_thres*soft_r, depth_diff, depth_thres*soft_r)\
                 / (depth_thres*soft_r)
    return visibility, light_dir, cam_depth_w