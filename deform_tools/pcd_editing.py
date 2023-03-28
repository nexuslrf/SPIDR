import torch
import numpy as np
import open3d as o3d
import argparse
import shutil
import open3d.core as o3c
import torch.nn.functional as F
from torch import Tensor
import os

def get_pcd(pth_file, pcd_file):
    if isinstance(pth_file, str):
        pth_file = torch.load(pth_file)
    
    xyz = pth_file["neural_points.xyz"]
    color = pth_file["neural_points.points_color"]
    normal = pth_file["neural_points.points_dir"]
    normal = F.normalize(normal, dim=-1)

    n_pts = xyz.shape[0]
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3c.Tensor(xyz.cpu().numpy())
    pcd.point.colors = o3c.Tensor((255*color[0].cpu().numpy()).astype(np.uint8))
    pcd.point.normals = o3c.Tensor(normal[0].cpu().numpy())
    pcd.point.quality = o3c.Tensor(np.arange(n_pts)[...,None].astype(np.float32))
    
    o3d.t.io.write_point_cloud(pcd_file, pcd)

def write_pth(pth, new_pth, new_pcd, rot_mask=None, rot_mat=None):
    # new_pcd: open3d.t.geometry.PointCloud | str
    if isinstance(pth, str):
        pth = torch.load(pth)
    if isinstance(new_pcd, str):
        new_pcd = o3d.t.io.read_point_cloud(new_pcd)
        pcd_idx = new_pcd.point['quality'].numpy()[...,0].astype(np.int64)
        pcd_order = np.argsort(pcd_idx)
        new_pcd = new_pcd.select_by_index(pcd_order)

    pth['neural_points.ori_normal'] = pth['neural_points.points_dir']

    point_num = pth['neural_points.xyz'].shape[0]
    pth['neural_points.xyz'] = torch.from_numpy(new_pcd.point.positions.numpy().astype(np.float32))
    pth['neural_points.points_dir'] = torch.from_numpy(new_pcd.point.normals.numpy().astype(np.float32))[None,...]

    if rot_mask is not None and rot_mat is not None:
        all_rotate = np.array([np.identity(3)] * point_num)
        if isinstance(rot_mask, list):
            for mask, mat in zip(rot_mask, rot_mat):
                all_rotate[mask] = mat
        else:
            all_rotate[rot_mask] = rot_mat
        pth["neural_points.Rdeform"] = torch.from_numpy(all_rotate.astype(np.float32))[None,...]

    torch.save(pth, new_pth)
