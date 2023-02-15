import torch
import torch.nn.functional as F
import open3d as o3d
import numpy as np
from tqdm import tqdm
import argparse
import mcubes

def marching_cube(xyz, sdf, mask=None, out_file='mesh.obj', threshold=0.002, conv_iters=60, pad=1):
    if mask is not None:
        # fill 1 for empty voxel
        sdf_fill1 = sdf.clone()
        sdf_fill1[~mask]=1

        # a 3d max pooling propogation
        cnt_tensor = mask.float()[None, None,...].cuda()
        sdf_tensor = sdf.float()[None, None, ...].cuda()
        n_mask = (mask == 0)[None, None, ...].cuda()
        ori_n_mask = (mask == 0)[None, None, ...]
        pbar = tqdm()
        n_mask_curr = n_mask.sum().item()
        n_mask_prev = n_mask_curr + 1

        addition_iters = 5
        k = 0
        while n_mask_prev > n_mask_curr or k < addition_iters:
            n_mask_prev = n_mask_curr
            sdf_out_pos = F.max_pool3d(sdf_tensor, (3,3,3), stride=1, padding=1)
            sdf_out_neg = -F.max_pool3d(-sdf_tensor, (3,3,3), stride=1, padding=1)
            blend_mask = (sdf_out_pos * sdf_out_neg) >= 0 # possible situation: pos = neg = 0
            sign_mask = (sdf_out_pos + sdf_out_neg) >= 0
            sdf_mid = (2*sdf_out_pos + sdf_out_neg) * 0.3
            pos_mask = sign_mask * blend_mask
            neg_mask = (~sign_mask) * blend_mask
            sdf_out = sdf_out_pos * pos_mask + neg_mask * sdf_out_neg #+ sdf_mid * (~blend_mask)
            # sdf_tensor[n_mask] = sdf_out[n_mask]
            sdf_tensor[ori_n_mask] = sdf_out[ori_n_mask]
            cnt_out = F.max_pool3d(cnt_tensor, (3,3,3), stride=1, padding=1)
            n_mask = (cnt_out == 0) | (~blend_mask & n_mask)

            cnt_tensor = (cnt_out > 0).float()
            n_mask_curr = n_mask.sum().item()
            if n_mask_prev > n_mask_curr:
                k += 1

            pbar.update(1)
        pbar.close()

        weight = torch.ones(1, 1, 3, 3, 3).cuda()
        for i in [0,2]:
            for j in [0,2]:
                for k in [0,2]:
                    weight[0,0,i,j,k] = 0.

        # a 3d convolution propogation
        for i in tqdm(range(conv_iters)):
            sdf_out = F.conv3d(sdf_tensor, weight, stride=1, padding=1)
            cnt_out = F.conv3d(cnt_tensor, weight, stride=1, padding=1)
            sdf_out = torch.where(cnt_out>=1, sdf_out / cnt_out, cnt_out)
            sdf_tensor[ori_n_mask] = sdf_out[ori_n_mask]
            n_mask = (cnt_out < 0.5)
            cnt_tensor = (cnt_out > 0.5).float()
        
        sdf_conv = sdf_tensor.cpu()[0,0]

        fc_mask = sdf_conv > 0
        sdf = sdf_fill1 * fc_mask + sdf_conv * (~fc_mask)
    
    # Note: min_xyz is the coordinate of the voxel with index (0,0,0)
    min_xyz = xyz[0,0,0][None,...].numpy()
    vsize = (xyz[1,1,1] - xyz[0,0,0])[None,...].numpy()
    pad_offset = 0

    if pad > 0:
        sdf = F.pad(sdf, (pad,pad,pad,pad,pad,pad), value=1)
        pad_offset = vsize * pad

    vertices, triangles = mcubes.marching_cubes(-sdf.numpy(), -threshold) # minus for correcting the orientation
    print("Marching cube finished: {} vertices, {} triangles".format(vertices.shape[0], triangles.shape[0]))
    vertices = vertices * vsize + min_xyz - pad_offset
    mcubes.export_obj(vertices, triangles, out_file)

def mesh_fix(mesh_file):
    # Note: some extracted mesh may not be watertight. This function may help u get a watertight mesh.
    # addtional package:
    import pyvista as pv
    from pymeshfix import MeshFix

    orig_mesh = pv.read(mesh_file)
    meshfix = MeshFix(orig_mesh)
    holes = meshfix.extract_holes()
    meshfix.repair(joincomp=True)

    vert, faces = meshfix.v, meshfix.f
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    mesh = pv.PolyData(vert, triangles)
    pv.save_meshio(mesh_file[:-4]+'_fix.obj', mesh)



def closest_pt_point_triangle(p, a, b, c):
    # Based on algorithm shown in:
    # * Ericson C. Real-time collision detection. Crc Press; 2004 Dec 22.
    dot = lambda a, b: (a*b).sum(-1)

    if isinstance(p, torch.Tensor):
        import torch as fnp
    else:
        import numpy as fnp
    # Check if P in vertex region outside A
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    coord = fnp.zeros_like(p)
    a_mask =  (d1 <= 0.0) * (d2 <= 0.0)
    coord[a_mask,..., 0] = 1 # barycentric coordinates (1,0,0)
    active_mask = ~a_mask
    # Check if P in vertex region outside B
    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    b_mask = (d3 >= 0.0) * (d4 <= d3) * active_mask
    coord[b_mask,..., 1] = 1 # barycentric coordinates (0,1,0)
    active_mask = active_mask * ~b_mask
    # Check if P in vertex region outside C
    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    c_mask =  (d6 >= 0.0) * (d5 <= d6) * active_mask
    coord[c_mask,..., 2] = 1 # barycentric coordinates (0,0,1)
    active_mask = active_mask * ~c_mask
    # Check if P in edge region of AB, if so return projection of P onto AB
    vc = d1*d4 - d3*d2
    ab_mask = (vc <= 0.0) * (d1 >= 0.0) * (d3 <= 0.0) * active_mask
    v = d1 / (d1 - d3)
    coord[ab_mask,...,0] = 1-v[ab_mask]
    coord[ab_mask,...,1] = v[ab_mask]  # barycentric coordinates (1-v,v,0)
    active_mask = active_mask * ~ab_mask
    # Check if P in edge region of AC, if so return projection of P onto AC
    vb = d5*d2 - d1*d6
    ac_mask = (vb <= 0.0) * (d2 >= 0.0) * (d6 <= 0.0) * active_mask
    w = d2 / (d2 - d6)
    coord[ac_mask,...,0] = 1 - w[ac_mask]
    coord[ac_mask,...,2] = w[ac_mask]
    active_mask = active_mask * ~ac_mask # barycentric coordinates (1-w,0,w)
    # Check if P in edge region of BC, if so return projection of P onto BC
    va = d3*d6 - d5*d4
    bc_mask = (va <= 0.0) * ((d4 - d3) >= 0.0) * ((d5 - d6) >= 0.0) * active_mask
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    coord[bc_mask,..., 1] = 1 - w[bc_mask]
    coord[bc_mask,..., 2] = w[bc_mask] # barycentric coordinates (0,1-w,w)
    active_mask = active_mask * ~bc_mask
    # P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1. - v - w
    coord[active_mask,...,0] = u[active_mask]
    coord[active_mask,...,1] = v[active_mask]
    coord[active_mask,...,2] = w[active_mask]
    # = u*a + v*b + w*c, u = va * denom = 1.0-v-w
    return coord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vol_file', type=str, default='')
    parser.add_argument('--out_file', type=str, default='')
    args = parser.parse_args()

    device = torch.device('cpu')
    saved_vol = torch.load(args.vol_file, map_location=device)
    xyz = saved_vol["xyz"]
    sdf = saved_vol['sdf']
    mask = saved_vol['mask']

    marching_cube(xyz, sdf, mask, args.out_file)

if __name__ == '__main__':
    main()
