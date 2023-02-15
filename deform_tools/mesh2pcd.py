import igl
import open3d as o3d
import open3d.core as o3c
import numpy as np
from numpy.linalg import norm
import os
import argparse
from mesh_editing import closest_pt_point_triangle


def mesh2pcd(mesh_file, mesh_deformed_file, pcd_file, pcd_deformed_file, offset=0):
    # read mesh
    V, F = igl.read_triangle_mesh(mesh_file)
    V_deformed, _ = igl.read_triangle_mesh(mesh_deformed_file)
    print(f"read triangle mesh: V={len(V)}, F={len(F)}")

    # read pointcloud
    pcd = o3d.t.io.read_point_cloud(pcd_file)
    pc = pcd.point.positions.numpy()
    pcn = pcd.point.normals.numpy()
    print(f"read pcd: V={len(pc)}")

    V_displacement = V_deformed - V

    # compute correspondence between pointcloud and mesh
    _, indices, _ = igl.point_mesh_squared_distance(pc, V, F)
    corner1_idx = F[indices, 0]
    corner2_idx = F[indices, 1]
    corner3_idx = F[indices, 2]

    normal_v = igl.per_vertex_normals(V, F)
    normal_v /= np.linalg.norm(normal_v, axis=1)[:, None].clip(min=1-4)

    V = V - normal_v * offset # shrink/expand the mesh

    # projected barycentric coordinates of pointcloud onto triangles
    # `igl.barycentric_coordinates_tri` is not stable. Use the ours instead.
    theta = closest_pt_point_triangle(    
        pc, 
        V[corner1_idx],
        V[corner2_idx],
        V[corner3_idx]
    )

    # use the deformed mesh to guide the pointcloud translation
    projected_pc = theta[:, 0][:, None] * V[corner1_idx] + \
               theta[:, 1][:, None] * V[corner2_idx] + \
               theta[:, 2][:, None] * V[corner3_idx]

    normal =  pc - projected_pc

    normal_d = np.linalg.norm(normal, axis=1)
    normal = normal / normal_d[:, None].clip(min=1e-4)

    projected_normal = theta[:, 0][:, None] * normal_v[corner1_idx] + \
                    theta[:, 1][:, None] * normal_v[corner2_idx] + \
                    theta[:, 2][:, None] * normal_v[corner3_idx]
    projected_normal /= np.linalg.norm(projected_normal, axis=1)[:, None].clip(min=1e-4)
    d_sign = ((normal * projected_normal).sum(-1) > 0) * 2. - 1

    normal_v_deformed = igl.per_vertex_normals(V_deformed, F)
    normal_v_deformed /= np.linalg.norm(normal_v_deformed, axis=1)[:, None]

    V_deformed = V_deformed - normal_v_deformed * offset

    projected_normal_deformed = theta[:, 0][:, None] * normal_v_deformed[corner1_idx] + \
                                theta[:, 1][:, None] * normal_v_deformed[corner2_idx] + \
                                theta[:, 2][:, None] * normal_v_deformed[corner3_idx]
    projected_normal_deformed /= np.linalg.norm(projected_normal_deformed, axis=1)[:, None].clip(min=1e-4)
    projected_pc_deformed = \
        theta[:, 0][:, None] * V_deformed[corner1_idx] + \
        theta[:, 1][:, None] * V_deformed[corner2_idx] + \
        theta[:, 2][:, None] * V_deformed[corner3_idx]

    projected_normal_displacement = np.nan_to_num(projected_normal_deformed - projected_normal)

    ## original solution:
    # 
    # normal_deformed = normal + projected_normal_displacement
    # normal_deformed /= np.linalg.norm(normal_deformed, axis=1)[:, None].clip(min=1e-4)
    # new_pc = projected_pc_deformed + normal_d[:, None] * normal_deformed

    # if we use gt mesh, this works better
    normal_deformed = projected_normal_deformed
    new_pc = projected_pc_deformed + normal_d[:, None] * normal_deformed * d_sign[:, None]

    new_normal = pcn + projected_normal_displacement
    pcn /= np.linalg.norm(pcn, axis=1)[:, None].clip(min=1e-4)
    new_normal = pcn + projected_normal_displacement
    new_normal /= np.linalg.norm(new_normal, axis=1)[:, None]

    # write pointcloud
    pcd.point.positions = o3c.Tensor(new_pc)
    pcd.point.normals = o3c.Tensor(new_normal)

    o3d.t.io.write_point_cloud(pcd_deformed_file, pcd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str,
                        help='path of the input mesh file')
    parser.add_argument('--mesh_deformed', type=str,
                        help='path of the input deformed mesh file')
    parser.add_argument('--pcd', type=str,
                        help='path of the input pointcloud file')
    parser.add_argument('--pcd_deformed', type=str,
                        help='path of the output pointcloud file')
    parser.add_argument('--offset', type=float, default=0,
                        help='offset of the pcd points to mesh faces')
    args = parser.parse_args()

    mesh2pcd(args.mesh, args.mesh_deformed, args.pcd, args.pcd_deformed, args.offset)

