# for pytorch method
import numpy as np
# import pytorch3d as p3d
import torch
import torch.nn.functional as F
# from pytorch3d.io import load_obj, load_objs_as_meshes, load_ply
# from pytorch3d.renderer import (DirectionalLights, FoVPerspectiveCameras,
#                                 Materials, MeshRasterizer, MeshRenderer,
#                                 MeshRendererWithFragments, PerspectiveCameras,
#                                 PointLights, RasterizationSettings,
#                                 HardPhongShader, HardGouraudShader, Textures)


# pyrender
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([
        [-1,0,0,0],
        [ 0,0,1,0],
        [ 0,1,0,0],
        [ 0,0,0,1]]) @ c2w
    c2w = c2w #@ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return c2w

trans_t = lambda t : np.asarray([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.asarray([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.asarray([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)

blender2opencv = np.array([
    [1,  0,  0, 0], 
    [0, -1,  0, 0], 
    [0,  0, -1, 0], 
    [0,  0,  0, 1]]) 

def pose_spherical_topy3d(theta, phi, radius):
    pyrender_c2w = pose_spherical(theta, phi, radius)
    pytorch3d_c2w = pyrender_c2w.copy()
    R = pytorch3d_c2w[:3, :3].T
    print('camera_position:', pytorch3d_c2w[:3, 3:])
    T = (-R@pytorch3d_c2w[:3, 3:])
    
    R_exchangeAxis = np.array([
        [ -1,0,0,0],
        [ 0,1,0,0],
        [ 0,0,-1,0],
        [ 0,0,0,1]])
    R_pyrender = np.array([
        [-1,0,0,0],
        [ 0,0,1,0],
        [ 0,1,0,0],
        [ 0,0,0,1]])
    w2c = np.eye(4)
    w2c[:3,:3] = R
    w2c[:3, 3:] = T
    # w2c = R_exchangeAxis@R_pyrender@w2c
    w2c = R_exchangeAxis@w2c
    R, T = w2c[:3, :3],  w2c[:3, 3:]

    return R.T, T.T

def to_pt3d_pose(campos, camrot):
    R = camrot.T
    T = (-R@campos)
    
    R_exchangeAxis = np.array([
        [ -1,0,0,0],
        [ 0,1,0,0],
        [ 0,0,-1,0],
        [ 0,0,0,1]])

    w2c = np.eye(4)
    w2c[:3,:3] = R
    w2c[:3, 3:] = T
    # w2c = R_exchangeAxis@R_pyrender@w2c
    w2c = R_exchangeAxis@w2c
    R, T = w2c[:3, :3],  w2c[:3, 3:]

    return R.T, T.T

class MeshRenderer:
    def __init__(self, mesh_path, device='cuda'):
        self.mesh_path = mesh_path
        self.device = device
        self.mesh = load_objs_as_meshes([self.mesh_path],device=device)
        # mesh.verts_list()[0] = (torch.from_numpy(rot_xy[:3, :3]).to(device)@mesh.verts_list()[0].T).T
        verts_rgb = torch.ones_like(self.mesh.verts_list()[0])[None] # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb.to(device),faces_uvs=self.mesh.faces_list()[0].to(device),maps=torch.ones(800,800,3).to(device))
        self.mesh.textures = textures
        self.blender2opencv = torch.tensor([
                    [1,  0,  0], 
                    [0, -1,  0], 
                    [0,  0, -1]], device=device) 
        self.R_exchangeAxis = torch.tensor([
            [ -1,0,0,0],
            [ 0,1,0,0],
            [ 0,0,-1,0],
            [ 0,0,0,1]], device=device)


    def render_depth(self, campos, camrot, focal, height, width, near=2, far=6, depth_only=True):
        '''
        campos: 3
        camrot: 3,3
        '''
        camrot = camrot @ self.blender2opencv.to(camrot.dtype)
        # R, T = to_pt3d_pose(campos, camrot)
        # R, T = torch.from_numpy(R).float().unsqueeze(0), torch.from_numpy(T).float()
        R = camrot.T
        T = (-R@campos[...,None])
        
        w2c = torch.eye(4, dtype=campos.dtype, device=campos.device)
        w2c[:3,:3] = R
        w2c[:3, 3:] = T
        w2c = self.R_exchangeAxis.to(camrot.dtype)@w2c
        R, T = w2c[:3, :3].float().T[None,...],  w2c[:3, 3:].float().T

        fov = 2*np.arctan2(width, 2*focal)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov*180/np.pi, znear=near, zfar=far)
        raster_settings = RasterizationSettings(
            image_size=(height, width), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
        )
        fragments = rasterizer(self.mesh)
        depths = fragments.zbuf[0]
        images = None

        if not depth_only:
            lights = PointLights(device=self.device, location=[[0.0, 0.0, 100.]])
            shader=HardGouraudShader(
                device=self.device, 
                cameras=cameras,
                lights=lights
            )
            images = shader(fragments, self.mesh)[0]

        return images,depths

    def render_depth_2(self,theta,phi,radius):
        # theta: 0-180, phi: 0-360
        R, T = pose_spherical_topy3d(theta, phi, radius)
        R, T = torch.from_numpy(R).float().unsqueeze(0), torch.from_numpy(T).float()
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov*180/np.pi, znear=0.01, zfar=1000)
        raster_settings = RasterizationSettings(
            image_size=800, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 100.]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=HardGouraudShader(
                device=self.device, 
                cameras=cameras,
                lights=lights
            )
        )
        images,fragments = renderer(self.mesh)
        depths = fragments.zbuf
        return images[0],depths[0]


if __name__=="__main__":

    import sys
    import os
    import pathlib
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '../..'))
    from utils.lighting import gen_light_xyz

    width = height = 800
    radius = 4
    fov = 0.6911112070083618
    focal = 0.5 * height / np.tan(0.5 * fov) # focal: distance between camera and pixel plane
    intrinsic = np.array([
        [focal, 0, width / 2], 
        [0, focal, height / 2], 
        [0, 0, 1]])

    # light
    l_radius = 100
    df = 1
    l_focal = l_radius/radius *focal / df
    l_fov = 2* np.arctan2(0.5*width/df, l_focal)
    
    import imageio

    lxyz, area = gen_light_xyz(16, 32, l_radius)
    phis = -np.arcsin(lxyz[..., 2]/l_radius)*180/np.pi
    thetas = np.arctan2(lxyz[...,0], lxyz[...,1])*180/np.pi # (16,32)
    light_shape = phis.shape
    light_poses = np.stack([
            pose_spherical(thetas[i,j], phis[i,j], l_radius) @ blender2opencv 
                for i in range(light_shape[0]) for j in range(light_shape[1])], 0)

    mesh_path = '/data/stu01/ruofan/d-pointnerf/checkpoints/nerfsynth/lego/lego.obj'

    theta,phi,radius = -45., -80., 4.
    pose = pose_spherical(theta,phi,radius)
    campos = pose[:3,3:]
    camrot = pose[:3,:3]

    renderer_3d = MeshRenderer(mesh_path)
    for i in range(8):
        pose = light_poses[i]
        campos = torch.from_numpy(pose[:3,-1]).cuda()
        camrot = torch.from_numpy(pose[:3,:3]).cuda()
        # img,dp = renderer_3d.render_depth(theta,phi,radius)
        img_,dp_ = renderer_3d.render_depth(campos, camrot, l_focal, 800,800, 98, 102)
        # save for vis

        # torch3d_color = img.cpu().numpy()
        # torch3d_depth = (dp/dp.max()).squeeze().cpu().numpy()
        # torch3d_depth[torch3d_depth<=0]=0

        torch3d_color_ = img_.cpu().numpy()
        torch3d_depth_ = (dp_/dp_.max()).squeeze().cpu().numpy()
        torch3d_depth_[torch3d_depth_<=0]=0

        imageio.imwrite(f"color_{i}.png", torch3d_color_[...,:3])
        imageio.imwrite(f"depth_{i}.png", torch3d_depth_)
    

    # imageio.imwrite("color.png", np.hstack([torch3d_color_[...,:3],torch3d_color[...,:3]]))
    # imageio.imwrite("depth.png", np.hstack([torch3d_depth_,torch3d_depth]))






