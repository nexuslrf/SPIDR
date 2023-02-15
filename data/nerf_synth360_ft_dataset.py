from models.mvs.mvs_utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
from kornia import create_meshgrid
import time
import json
from . import data_utils
from plyfile import PlyData, PlyElement

from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
import h5py

from data.base_dataset import BaseDataset
import configparser
import tqdm
from os.path import join
import cv2
# import torch.nn.functional as F
from .data_utils import get_dtu_raydir
from utils import lighting

FLIP_Z = np.asarray([
    [1,0,0],
    [0,1,0],
    [0,0,-1],
], dtype=np.float32)

def colorjitter(img, factor):
    # brightness_factor,contrast_factor,saturation_factor,hue_factor
    # img = F.adjust_brightness(img, factor[0])
    # img = F.adjust_contrast(img, factor[1])
    img = F.adjust_saturation(img, factor[2])
    img = F.adjust_hue(img, factor[3]-1.0)

    return img


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
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



def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    c2w = torch.FloatTensor(c2w)
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)],
                             -1)  # (H, W, 3)

    return directions

class NerfSynth360FtDataset(BaseDataset):

    def initialize(self, opt, img_wh=[800,800], downSample=1.0, max_len=-1, norm_w2c=None, norm_c2w=None):
        self.opt = opt
        self.data_dir = opt.data_root
        self.scan = opt.scan
        self.split = opt.split
        downSample = opt.down_sample
        self.img_wh = (int(opt.img_wh[0] * downSample), int(opt.img_wh[1] * downSample))
        self.downSample = downSample

        self.scale_factor = 1.0 / 1.0
        self.max_len = max_len

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])
        self.near_far = np.array([self.opt.near_plane, self.opt.far_plane])

        if not self.opt.bg_color or self.opt.bg_color == 'black':
            self.bg_color = (0, 0, 0)
        elif self.opt.bg_color == 'white':
            self.bg_color = (1, 1, 1)
        elif self.opt.bg_color == 'random':
            self.bg_color = 'random'
        else:
            self.bg_color = [float(one) for one in self.opt.bg_color.split(",")]
        self.define_transforms()
        meta_split = "train" if self.split in ["render","light"] else self.split
        with open(os.path.join(self.data_dir, self.scan, f'transforms_{meta_split}.json'), 'r') as f:
            self.meta = json.load(f)
        with open(os.path.join(self.data_dir, self.scan, f'transforms_test.json'), 'r') as f:
            self.testmeta = json.load(f)
        self.id_list = [i for i in range(len(self.meta["frames"]))]
        self.test_id_list = [i for i in range(len(self.testmeta["frames"]))]
        self.norm_w2c, self.norm_c2w = torch.eye(4, device="cuda", dtype=torch.float32), torch.eye(4, device="cuda", dtype=torch.float32)
        if opt.normview > 0:
            _, _ , w2cs, c2ws = self.build_proj_mats(list=self.test_id_list)
            norm_w2c, norm_c2w = self.normalize_cam(w2cs, c2ws)
        if opt.normview >= 2:
            self.norm_w2c, self.norm_c2w = torch.as_tensor(norm_w2c, device="cuda", dtype=torch.float32), torch.as_tensor(norm_c2w, device="cuda", dtype=torch.float32)
            norm_w2c, norm_c2w = None, None
        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = self.build_proj_mats(norm_w2c=norm_w2c, norm_c2w=norm_c2w)
        
        if self.split == 'render':
            self.get_render_poses()
            print("render only, pose total:", self.total)
        elif self.split == 'light':
            self.get_light_poses()
            print("light view, pose total:", self.total)
        else:
            self.build_init_metas()
            self.read_meta()
            self.total = len(self.id_list)
            print("dataset total:", self.split, self.total)

    def get_render_poses(self):
        stride = 20 #self.opt.render_stride
        radius = 4 #self.opt.render_radius
        self.render_poses = np.stack([pose_spherical(angle, -30.0, radius) @ self.blender2opencv for angle in np.linspace(-180, 180, stride + 1)[:-1]], 0)
        self.total = len(self.render_poses)
    
    def get_light_poses(self, lxyz=None):
        radius_cam = 4
        radius = self.opt.light_radius
        if lxyz is None:
            lxyz, area = lighting.gen_light_xyz(
                self.opt.light_env_h, self.opt.light_env_w, radius)
        phis = -np.arcsin(lxyz[..., 2]/radius)*180/np.pi
        thetas = np.arctan2(lxyz[...,0], lxyz[...,1])*180/np.pi # (16,32)
        light_shape = phis.shape
        self.light_poses = np.stack([
            pose_spherical(thetas[i,j], phis[i,j], radius) @ self.blender2opencv 
                for i in range(light_shape[0]) for j in range(light_shape[1])], 0)
        self.light_t_offset = radius - radius_cam
        self.near_far = self.near_far + self.light_t_offset
        intrinsic_cam = self.intrinsics[0]
        cam_focal_x, cam_focal_y = intrinsic_cam[0,0], intrinsic_cam[1,1]
        focal_x = radius / radius_cam * cam_focal_x
        focal_y = radius / radius_cam * cam_focal_y
        self.focal = focal_x
        self.intrinsics = \
            [np.array([[focal_x, 0, self.width / 2], [0, focal_y, self.height / 2], [0, 0, 1]])]
        self.total = len(self.light_poses)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # ['random', 'random2', 'patch'], default: no random sample
        parser.add_argument('--random_sample',
                            type=str,
                            default='none',
                            help='random sample pixels')
        parser.add_argument('--random_sample_size',
                            type=int,
                            default=1024,
                            help='number of random samples')
        parser.add_argument('--init_view_num',
                            type=int,
                            default=3,
                            help='number of random samples')
        parser.add_argument('--shape_id', type=int, default=0, help='shape id')
        parser.add_argument('--trgt_id', type=int, default=0, help='shape id')
        parser.add_argument('--num_nn',
                            type=int,
                            default=1,
                            help='number of nearest views in a batch')
        parser.add_argument(
            '--near_plane',
            type=float,
            default=2.125,
            help=
            'Near clipping plane, by default it is computed according to the distance of the camera '
        )
        parser.add_argument(
            '--far_plane',
            type=float,
            default=4.525,
            help=
            'Far clipping plane, by default it is computed according to the distance of the camera '
        )

        parser.add_argument(
            '--bg_color',
            type=str,
            default="white",
            help=
            'background color, white|black(None)|random|rgb (float, float, float)'
        )

        parser.add_argument(
            '--scan',
            type=str,
            default="scan1",
            help=''
        )
        parser.add_argument(
                    '--full_comb',
                    type=int,
                    default=0,
                    help=''
                )

        parser.add_argument('--inverse_gamma_image',
                            type=int,
                            default=-1,
                            help='de-gamma correct the input image')
        parser.add_argument('--pin_data_in_memory',
                            type=int,
                            default=-1,
                            help='load whole data in memory')
        parser.add_argument('--normview',
                            type=int,
                            default=0,
                            help='load whole data in memory')
        parser.add_argument(
            '--id_range',
            type=int,
            nargs=3,
            default=(0, 385, 1),
            help=
            'the range of data ids selected in the original dataset. The default is range(0, 385). If the ids cannot be generated by range, use --id_list to specify any ids.'
        )
        parser.add_argument(
            '--id_list',
            type=int,
            nargs='+',
            default=None,
            help=
            'the list of data ids selected in the original dataset. The default is range(0, 385).'
        )
        parser.add_argument(
            '--split',
            type=str,
            default="train",
            help=
            'train, val, test'
        )
        parser.add_argument("--half_res", action='store_true',
                            help='load blender synthetic data at 400x400 instead of 800x800')
        parser.add_argument("--testskip", type=int, default=8,
                            help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
        parser.add_argument('--dir_norm',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--train_load_num',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')
        parser.add_argument('--down_sample', type=float, default=1.)
        parser.add_argument(
            '--img_wh', type=int, nargs=2, default=(800, 800),
            # default=(1088, 640),
            help='resize target of the image'
        )
        return parser

    def normalize_cam(self, w2cs, c2ws):
        # cam_xyz = c2ws[..., :3, 3]
        # rtp = self.bcart2sphere(cam_xyz)
        # print(rtp.shape)
        # rtp = np.mean(rtp, axis=0)
        # avg_xyz = self.sphere2cart(rtp)
        # euler_lst = []
        # for i in range(len(c2ws)):
        #     euler_angles = self.matrix2euler(c2ws[i][:3,:3])
        #     print("euler_angles", euler_angles)
        #     euler_lst += [euler_angles]
        # euler = np.mean(np.stack(euler_lst, axis=0), axis=0)
        # print("euler mean ",euler)
        # M = self.euler2matrix(euler)
        # norm_c2w = np.eye(4)
        # norm_c2w[:3,:3] = M
        # norm_c2w[:3,3] = avg_xyz
        # norm_w2c = np.linalg.inv(norm_c2w)
        # return norm_w2c, norm_c2w
        index = 0
        return w2cs[index], c2ws[index]

    def define_transforms(self):
        self.transform = T.ToTensor()



    def get_campos_ray(self):
        centerpixel = np.asarray(self.img_wh).astype(np.float32)[None, :] // 2
        camposes = []
        centerdirs = []
        for i, idx in enumerate(self.id_list):
            c2w = self.cam2worlds[i].astype(np.float32)
            campos = c2w[:3, 3]
            camrot = c2w[:3, :3]
            raydir = get_dtu_raydir(centerpixel, self.intrinsics[0].astype(np.float32), camrot, True)
            camposes.append(campos)
            centerdirs.append(raydir)
        camposes = np.stack(camposes, axis=0)  # 2091, 3
        centerdirs = np.concatenate(centerdirs, axis=0)  # 2091, 3
        # print("camposes", camposes.shape, centerdirs.shape)
        return torch.as_tensor(camposes, device="cuda", dtype=torch.float32), torch.as_tensor(centerdirs, device="cuda",
                                                                                              dtype=torch.float32)



    def build_init_metas(self):
        self.view_id_list = []
        cam_xyz_lst = [c2w[:3,3] for c2w in self.cam2worlds]
        _, _, w2cs, c2ws = self.build_proj_mats(meta=self.testmeta, list=self.test_id_list)
        test_cam_xyz_lst = [c2w[:3,3] for c2w in c2ws]

        if self.split=="train":
            cam_xyz = np.stack(cam_xyz_lst, axis=0)
            test_cam_xyz = np.stack(test_cam_xyz_lst, axis=0)
            triangles = data_utils.triangluation_bpa(cam_xyz, test_pnts=test_cam_xyz, full_comb=self.opt.full_comb>0)
            self.view_id_list = [triangles[i] for i in range(len(triangles))]
            if self.opt.full_comb<0:
                with open(f'../data/nerf_synth_configs/list/lego360_init_pairs.txt') as f:
                    for line in f:
                        str_lst = line.rstrip().split(',')
                        src_views = [int(x) for x in str_lst]
                        self.view_id_list.append(src_views)


    def load_init_points(self):
        points_path = os.path.join(self.data_dir, self.scan, "colmap_results/dense/fused.ply")
        # points_path = os.path.join(self.data_dir, self.scan, "exported/pcd_te_1_vs_0.01_jit.ply")
        assert os.path.exists(points_path)
        plydata = PlyData.read(points_path)
        # plydata (PlyProperty('x', 'double'), PlyProperty('y', 'double'), PlyProperty('z', 'double'), PlyProperty('nx', 'double'), PlyProperty('ny', 'double'), PlyProperty('nz', 'double'), PlyProperty('red', 'uchar'), PlyProperty('green', 'uchar'), PlyProperty('blue', 'uchar'))
        print("plydata", plydata.elements[0])
        x,y,z=torch.as_tensor(plydata.elements[0].data["x"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["y"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["z"].astype(np.float32), device="cuda", dtype=torch.float32)
        points_xyz = torch.stack([x,y,z], dim=-1).to(torch.float32)

        # np.savetxt(os.path.join(self.data_dir, self.scan, "exported/pcd.txt"), points_xyz.cpu().numpy(), delimiter=";")
        if self.opt.comb_file is not None:
            file_points = np.loadtxt(self.opt.comb_file, delimiter=";")
            print("file_points", file_points.shape)
            comb_xyz = torch.as_tensor(file_points[...,:3].astype(np.float32), device=points_xyz.device, dtype=points_xyz.dtype)
            points_xyz = torch.cat([points_xyz, comb_xyz], dim=0)
        # np.savetxt("/home/xharlie/user_space/codes/testNr/checkpoints/pcolallship360_load_confcolordir_KNN8_LRelu_grid320_333_agg2_prl2e3_prune1e4/points/save.txt", points_xyz.cpu().numpy(), delimiter=";")
        return points_xyz



    def build_proj_mats(self, meta=None, list=None, norm_w2c=None, norm_c2w=None):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        list = self.id_list if list is None else list
        meta = self.meta if meta is None else meta
        focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        self.focal = focal
        self.near_far = np.array([self.opt.near_plane, self.opt.far_plane])
        for vid in list:
            frame = meta['frames'][vid]
            c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
            if norm_w2c is not None:
                c2w = norm_w2c @ c2w
            w2c = np.linalg.inv(c2w)
            cam2worlds.append(c2w)
            world2cams.append(w2c)

            intrinsic = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy().astype(np.float32))

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            proj_mats += [(proj_mat_l, self.near_far)]

        proj_mats, intrinsics = np.stack(proj_mats), np.stack(intrinsics)
        world2cams, cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
        return proj_mats, intrinsics, world2cams, cam2worlds



    def define_transforms(self):
        self.transform = T.ToTensor()


    def read_meta(self):

        w, h = self.img_wh
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.blackimgs = []
        self.whiteimgs = []
        self.depths = []
        self.alphas = []

        self.view_id_dict = {}
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)

        count = 0
        for i, idx in tqdm.tqdm(enumerate(self.id_list)):
            frame = self.meta['frames'][idx]

            image_path = os.path.join(self.data_dir, self.scan, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            self.depths += [(img[-1:, ...] > 0.1).numpy().astype(np.float32)]
            self.alphas += [img[-1:].numpy().astype(np.float32)]
            self.blackimgs += [img[:3] * img[-1:]]
            self.whiteimgs += [img[:3] * img[-1:] + (1 - img[-1:])]


            # ray directions for all pixels, same for all images (same H, W, focal)

            # rays_o, rays_d = get_rays(self.directions, self.cam2worlds[i])  # both (h*w, 3)
            #
            # self.all_rays += [torch.cat([rays_o, rays_d,
            #                              self.near_far[0] * torch.ones_like(rays_o[:, :1]),
            #                              self.near_far[1] * torch.ones_like(rays_o[:, :1])], 1)]  # (h*w, 8)
            self.view_id_dict[idx] = i
        self.poses = self.cam2worlds


    def __len__(self):
        if self.split == 'train':
            return len(self.id_list) if self.max_len <= 0 else self.max_len
        return len(self.id_list) if self.max_len <= 0 else self.max_len


    def name(self):
        return 'NerfSynthFtDataset'


    def __del__(self):
        print("end loading")

    def normalize_rgb(self, data):
        # to unnormalize image for visualization
        # data C, H, W
        C, H, W = data.shape
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        return (data - mean) / std

    def get_init_item(self, idx, crop=False):
        sample = {}
        init_view_num = self.opt.init_view_num
        view_ids = self.view_id_list[idx]
        if self.split == 'train':
            view_ids = view_ids[:init_view_num]

        affine_mat, affine_mat_inv = [], []
        mvs_images, imgs, depths_h, alphas = [], [], [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i in view_ids:
            vid = self.view_id_dict[i]
            # mvs_images += [self.normalize_rgb(self.blackimgs[vid])]
            # mvs_images += [self.whiteimgs[vid]]
            mvs_images += [self.blackimgs[vid]]
            imgs += [self.whiteimgs[vid]]
            proj_mat_ls, near_far = self.proj_mats[vid]
            intrinsics.append(self.intrinsics[vid])
            w2cs.append(self.world2cams[vid])
            c2ws.append(self.cam2worlds[vid])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            depths_h.append(self.depths[vid])
            alphas.append(self.alphas[vid])
            near_fars.append(near_far)

        for i in range(len(affine_mat)):
            view_proj_mats = []
            ref_proj_inv = affine_mat_inv[i]
            for j in range(len(affine_mat)):
                if i == j:  # reference view
                    view_proj_mats += [np.eye(4)]
                else:
                    view_proj_mats += [affine_mat[j] @ ref_proj_inv]
            # view_proj_mats: 4, 4, 4
            view_proj_mats = np.stack(view_proj_mats)
            proj_mats.append(view_proj_mats[:, :3])
        # (4, 4, 3, 4)
        proj_mats = np.stack(proj_mats)
        imgs = np.stack(imgs).astype(np.float32)
        mvs_images = np.stack(mvs_images).astype(np.float32)

        depths_h = np.stack(depths_h)
        alphas = np.stack(alphas)
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        # view_ids_all = [target_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        # c2ws_all = self.cam2worlds[self.remap[view_ids_all]]

        sample['images'] = imgs  # (V, 3, H, W)
        sample['mvs_images'] = mvs_images  # (V, 3, H, W)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['alphas'] = alphas.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars_depth'] = near_fars.astype(np.float32)[0]
        sample['near_fars'] = np.tile(self.near_far.astype(np.float32)[None,...],(len(near_fars),1))
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        # sample['light_id'] = np.array(light_idx)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        # sample['scan'] = scan
        # sample['c2ws_all'] = c2ws_all.astype(np.float32)


        for key, value in sample.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                    sample[key] = value.unsqueeze(0)

        return sample



    def __getitem__(self, id, crop=False, full_img=False):
        item = {}
        if self.bg_color:
            if self.bg_color == 'random':
                val = np.random.rand()
                if val > 0.5:
                    item['bg_color'] = torch.FloatTensor([1, 1, 1])
                    img = self.whiteimgs[id]
                else:
                    item['bg_color'] = torch.FloatTensor([0, 0, 0])
                    img = self.blackimgs[id]
            else:
                item['bg_color'] = torch.FloatTensor(self.bg_color)
                img = self.whiteimgs[id]
        else:
            img = self.whiteimgs[id]

        w2c = self.world2cams[id]
        c2w = self.cam2worlds[id]
        intrinsic = self.intrinsics[id]
        proj_mat_ls, near_far = self.proj_mats[id]
        alpha = self.alphas[id][0]

        gt_image = np.transpose(img, (1,2,0))
        # print("gt_image", gt_image.shape)
        width, height = gt_image.shape[1], gt_image.shape[0]
        camrot = (c2w[0:3, 0:3])
        campos = c2w[0:3, 3]
        # print("camrot", camrot, campos)

        item["intrinsic"] = intrinsic
        # item["intrinsic"] = sample['intrinsics'][0, ...]
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float() # @ FLIP_Z
        item["c2w"] = torch.from_numpy(c2w).float()
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([near_far[1]]).view(1, 1)
        item['near'] = torch.FloatTensor([near_far[0]]).view(1, 1)
        item['h'] = height
        item['w'] = width
        item['depths_h'] = self.depths[id]
        # bounding box
        if full_img:
            item['images'] = img[None,...]
        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            vx, vy = (alpha > 0).nonzero()
            idx = np.random.randint(0, vx.shape[0])
            indx = max(vx[idx] - subsamplesize//2, 0)
            indy = max(vy[idx] - subsamplesize//2, 0)
            dilation = np.random.randint(0,3)+1
            # indx = np.random.randint(0, width - subsamplesize + 1)
            # indy = np.random.randint(0, height - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, min(indx + dilation*subsamplesize, width), dilation).astype(np.float32),
                np.arange(indy, min(indy + dilation*subsamplesize, height), dilation).astype(np.float32))
        elif self.opt.random_sample == "random":
            px = np.random.randint(0,
                                   width,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.randint(0,
                                   height,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "random2":
            px = np.random.uniform(0,
                                   width - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.uniform(0,
                                   height - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "proportional_random":
            raise Exception("no gt_mask, no proportional_random !!!")
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32))

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
        item["pixel_idx"] = pixelcoords
        # print("pixelcoords", pixelcoords.reshape(-1,2)[:10,:])
        raydir = get_dtu_raydir(pixelcoords, item["intrinsic"], camrot, self.opt.dir_norm > 0)
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()
        py, px = py.astype(np.int32), px.astype(np.int32)
        gt_image = gt_image[py, px].reshape((-1, 3))
        # gt_mask = gt_mask[py, px, :]
        item['gt_image'] = gt_image
        item['alpha'] = alpha[py, px].reshape((-1, 1))
        item['id'] = id

        return item



    def get_item(self, idx, crop=False, full_img=False):
        item = self.__getitem__(idx, crop=crop, full_img=full_img)

        for key, value in item.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                item[key] = value.unsqueeze(0)
        return item



    def get_dummyrot_item(self, idx, crop=False):

        item = {}
        width, height = self.width, self.height
        if self.split == 'render':             
            transform_matrix = self.render_poses[idx]
            t_offset = 0
        elif self.split == 'light':
            transform_matrix = self.light_poses[idx]
            t_offset = self.light_t_offset
        else:
            assert False, "dummyrot only supports two splits"
        camrot = transform_matrix[0:3, 0:3]
        campos = transform_matrix[0:3, 3]
        

        item["focal"] = self.focal
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float()
        item['lightpos'] = item["campos"]
        item['intrinsic'] = self.intrinsics[0]

 
        # near far
        item['far'] = torch.FloatTensor([self.opt.far_plane]).view(1, 1) + t_offset
        item['near'] = torch.FloatTensor([self.opt.near_plane]).view(1, 1) + t_offset
        item['h'] = self.height
        item['w'] = self.width

        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            indx = np.random.randint(0, width - subsamplesize + 1)
            indy = np.random.randint(0, height - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, indx + subsamplesize).astype(np.float32),
                np.arange(indy, indy + subsamplesize).astype(np.float32))
        elif self.opt.random_sample == "random":
            px = np.random.randint(0,
                                   width,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.randint(0,
                                   height,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "random2":
            px = np.random.uniform(0,
                                   width - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.uniform(0,
                                   height - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "proportional_random":
            raise Exception("no gt_mask, no proportional_random !!!")
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32))

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
        item["pixel_idx"] = pixelcoords
        # print("pixelcoords", pixelcoords.reshape(-1,2)[:10,:])
        raydir = get_dtu_raydir(pixelcoords, self.intrinsics[0], camrot, self.opt.dir_norm > 0)
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()
        item['id'] = idx

        if self.bg_color:
            if self.bg_color == 'random':
                val = np.random.rand()
                if val > 0.5:
                    item['bg_color'] = torch.FloatTensor([1, 1, 1])
                else:
                    item['bg_color'] = torch.FloatTensor([0, 0, 0])
            else:
                item['bg_color'] = torch.FloatTensor(self.bg_color)


        for key, value in item.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            item[key] = value.unsqueeze(0)

        return item

