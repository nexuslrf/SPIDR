import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import glob
import copy
import torch
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from models.mvs.mvs_points_model import MvsPointsModel
from models.mvs import mvs_utils, filter_utils
from models.rendering.mesh_renderer import MeshRenderer
from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
torch.manual_seed(0)
np.random.seed(0)
import random
import cv2
from PIL import Image
from tqdm import tqdm
import gc
from ft_helper import update_iter_opt, create_all_bg
from deform_tools.mesh_editing import marching_cube

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)

def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [item[:, mask, ...] if item is not None else None for item in seconddim_lst]
    return first_lst, second_lst



def render_vid(model, dataset, visualizer, opt, bg_info, steps=0, gen_vid=True, train_log=False, model_fixed=None):
    print('-----------------------------------Rendering-----------------------------------')
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step))
    rand_i = random.randint(0, total_num-1)
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    s, e = (0, total_num) if not train_log else (rand_i, rand_i + 1)
    for i in range(s, e):
        data = dataset.get_dummyrot_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        # cam_posts.append(data['campos'])
        # cam_dirs.append(data['raydir'] + data['campos'][None,...])
        # continue
        visuals = None
        stime = time.time()

        for k in range(0, height * width, chunk_size):
            start = k
            end = min([k + chunk_size, height * width])
            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            # print("tmpgts", tmpgts["gt_image"].shape)
            # print(data["pixel_idx"])
            model.set_input(data)
            if opt.bgmodel.endswith("plane"):
                img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                if len(bg_ray_lst) > 0:
                    bg_ray_all = bg_ray_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                    bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                data["bg_ray"] = bg_ray

            model.test(model_fixed=model_fixed)
            curr_visuals = model.get_current_visuals(data=data)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if key == "gt_image" or value is None: continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                    visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if key == "gt_image": continue
                    visuals[key][start:end, :] = value.cpu().numpy()

        for key, value in visuals.items():
            visualizer.print_details("{}:{}".format(key, visuals[key].shape))
            visuals[key] = visuals[key].reshape(height, width, -1)
            if key == 'coarse_depth':
                visuals[key] = visuals[key].clip(*dataset.near_far)
        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, i, opt=opt) # SAVE_IMG

    # visualizer.save_neural_points(200, np.concatenate(cam_posts, axis=0),None, None, save_ref=False)
    # visualizer.save_neural_points(200, np.concatenate(cam_dirs, axis=0),None, None, save_ref=False)
    # print("vis")
    # exit()

    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num), 0)
        print('--------------------------------Finish generating vid--------------------------------')

    return


def test(model, dataset, visualizer, opt, bg_info, test_steps=0, gen_vid=False, lpips=True, train_log=False, model_fixed=None):
    print('-----------------------------------Testing-----------------------------------')
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step)) # 1 if test_steps == 10000 else opt.test_num_step
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    model_rand_smpl_size = model.opt.random_sample_size
    height = dataset.height
    width = dataset.width
    visualizer.reset()
    count = 0
    if not train_log:
        test_end_id = total_num if opt.test_end_id < 0 else opt.test_end_id
        test_iter = range(opt.test_start_id, test_end_id, opt.test_num_step)
    else:
        if len(opt.test_ids) > 0:
            test_iter = opt.test_ids
        else:
            rand_i = random.randint(0, total_num-1)
            test_iter = [rand_i]
    for i in test_iter: # 1 if test_steps == 10000 else opt.test_num_step
        if i > total_num-1: i  = i % total_num
        if opt.sample_debug: i = 4 #46 #100 #42 # 95
        data = dataset.get_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        edge_mask = torch.zeros([height, width], dtype=torch.bool)
        edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
        edge_mask=edge_mask.reshape(-1) > 0
        np_edge_mask=edge_mask.numpy().astype(bool)
        totalpixel = pixel_idx.shape[1]
        tmpgts = {}
        tmpgts["gt_image"] = data['gt_image'].clone()
        tmpgts["gt_mask"] = data['gt_mask'].clone() if "gt_mask" in data else None

        # data.pop('gt_image', None)
        data.pop('gt_mask', None)

        visuals = None
        stime = time.time()
        ray_masks = []

        chunk_range = range(0, totalpixel, chunk_size)
        B = raydir.shape[0]
        if model.opt.random_sample == 'patch':
            raydir = raydir.reshape(raydir.shape[0], height, width, 3)
            pixel_idx = pixel_idx.reshape(pixel_idx.shape[0], height, width, 2)
            chunk_range = [(r,c) for r in range(0, height, patch_size) for c in range(0, width, patch_size)]

        for k in chunk_range:
            if model.opt.random_sample != 'patch':
                start = k
                end = min([k + chunk_size, totalpixel])
                data['raydir'] = raydir[:, start:end, :]
                data["pixel_idx"] = pixel_idx[:, start:end, :]
            else:
                r,c = k # (588, 316)
                if opt.sample_debug:
                    r = 120 #403 - 60 #602 #338 #301 #267 #457 #547 #561 #450 # 463 # 247 
                    c = 322 #450 #377 #321 #329 #381 #516 #432 #417 #356 # 257 # 439 
                    patch_size = 40
                data['raydir'] = raydir[:, r:r+patch_size, c:c+patch_size, :]#.reshape(B, -1, 3)
                data['pixel_idx'] = pixel_idx[:, r:r+patch_size, c:c+patch_size, :]#.reshape(B, -1, 2)
                
                _, p_h, p_w, _ = data['raydir'].shape
                data['raydir'] = data['raydir'].reshape(B, -1, 3)
                data['pixel_idx'] = data['pixel_idx'].reshape(B, -1, 2)
                model.opt.random_sample_size = (p_h, p_w)

            model.set_input(data)

            if opt.bgmodel.endswith("plane"):
                img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                if len(bg_ray_lst) > 0:
                    bg_ray_all = bg_ray_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                    bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, 
                                            intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                data["bg_ray"] = bg_ray

                # xyz_world_sect_plane_lst.append(xyz_world_sect_plane)
            model.test(model_fixed=model_fixed)
            curr_visuals = model.get_current_visuals(data=data)

            # print("loss", mse2psnr(torch.nn.MSELoss().to("cuda")(curr_visuals['coarse_raycolor'], tmpgts["gt_image"].view(1, -1, 3)[:, start:end, :].cuda())))
            # print("sum", torch.sum(torch.square(tmpgts["gt_image"].view(1, -1, 3)[:, start:end, :] - tmpgts["gt_image"].view(height, width, 3)[data["pixel_idx"][0,...,1].long(), data["pixel_idx"][0,...,0].long(),:])))
            chunk_pixel_id = data["pixel_idx"].cpu().numpy().astype(np.int32)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    chunk = value.cpu().numpy()
                    channel_dim = chunk.shape[-1]
                    visuals[key] = np.zeros((height, width, channel_dim)).astype(chunk.dtype)
                    # print(key)
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = value.cpu().numpy()
            if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                ray_masks.append(model.output["ray_mask"] > 0)
            torch.cuda.empty_cache()
        if len(ray_masks) > 0:
            ray_masks = torch.cat(ray_masks, dim=1)
        # visualizer.save_neural_points(data["id"].cpu().numpy()[0], (raydir.cuda() + data["campos"][:, None, :]).squeeze(0), None, data, save_ref=True)
        # exit()
        # print("curr_visuals",curr_visuals)
        pixel_idx=pixel_idx.to(torch.long)
        gt_image = torch.zeros((height*width, 3), dtype=torch.float32)
        gt_image[edge_mask, :] = tmpgts['gt_image'].clone()
        if 'gt_image' in model.visual_names:
            visuals['gt_image'] = gt_image
        if 'gt_mask' in curr_visuals:
            visuals['gt_mask'] = np.zeros((height, width, 3)).astype(chunk.dtype)
            visuals['gt_mask'][np_edge_mask,:] = tmpgts['gt_mask']
        if 'ray_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            print(visuals['ray_masked_coarse_raycolor'].shape, ray_masks.cpu().numpy().shape)
            visuals['ray_masked_coarse_raycolor'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        if 'ray_depth_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_depth_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            visuals['ray_depth_masked_coarse_raycolor'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'ray_depth_masked_gt_image' in model.visual_names:
            visuals['ray_depth_masked_gt_image'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['ray_depth_masked_gt_image'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'gt_image_ray_masked' in model.visual_names:
            visuals['gt_image_ray_masked'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['gt_image_ray_masked'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        if 'coarse_depth' in model.visual_names:
            visuals['coarse_depth'] = visuals['coarse_depth'].clip(*dataset.near_far)
        for key, value in visuals.items():
            if key in opt.visual_items:
                visualizer.print_details("{}:{}".format(key, visuals[key].shape))
                visuals[key] = visuals[key].reshape(height, width, -1)


        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, i, opt=opt, train_log=train_log)

        acc_dict = {}
        if "coarse_raycolor" in opt.test_color_loss_items:
            loss = torch.nn.MSELoss().to("cuda")(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), gt_image.view(1, -1, 3).cuda())
            acc_dict.update({"coarse_raycolor": loss})
            print("coarse_raycolor", loss, mse2psnr(loss))

        if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
            masked_gt = tmpgts["gt_image"].view(1, -1, 3).cuda()[ray_masks,:].reshape(1, -1, 3)
            ray_masked_coarse_raycolor = torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3)[:,edge_mask,:][ray_masks,:].reshape(1, -1, 3)

            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_gt")
            # filepath = os.path.join("/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # tmpgtssave = tmpgts["gt_image"].view(1, -1, 3).clone()
            # tmpgtssave[~ray_masks,:] = 1.0
            # img = np.array(tmpgtssave.view(height,width,3))
            # save_image(img, filepath)
            #
            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_coarse_raycolor")
            # filepath = os.path.join(
            #     "/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # csave = torch.zeros_like(tmpgts["gt_image"].view(1, -1, 3))
            # csave[~ray_masks, :] = 1.0
            # csave[ray_masks, :] = torch.as_tensor(visuals["coarse_raycolor"]).view(1, -1, 3)[ray_masks,:]
            # img = np.array(csave.view(height, width, 3))
            # save_image(img, filepath)

            loss = torch.nn.MSELoss().to("cuda")(ray_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_masked_coarse_raycolor", loss, mse2psnr(loss)))

        if "ray_depth_mask" in model.output and "ray_depth_masked_coarse_raycolor" in opt.test_color_loss_items:
            ray_depth_masks = model.output["ray_depth_mask"].reshape(model.output["ray_depth_mask"].shape[0], -1)
            masked_gt = torch.masked_select(tmpgts["gt_image"].view(1, -1, 3).cuda(), (ray_depth_masks[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
            ray_depth_masked_coarse_raycolor = \
                torch.masked_select(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), 
                                    ray_depth_masks[..., None].expand(-1, -1, 3).reshape(1, -1, 3))

            loss = torch.nn.MSELoss().to("cuda")(ray_depth_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_depth_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_depth_masked_coarse_raycolor", loss, mse2psnr(loss)))
        print(acc_dict.items())
        visualizer.accumulate_losses(acc_dict)
        count+=1

    if opt.brdf_rendering:
        model.net_ray_marching.module.brdf_renderer.vis_light(visualizer)
    visualizer.print_losses(count)
    psnr = visualizer.get_psnr(opt.test_color_loss_items[0])
    # visualizer.reset()

    print('--------------------------------Finish Test Rendering--------------------------------')
    if 'gt_image' in opt.visual_items:
        report_metrics(visualizer.image_dir, visualizer.image_dir, visualizer.image_dir, 
            ["psnr", "ssim", "lpips", "vgglpips", "rmse"] if lpips else ["psnr", "ssim", "rmse"], 
            [i for i in range(0, total_num, opt.test_num_step)], 
            imgStr="step-%04d-{}.png".format(opt.visual_items[0]),gtStr="step-%04d-{}.png".format('gt_image'))



    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num, opt.test_num_step), test_steps)
        print('--------------------------------Finish generating vid--------------------------------')
    model.opt.random_sample_size = model_rand_smpl_size
    return psnr

def bake_depth(model, opt, step):
    test_opt = copy.deepcopy(opt)
    test_opt.split = 'light'
    test_opt.nerf_splits = ['light']
    test_opt.visual_items = ['coarse_depth']
    test_opt.name = opt.name + "/light_{}".format(step if opt.light_save_id <0 else opt.light_save_id)
    # test_opt.down_sample = 0.5
    dataset = create_dataset(test_opt)
    visualizer = Visualizer(test_opt)

    ori_visual_names, ori_visual_items = model.visual_names, model.opt.visual_items
    model.visual_names = ['coarse_depth']
    model.opt.visual_items = ['coarse_depth']
    model.opt.depth_only = True
    model.opt.compute_depth = True
    model.opt.is_train = 0
    model.opt.no_loss = 1

    model.eval()
    total_num = dataset.total
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    depth_maps = []
    if opt.depth_from_mesh:
        mesh_render = MeshRenderer(opt.mesh_path, device=model.device)
    with torch.no_grad():
        for i in tqdm(range(total_num)):
            data = dataset.get_dummyrot_item(i)
            raydir = data['raydir'].clone()
            pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
            # cam_posts.append(data['campos'])
            # cam_dirs.append(data['raydir'] + data['campos'][None,...])
            # continue
            depth = None
            stime = time.time()
            if opt.depth_from_mesh:
                _, depth = mesh_render.render_depth(data['campos'][0].cuda(), data['camrotc2w'][0].cuda(), 
                    data['focal'].item(), height, width, data['near'].item(), data['far'].item())
                depth = depth.cpu()
            else:
                for k in range(0, height * width, chunk_size):
                    start = k
                    end = min([k + chunk_size, height * width])
                    data['raydir'] = raydir[:, start:end, :]
                    data["pixel_idx"] = pixel_idx[:, start:end, :]
                    # print("tmpgts", tmpgts["gt_image"].shape)
                    # print(data["pixel_idx"])
                    model.set_input(data)
                    model.test()
                    curr_visuals = model.get_current_visuals(data=data)
                    chunk = curr_visuals['coarse_depth'].data.cpu()
                    if depth is None:
                        depth = torch.zeros(height * width, 1, dtype=chunk.dtype)
                        depth[start:end, :] = chunk
                    else:
                        depth[start:end, :] = chunk

            depth = depth.reshape(height, width, 1).clamp(*dataset.near_far)
            print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
            visualizer.display_current_results(
                {'coarse_depth': depth.numpy()}, i, opt=test_opt) # SAVE_IMG
            depth_maps.append(depth.detach())
            torch.cuda.empty_cache()

    depth_maps = torch.stack(depth_maps, 0)
    light_poses = torch.from_numpy(dataset.light_poses).float()
    light_cache = {'depth_maps': depth_maps, 'light_poses': light_poses, 'intrinsic': dataset.intrinsics[0]}
    torch.save(light_cache, os.path.join(opt.checkpoints_dir, test_opt.name, 'light_cache.pth'))
    model.visual_names = ori_visual_names
    model.opt.visual_items = ori_visual_items
    model.opt.depth_only = False
    model.opt.is_train = 1
    model.opt.no_loss = 0
    return light_cache

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def get_mesh(model, opt, step, chunk_size=16384, save_vol=False, mesh_suffix=""):
    point_xyz_w_tensor = model.neural_points.xyz[None,...]
    mc_vsize = opt.vsize * np.array([opt.marching_cube_scale])
    radius_limit_np, depth_limit_np, ranges_np, vsize_np, vdim_np, scaled_vsize_np, scaled_vdim_np, vscale_np, \
        ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu \
            = model.neural_points.querier.get_hyperparameters(mc_vsize, point_xyz_w_tensor, ranges=opt.ranges)
    xyz_grid = torch.stack(torch.meshgrid(*[torch.arange(d) for d in scaled_vdim_np]), dim=-1).to(point_xyz_w_tensor.device)
    xyz_grid = xyz_grid * scaled_vsize_gpu + ranges_gpu[:3] + scaled_vsize_gpu/2
    grid_shape = xyz_grid.shape[:-1]
    xyz_grid_flatten = xyz_grid.reshape(-1, 3)
    num_pts = xyz_grid_flatten.shape[0]
    sdf_flatten = torch.zeros_like(xyz_grid_flatten[...,0])
    pts_mask = []
    model.neural_points.querier.get_hyperparameters(opt.vsize, point_xyz_w_tensor, ranges=opt.ranges, recache=True)
    for i in tqdm(range(0, num_pts, chunk_size)):
        pts_chunk = xyz_grid_flatten[i:i+chunk_size]
        output, chunk_pts_mask = model.net_ray_marching(queried_xyz=pts_chunk, xyz_forward=True, empty_check=True)
        pts_mask.append(chunk_pts_mask)
        if output is not None:
            sdf_flatten[i:i+chunk_size][chunk_pts_mask] = output['sdf'][...,0].detach()
        torch.cuda.empty_cache()
    pts_mask = torch.cat(pts_mask, -1)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    xyz = xyz_grid.cpu()
    sdf = sdf_flatten.reshape(grid_shape).cpu()
    mask = pts_mask.reshape(grid_shape).cpu()
    if save_vol:
        save_filename = '{}_vol.pth'.format(step)
        save_path = os.path.join(save_dir, save_filename)
        torch.save({'xyz': xyz, 'sdf': sdf, 'mask': mask}, save_path)
    
    save_filename = '{}_mesh{}.obj'.format(step, mesh_suffix)
    save_filename = os.path.join(save_dir, save_filename)
    marching_cube(xyz, sdf, mask, save_filename, opt.marching_cube_thresh, opt.marching_cube_smooth_iter)
    
    

def pts_density_n_normal(model, opt, step='last'):
    '''
    Extract density values (and normals) for each stored neural point.
    '''
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    output = model.density_n_normal(chunk_size)
    model.neural_points.save_pcd(output, epoch=step)
    

def create_render_dataset(test_opt, opt, total_steps, test_num_step=1):
    test_opt.nerf_splits = ["render"]
    test_opt.split = "render"
    test_opt.name = opt.name + "/vid_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    # test_opt.random_sample_size = 30
    test_dataset = create_dataset(test_opt)
    return test_dataset

def main():
    torch.backends.cudnn.benchmark = True

    opt = TrainOptions().parse()
    cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
                              gpu_ids else torch.device('cpu'))
    print("opt.color_loss_items ", opt.color_loss_items)

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED +
              '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +
            fmt.END)
    if not opt.bake_light:
        if 'normal_map' in opt.visual_items: opt.return_normal_map = True
        if 'agg_normal' in opt.visual_items: opt.return_agg_normal = True
        if 'pred_normal' in opt.visual_items: opt.return_pred_normal = True
    visualizer = Visualizer(opt)
    # train_dataset = create_dataset(opt)
    img_lst=None
    with torch.set_grad_enabled(opt.which_agg_model == 'sdfmlp'):
        print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")
        if opt.bgmodel.endswith("plane"):
            _, _, _, _, _, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(train_dataset, visualizer, opt)

        resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resume_iter == "best":
            opt.resume_iter = "latest"
        resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
        if resume_iter is None:
            visualizer.print_details("No previous checkpoints at iter {} !!", resume_iter)
            exit()
        else:
            opt.resume_iter = resume_iter
            visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            visualizer.print_details('test at {} iters'.format(opt.resume_iter))
            visualizer.print_details(f"Iter: {resume_iter}")
            visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        opt.mode = 2
        opt.load_points=1
        opt.resume_dir=resume_dir
        opt.resume_iter = resume_iter
        opt.is_train=False

    model = create_model(opt)
    model.setup(opt) #, train_len=len(train_dataset))

    # TODO add model_fixed
    model_fixed = None
    if opt.fixed_weight_copy:
        print('+'*16, 'create freezed model', '+'*16)
        opt_fixed = copy.deepcopy(opt)
        opt_fixed.is_train = False
        opt_fixed.depth_only = True
        opt_fixed.is_train = 0
        opt_fixed.no_loss = 1
        opt_fixed.use_microfacet_mlp = False
        opt_fixed.use_albedo_mlp = False
        opt_fixed.mlp_fast_forward = True
        if opt.resume_fixed_iter != '':
            opt_fixed.resume_iter = opt.resume_fixed_iter
        model_fixed = create_model(opt_fixed)
        model_fixed.setup(opt_fixed)
        model_fixed.eval()

    update_iter_opt(opt, int(resume_iter), model)

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    # test_opt.random_sample_size = min(48, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.prob = 0
    # test_opt.split = "test"
    test_opt.nerf_splits = [test_opt.split]
    test_opt.test_num_step = opt.test_num_step
    if test_opt.split == 'test':
        test_opt.name = opt.name + "/test_{}".format(resume_iter)
    elif test_opt.split == 'render':
        test_opt.name = opt.name + "/vid_{}".format(resume_iter)

    visualizer.reset()

    fg_masks = None
    test_bg_info = None
    if opt.bgmodel.endswith("plane"):
        test_dataset = create_dataset(test_opt)
        bg_ray_test_lst = create_all_bg(test_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst)
        test_bg_info = [img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_test_lst]
        del test_dataset
        # if opt.vid > 0:
        #     render_dataset = create_render_dataset(test_opt, opt, resume_iter, test_num_step=opt.test_num_step)
    if opt.marching_cube:
        get_mesh(model, opt, step=resume_iter)
        exit()
    
    if opt.den_n_normal:
        pts_density_n_normal(model, opt, step=resume_iter)
        exit()

    if opt.bake_light:
        bake_depth(model, test_opt, resume_iter)
        exit()
    ############ initial test ###############
    # import IPython; IPython.embed()
    with torch.set_grad_enabled(opt.which_agg_model == 'sdfmlp'):
        test_dataset = create_dataset(test_opt)
        model.opt.is_train = 0
        model.opt.no_loss = 1
        
        if test_opt.split in ['render', 'light']:
            render_vid(model, test_dataset, Visualizer(test_opt),
                    test_opt, bg_info=None, steps=resume_iter, model_fixed=model_fixed)
        else:
            test(model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, 
                    test_steps=resume_iter, model_fixed=model_fixed)

if __name__ == '__main__':
    main()
