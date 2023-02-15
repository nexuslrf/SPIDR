import os
import random
import numpy as np
import torch
from torch_scatter import scatter_max
from models import create_model
from models.mvs import mvs_utils, filter_utils
from tqdm import tqdm

def gen_points_filter_embeddings(dataset, visualizer, opt):
    print('-----------------------------------Generate Points-----------------------------------')
    opt.is_train=False
    opt.mode = 1
    model = create_model(opt)
    model.setup(opt)

    model.eval()
    cam_xyz_all = []
    intrinsics_all = []
    extrinsics_all = []
    confidence_all = []
    points_mask_all = []
    intrinsics_full_lst = []
    confidence_filtered_all = []
    near_fars_all = []
    gpu_filter = True
    cpu2gpu= len(dataset.view_id_list) > 300

    imgs_lst, HDWD_lst, c2ws_lst, w2cs_lst, intrinsics_lst = [],[],[],[],[]
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset.view_id_list))):
            data = dataset.get_init_item(i)
            model.set_input(data)
            # intrinsics    1, 3, 3, 3
            points_xyz_lst, photometric_confidence_lst, point_mask_lst, intrinsics_lst, extrinsics_lst, HDWD, c2ws, w2cs, intrinsics, near_fars  = model.gen_points()
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            B, N, C, H, W, _ = points_xyz_lst[0].shape
            # print("points_xyz_lst",points_xyz_lst[0].shape)
            cam_xyz_all.append((points_xyz_lst[0].cpu() if cpu2gpu else points_xyz_lst[0]) if gpu_filter else points_xyz_lst[0].cpu().numpy())
            # intrinsics_lst[0] 1, 3, 3
            intrinsics_all.append(intrinsics_lst[0] if gpu_filter else intrinsics_lst[0])
            extrinsics_all.append(extrinsics_lst[0] if gpu_filter else extrinsics_lst[0].cpu().numpy())
            if opt.manual_depth_view !=0:
                confidence_all.append((photometric_confidence_lst[0].cpu() if cpu2gpu else photometric_confidence_lst[0]) if gpu_filter else photometric_confidence_lst[0].cpu().numpy())
            points_mask_all.append((point_mask_lst[0].cpu() if cpu2gpu else point_mask_lst[0]) if gpu_filter else point_mask_lst[0].cpu().numpy())
            imgs_lst.append(data["images"].cpu())
            HDWD_lst.append(HDWD)
            c2ws_lst.append(c2ws)
            w2cs_lst.append(w2cs)
            intrinsics_full_lst.append(intrinsics)
            near_fars_all.append(near_fars[0,0])
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            # #################### start query embedding ##################
        torch.cuda.empty_cache()
        if opt.manual_depth_view != 0:
            if gpu_filter:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks_gpu(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, opt, vis=True, return_w=True, cpu2gpu=cpu2gpu, near_fars_all=near_fars_all)
            else:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks(cam_xyz_all, [intr.cpu().numpy() for intr in intrinsics_all], extrinsics_all, confidence_all, points_mask_all, opt)
            # print(xyz_ref_lst[0].shape) # 224909, 3
        else:
            cam_xyz_all = [cam_xyz_all[i].reshape(-1,3)[points_mask_all[i].reshape(-1),:] for i in range(len(cam_xyz_all))]
            xyz_world_all = [np.matmul(np.concatenate([cam_xyz_all[i], np.ones_like(cam_xyz_all[i][..., 0:1])], axis=-1), np.transpose(np.linalg.inv(extrinsics_all[i][0,...])))[:, :3] for i in range(len(cam_xyz_all))]
            xyz_world_all, cam_xyz_all, confidence_filtered_all = filter_utils.range_mask_lst_np(xyz_world_all, cam_xyz_all, confidence_filtered_all, opt)
            del cam_xyz_all
        # for i in range(len(xyz_world_all)):
        #     visualizer.save_neural_points(i, torch.as_tensor(xyz_world_all[i], device="cuda", dtype=torch.float32), None, data, save_ref=opt.load_points==0)
        # exit()
        # xyz_world_all = xyz_world_all.cuda()
        # confidence_filtered_all = confidence_filtered_all.cuda()
        points_vid = torch.cat([torch.ones_like(xyz_world_all[i][...,0:1]) * i for i in range(len(xyz_world_all))], dim=0)
        xyz_world_all = torch.cat(xyz_world_all, dim=0) if gpu_filter else torch.as_tensor(
            np.concatenate(xyz_world_all, axis=0), device="cuda", dtype=torch.float32)
        confidence_filtered_all = torch.cat(confidence_filtered_all, dim=0) if gpu_filter else torch.as_tensor(np.concatenate(confidence_filtered_all, axis=0), device="cuda", dtype=torch.float32)
        print("xyz_world_all", xyz_world_all.shape, points_vid.shape, confidence_filtered_all.shape)
        torch.cuda.empty_cache()
        # visualizer.save_neural_points(0, xyz_world_all, None, None, save_ref=False)
        # print("vis 0")

        print("%%%%%%%%%%%%%  getattr(dataset, spacemin, None)", getattr(dataset, "spacemin", None))
        if getattr(dataset, "spacemin", None) is not None:
            mask = (xyz_world_all - dataset.spacemin[None, ...].to(xyz_world_all.device)) >= 0
            mask *= (dataset.spacemax[None, ...].to(xyz_world_all.device) - xyz_world_all) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            first_lst, second_lst = masking(mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
        # visualizer.save_neural_points(50, xyz_world_all, None, None, save_ref=False)
        # print("vis 50")
        if getattr(dataset, "alphas", None) is not None:
            vishull_mask = mvs_utils.alpha_masking(xyz_world_all, dataset.alphas, dataset.intrinsics, dataset.cam2worlds, dataset.world2cams, dataset.near_far if opt.ranges[0] < -90.0 and getattr(dataset,"spacemin",None) is None else None, opt=opt)
            first_lst, second_lst = masking(vishull_mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            print("alpha masking xyz_world_all", xyz_world_all.shape, points_vid.shape)
        # visualizer.save_neural_points(100, xyz_world_all, None, data, save_ref=opt.load_points == 0)
        # print("vis 100")

        if opt.vox_res > 0:
            xyz_world_all, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(xyz_world_all.cuda() if len(xyz_world_all) < 99999999 else xyz_world_all[::(len(xyz_world_all)//99999999+1),...].cuda(), opt.vox_res)
            points_vid = points_vid[sampled_pnt_idx,:]
            confidence_filtered_all = confidence_filtered_all[sampled_pnt_idx]
            print("after voxelize:", xyz_world_all.shape, points_vid.shape)
            xyz_world_all = xyz_world_all.cuda()

        xyz_world_all = [xyz_world_all[points_vid[:,0]==i, :] for i in range(len(HDWD_lst))]
        confidence_filtered_all = [confidence_filtered_all[points_vid[:,0]==i] for i in range(len(HDWD_lst))]
        cam_xyz_all = [(torch.cat([xyz_world_all[i], torch.ones_like(xyz_world_all[i][...,0:1])], dim=-1) @ extrinsics_all[i][0].t())[...,:3] for i in range(len(HDWD_lst))]
        points_embedding_all, points_color_all, points_dir_all, points_conf_all = [], [], [], []
        for i in tqdm(range(len(HDWD_lst))):
            if len(xyz_world_all[i]) > 0:
                embedding, color, dir, conf = \
                    model.query_embedding(HDWD_lst[i], torch.as_tensor(cam_xyz_all[i][None, ...], device="cuda", dtype=torch.float32), 
                                        torch.as_tensor(confidence_filtered_all[i][None, :, None], device="cuda", dtype=torch.float32) \
                                            if len(confidence_filtered_all) > 0 else None, 
                                        imgs_lst[i].cuda(), c2ws_lst[i], w2cs_lst[i], intrinsics_full_lst[i], 0, pointdir_w=True)
                points_embedding_all.append(embedding)
                points_color_all.append(color)
                points_dir_all.append(dir)
                points_conf_all.append(conf)

        xyz_world_all = torch.cat(xyz_world_all, dim=0)
        points_embedding_all = torch.cat(points_embedding_all, dim=1)
        points_color_all = torch.cat(points_color_all, dim=1) if points_color_all[0] is not None else None
        points_dir_all = torch.cat(points_dir_all, dim=1) if points_dir_all[0] is not None else None
        points_conf_all = torch.cat(points_conf_all, dim=1) if points_conf_all[0] is not None else None

        visualizer.save_neural_points(200, xyz_world_all, points_color_all, data, save_ref=opt.load_points == 0)
        print("vis")
        model.cleanup()
        del model
    return xyz_world_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all, \
        [img[0].cpu() for img in imgs_lst], [c2w for c2w in c2ws_lst], [w2c for w2c in w2cs_lst] , intrinsics_all, [list(HDWD) for HDWD in HDWD_lst]


def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [item[:, mask, ...] if item is not None else None for item in seconddim_lst]
    return first_lst, second_lst


def probe_hole(model, dataset, visualizer, opt, bg_info, test_steps=0, opacity_thresh=0.7):
    print('-----------------------------------Probing Holes-----------------------------------')
    add_xyz = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_conf = torch.zeros([0, 1], device="cuda", dtype=torch.float32)
    add_color = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_dir = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_embedding = torch.zeros([0, opt.point_features_dim], device="cuda", dtype=torch.float32)
    add_roughness = torch.zeros([0, 1], device="cuda", dtype=torch.float32)
    add_specular = torch.zeros([0, 1], device="cuda", dtype=torch.float32)
    add_normal = torch.zeros([0, 3], device="cuda", dtype=torch.float32)

    kernel_size = model.opt.kernel_size
    model.net_ray_marching.module.neural_points.build_occ = True
    if opt.prob_kernel_size is not None:
        tier = np.sum(np.asarray(opt.prob_tiers) < test_steps)
        print("cal by tier", tier)
        model.opt.query_size = np.asarray(opt.prob_kernel_size[tier*3:tier*3+3])
        print("prob query size =", model.opt.query_size)
    model.opt.prob = 1
    total_num = len(model.top_ray_miss_ids) -1 if opt.prob_mode == 0 and opt.prob_num_step > 1 else len(dataset)

    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    height = dataset.height
    width = dataset.width
    visualizer.reset()
    prob_maps = {}
    max_num = len(dataset) // opt.prob_num_step
    take_top = False
    if opt.prob_top == 1 and opt.prob_mode <= 0: # and opt.far_thresh <= 0:
        if getattr(model, "top_ray_miss_ids", None) is not None:
            mask = model.top_ray_miss_loss[:-1] > 0.0
            frame_ids = model.top_ray_miss_ids[:-1][mask][:max_num]
            print(len(frame_ids), max_num)
            print("prob frame top_ray_miss_loss:", model.top_ray_miss_loss)
            take_top = True
        else:
            print("model has no top_ray_miss_ids")
    else:
        frame_ids = list(range(len(dataset)))[:max_num]
        random.shuffle(frame_ids)
        frame_ids = frame_ids[:max_num]
    print("{}/{} has holes, id_lst to prune".format(len(frame_ids), total_num), frame_ids, opt.prob_num_step)
    print("take top:", take_top, "; prob frame ids:", frame_ids)
    with tqdm(range(len(frame_ids))) as pbar:
        for j in pbar:
            i = frame_ids[j]
            pbar.set_description("Processing frame id %d" % i)
            data = dataset.get_item(i)
            bg = data['bg_color'][None, :].cuda()
            raydir = data['raydir'].clone()
            pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
            edge_mask = torch.zeros([height, width], dtype=torch.bool, device='cuda')
            edge_mask[pixel_idx[0, ..., 1].to(torch.long), pixel_idx[0, ..., 0].to(torch.long)] = 1
            edge_mask = edge_mask.reshape(-1) > 0
            totalpixel = pixel_idx.shape[1]
            gt_image_full = data['gt_image'].cuda()

            probe_keys = [
                "coarse_raycolor", "ray_mask", "ray_max_sample_loc_w", "ray_max_far_dist", 
                "ray_max_shading_opacity", "shading_avg_color", "shading_avg_dir", "shading_avg_conf", 
                "shading_avg_embedding", "shading_avg_roughness", "shading_avg_specular", 
                "shading_avg_normal"]
            prob_maps = {}
            for k in range(0, totalpixel, chunk_size):
                start = k
                end = min([k + chunk_size, totalpixel])
                data['raydir'] = raydir[:, start:end, :]
                data["pixel_idx"] = pixel_idx[:, start:end, :]
                model.set_input(data)
                output = model.test()
                chunk_pixel_id = data["pixel_idx"].to(torch.long)
                output["ray_mask"] = output["ray_mask"][..., None]
                if opt.low_trans_as_miss > 0: 
                    output["ray_mask"] = (output['coarse_is_background'] < opt.low_trans_as_miss) & output['ray_mask']
                torch.cuda.empty_cache()
                for key in probe_keys:
                    if "ray_max_shading_opacity" not in output and key != 'coarse_raycolor':
                        break
                    if key not in output:
                        continue
                    if output[key] is None:
                        prob_maps[key] = None
                    else:
                        if key not in prob_maps.keys():
                            C = output[key].shape[-1]
                            prob_maps[key] = torch.zeros((height, width, C), device="cuda", dtype=output[key].dtype)
                        prob_maps[key][chunk_pixel_id[0, ..., 1], chunk_pixel_id[0, ..., 0], :] = output[key]

            gt_image = torch.zeros((height * width, 3), dtype=torch.float32, device=prob_maps["ray_mask"].device)
            gt_image[edge_mask, :] = gt_image_full
            gt_image = gt_image.reshape(height, width, 3)
            miss_ray_mask = (prob_maps["ray_mask"] < 1) * (torch.norm(gt_image - bg, dim=-1, keepdim=True) > 0.002)
            miss_ray_inds = (edge_mask.reshape(height, width, 1) * miss_ray_mask).squeeze(-1).nonzero() # N, 2

            neighbor_inds = bloat_inds(miss_ray_inds, 1, (height, width))
            neighboring_miss_mask = torch.zeros_like(gt_image[..., 0])
            neighboring_miss_mask[neighbor_inds[..., 0], neighbor_inds[...,1]] = 1
            if opt.far_thresh > 0:
                far_ray_mask = (prob_maps["ray_mask"] > 0) * (prob_maps["ray_max_far_dist"] > opt.far_thresh) * (torch.norm(gt_image - prob_maps["coarse_raycolor"], dim=-1, keepdim=True) < 0.1)
                neighboring_miss_mask += far_ray_mask.squeeze(-1)
            neighboring_miss_mask = (prob_maps["ray_mask"].squeeze(-1) > 0) * neighboring_miss_mask * (prob_maps["ray_max_shading_opacity"].squeeze(-1) > opacity_thresh) > 0
            neighbor_inds = neighbor_inds.reshape(-1, 9, 2)
            inds_mask = neighboring_miss_mask[neighbor_inds[..., 0], neighbor_inds[...,1]]
            fill_holes = inds_mask.sum(-1)
            if (fill_holes > 0).any():
                neighbor_inds = neighbor_inds[fill_holes > 0]
                fill_holes, inds_mask = fill_holes[fill_holes > 0], inds_mask[fill_holes > 0]
                voxel_size = opt.vsize[0] * opt.vscale[0]
                t_expand = data['campos'].new_tensor([-voxel_size, 0, voxel_size])[None,...]
                neighbor_xyz = prob_maps["ray_max_sample_loc_w"][neighbor_inds[..., 0], neighbor_inds[...,1]]
                neighbor_raydir = raydir.reshape_as(prob_maps["ray_max_sample_loc_w"])[neighbor_inds[..., 0], neighbor_inds[...,1]].to(neighbor_xyz.device)
                neighbor_t = ((neighbor_xyz - data['campos'][0]) / neighbor_raydir)[..., 0]
                avg_t = (neighbor_t * inds_mask).sum(-1, keepdim=True) / fill_holes[...,None] + t_expand[None,:]
                hole_xyz = (neighbor_raydir[:,4:5,:] * avg_t[...,None] + data['campos'][0]).reshape(-1,3)
                add_xyz = torch.cat([add_xyz, hole_xyz], dim=0)
                f_expand = data['campos'].new_zeros(1,3,1)
                hole_avg = lambda ten: ((
                        (ten[neighbor_inds[..., 0], neighbor_inds[...,1]] * inds_mask[...,None]).sum(-2) 
                            / fill_holes[...,None])[:,None,...] + f_expand).reshape(-1,ten.shape[-1])
                add_conf = torch.cat([add_conf, hole_avg(prob_maps["shading_avg_conf"])], dim=0) if prob_maps["shading_avg_conf"] is not None else None
                add_color = torch.cat([add_color, hole_avg(prob_maps["shading_avg_color"])], dim=0) if prob_maps["shading_avg_color"] is not None else None
                add_dir = torch.cat([add_dir, hole_avg(prob_maps["shading_avg_dir"])], dim=0) if prob_maps["shading_avg_dir"] is not None else None
                add_embedding = torch.cat([add_embedding, hole_avg(prob_maps["shading_avg_embedding"])], dim=0) if prob_maps["shading_avg_embedding"] is not None else None
                add_roughness = torch.cat([add_roughness, hole_avg(prob_maps["shading_avg_roughness"])], dim=0) \
                    if "shading_avg_roughness" in prob_maps and prob_maps["shading_avg_roughness"] is not None else None
                add_specular = torch.cat([add_specular, hole_avg(prob_maps["shading_avg_specular"])], dim=0) \
                    if "shading_avg_specular" in prob_maps and prob_maps["shading_avg_specular"] is not None else None
                add_normal = torch.cat([add_normal, hole_avg(prob_maps["shading_avg_normal"])], dim=0) \
                    if "shading_avg_normal" in prob_maps and prob_maps["shading_avg_normal"] is not None else None

            add_xyz = torch.cat([add_xyz, prob_maps["ray_max_sample_loc_w"][neighboring_miss_mask]], dim=0)
            add_conf = torch.cat([add_conf, prob_maps["shading_avg_conf"][neighboring_miss_mask]], dim=0) if prob_maps["shading_avg_conf"] is not None else None
            add_color = torch.cat([add_color, prob_maps["shading_avg_color"][neighboring_miss_mask]], dim=0) if prob_maps["shading_avg_color"] is not None else None
            add_dir = torch.cat([add_dir, prob_maps["shading_avg_dir"][neighboring_miss_mask]], dim=0) if prob_maps["shading_avg_dir"] is not None else None
            add_embedding = torch.cat([add_embedding, prob_maps["shading_avg_embedding"][neighboring_miss_mask]], dim=0)
            add_roughness = torch.cat([add_roughness, prob_maps["shading_avg_roughness"][neighboring_miss_mask]], dim=0) \
                if "shading_avg_roughness" in prob_maps and prob_maps["shading_avg_roughness"] is not None else None
            add_specular = torch.cat([add_specular, prob_maps["shading_avg_specular"][neighboring_miss_mask]], dim=0) \
                if "shading_avg_specular" in prob_maps and prob_maps["shading_avg_specular"] is not None else None
            add_normal = torch.cat([add_normal, prob_maps["shading_avg_normal"][neighboring_miss_mask]], dim=0) \
                if "shading_avg_normal" in prob_maps and prob_maps["shading_avg_normal"] is not None else None
            if len(add_xyz) > -1:
                output = prob_maps["coarse_raycolor"].permute(2,0,1)[None, None,...]
                visualizer.save_ref_views({"images": output}, i, subdir="prob_img_{:04d}".format(test_steps))

    add_conf = add_conf * opt.prob_mul 
    model.opt.kernel_size = kernel_size
    if opt.bgmodel.startswith("planepoints"):
        mask = dataset.filter_plane(add_xyz)
        first_lst, _ = masking(mask, 
                [add_xyz, add_embedding, add_color, add_dir, add_conf, add_roughness, add_specular, add_normal], [])
        add_xyz, add_embedding, add_color, add_dir, add_conf, add_roughness, add_specular, add_normal = first_lst
    if len(add_xyz) > 0:
        visualizer.save_neural_points("prob{:04d}".format(test_steps), add_xyz, None, None, save_ref=False)
        visualizer.print_details("vis added points to probe folder")
    if opt.prob_mode == 0 and opt.prob_num_step > 1:
        model.reset_ray_miss_ranking()
    del visualizer, prob_maps
    model.opt.prob = 0

    return add_xyz, add_embedding, add_color, add_dir, add_conf, add_roughness, add_specular, add_normal

def vox_growing(model, opt, add_neighbor_vox=True, k_neighbors=2, chunk_size=16384):
    neural_points = model.neural_points
    points_xyz = neural_points.xyz
    radius_limit_np, depth_limit_np, ranges_np, vsize_np, vdim_np, scaled_vsize_np, scaled_vdim_np, vscale_np, \
        ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu \
        = neural_points.querier.get_hyperparameters(opt.vsize, points_xyz[None, ...], ranges=opt.ranges)
    neural_points.querier.update_occ_vox(points_xyz[None,...], opt.P)
    vox_vol = neural_points.querier.occ_numpnts_tensor # [B, max_o]
    valid_vox_mask = vox_vol > 0
    valid_vox_vol = vox_vol[valid_vox_mask]
    vox_vol_grid = torch.zeros_like(neural_points.querier.coor_occ_tensor, dtype=valid_vox_vol.dtype)
    valid_vox_coord = neural_points.querier.occ_2_coor_tensor[valid_vox_mask].long() # [N, 3]
    vox_vol_grid[:, valid_vox_coord[...,0], valid_vox_coord[...,1], valid_vox_coord[...,2]] \
        = valid_vox_vol[None,...]
    valid_vox_center_xyz = valid_vox_coord * scaled_vsize_gpu + ranges_gpu[:3] + scaled_vsize_gpu/2
    valid_occ_2_pnts = neural_points.querier.occ_2_pnts_tensor[valid_vox_mask].long() # [N, P]
    
    edge_vox_mask = (valid_vox_vol < max(opt.P * 0.2, 1.1))

    if opt.vox_growing_thresh > 0:
        num_pts = valid_vox_center_xyz.shape[0]
        sdf_flatten = torch.zeros_like(valid_vox_center_xyz[...,0])
        grad_flatten = torch.zeros_like(valid_vox_center_xyz)

        pts_mask = []
        for i in tqdm(range(0, num_pts, chunk_size)):
            pts_chunk = valid_vox_center_xyz[i:i+chunk_size]
            output, chunk_pts_mask = model.net_ray_marching(queried_xyz=pts_chunk, xyz_forward=True, empty_check=True)
            pts_mask.append(chunk_pts_mask)
            if output is not None:
                sdf_flatten[i:i+chunk_size][chunk_pts_mask] = output['sdf'][...,0].detach()
                grad_flatten[i:i+chunk_size][chunk_pts_mask] = output['pred_gradient'].detach()
            torch.cuda.empty_cache()
        pts_mask = torch.cat(pts_mask, -1)
        # pick vox with close-to-zero sdf values based on beta
        beta = (model.aggregator.sdf_beta.detach().abs() + model.aggregator.sdf_beta_min).item()
        sdf_thresh = -beta * np.log(2 * opt.vox_growing_thresh)
    
        # adding points to low vol vox...
        max_vsize = max(max(opt.vsize), sdf_thresh)
        edge_vox_mask = (sdf_flatten < sdf_thresh) * (sdf_flatten > -max_vsize) * edge_vox_mask
    # import IPython; IPython.embed()
    edge_vox_coord = valid_vox_coord[edge_vox_mask] # P, 3

    neighbor_coord = bloat_inds(edge_vox_coord, 1, scaled_vdim_np).reshape(-1, 27, 3)

    vox_nvol = vox_vol_grid[0, neighbor_coord[...,0], neighbor_coord[...,1], neighbor_coord[...,2]].sum(-1)
    vox_nvol_mask = vox_nvol <= opt.P * 2 # ~ 0.3 * 9 * opt.P
    vox_mask = torch.zeros_like(edge_vox_mask).masked_scatter_(edge_vox_mask, vox_nvol_mask)
    masked_vox_coord = edge_vox_coord[vox_nvol_mask]
    # 9 spheres in a unit cube --> r = sqrt(3) - 3/2
    r = 3**0.5 - 1.5
    pos_perm = points_xyz.new_tensor([r, 1-r])
    candidate_pnt_pos = torch.stack(torch.meshgrid([pos_perm,pos_perm,pos_perm]), dim=-1).reshape(-1, 3) # 8, 3
    candidate_pnt_pos = torch.cat([points_xyz.new_tensor([[0.5, 0.5, 0.5]]), candidate_pnt_pos], dim=0) *\
                            scaled_vsize_gpu + ranges_gpu[:3] # 9, 3
    inc_vox_pnt_inds = valid_occ_2_pnts[vox_mask]
    inc_vox_pnt_mask = (inc_vox_pnt_inds >= 0)
    valid_inc_vox_pnt_inds = inc_vox_pnt_inds[inc_vox_pnt_mask] # S Note S >= P
    valid_pnt_2_vox_inds = inc_vox_pnt_mask.nonzero()[..., 0] # S
    inc_vox_pnt = neural_points.xyz[valid_inc_vox_pnt_inds] # S, 3
    inc_vox_coord_per_pnt = masked_vox_coord[...,None,:].expand(*inc_vox_pnt_inds.shape, 3)[inc_vox_pnt_mask] # S, 3
    candidate_pnt = inc_vox_coord_per_pnt[...,None,:] * scaled_vsize_gpu + candidate_pnt_pos[None,...] # S, 9, 3
    min_dist, pos_ind_per_pnt = ((candidate_pnt - inc_vox_pnt[...,None,:])**2).sum(-1).min(-1)  # S, 9, 3 -sum-> S,9 -min-> S
    max_min_dist = points_xyz.new_zeros(masked_vox_coord.shape[0]) # P
    _, argmax_min_dist = scatter_max(min_dist, valid_pnt_2_vox_inds, dim=-1, out=max_min_dist) # P
    add_vox_pnt_xyz = candidate_pnt[argmax_min_dist, pos_ind_per_pnt[argmax_min_dist]] # [P, 3]
    vox_vol_grid[0, inc_vox_coord_per_pnt[...,0], inc_vox_coord_per_pnt[...,1], inc_vox_coord_per_pnt[...,2]] += 1
    print(f"Add_vox_pnt: {add_vox_pnt_xyz.shape[0]}")
    # find neighbors
    if add_neighbor_vox and opt.vox_growing_thresh > 0:
        # edge_vox_mask = (sdf_flatten.abs() < sdf_thresh) * (valid_vox_vol < max(opt.P * 0.2, 1.1)) # P
        # edge_vox_coord = valid_vox_coord[edge_vox_mask]
        ### dir filtering
        # edge_vox_normal = grad_flatten[edge_vox_mask]
        # dir = F.normalize((neighbor_coord - edge_vox_coord[...,None,:]) * scaled_vsize_gpu, dim=-1)
        # cos = (dir * edge_vox_normal[...,None,:]).sum(-1)
        # dir_mask = cos.abs() < 0.25 # ~5/12*pi
        # filtered_neighbor_coord = neighbor_coord[dir_mask]
        ###
        edge_vox_mask = (valid_vox_vol < max(opt.P * 0.3, 1.1))
        if opt.vox_growing_thresh > 0:
            edge_vox_mask = (sdf_flatten < sdf_thresh) * (sdf_flatten > -max_vsize) * edge_vox_mask
        edge_vox_coord = valid_vox_coord[edge_vox_mask] # P, 3
        if edge_vox_coord.shape[0] > 0:
            neighbor_coord = bloat_inds(edge_vox_coord, 1, scaled_vdim_np).reshape(-1, 27, 3)

            # unique coord
            uni_neighbor_coord, uni_2_neighbor = neighbor_coord.reshape(-1,3).unique(dim=0, return_inverse=True) # N, 3; P * 27 
            uni_neighbor_vol = vox_vol_grid[0, uni_neighbor_coord[...,0], uni_neighbor_coord[...,1], uni_neighbor_coord[...,2]] # N
            unocc_neighbor_mask = uni_neighbor_vol <= 0
            unocc_coord = uni_neighbor_coord[unocc_neighbor_mask] # M, 3
            unocc_coord_neighbor = bloat_inds(unocc_coord, 1, scaled_vdim_np).reshape(-1, 27, 3) # M, 27, 3
            unocc_coord_vol = vox_vol_grid[0, unocc_coord_neighbor[...,0], unocc_coord_neighbor[...,1], unocc_coord_neighbor[...,2]]\
                                .sum(-1, dtype=uni_neighbor_vol.dtype) # M
            # M -> N -> P*27
            uni_neighbor_vol += opt.P * 26 # N
            uni_neighbor_vol[unocc_neighbor_mask] = unocc_coord_vol # M -> N
            neighbor_coord_vol = uni_neighbor_vol[uni_2_neighbor].reshape(-1, 27) # N -> P*27 -> P, 27
            neighbor_vol_kmin, vol_idx = neighbor_coord_vol.reshape(-1, 27).topk(k_neighbors, dim=-1, largest=False) # P, K
            neighbor_kmin_mask = neighbor_vol_kmin < opt.P * 1 # ~ 0.3 * 9 * opt.P
            sel_neighbor_coord = neighbor_coord[torch.arange(vol_idx.shape[0])[:,None].expand_as(vol_idx), vol_idx, :] # P, K, 3
            sel_neighbor_coord = sel_neighbor_coord[neighbor_kmin_mask].unique(dim=0)

            add_neighbor_pnt_xyz = sel_neighbor_coord * scaled_vsize_gpu + ranges_gpu[:3] + scaled_vsize_gpu/2
            add_vox_pnt_xyz = torch.cat([add_vox_pnt_xyz, add_neighbor_pnt_xyz], dim=0)
            print(f"Add_neighbor_pnt: {add_neighbor_pnt_xyz.shape[0]}")
    return add_vox_pnt_xyz

def bloat_inds(inds: torch.Tensor, shift, ranges):
    '''
    inds: N,D
    '''
    inds = inds[:,None,:]
    n_dim = len(ranges)
    s_mesh = [torch.arange(-shift, shift+1, dtype=torch.long) for i in range(n_dim)]
    shift_inds = torch.stack(torch.meshgrid(*s_mesh),dim=-1).reshape(1, -1, n_dim).cuda()
    inds = (inds + shift_inds).reshape(-1, n_dim)
    for i in range(n_dim):
        inds[...,i].clamp_(min=0, max=ranges[i]-1)
    return inds

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")] # this is total_steps
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def create_all_bg(dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, dummy=False):
    total_num = dataset.total
    height = dataset.height
    width = dataset.width
    bg_ray_lst = []
    random_sample = dataset.opt.random_sample
    for i in range(0, total_num):
        dataset.opt.random_sample = "no_crop"
        if dummy:
            data = dataset.get_dummyrot_item(i)
        else:
            data = dataset.get_item(i)
        raydir = data['raydir'].clone()
        # print("data['pixel_idx']",data['pixel_idx'].shape) # 1, 512, 640, 2
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        start=0
        end = height * width

        data['raydir'] = raydir[:, start:end, :]
        data["pixel_idx"] = pixel_idx[:, start:end, :]
        model.set_input(data)

        xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
        bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"])
        bg_ray = bg_ray.reshape(bg_ray.shape[0], height, width, 3) # 1, 512, 640, 3
        bg_ray_lst.append(bg_ray)
    dataset.opt.random_sample = random_sample
    return bg_ray_lst


def save_points_conf(visualizer, xyz, points_color, points_conf, total_steps):
    print("total:", xyz.shape, points_color.shape, points_conf.shape)
    colors, confs = points_color[0], points_conf[0,...,0]
    pre = -1000
    for i in range(12):
        thresh = (i * 0.1) if i <= 10 else 1000
        mask = ((confs <= thresh) * (confs > pre)) > 0
        thresh_xyz = xyz[mask, :]
        thresh_color = colors[mask, :]
        visualizer.save_neural_points(f"{total_steps}-{thresh}", thresh_xyz, thresh_color[None, ...], None, save_ref=False)
        pre = thresh
    exit()

def nearest_view(campos, raydir, xyz, id_list):
    cam_ind = torch.zeros([0,1], device=campos.device, dtype=torch.long)
    step=10000
    for i in range(0, len(xyz), step):
        dists = xyz[i:min(len(xyz),i+step), None, :] - campos[None, ...] # N, M, 3
        dists_norm = torch.norm(dists, dim=-1) # N, M
        dists_dir = dists / (dists_norm[...,None]+1e-6) # N, M, 3
        dists = dists_norm / 200 + (1.1 - torch.sum(dists_dir * raydir[None, :],dim=-1)) # N, M
        cam_ind = torch.cat([cam_ind, torch.argmin(dists, dim=1).view(-1,1)], dim=0) # N, 1
    return cam_ind


def update_iter_opt(opt, total_steps, model):
    while len(opt.sample_inc_iters) > 0 and total_steps > opt.sample_inc_iters[0]:
        if len(opt.sample_stepsize_list) > 0:
            model.opt.sample_stepsize = opt.sample_stepsize_list[0]
            opt.sample_stepsize_list = opt.sample_stepsize_list[1:]
        if len(opt.SR_list) > 0:
            model.opt.SR = opt.SR_list[0]
            opt.SR_list = opt.SR_list[1:]
        opt.sample_inc_iters = opt.sample_inc_iters[1:]
    
    if total_steps < opt.diffuse_only_iters:
        opt.diffuse_branch_only = True
        model.opt.diffuse_branch_only = True
    else:
        opt.diffuse_branch_only = False
        model.opt.diffuse_branch_only = False

    