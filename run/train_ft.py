import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import glob
import copy
import torch
import torch.nn.functional as F
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from models.mvs.mvs_points_model import MvsPointsModel
from models.mvs import mvs_utils, filter_utils
from pprint import pprint
from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
# from render_vid import render_vid
torch.manual_seed(0)
np.random.seed(0)
import random
import cv2
from PIL import Image
from tqdm import tqdm
# from models.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
# from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from test_ft import test, render_vid
from models.helpers.networks import init_seq
from ft_helper import gen_points_filter_embeddings, probe_hole, vox_growing, get_latest_epoch, \
        create_all_bg, nearest_view, update_iter_opt

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)

def create_render_dataset(test_opt, opt, total_steps, test_num_step=1):
    test_opt.nerf_splits = ["render"]
    test_opt.split = "render"
    test_opt.name = opt.name + "/vid_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_opt.random_sample_size = 30
    test_dataset = create_dataset(test_opt)
    return test_dataset

def create_test_dataset(test_opt, opt, total_steps, prob=None, test_num_step=1):
    test_opt.prob = prob if prob is not None else test_opt.prob
    test_opt.nerf_splits = ["test"]
    test_opt.split = "test"
    test_opt.name = opt.name + "/test_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_dataset = create_dataset(test_opt)
    return test_dataset

def create_comb_dataset(test_opt, opt, total_steps, prob=None, test_num_step=1):
    test_opt.prob = prob if prob is not None else test_opt.prob
    test_opt.nerf_splits = ["comb"]
    test_opt.split = "comb"
    test_opt.name = opt.name + "/comb_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_dataset = create_dataset(test_opt)
    return test_dataset

def init_brdf_config(model, opt, training=True):
    # only rgb loss + light loss
    # no agg_normal in vis
    # no pruning no probing
    model.opt.brdf_training = training
    opt.brdf_training = training
    opt.prune_iter = 0
    opt.prob_freq = 0


def main():
    torch.backends.cudnn.benchmark = True

    opt = TrainOptions().parse()
    cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
                              gpu_ids else torch.device('cpu'))
    print("opt.color_loss_items ", opt.color_loss_items)
    if opt.run_mode == 'lighting': opt.brdf_rendering = True
    if opt.prob_freq > 0: assert opt.vox_growing_iter % opt.prob_freq == 0
    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED +
              '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +
            fmt.END)
    visualizer = Visualizer(opt)
    train_dataset = create_dataset(opt)
    normRw2c = train_dataset.norm_w2c[:3,:3] # torch.eye(3, device="cuda") #
    img_lst=None
    best_PSNR=0.0
    best_iter=0
    points_xyz_all=None
    with torch.no_grad():
        print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")
        if len([n for n in glob.glob(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth") if os.path.isfile(n)]) > 0:
            if opt.bgmodel.endswith("plane"):
                _, _, _, _, _, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(train_dataset, visualizer, opt)

            resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
            if opt.resume_iter == "best":
                opt.resume_iter = "latest"
            resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
            if resume_iter is None:
                epoch_count = 1
                total_steps = 0
                visualizer.print_details("No previous checkpoints, start from scratch!!!!")
            else:
                opt.resume_iter = resume_iter
                states = torch.load(
                    os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)), map_location=cur_device)
                epoch_count = states['epoch_count']
                total_steps = states['total_steps']
                best_PSNR = states['best_PSNR'] if 'best_PSNR' in states else best_PSNR
                best_iter = states['best_iter'] if 'best_iter' in states else best_iter
                best_PSNR = best_PSNR.item() if torch.is_tensor(best_PSNR) else best_PSNR
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                visualizer.print_details('Continue training from {} epoch'.format(opt.resume_iter))
                visualizer.print_details(f"Iter: {total_steps}")
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                del states

            if int(resume_iter) == 0 and opt.which_agg_model == 'sdfmlp':
                opt.resume_points_only = True 
            opt.mode = 2
            opt.load_points=1
            opt.resume_dir=resume_dir
            opt.resume_iter = resume_iter
            opt.is_train=True
            model = create_model(opt)
        elif opt.load_points < 1:
            points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all, \
                img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(train_dataset, visualizer, opt)
            opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            opt.is_train=True
            opt.mode = 2
            model = create_model(opt)
        else: # not used
            load_points = opt.load_points
            opt.is_train = False
            opt.mode = 1
            opt.load_points = 0
            model = create_model(opt)
            model.setup(opt)
            model.eval()
            if load_points in [1,3]:
                points_xyz_all = train_dataset.load_init_points()
            if load_points == 2:
                points_xyz_all = train_dataset.load_init_depth_points(device="cuda", vox_res=100)
            if load_points == 3:
                depth_xyz_all = train_dataset.load_init_depth_points(device="cuda", vox_res=80)
                print("points_xyz_all",points_xyz_all.shape)
                print("depth_xyz_all", depth_xyz_all.shape)
                filter_res = 100
                pc_grid_id, _, pc_space_min, pc_space_max = mvs_utils.construct_vox_points_ind(points_xyz_all, filter_res)
                d_grid_id, depth_inds, _, _ = mvs_utils.construct_vox_points_ind(depth_xyz_all, filter_res, space_min=pc_space_min, space_max=pc_space_max)
                all_grid= torch.cat([pc_grid_id, d_grid_id], dim=0)
                min_id = torch.min(all_grid, dim=-2)[0]
                max_id = torch.max(all_grid, dim=-2)[0] - min_id
                max_id_lst = (max_id+1).cpu().numpy().tolist()
                mask = torch.ones(max_id_lst, device=d_grid_id.device)
                pc_maskgrid_id = (pc_grid_id - min_id[None,...]).to(torch.long)
                mask[pc_maskgrid_id[...,0], pc_maskgrid_id[...,1], pc_maskgrid_id[...,2]] = 0
                depth_maskinds = (d_grid_id[depth_inds,:] - min_id).to(torch.long)
                depth_maskinds = mask[depth_maskinds[...,0], depth_maskinds[...,1], depth_maskinds[...,2]]
                depth_xyz_all = depth_xyz_all[depth_maskinds > 0]
                visualizer.save_neural_points("dep_filtered", depth_xyz_all, None, None, save_ref=False)
                print("vis depth; after pc mask depth_xyz_all",depth_xyz_all.shape)
                points_xyz_all = [points_xyz_all, depth_xyz_all] if opt.vox_res > 0 else torch.cat([points_xyz_all, depth_xyz_all],dim=0)
                del depth_xyz_all, depth_maskinds, mask, pc_maskgrid_id, max_id_lst, max_id, min_id, all_grid

            if opt.ranges[0] > -99.0:
                ranges = torch.as_tensor(opt.ranges, device=points_xyz_all.device, dtype=torch.float32)
                mask = torch.prod(
                    torch.logical_and(points_xyz_all[..., :3] >= ranges[None, :3], points_xyz_all[..., :3] <= ranges[None, 3:]),
                    dim=-1) > 0
                points_xyz_all = points_xyz_all[mask]


            if opt.vox_res > 0:
                points_xyz_all = [points_xyz_all] if not isinstance(points_xyz_all, list) else points_xyz_all
                points_xyz_holder = torch.zeros([0,3], dtype=points_xyz_all[0].dtype, device="cuda")
                for i in range(len(points_xyz_all)):
                    points_xyz = points_xyz_all[i]
                    vox_res = opt.vox_res // (1.5**i)
                    print("load points_xyz", points_xyz.shape)
                    _, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(
                                points_xyz.cuda() if len(points_xyz) < 80000000 else points_xyz[::(len(points_xyz) // 80000000 + 1), ...].cuda(), vox_res)
                    points_xyz = points_xyz[sampled_pnt_idx, :]
                    print("after voxelize:", points_xyz.shape)
                    points_xyz_holder = torch.cat([points_xyz_holder, points_xyz], dim=0)
                points_xyz_all = points_xyz_holder



            if opt.resample_pnts > 0:
                if opt.resample_pnts == 1:
                    print("points_xyz_all",points_xyz_all.shape)
                    inds = torch.min(torch.norm(points_xyz_all, dim=-1, keepdim=True), dim=0)[1] # use the point closest to the origin
                else:
                    inds = torch.randperm(len(points_xyz_all))[:opt.resample_pnts, ...]
                points_xyz_all = points_xyz_all[inds, ...]

            campos, camdir = train_dataset.get_campos_ray()
            cam_ind = nearest_view(campos, camdir, points_xyz_all, train_dataset.id_list)
            unique_cam_ind = torch.unique(cam_ind)
            print("unique_cam_ind", unique_cam_ind.shape)
            points_xyz_all = [points_xyz_all[cam_ind[:,0]==unique_cam_ind[i], :] for i in range(len(unique_cam_ind))]

            featuredim = opt.point_features_dim
            points_embedding_all = torch.zeros([1, 0, featuredim], device=unique_cam_ind.device, dtype=torch.float32)
            points_color_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
            points_dir_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
            points_conf_all = torch.zeros([1, 0, 1], device=unique_cam_ind.device, dtype=torch.float32)
            print("extract points embeding & colors", )
            for i in tqdm(range(len(unique_cam_ind))):
                id = unique_cam_ind[i]
                batch = train_dataset.get_item(id, full_img=True)
                HDWD = [train_dataset.height, train_dataset.width]
                c2w = batch["c2w"][0].cuda()
                w2c = torch.inverse(c2w)
                intrinsic = batch["intrinsic"].cuda()
                # cam_xyz_all 252, 4
                cam_xyz_all = (torch.cat([points_xyz_all[i], torch.ones_like(points_xyz_all[i][...,-1:])], dim=-1) @ w2c.transpose(0,1))[..., :3]
                embedding, color, dir, conf = model.query_embedding(
                                                HDWD, cam_xyz_all[None,...], None, batch['images'].cuda(), 
                                                c2w[None, None,...], w2c[None, None,...], intrinsic[:, None,...], 0, pointdir_w=True)
                conf = conf * opt.default_conf if opt.default_conf > 0 and opt.default_conf < 1.0 else conf
                points_embedding_all = torch.cat([points_embedding_all, embedding], dim=1)
                points_color_all = torch.cat([points_color_all, color], dim=1)
                points_dir_all = torch.cat([points_dir_all, dir], dim=1)
                points_conf_all = torch.cat([points_conf_all, conf], dim=1)
                # visualizer.save_neural_points(id, cam_xyz_all, color, batch, save_ref=True)
            points_xyz_all=torch.cat(points_xyz_all, dim=0)
            visualizer.save_neural_points("init", points_xyz_all, points_color_all, None, save_ref=load_points == 0)
            print("vis")
            # visualizer.save_neural_points("cam", campos, None, None, None)
            # print("vis")
            # exit()

            opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            opt.is_train = True
            opt.mode = 2
            model = create_model(opt)

        if points_xyz_all is not None: # no loaded point cloud
            if opt.bgmodel.startswith("planepoints"): # not used
                gen_pnts, gen_embedding, gen_dir, gen_color, gen_conf = train_dataset.get_plane_param_points()
                visualizer.save_neural_points("pl", gen_pnts, gen_color, None, save_ref=False)
                print("vis pl")
                points_xyz_all = torch.cat([points_xyz_all, gen_pnts], dim=0)
                points_embedding_all = torch.cat([points_embedding_all, gen_embedding], dim=1)
                points_color_all = torch.cat([points_color_all, gen_dir], dim=1)
                points_dir_all = torch.cat([points_dir_all, gen_color], dim=1)
                points_conf_all = torch.cat([points_conf_all, gen_conf], dim=1)

            model.set_points(points_xyz_all.cuda(), points_embedding_all.cuda(), points_color=points_color_all.cuda(),
                             points_dir=points_dir_all.cuda(), points_conf=points_conf_all.cuda(),
                             Rw2c=normRw2c.cuda() if opt.load_points < 1 and opt.normview != 3 else None)
            epoch_count = 1
            total_steps = 0

            if opt.which_agg_model == 'sdfmlp':
                model.save_networks(total_steps)
                opt.mode = 2
                opt.load_points=1
                opt.resume_dir= model.save_dir
                opt.resume_iter = f'{total_steps}'
                opt.is_train=True
                opt.resume_points_only = True
                model = create_model(opt)

            del points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all
    
    brdf_start_step = 0
    if opt.brdf_rendering:
        brdf_start_step = opt.maximum_step
        opt.maximum_step += opt.brdf_step
    opt.sched_steps = total_steps - brdf_start_step
    model.setup(opt, train_len=len(train_dataset))

    if opt.brdf_rendering:
        if opt.brdf_mlp and opt.brdf_mlp_reinit:
            print('Reinitialize Color MLP!!!')
            if opt.diffuse_branch_channel > 0:
                init_seq(model.net_ray_marching.module.aggregator.diffuse_branch)
            init_seq(model.net_ray_marching.module.aggregator.color_branch)
        if opt.use_albedo_mlp and total_steps == brdf_start_step:
            print("Copy diffse mlp weights to albedo mlp!")
            diffuse_weights = model.net_ray_marching.module.aggregator.diffuse_branch.state_dict()
            model.net_ray_marching.module.aggregator.albedo_branch.load_state_dict(diffuse_weights)

    model_fixed = None
    if opt.fixed_weight_copy: # will not be used
        print('+'*16, 'create freezed model', '+'*16)
        opt_fixed = copy.deepcopy(opt)
        opt_fixed.is_train = False
        opt_fixed.depth_only = True 
        if any([('brdf_d' in k) or ('brdf_s' in k) for k in opt.color_loss_items]):
            opt_fixed.depth_only = False 
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

    model.train()
    data_loader = create_data_loader(opt, dataset=train_dataset)
    dataset_size = len(data_loader)
    visualizer.print_details('# training images = {}'.format(dataset_size))

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(48, opt.random_sample_size) \
        if not opt.fixed_weight_copy else 32
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.prob = 0
    test_opt.split = "test"

    with open('/tmp/.neural-volumetric.name', 'w') as f:
        f.write(opt.name + '\n')

    visualizer.reset()
    fg_masks = None
    bg_ray_train_lst, bg_ray_test_lst = [], []
    if opt.bgmodel.endswith("plane"):
        test_dataset = create_dataset(test_opt)
        bg_ray_train_lst = create_all_bg(train_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst)
        bg_ray_test_lst = create_all_bg(test_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst)
        test_bg_info = [img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_test_lst]
        del test_dataset
        if opt.vid > 0:
            render_dataset = create_render_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
            bg_ray_render_lst  = create_all_bg(render_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, dummy=True)
            render_bg_info =  [img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_render_lst]
    else:
        test_bg_info, render_bg_info = None, None
        img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = None, None, None, None, None

    ############ initial test ###############
    if total_steps == 0 and opt.maximum_step <= 0:
        with torch.no_grad():
            test_opt.nerf_splits = ["test"]
            test_opt.split = "test"
            test_opt.name = opt.name + "/test_{}".format(total_steps)
            test_opt.test_num_step = opt.test_num_step
            test_dataset = create_dataset(test_opt)
            model.opt.is_train = 0
            model.opt.no_loss = 1
            model.eval()
            test(model, test_dataset, Visualizer(test_opt, tb_writer=visualizer.tb_writer),
                         test_opt, test_bg_info, test_steps=total_steps, model_fixed=model_fixed)
            model.opt.no_loss = 0
            model.opt.is_train = 1
            model.train()
            exit()

    # if total_steps == 0 and (len(train_dataset.id_list) > 30 or len(train_dataset.view_id_list)  > 30):
    #     other_states = {
    #         'epoch_count': 0,
    #         'total_steps': total_steps,
    #     }
    #     model.save_networks(total_steps, other_states)
    #     visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, 0, total_steps))

    test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)

    if opt.brdf_rendering:
        init_brdf_config(model, opt, training=True)

    real_start=total_steps
    train_random_sample_size = opt.random_sample_size

    update_iter_opt(opt, total_steps, model)

    for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(data_loader):
            if opt.maximum_step is not None and total_steps >= opt.maximum_step:
                break
            if opt.prune_iter > 0 and real_start != total_steps and total_steps % opt.prune_iter == 0 and \
                    total_steps < (opt.maximum_step - 1) and total_steps > 0 \
                    and total_steps <= opt.prune_max_iter and total_steps >= opt.prune_min_iter:
                with torch.set_grad_enabled(opt.which_agg_model == 'sdfmlp'):
                    model.clean_optimizer()
                    model.clean_scheduler()
                    model.prune_points(opt.prune_thresh, total_steps)
                    model.setup_optimizer(opt)
                    model.init_scheduler(total_steps, opt) # TODO
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            if opt.prob_freq > 0 and real_start != total_steps and total_steps % opt.prob_freq == 0 \
                and total_steps < (opt.maximum_step - 1) and total_steps > 0 and total_steps < opt.prob_max_iter:                
                grow_point_exit = False
                if opt.prob_kernel_size is not None:
                    tier = np.sum(np.asarray(opt.prob_tiers) < total_steps)
                if (model.top_ray_miss_loss[0] > 1e-6 or opt.prob_mode != 0 or opt.far_thresh > 0) and (opt.prob_kernel_size is None or tier < (len(opt.prob_kernel_size) // 3)):
                    torch.cuda.empty_cache()
                    model.opt.is_train = 0
                    model.opt.no_loss = 1
                    with torch.set_grad_enabled(opt.which_agg_model == 'sdfmlp'):
                        prob_opt = copy.deepcopy(test_opt)
                        prob_opt.name = opt.name
                        # if opt.prob_type=0:
                        ori_train_sample_mode = train_dataset.opt.random_sample
                        train_dataset.opt.random_sample = "no_crop"
                        if opt.prob_mode <= 0:
                            train_dataset.opt.random_sample_size = min(32, train_random_sample_size)
                            prob_dataset = train_dataset
                        elif opt.prob_mode == 1:
                            prob_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=1)
                        else:
                            prob_dataset = create_comb_dataset(test_opt, opt, total_steps, test_num_step=1)
                        model.eval()
                        add_xyz, add_embedding, add_color, add_dir, add_conf, add_roughness, add_specular, add_normal = \
                            probe_hole(model, prob_dataset, Visualizer(prob_opt), prob_opt, None, test_steps=total_steps, opacity_thresh=opt.prob_thresh)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        if opt.prob_mode != 0:
                            del prob_dataset
                        # else:
                        if len(add_xyz) > 0:
                            print("len(add_xyz)", len(add_xyz))
                            model.grow_points(add_xyz, add_embedding, add_color, add_dir, add_conf, 
                                add_roughness, add_specular, add_normal)
                            length_added = len(add_xyz)
                            del add_xyz, add_embedding, add_color, add_dir, add_conf, add_roughness, add_specular
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            visualizer.print_details(
                                "$$$$$$$$$$$$$$$$$$$$$$$$$$           "+\
                                "add grow new points num: {}, all num: {}           $$$$$$$$$$$$$$$$".format(length_added, len(model.neural_points.xyz)))
                            # model.reset_optimizer(opt)
                            # model.reset_scheduler(total_steps, opt)
                            # model.cleanup()
                            # pprint(vars(model))
                            # del model
                            # visualizer.reset()
                            # gc.collect()
                            # torch.cuda.synchronize()
                            # torch.cuda.empty_cache()
                            # input("Press Enter to continue...")
                            # opt.is_train = 1
                            # opt.no_loss = 0
                            # model = create_model(opt)
                            #
                            # model.setup(opt, train_len=len(train_dataset))
                            # model.train()
                            #
                            # if total_steps > 0:
                            #     for scheduler in model.schedulers:
                            #         for i in range(total_steps):
                            #             scheduler.step()

                            grow_point_exit = True
                        
                    train_dataset.opt.random_sample = ori_train_sample_mode
                    train_dataset.opt.random_sample_size = train_random_sample_size
                else:
                    visualizer.print_details(
                        'nothing to probe, max ray miss is only {}'.format(model.top_ray_miss_loss[0]))
                if opt.vox_growing_iter > 0 and total_steps % opt.vox_growing_iter == 0:
                    add_point_xyz = vox_growing(model, opt)
                    if add_point_xyz.shape[0] > 0:
                        add_point_attr = model.neural_points.interp_pnt_attr(add_point_xyz) # a long tuple
                        model.grow_points(*add_point_attr)
                        visualizer.print_details(
                                "$$$$$$$$$$$$$$$$ VoxGrowing           "+\
                                "add grow new points num: {}, all num: {}   $$$$$$$$$$$$$$$$".format(add_point_xyz.shape[0], len(model.neural_points.xyz)))
                        torch.cuda.empty_cache()
                        grow_point_exit = True
                        # (add_xyz, add_embedding, add_color, add_dir, add_conf, add_roughness, add_specular, add_normal)

                if grow_point_exit:
                    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
                    other_states = {
                        "best_PSNR": best_PSNR,
                        "best_iter": best_iter,
                        'epoch_count': epoch,
                        'total_steps': total_steps,
                    }
                    print("states",other_states)

                    # model.save_networks(total_steps, other_states, back_gpu=False)
                    # exit()
                    # TODO: prevent exit. may need to reset optimizer and scheduler
                    model.clean_optimizer()
                    model.clean_scheduler()
                    model.setup_optimizer(opt)
                    model.init_scheduler(total_steps, opt)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                model.opt.no_loss = 0
                model.opt.is_train = 1
                model.train()

                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            update_iter_opt(opt, total_steps, model)

            total_steps += 1
            if opt.which_agg_model == 'sdfmlp': 
                model.opt.cos_anneal_ratio = min(1.0, total_steps/opt.cos_anneal_iters)
                model.opt.normal_anneal_ratio = min(1.0, total_steps/opt.normal_anneal_iters)
            if opt.brdf_rendering and opt.roughness_anneal_iters > 0:
                model.opt.roughness_anneal_ratio = min(1.0, (total_steps-brdf_start_step) /opt.roughness_anneal_iters)
            model.set_input(data)
            if opt.bgmodel.endswith("plane"):
                if len(bg_ray_train_lst) > 0:
                    bg_ray_all = bg_ray_train_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(model.input)
                    bg_ray, fg_masks = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks=fg_masks)
                data["bg_ray"] = bg_ray
            model.optimize_parameters(total_steps=total_steps, backward=True, model_fixed=model_fixed) # TODO
            if model.output['ray_mask'].any():
                losses = model.get_current_losses()
                visualizer.accumulate_losses(losses)

            if opt.lr_policy.startswith("iter"):
                model.update_learning_rate(opt=opt, total_steps=total_steps) # steps here only used for print!

            if total_steps and total_steps % opt.print_freq == 0:
                if opt.show_tensorboard:
                    visualizer.plot_current_losses_with_tb(total_steps, losses)
                if opt.which_agg_model== 'sdfmlp':
                    print(f'beta: {model.aggregator.sdf_act_val}')
                if opt.brdf_rendering:
                    print(f'light probe: max: {model.net_ray_marching.module.brdf_renderer.light_max}'+ \
                        f' min: {model.net_ray_marching.module.brdf_renderer.light_min}')
                visualizer.print_losses(total_steps)
                visualizer.reset()

            if total_steps and total_steps % opt.img_freq == 0:
                test_opt.name = opt.name + "/test_{}".format(total_steps)
                rnm, ran, rpn = False, False, False
                if 'normal_map' in opt.visual_items: rnm = opt.return_normal_map; opt.return_normal_map = True
                if 'agg_normal' in opt.visual_items: ran = opt.return_agg_normal; opt.return_agg_normal = True
                if 'pred_normal' in opt.visual_items: rpn = opt.return_pred_normal; opt.return_pred_normal = True

                torch.cuda.empty_cache()
                model.opt.is_train, model.opt.no_loss = 0, 1
                if opt.brdf_rendering: init_brdf_config(model, opt, training=False)
                with torch.no_grad():
                    test_psnr = test(model, test_dataset, Visualizer(test_opt, tb_writer=visualizer.tb_writer),
                            test_opt, test_bg_info, test_steps=total_steps, lpips=False, train_log=True, model_fixed=model_fixed)
                model.opt.is_train, model.opt.no_loss = 1, 0
                if opt.brdf_rendering: init_brdf_config(model, opt, training=True)
                opt.return_normal_map, opt.return_agg_normal, opt.return_pred_normal\
                     = rnm, ran, rpn

            if hasattr(opt, "save_point_freq") and total_steps and \
                    total_steps % opt.save_point_freq == 0 and \
                    (opt.prune_iter > 0 and total_steps <= opt.prune_max_iter or opt.save_point_freq==1):
                visualizer.save_neural_points(total_steps, model.neural_points.xyz, model.neural_points.points_embeding, data, save_ref=opt.load_points==0)
                visualizer.print_details('saving neural points at total_steps {})'.format(total_steps))

            try:
                if total_steps == 10000 or (total_steps % opt.save_iter_freq == 0 and total_steps > 0):
                    other_states = {
                        "best_PSNR": best_PSNR,
                        "best_iter": best_iter,
                        'epoch_count': epoch,
                        'total_steps': total_steps,
                    }
                    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
                    model.save_networks(total_steps, other_states)
            except Exception as e:
                visualizer.print_details(e)


            if opt.vid > 0 and total_steps % opt.vid == 0 and total_steps > 0:
                torch.cuda.empty_cache()
                test_dataset = create_render_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
                model.opt.is_train = 0
                model.opt.no_loss = 1
                with torch.no_grad():
                    render_vid(model, test_dataset, Visualizer(test_opt, tb_writer=visualizer.tb_writer),
                                 test_opt, render_bg_info, steps=total_steps, model_fixed=model_fixed)
                model.opt.no_loss = 0
                model.opt.is_train = 1
                del test_dataset

            if (total_steps % opt.test_freq == 0 and total_steps < (opt.maximum_step - 1) and total_steps > 0): #total_steps == 10000 or 
                # test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
                test_opt.name = opt.name + "/test_{}".format(total_steps)
                rnm, ran, rpn = False, False, False
                if 'normal_map' in opt.visual_items: rnm = opt.return_normal_map; opt.return_normal_map = True
                if 'agg_normal' in opt.visual_items: ran = opt.return_agg_normal; opt.return_agg_normal = True
                if 'pred_normal' in opt.visual_items: rpn = opt.return_pred_normal; opt.return_pred_normal = True
                
                torch.cuda.empty_cache()
                model.opt.is_train, model.opt.no_loss = 0, 1
                if opt.brdf_rendering: init_brdf_config(model, opt, training=False)
                with torch.no_grad():
                    if opt.test_train == 0:
                        test_psnr = test(model, test_dataset, Visualizer(test_opt, tb_writer=visualizer.tb_writer),
                                     test_opt, test_bg_info, test_steps=total_steps, lpips=True, model_fixed=model_fixed)
                    else:
                        train_dataset.opt.random_sample = "no_crop"
                        test_psnr = test(model, train_dataset, Visualizer(test_opt, tb_writer=visualizer.tb_writer),
                                     test_opt, test_bg_info, test_steps=total_steps, lpips=True, model_fixed=model_fixed)
                        train_dataset.opt.random_sample = opt.random_sample
                model.opt.is_train, model.opt.no_loss = 1, 0
                if opt.brdf_rendering: init_brdf_config(model, opt, training=True)
                opt.return_normal_map, opt.return_agg_normal, opt.return_pred_normal\
                     = rnm, ran, rpn
                # del test_dataset
                best_iter = total_steps if test_psnr > best_PSNR else best_iter
                best_PSNR = max(test_psnr, best_PSNR)
                visualizer.print_details(f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_iter: {best_iter}")
            model.train()

        # try:
        #     print("saving the model at the end of epoch")
        #     other_states = {'epoch_count': epoch, 'total_steps': total_steps}
        #     model.save_networks('latest', other_states)
        #
        # except Exception as e:
        #     print(e)

        if opt.maximum_step is not None and total_steps >= opt.maximum_step:
            visualizer.print_details('{}: End of stepts {} / {} \t Time Taken: {} sec'.format(
                opt.name, total_steps, opt.maximum_step,
                time.time() - epoch_start_time))
            break

    del train_dataset
    other_states = {
        'epoch_count': epoch,
        'total_steps': total_steps,
    }
    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
    model.save_networks(total_steps, other_states)

    torch.cuda.empty_cache()
    test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=50)
    model.opt.no_loss = 1
    model.opt.is_train = 0

    visualizer.print_details("full datasets test:")
    with torch.no_grad():
        test_psnr = test(model, test_dataset, Visualizer(test_opt, tb_writer=visualizer.tb_writer),
                     test_opt, test_bg_info, test_steps=total_steps, gen_vid=True, lpips=True, model_fixed=model_fixed)
    best_iter = total_steps if test_psnr > best_PSNR else best_iter
    best_PSNR = max(test_psnr, best_PSNR)
    visualizer.print_details(
        f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_iter: {best_iter}")
    exit()


if __name__ == '__main__':
    main()
