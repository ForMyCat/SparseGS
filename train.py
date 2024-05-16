import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2
import diptest
from icecream import ic
from guidance.sd_utils import StableDiffusion
from random import randint
from utils.loss_utils import l1_loss, ssim, local_pearson_loss, pearson_depth_loss, mask_l1_loss
from utils.prune_utils import calc_diff
from scipy import stats
import matplotlib.pyplot as plt
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, normalize
import time
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import getWorld2View2
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import copy
os.environ['QT_QPA_PLATFORM']='offscreen'
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, step, max_cameras, prune_sched):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, step=step, max_cameras=max_cameras)
    gaussians.training_setup(opt)
    

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    last_prune_iter = None
    print(prune_sched)

    if dataset.lambda_diffusion:
        guidance_sd = StableDiffusion(device="cuda")
        guidance_sd.get_text_embeds([""], [""])
        print(f"[INFO] loaded SD!")

    warp_cam_stack = None
    
    save_cc = 0
    diff_cam = copy.deepcopy(scene.getTrainCameras()[0])
    for iteration in range(first_iter, opt.iterations + 1):        
        save_cc += 1
        '''    
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        '''
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_idxs = list(np.arange(len(viewpoint_stack)))
        rand = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(rand)
        viewpoint_idx = viewpoint_idxs.pop(rand)



        pick_diff_cam = ((randint(1, 100) <= (args.SDS_freq*100)) and dataset.lambda_diffusion and iteration > (opt.iterations*2/3))
        if pick_diff_cam: # A diffusion cam is picked
            diff_pose = scene.getRandEllipsePose(viewpoint_idx, 0, z_variation=0)
            diff_cam.world_view_transform = torch.tensor(getWorld2View2(diff_pose[:3, :3].T, diff_pose[:3, 3], diff_cam.trans, diff_cam.scale)).transpose(0, 1).cuda()
            diff_cam.full_proj_transform = (diff_cam.world_view_transform.unsqueeze(0).bmm(diff_cam.projection_matrix.unsqueeze(0))).squeeze(0)
            diff_cam.camera_center = diff_cam.world_view_transform.inverse()[3, :3]
            diff_render_pkg = render(diff_cam, gaussians, pipe, background)
            diff_image = diff_render_pkg["render"]

        
        pick_warp_cam = ((randint(1, 10) <= 4) and (dataset.lambda_reg > 0) and iteration > (dataset.warp_reg_start_itr))
        if pick_warp_cam: # A warping cam is picked
            if not warp_cam_stack:
                warp_cam_stack = scene.getFtCameras().copy()
            warp_cam = warp_cam_stack.pop(randint(0, len(warp_cam_stack)-1))
            warp_render_pkg = render(warp_cam, gaussians, pipe, background)
            warp_image, warp_viewspace_point_tensor, warp_visibility_filter, warp_radii = warp_render_pkg["render"], warp_render_pkg["viewspace_points"], warp_render_pkg["visibility_filter"], warp_render_pkg["radii"]
            reg_gt_image = warp_cam.original_image.cuda()
            reg_mask = warp_cam.warp_mask



        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

       
        diffusion_loss = None
        lp_loss = None
        reg_loss = None
        pearson_loss = None

        if dataset.lambda_pearson > 0 and (not pick_warp_cam):
            pearson_loss = pearson_depth_loss(depth.squeeze(0), viewpoint_cam.depth)
            loss += dataset.lambda_pearson * pearson_loss
        
        if dataset.lambda_local_pearson > 0 and (not pick_warp_cam):
            lp_loss = local_pearson_loss(depth.squeeze(0), viewpoint_cam.depth, dataset.box_p, dataset.p_corr)
            loss += dataset.lambda_local_pearson * lp_loss
        
        if pick_diff_cam:
            diffusion_loss = guidance_sd.train_step(diff_image.unsqueeze(0), dataset.step_ratio)
            loss += dataset.lambda_diffusion * diffusion_loss

        if pick_warp_cam:
            reg_Ll1 = mask_l1_loss(warp_image, reg_gt_image, reg_mask)
            reg_loss = (1.0 - opt.lambda_dssim) * reg_Ll1 + opt.lambda_dssim * (1.0 - ssim(warp_image, reg_gt_image))
            loss += dataset.lambda_reg * reg_loss 

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            losses = [lp_loss, pearson_loss, diffusion_loss, reg_loss]
            names = [ "Local_Depth", "Global Depth", "Diffusion", "Warp Reg"]
            
            # Update warping at iteration warp_reg_start_itr
            if (iteration in [dataset.warp_reg_start_itr]) and ((dataset.lambda_reg > 0)):
                src_cam_stack = scene.getTrainCameras().copy()
                src_depth_dict = dict()
                for each_src_cam in src_cam_stack:
                    src_render_pkg = render(each_src_cam, gaussians, pipe, background)
                    src_depth = src_render_pkg["alpha_depth"]
                    src_depth_dict[each_src_cam.uid] = src_depth
                _warp_cam_stack = scene.getFtCameras()
                for _cam in _warp_cam_stack:
                    _uid = _cam.src_uid
                    _depth_min = src_depth_dict[_uid].min().item()
                    _depth_max = src_depth_dict[_uid].max().item()
                    _cam.generate_warp_gt(_depth_min, _depth_max)

            if iteration % 10 == 0:
                postfix_dict = {"EMA Loss": f"{ema_loss_for_log:.{7}f}",
                                          "Total Loss": f"{loss:.{7}f}"}

                for l,n in zip(losses, names):
                    if l is not None:
                        postfix_dict[n] = f"{l:.{7}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            tr_dict = {names[i]: losses[i] for i in range(len(losses))}
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, tr_dict,  iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and (not pick_warp_cam):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Densification for warp Cam
            if iteration < opt.densify_until_iter and (pick_warp_cam):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[warp_visibility_filter] = torch.max(gaussians.max_radii2D[warp_visibility_filter], warp_radii[warp_visibility_filter])
                gaussians.add_densification_stats(warp_viewspace_point_tensor, warp_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
            if iteration in prune_sched:
                os.makedirs(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}"), exist_ok=True)
                os.makedirs(os.path.join(dataset.model_path, f"modes_{iteration}"), exist_ok=True)
                scene.save(iteration-1)
                prune_floaters(scene.getTrainCameras().copy(), gaussians, pipe, background, dataset, iteration)
                scene.save(iteration+1)
                last_prune_iter = iteration
            
            if last_prune_iter is not None and not (iteration == last_prune_iter) and iteration - last_prune_iter > dataset.densify_lag and iteration - last_prune_iter < dataset.densify_period + dataset.densify_lag and iteration % 100 == 0:                
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, 20)
                print('Densifying')
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

#   pixel_thresh is the stopping criteria, or lower bound of how many pixels we want to prune, if lower then stop bc we dont have much more to prune
#   thresh_bin  is the upper bound so that we dont prune too much at once
#   prune_stop is how far we want to prune from the initial number of pixels pruned 
# 

def calc_alpha(means2D, conic_opac, x, y):
    dx = x - means2D[:,0]
    dy = y - means2D[:,1]
    power = -0.5*(conic_opac[:,0]*(dx*dx) + conic_opac[:,2]*(dy*dy)) - conic_opac[:,1]*dx*dy
    alpha = power
    alpha[power > 0] = -100
    return alpha

def prune_floaters(viewpoint_stack, gaussians, pipe, background, dataset, iteration):
     with torch.no_grad():
        N = gaussians.get_opacity.shape[0]
        ctrs = [0]*len(viewpoint_stack)

        num_pixels_init = [None]*len(viewpoint_stack)
        #mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        os.makedirs(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"modes_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"depth_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"diff_{iteration}"), exist_ok=True)


        mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")

        plt.figure(figsize=(25,20))
        
        dips = []        
        point_lists = []
        means2Ds = []
        conic_opacities = []
        mode_ids = []
        diffs = []
        names = []
        for view in viewpoint_stack:
            names.append(view.image_name)
            render_pkg = render(view, gaussians, pipe, background, ret_pts=True)
            mode_id, mode, point_list, depth, means2D, conic_opacity = render_pkg["mode_id"], render_pkg["modes"], render_pkg["point_list"], render_pkg["alpha_depth"], render_pkg["means2D"], render_pkg["conic_opacity"] 
            diff = calc_diff(mode, depth)
            plt.imsave(os.path.join(dataset.model_path, f"modes_{iteration}", f"{view.image_name}.png" ), mode.cpu().numpy().squeeze(), cmap='jet')
            plt.imsave(os.path.join(dataset.model_path, f"depth_{iteration}", f"{view.image_name}.png" ), depth.cpu().numpy().squeeze(), cmap='jet')
            point_lists.append(point_list)
            means2Ds.append(means2D)
            conic_opacities.append(conic_opacity)
            mode_ids.append(mode_id)
            diffs.append(diff)
            dips.append(diptest.dipstat(diff[diff > 0].cpu().numpy()))

        dips = np.array(dips)
        avg_dip = dips.mean()
        perc = dataset.prune_perc*100*np.exp(-1*dataset.prune_exp*avg_dip)


        if (perc < 80):
            perc = 80
        print(f'Percentile {perc}')

        for name, mode_id, point_list, diff, means2D, conic_opacity in zip(names, mode_ids, point_lists, diffs, means2Ds, conic_opacities):
            submask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
           
            diffpos = diff[diff > 0]
            threshold = np.percentile(diffpos.cpu().numpy(), perc)
            pruned_modes_mask = (diff > threshold).squeeze()
            cv2.imwrite(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}",f"{name}.png"), pruned_modes_mask.cpu().numpy().squeeze().astype(np.uint8)*255)

            pixel_y, pixel_x = torch.meshgrid(torch.arange(pruned_modes_mask.shape[0]), torch.arange(pruned_modes_mask.shape[1]), indexing='ij')
            pixel_y = pixel_y.to('cuda')
            pixel_x = pixel_x.to('cuda')
            prune_mode_ids = mode_id[:,pruned_modes_mask] # subselect the mode idxs
            pixel_x = pixel_x[pruned_modes_mask]
            pixel_y = pixel_y[pruned_modes_mask]

            neg_mask = (prune_mode_ids == -1).any(dim=0)
            prune_mode_ids = prune_mode_ids[:,~neg_mask]
            pixel_x = pixel_x[~neg_mask]
            pixel_y = pixel_y[~neg_mask]

            selected_gaussians = set()
            for j in range(prune_mode_ids.shape[-1]):
                x = pixel_x[j]
                y = pixel_y[j]
                gausses = point_list[prune_mode_ids[0,j]:prune_mode_ids[1,j]+1].long()
                c_opacs = conic_opacity[gausses]
                m2Ds = means2D[gausses]
                test_alpha = calc_alpha(m2Ds, c_opacs, x, y)
 
                alpha_mask = test_alpha > dataset.power_thresh
      
                gausses = gausses[alpha_mask]
       
                selected_gaussians.update(gausses.tolist())
      
            submask[list(selected_gaussians)] = True
            
            print(f"submask {torch.count_nonzero(submask)}")

            mask = mask | submask

            num_points_pruned = submask.sum()
            print(f'Pruning {num_points_pruned} gaussians')

        print(gaussians.get_xyz.shape[0])
        gaussians.prune_points(mask)
        print(gaussians.get_xyz.shape[0])
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        print("Tensorboard Found!")
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, tr_dict, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for k,v in tr_dict.items():
            if v is not None:
                tb_writer.add_scalar('train_loss_patches/' + k, v.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_cameras", type=int, default=None)
    parser.add_argument("--prune_sched", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    dataset = lp.extract(args)
    
    print("Optimizing " + dataset.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.step, args.max_cameras, args.prune_sched)

    # All done
    print("\nTraining complete.")
