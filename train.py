#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, nearMean_map
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from utils.general_utils import get_expon_lr_func
import torch.nn.functional as F

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint:
            print("start from fsssssssine stage, skip coarse stage.")
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    #
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=16,
                                                collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=16,
                                                collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False
        #
    count = 0
    for iteration in range(first_iter, final_iter + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    print("3rd view", custom_cam.world_view_transform)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage,
                                       cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size, shuffle=True,
                                                        num_workers=32, collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size:

                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))
        # breakpoint()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        rnd_depth = []
        gt_depths = []
        radii_list = []
        flow_gtsss = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        """
        Dumb way to append feature maps
        """
        feature_maps = []
        ################################ Gaussian parameter at t2

        # Initialize 't2' dictionary to store the latest values across iterations
        t2 = {
            "proj_2D_t_2": None,
            "gs_per_pixel": None,
            "weight_per_gs_pixel": None
        }
        """
        Load data semantic features
                gt_feature_map = viewpoint_cam.semantic_feature.cuda()
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 
        if dataset.speedup:
            feature_map = cnn_decoder(feature_map)
        Ll1_feature = l1_loss(feature_map, gt_feature_map) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 1.0 * Ll1_feature 
        """
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            image, viewspace_point_tensor, visibility_filter, radii, depth, feature_map = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], \
                                                                                            render_pkg["depth"], render_pkg["feature_map"]

            #image, viewspace_point_tensor, visibility_filter, radii, proj_2D_t_1, gs_per_pixel, weight_per_gs_pixel, x_mu, cov2D_inv_t_1, cov2D_t_2 = \
            #render_pkg["render"], render_pkg[
             #   "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["proj_2D"], \
            #render_pkg["gs_per_pixel"].long(), render_pkg["weight_per_gs_pixel"], render_pkg["x_mu"], render_pkg[
             #   "conic_2D"].detach(), render_pkg["conic_2D_inv"]

            images.append(image.unsqueeze(0))
            #feature_maps.append((feature_map.unsqueeze(0)))
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
                #flow_gt = viewpoint_cam.flow.cuda()
                depth_gt = viewpoint_cam.depth.cuda()
            else:
                gt_image = viewpoint_cam['image'].cuda()
                # flow_gt = viewpoint_cam["flow"].cuda()
                flow_gt = viewpoint_cam.flow.cuda()
                depth_gt = viewpoint_cam.depth.cuda()

            gt_images.append(gt_image.unsqueeze(0))
            flow_gtsss.append(flow_gt.unsqueeze(0))
            gt_depths.append(depth_gt.unsqueeze(0))

            # flow_gtsss.append(flow_gt.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        """
        Adding Depth guided supervision loss

        NOTE: Carefull with the deoth batch for the momemnt keeping batch size 1
        """
        rendered_depth = render_pkg["depth"]
        rnd_depth.append(rendered_depth.unsqueeze(0))

        # if scene.dataset_type != "PanopticSports":
        #   gt_depth = viewpoint_cam.depth.cuda()
        # else:
        #   gt_depth = viewpoint_cam['depth'].cuda()
        # gt_depths.append(gt_depth.unsqueeze(0))

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        flow_gt_tensor = torch.cat(flow_gtsss, 0)
        depth_gt_tensorr = torch.cat(gt_depths, 0)
        rendered_depth_tensor = torch.cat(rnd_depth, 0)
        #rendered_feature_tensor = torch.cat(feature_maps, 0)

        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])


        """



        import matplotlib.pyplot as plt

        # Move tensor to CPU if it's on CUDA and remove the batch dimension
        tensor_cpu = flow_gt_tensor.squeeze(0).cpu().numpy()  # Shape will now be (1200, 1900, 2)

        flow_x = tensor_cpu[0]  # Horizontal component (1200, 1900)
        flow_y = tensor_cpu[1]  # Vertical component (1200, 1900)

        # Compute magnitude and angle
        magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
        angle = np.arctan2(flow_y, flow_x)  # Angle in radians

        # Normalize the magnitude
        import cv2
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        # Create an HSV image where:
        # - H (hue) is the angle (converted to degrees)
        # - S (saturation) is set to 1 (full saturation)
        # - V (value) is the normalized magnitude
        hsv = np.zeros((flow_x.shape[0], flow_x.shape[1], 3), dtype=np.float32)
        hsv[..., 0] = (angle + np.pi) * (180 / np.pi) / 2  # Map angle to [0, 180]
        hsv[..., 1] = 1  # Full saturation
        hsv[..., 2] = magnitude  # Value = magnitude

        # Convert HSV to RGB for visualization
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Visualize
        plt.imshow(rgb_flow)
        plt.title('Optical Flow (HSV Representation)')
        plt.axis('off')
        plt.show()
        # breakpoint()
 # Computes EPE for all pixels
        """

        """        #L_FLOW_LOSS = end_point_error()

        #depth_tensor = torch.cat(rnd_depth,0)
        #gt_depth_tensor = torch.cat(gt_depths,0)

        #depth_loss = l1_loss(depth_tensor, gt_depth_tensor[:,:3,:,:])
        #depth_weight_scale = 0.5

        cov2D_t_2_mtx = torch.zeros([cov2D_t_2.shape[0], 2, 2]).cuda()
        cov2D_t_2_mtx[:, 0, 0] = cov2D_t_2[:, 0]
        cov2D_t_2_mtx[:, 0, 1] = cov2D_t_2[:, 1]
        cov2D_t_2_mtx[:, 1, 0] = cov2D_t_2[:, 1]
        cov2D_t_2_mtx[:, 1, 1] = cov2D_t_2[:, 2]

        cov2D_inv_t_1_mtx = torch.zeros([cov2D_inv_t_1.shape[0], 2, 2]).cuda()
        cov2D_inv_t_1_mtx[:, 0, 0] = cov2D_inv_t_1[:, 0]
        cov2D_inv_t_1_mtx[:, 0, 1] = cov2D_inv_t_1[:, 1]
        cov2D_inv_t_1_mtx[:, 1, 0] = cov2D_inv_t_1[:, 1]
        cov2D_inv_t_1_mtx[:, 1, 1] = cov2D_inv_t_1[:, 2]

        # B_t_2
        U_t_2 = torch.svd(cov2D_t_2_mtx)[0]
        S_t_2 = torch.svd(cov2D_t_2_mtx)[1]
        V_t_2 = torch.svd(cov2D_t_2_mtx)[2]
        B_t_2 = torch.bmm(torch.bmm(U_t_2, torch.diag_embed(S_t_2) ** (1 / 2)), V_t_2.transpose(1, 2))

        # B_t_1 ^(-1)
        U_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[0]
        S_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[1]
        V_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[2]
        B_inv_t_1 = torch.bmm(torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1) ** (1 / 2)),
                              V_inv_t_1.transpose(1, 2))

        # calculate B_t_2*B_inv_t_1
        B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)

        # calculate cov2D_t_2*cov2D_inv_t_1
        # cov2D_t_2cov2D_inv_t_1 = torch.zeros([cov2D_inv_t_2.shape[0],2,2]).cuda()
        # cov2D_t_2cov2D_inv_t_1[:, 0, 0] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1]
        # cov2D_t_2cov2D_inv_t_1[:, 0, 1] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 2]
        # cov2D_t_2cov2D_inv_t_1[:, 1, 0] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 1]
        # cov2D_t_2cov2D_inv_t_1[:, 1, 1] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 2]

        # isotropic version of GaussianFlow
        # predicted_flow_by_gs = (proj_2D_next[gs_per_pixel] - proj_2D[gs_per_pixel].detach()) * weights.detach()

        # full formulation of GaussianFlow
        proj_2D_t_2 = t2["proj_2D_t_1"]
        cov_multi = (B_t_2_B_inv_t_1[gs_per_pixel] @ x_mu.permute(0, 2, 3, 1).unsqueeze(-1).detach()).squeeze()
        predicted_flow_by_gs = (cov_multi + proj_2D_t_2[gs_per_pixel] - proj_2D_t_1[
            gs_per_pixel].detach() - x_mu.permute(0, 2, 3, 1).detach()) * weight_per_gs_pixel.unsqueeze(-1).detach()

        # print("sahata")
        # flow supervision loss
        # large_motion_msk = torch.norm(flow_gt_tensor, p=2,
        #                            dim=-1) >= 0.3  # flow_thresh  # flow_thresh = 0.1 or other value to filter out noise, here we assume that we have already loaded pre-computed optical flow somewhere as pseudo GT
        large_motion_msk_proper = torch.norm(flow_gt_tensor) >= 0.3
        # now_what_is = (predicted_flow_by_gs.sum(0)[large_motion_msk_proper].permute(0, 3, 1, 2))  # .mean()
        gt_macha_flow =  flow_gt_tensor.expand(20, -1, -1, -1)  #

        #mse_loss = torch.nn.functional.mse_loss(predicted_flow_by_gs, gt_macha_flow)
        #lflow = end_point_error(predicted_flow_by_gs, gt_macha_flow)

        def end_point_error(predicted_flow, ground_truth_flow):
            # Ensure both tensors have the same shape
            assert predicted_flow.shape == ground_truth_flow.shape, "Predicted and ground truth flows must have the same shape."

            # Compute the L2 norm (EPE) between predicted and ground truth flow
            epe_per_pixel = torch.norm(predicted_flow - ground_truth_flow, dim=1)  # (batch_size, height, width)
            return epe_per_pixel.mean()  # Mean EPE for the entire batch

        # Example usage:
        predicted_flow = predicted_flow_by_gs.permute(0, 3, 1,
                                                2)  # Ensure predicted flow is in (batch_size, 2, height, width)
        #ground_truth_flow = ground_truth_flow.expand(predicted_flow.shape[0], -1, -1,
         #                                            -1)  # Expand ground truth to match batch size

        # Calculate the loss
        epe_loss = end_point_error(predicted_flow, gt_macha_flow)
        print(f'EPE Loss: {epe_loss.item()}')
        """

        # what_is_this = predicted_flow_by_gs.sum(0)[large_motion_msk] #.mean() #, p=2, dim=-1).mean())
        """        Lflow = torch.norm(
            (flow_gt_tensor - predicted_flow_by_gs.sum(0)[large_motion_msk_proper].permute(0, 3, 1, 2)), p=2,
            dim=-1).mean()
        # Lflow = torch.norm(flow_gt_tensor - now_what_is.permute(0, 3, 1, 2) ) #.sum(0))#[large_motion_msk], p=2, dim=-1).mean()
        # flow_weight = 0.1
        # loss = loss + flow_weight * Lflow

        Ll1 = Ll1 / Ll1.detach().abs().max()
        flow_loss_normalized = Lflow / Lflow.detach().abs().max()"""

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        # norm
        """
        Dumb way to supervise depth loss |loss = 0.8 *Ll1 + 0.2 * depth_loss

        gt_image_tensor
        gt_depth_map = viewpoint_cam.invdepthmap.cuda()
        depth_mask = viewpoint_cam.depth_mask.cuda()
        """

        """        Lssim = (1.0 - ssim(image_tensor, gt_image_tensor))
        photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
        loss = photo_loss.clone()
        """


        depth_l1_weight_init = 0.01 #1.0
        depth_l1_weight_final = 0.001
        depth_l1_weight = get_expon_lr_func(depth_l1_weight_init, depth_l1_weight_final,
                                            max_steps=opt.iterations)

        loss = Ll1

        #gt_feature_map = viewpoint_cam.flow.cuda()
        #flow_gt_tensor
        feature_map_tensor_ge = F.interpolate(feature_maps.unsqueeze(0),
                                    size=(flow_gt_tensor.shape[1], flow_gt_tensor.shape[2]), mode='bilinear',
                                    align_corners=True).squeeze(0)
        print("Logging Feature supervision")
        if dataset.speedup:
            """
            To do implement own encoder decoder
            """
            #feature_map = cnn_decoder(feature_map)
        Ll1_feature = l1_loss(feature_map_tensor_ge, flow_gt_tensor)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 1.0 * Ll1_feature

        if stage == "fine" and iteration % 100 == 0:
            print("Logging Depth supervision")
            Ll1depth_pure = torch.abs((rendered_depth_tensor - depth_gt_tensorr).mean())


            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()



        else:
            Ll1depth = 0.0


        # loss = 0.8 *Ll1 + 0.2 * depth_loss

        #loss = Ll1  ##########################################################################original_loss#######################################
        # loss += depth_weight_scale * depth_loss
        """
        Depth regularizer
        """
        # nearDepthMean_map = nearMean_map(depth_tensor.detach(), gt_image_tensor[:,:3,:,:], kernelsize=3)
        # loss = loss + l2_loss(nearDepthMean_map, depth_tensor) * 1.0
        "Original fine iteration loss"
        """
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss



            Adapting floww supervision 


        """

        # flow_weight could be 1, 0.1, ... whatever you want.
        #

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes,
                                                   hyper.plane_tv_weight)
            loss += tv_loss

        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            print("IS SSIM Loss bein opened")
            loss += opt.lambda_dssim * (1.0 - ssim_loss)

        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        # loss = Ll1 #+ tv_loss#+ epe_loss * 0.3 + tv_loss

        loss.backward()
        # Collect all the parameters that require gradients
        params = []
        for param in gaussians.__dict__.values():
            if isinstance(param, torch.Tensor) and param.requires_grad:
                params.append(param)

        # Apply gradient clipping to these parameters
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        # Gradient Norm Logging
        total_norm = 0
        for param in params:
            if param.grad is not None:
                # Calculate the norm of the gradient for this parameter
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                print(f"Gradient norm for param: {param_norm.item()}")  # Log the norm of each parameter's gradient

        total_norm = total_norm ** (1. / 2)
        print(f"Total Gradient Norm: {total_norm}")  # Log the total gradient norm

        # Check for NaNs in gradients and parameters
        for param in params:
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {param}")
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradient of parameter: {param}")
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar -original
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log

            # ema_loss_for_log = 0.4 * photo_loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                                          "point": f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                        or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration % 100 == 99):
                    # breakpoint()
                    render_training_image(scene, gaussians, [test_cams[iteration % len(test_cams)]], render, pipe,
                                          background, stage + "test", iteration, timer.get_elapsed_time(),
                                          scene.dataset_type)
                    render_training_image(scene, gaussians, [train_cams[iteration % len(train_cams)]], render, pipe,
                                          background, stage + "train", iteration, timer.get_elapsed_time(),
                                          scene.dataset_type)
                    # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                                opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (
                                            opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (
                                opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (
                                            opt.densify_until_iter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and \
                        gaussians.get_xyz.shape[0] < 360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5,
                                      scene.model_path, iteration, stage)
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and \
                        gaussians.get_xyz.shape[0] > 200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[
                    0] < 360000 and opt.add_point:
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                # gaussians.

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations, timer)


def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname


        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #
        validation_configs = ({'name': 'test',
                               'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in
                                           range(10, 5000, 299)]},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, stage=stage, cam_type=dataset_type, *renderArgs)[
                            "render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage + "/" + config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage + "/" + config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask

                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate',
                                 scene.gaussians._deformation_table.sum() / scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram",
                                    scene.gaussians._deformation_accum.mean(dim=-1) / 100, iteration, max_bins=500)

        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 14000, 17000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[2500, 3000, 5000, 7000, 10000, 14000, 17000, 20000, 30000, 45000, 50000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
