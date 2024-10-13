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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
"""
import os
import imageio
import torch
import numpy as np
from time import time
from tqdm import tqdm

    def render_mesh(self, pose):

        h = w = self.opt.output_size

        v = self.v + self.deform
        f = self.f

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(-1).squeeze(0) # [H, W] important to enable gradients!
        
        if self.albedo is None:
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, H, W, 3]
            xyzs = xyzs.view(-1, 3)
            mask = (alpha > 0).view(-1)
            image = torch.zeros_like(xyzs, dtype=torch.float32)
            if mask.any():
                masked_albedo = torch.sigmoid(self.mlp(self.encoder(xyzs[mask].detach(), bound=1)))
                image[mask] = masked_albedo.float()
        else:
            texc, texc_db = dr.interpolate(self.vt.unsqueeze(0), rast, self.ft, rast_db=rast_db, diff_attrs='all')
            image = torch.sigmoid(dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]

        image = image.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous() # [3, H, W]
        image = alpha * image + (1 - alpha)

        return image, alpha





def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_maps")
    

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)  # Directory to save depth maps

    render_images = []
    depth_images = []  # Store depth maps
    gt_list = []
    render_list = []

    print("point nums:", gaussians._xyz.shape[0])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            time1 = time()

        # Rendering Gaussian splats
        depth_map = render(view, gaussians, pipeline, background, cam_type=cam_type)["depths"]
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]

        # Move tensor to CPU if it's on CUDA, then convert to NumPy
        if torch.is_tensor(rendering):
            rendering = rendering.cpu().numpy()

        render_images.append(to8b(rendering[0]).transpose(1, 2, 0))
        render_list.append(rendering)

        # Get the depth map from the rendering function
        depth_map = render(view, gaussians, pipeline, background, cam_type=cam_type).get("depths")

        # If depth map is available, process it
        if depth_map is not None:
            # Move tensor to CPU if it's on CUDA, then convert to NumPy
            if torch.is_tensor(depth_map):
                depth_map = depth_map.cpu().numpy()

            depth_images.append(to8b(depth_map))  # Convert depth map to 8-bit for saving

        # Handling ground truth (GT) images
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt = view['image'].cuda()

            # Move GT image to CPU if it's on CUDA
            if torch.is_tensor(gt):
                gt = gt.cpu().numpy()

            gt_list.append(gt)

    time2 = time()
    print("FPS:", (len(views) - 1) / (time2 - time1))

    # Save ground truth images
    multithread_write(gt_list, gts_path)

    # Save rendered images
    multithread_write(render_list, render_path)

    # Save depth maps
    if depth_images:
        multithread_write(depth_images, depth_path)  # Save depth maps to separate directory

    # Create video from rendered images at 10 FPS
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images,
                     fps=10)

    # Create video from depth maps at 10 FPS (if depth maps were generated)
    if depth_images:
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_depth.mp4'), depth_images,
                         fps=10)
                         
                         --model_path
"/home/uchihadj/ECCV_workshop/4DGaussians/output/CVPR_2025_Benchmark/first_20_3_Frame"
--skip_video
--configs
arguments/dynerf/default.py

"""


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_maps")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    render_images = []
    gt_list = []
    render_list = []
    depth_list = []
    depth_images = []
    # breakpoint()
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        # breakpoint()

        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1,2,0))

        rendering_d = render(view, gaussians, pipeline, background, cam_type=cam_type)["depth"]

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm

        # Assuming `rendering_d` is your depth tensor of shape (1, 1200, 1900) on CUDA
        tensor_cpu = rendering_d.squeeze().cpu().numpy()  # Squeeze to remove the first dimension (1, 1200, 1900) -> (1200, 1900)

        # Step 1: Normalize the depth values between 0 and 1
        tensor_cpu_normalized = (tensor_cpu - np.min(tensor_cpu)) / (np.max(tensor_cpu) - np.min(tensor_cpu))

        # Step 2: Apply the 'magma' colormap, trubo, plasma, Spectral
        colormap = cm.get_cmap('inferno')
        colored_depth = colormap(tensor_cpu_normalized)  # This returns a (1200, 1900, 4) RGBA image

        # Step 3: Convert the RGBA image to RGB (removing the alpha channel)
        colored_depth_rgb = colored_depth[:, :, :3]  # (1200, 1900, 3)

        # Step 4: Convert the RGB image back to a PyTorch tensor
        colored_depth_tensor = torch.from_numpy(colored_depth_rgb).permute(2, 0, 1).float()  # (3, 1200, 1900)

        # (Optional) Move the tensor back to CUDA if needed
        # colored_depth_tensor = colored_depth_tensor.cuda()

        # Now, `colored_depth_tensor` is your tensor representing the depth visualization in 'magma' colormap
        # You can now save or use this tensor as needed
        depth_images.append(to8b(colored_depth_tensor).transpose(1, 2, 0))
        torchvision.utils.save_image(colored_depth_tensor, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
                #flow_ =
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)
        # if idx >= 10:
            # break
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    # print("writing training images.")

    multithread_write(gt_list, gts_path)
    # print("writing rendering images.")

    multithread_write(render_list, render_path)


    multithread_write(depth_list, depth_path)

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'depth_video_rgb.mp4'), depth_images,
                     fps=5)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=5)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=5000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(args.iteration)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)