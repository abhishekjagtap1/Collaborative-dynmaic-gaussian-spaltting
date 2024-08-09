
## ECCV Benchmarking Time

- **Use Pre trained gaussian point cloud for initialization -> observed PSNR upto 38 beating everyone in the race**
- Use `getprojectionmatrix2` for training and evaluation. Update `NoneType` error when opening SIBR viewer using this setting.
- Do not downsample points. Instead, reduce the bounds and use point painting of the same LiDAR, not a general one. 

  

##  Results

### Ground Truth Video
![GT Video](assets/GT_two_agents.gif)

### 3D Reconstruction Video
![3D Reconstruction Video](assets/3DReconstruction_video.gif)


### Novel View Synthesis Showing Collaboration Video
![Novel View Synthesis Showing Collaboration](assets/Novel_view_showing_collboration.gif)

### Novel View Synthesis [Left - Infrastructure View, Right - Vehicle trying to view left side of the scene]
![Novel View Synthesis Showing Collaboration](assets/Novel_view_using_infrastructure.gif)

### Freeze Time - Novel View Synthesis [Collaboration Between Infrastructrure and Vehicle]
![Novel View Synthesis Showing Collaboration (Freezed Time)](assets/FreezeTime.gif)

### 3 Agents Collaboration - Novel View Synthesis [Collaboration Between Infrastructrure and Vehicle]
[Collaboration of 3 AGENTS](assets/collaboration_3_agents.gif)

# Recent Updates

1) **Broken View Fixed:**
   - For Rendering a 3D scene from different agents, camera intrinsics needs to be adapted accordingly, as different agents have different intrinsic matrix
   - Dumb way to fix is by - changing intrinsic matrix via image_index - Works for now 
   - **New Updates**:
   - **Fixed SIBR viewer:** by changing ```FovX = focal2fov(self.dataset.focal[0], image.shape[2]) * 1.9``` ```FovY = focal2fov(self.dataset.focal[0], image.shape[1]) * 1.9``` in scene/datsaset.py:30 
   - Use ```./install/bin/SIBR_remoteGaussian_app --port 6018 --path /home/uchihadj/TUMtraf/4DGaussians/data/junk_combined_data/0/``` for visualizing training
   - Use ```./install/bin/SIBR_gaussianViewer_app -m /home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/Without_collaboration_SIBR/SIBR_viewer_per_time/```  for rendering

2) **How to accurately measure novel view synthesis results** - Benchmark LPIPS and PSNR and other metric using different poses
   - Currently working on it
3) **Benchmark our model performance on Nuscenes** - Pending
4) **Visulize With and Without collaboration results** - Under Training
5)  **How to avoid floaters when visulizing** - Crop or remove gaussian outliers using some post-processing scripts
6)  **How to Improve Dynamic gaussian performace:** Currently training is done only for 14K iteration, Need to increase to 20K and hope my ```scene/deforamtion.py``` does not throw any NAN errors
  
# To-Do's
1) **Rendering or SIBR viewer still requires a colmap path to specify dataset type:**

## Environmental Setups

Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.

```bash
conda create -n Gaussians4D python=3.7 
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

In our environment, we use pytorch=1.13.1+cu116.

## Data Preparation -TUMTRAF 


**For training on TUMTRAF scenes:**
If you want to train on TUMTRAf, Download the data from https://innovation-mobility.com/en/project-providentia/a9-dataset/ 

Arrange the number of agents accordingly
```
├── data
|   | multipleview
│     | TUMTRAF_Scene_1 
│   	  | V2X_Agent_infra-01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | V2X_Agent_ego_cav-01
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
```
Refer to heavily documented code at ```scene/neural_3D_dataset_NDC.py:500 ``` for loading data from different agents along with their poses.

Note: Calib/Poses are used from TUMTRAF dataset folder
Note: ```utils/Initialization_utils``` is used to preprocess the lidar point cloud for initialization

Run 
```python train.py -s  data/TUMTRAF_Scene_1/ --port 6018 --expname "ECCV_2025/less_dynamic_scene_without_collaboration" --configs arguments/dynerf/default.py  --debug_from 170 --checkpoint_iterations 3000```

### Story of my life: Insights useful for training (Most of the bugs and errors has been rectified)

1) **Projection Matrix:**
    - Don't use the projection matrix given by TUmtraf as it messes up the entire densification process. No LiDAR points will be visible in the projection view, making Gaussian splat initialization useless.

2) **Extrinsics for South_2:**
    - I have used extrinsics for south_2 by multiplying with base:
        ```python
        extrinsic_matrix_lidar_to_camera_south_2 = np.matmul(transformation_matrix_base_to_camera_south_2, transformation_matrix_lidar_to_base_south_2)
        ```
    - As a result, I observe the point cloud rotated and translated incorrectly. It's the same for vehicle and south_1. See `outputs/multiview/less_images_2_trail_coarse_train_render`.
    - At least I have a few points in the projection view, so Gaussian splats densify (although a bit late).
    - To achieve proper 3D geometric consistency with geo-consistent splats, correct extrinsics matrix conversion between sensors needs to be performed. I have tried conversions from Sanath, Lei, and mine until now.

3) **Field of View (FOV):**
    - For now, as of May 13, FOV is hardcoded. Need to update the data-parsing pipeline to handle it correctly.

4) **Dataloader Issues:**
    - Reset the dataloader into a random dataloader. Not sure what is causing this. It might be affecting training time, causing NaN, or affecting my SIBR viewer for visualization.
    - If I use `--configs arguments/hypernerf/default.py` for training, it fixes the issue of the random dataloader (not sure how, but it works). I will check training parameters later.
    - Commenting out `dataloader=true` in `arguments/hypernerf/default.py:13` might also fix the issue.

5) **Static Gaussian Splats:**
    - Until 3000 iterations, static Gaussian splats work properly with a PSNR of up to 27.

6) **Dynamic Gaussian Splats and NaN Issues:**
    - As soon as it starts including time render dine stage and densifying dynamic Gaussian splats (from 3000 to 15000 iterations), I end up with a NaN at the 180th iteration, now at the 500th iteration.
    - As of May 13, still trying to figure out what is causing the NaN to occur at exactly 500 iterations.
    - Hacks:
        - Skip and change the loss formula for that iteration. This does not work since in the next steps, Gaussian densification points or shape changes as it is not updated.
        - Scaling loss by a factor of 1000 to avoid small number convergence (need to check if this works). Found out `tv_loss` is causing the NaN thanks to hack 2.
        - Probably, time_regularization is not proper.
    
    **Attempt 2:**
    - Set `no_dr` and `no_ds` to true in `arguments/__init__.py` to check if NaN is avoided by using this. This works (no NaN), but the PSNR drops abruptly and never recovers.
    - Set `render_process` to false for faster training.
    - `scene/deformation.py:120` is important: How to increase PSNR and what happens with `no_ds` and `no_dr`.
    - Initial understanding: The bounds of the scene might be causing the problem.

    **Attempt 3:**
    - Trying to use only two views to avoid deformation problems along with scaling and rotation.
    - Changed coarse iteration to 200 and batch size in `arguments/hypernerf/default.py:20`.
    - Changed `gaussian_renderer/__init__.py:86`.
    - Is NaN caused by static Gaussians not being trained properly when going for the fine stage?
    - Is my val poses 300 too much?
    - I get NaN because of `scene/hexplane.py:152`, caused by bounds.

7) **SIBR Viewer Issues:**
    - SIBR viewer is now not working, so I cannot visualize the training process, making my life more difficult.
    - I might have been using the corrupted SIBR built app as I am just copying it from my previous repos.
    - A fresh install of SIBR viewer with dynamic rendering support needs to be configured.

## Coupling LiDAR and Camera Tightly for Gaussian Initialization

1) Using `camera_to_lidar_extrinsics` at `scene/multipleview_dataset.py:111` results in proper alignment. See `/home/uchihadj/CVPR_2025/4DGaussians/arguments/multipleview/perfect_lidar_and_south_camera_init.jpg`.

2) Please do the same for south_1 and vehicle.

3) Best progress I have made all day: Please use `/home/uchihadj/CVPR_2025/4DGaussians/output/multipleview/SIBR_viewer` for LiDAR-based Gaussian for infrastructure sensors.

4) Start debugging NaN by solving rotation and translation errors in the data loading. A backup of this moment is kept at `backup_multiview_alligned_lidar_infra.py`.

5) NaN is caused by transformation errors from the previous code. Hope this script now works -> No NaN when using two infra and updated pose in video_cams.

**PS:** When will I solve all this problem?

**For multipleviews scenes:**
If you want to train your own dataset of multipleviews scenes,you can orginize your dataset as follows:

```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
```
After that,you can use the  `multipleviewprogress.sh` we provided to generate related data of poses and pointcloud.You can use it as follows:
```bash
bash multipleviewprogress.sh (youe dataset name)
```
You need to ensure that the data folder is orginized as follows after running multipleviewprogress.sh:
```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
│   	  | sparse_
│     		  ├── cameras.bin
│     		  ├── images.bin
│     		  ├── ...
│   	  | points3D_multipleview.ply
│   	  | poses_bounds_multipleview.npy
```


## Training


For training on Tumtraf scenes,you are supposed to build a configuration file named (you dataset name).py under "./arguments/mutipleview",after that,run
```python
python train.py -s  data/multipleview/(your dataset name) --port 6017 --expname "multipleview/(your dataset name)" --configs arguments/multipleview/(you dataset name).py 
```


For your custom datasets, install nerfstudio and follow their [COLMAP](https://colmap.github.io/) pipeline. You should install COLMAP at first, then:

```python
pip install nerfstudio
# computing camera poses by colmap pipeline
ns-process-data images --data data/your-data --output-dir data/your-ns-data
cp -r data/your-ns-data/images data/your-ns-data/colmap/images
python train.py -s data/your-ns-data/colmap --port 6017 --expname "custom" --configs arguments/hypernerf/default.py 
```
You can customize your training config through the config files.

## Checkpoint

Also, you can training your model with checkpoint.

```python
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py --checkpoint_iterations 200 # change it.
```

Then load checkpoint with:

```python
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py --start_checkpoint "output/dnerf/bouncingballs/chkpnt_coarse_200.pth"
# finestage: --start_checkpoint "output/dnerf/bouncingballs/chkpnt_fine_200.pth"
```

## Rendering

Run the following script to render the images.

```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py  &
```

## Evaluation

You can just run the following script to evaluate the model.

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```


## Viewer
[Watch me](./docs/viewer_usage.md)
## Scripts

There are some helpful scripts, please feel free to use them.

`export_perframe_3DGS.py`:
get all 3D Gaussians point clouds at each timestamps.

usage:

```python
python export_perframe_3DGS.py --iteration 14000 --configs arguments/dnerf/lego.py --model_path output/dnerf/lego 
```

You will a set of 3D Gaussians are saved in `output/dnerf/lego/gaussian_pertimestamp`.

`weight_visualization.ipynb`:

visualize the weight of Multi-resolution HexPlane module.

`merge_many_4dgs.py`:
merge your trained 4dgs.
usage:

```python
export exp_name="dynerf"
python merge_many_4dgs.py --model_path output/$exp_name/sear_steak
```

`colmap.sh`:
generate point clouds from input data

```bash
bash colmap.sh data/hypernerf/virg/vrig-chicken hypernerf 
bash colmap.sh data/dynerf/sear_steak llff
```

**Blender** format seems doesn't work. Welcome to raise a pull request to fix it.

`downsample_point.py` :downsample generated point clouds by sfm.

```python
python scripts/downsample_point.py data/dynerf/sear_steak/colmap/dense/workspace/fused.ply data/dynerf/sear_steak/points3D_downsample2.ply
```


## Contributions

**This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.**

---

Some source code of ours is borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [k-planes](https://github.com/Giodiro/kplanes_nerfstudio),[HexPlane](https://github.com/Caoang327/HexPlane), [TiNeuVox](https://github.com/hustvl/TiNeuVox). We sincerely appreciate the excellent works of these authors.



## Acknoledgements

Some insights about neural voxel grids and dynamic scenes reconstruction originate from [TiNeuVox](https://github.com/hustvl/TiNeuVox). If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.

```
@article{wu20234dgaussians,
  title={4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
  author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei Wei and Liu, Wenyu and Tian, Qi and Wang Xinggang},
  journal={arXiv preprint arXiv:2310.08528},
  year={2023}
}

@inproceedings{TiNeuVox,
  author = {Fang, Jiemin and Yi, Taoran and Wang, Xinggang and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Nie\ss{}ner, Matthias and Tian, Qi},
  title = {Fast Dynamic Radiance Fields with Time-Aware Neural Voxels},
  year = {2022},
  booktitle = {SIGGRAPH Asia 2022 Conference Papers}
}
```
