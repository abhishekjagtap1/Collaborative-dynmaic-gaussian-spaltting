# Important Notes:

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


## ECCV Benchmarking Time

- Use `getprojectionmatrix2` for training and evaluation. Update `NoneType` error when opening SIBR viewer using this setting.
- Do not downsample points. Instead, reduce the bounds and use point painting of the same LiDAR, not a general one.


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

## Data Preparation

**For synthetic scenes:**
The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0).

**For real dynamic scenes:**
The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets). 

Meanwhile, [Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video) could be downloaded from their official websites. To save the memory, you should extract the frames of each video and then organize your dataset as follows.

```
├── data
│   | dnerf 
│     ├── mutant
│     ├── standup 
│     ├── ...
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```

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

For training synthetic scenes such as `bouncingballs`, run

```
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
```

For training dynerf scenes such as `cut_roasted_beef`, run
```python
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# Finally, train.
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```
For training hypernerf scenes such as `virg/broom`: Pregenerated point clouds by COLMAP are provided [here](https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view). Just download them and put them in to correspond folder, and you can skip the former two steps. Also, you can run the commands directly.

```python
# First, computing dense point clouds by COLMAP
bash colmap.sh data/hypernerf/virg/broom2 hypernerf
# Second, downsample the point clouds generated in the first step. 
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply
# Finally, train.
python train.py -s  data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py 
```

For training multipleviews scenes,you are supposed to build a configuration file named (you dataset name).py under "./arguments/mutipleview",after that,run
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

## Acknowledgement

We would like to express our sincere gratitude to [@zhouzhenghong-gt](https://github.com/zhouzhenghong-gt/) for his revisions to our code and discussions on the content of our paper.

## Citation

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
