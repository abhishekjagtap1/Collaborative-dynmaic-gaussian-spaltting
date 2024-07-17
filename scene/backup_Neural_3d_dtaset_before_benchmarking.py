import concurrent.futures
import gc
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses



def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    video_images_path = video_path.split('.')[0]
    image_path = os.path.join(video_images_path,"images")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
        while video_frames.isOpened():
            ret, video_frame = video_frames.read()
            if ret:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = Image.fromarray(video_frame)
                if downsample != 1.0:

                    img = video_frame.resize(img_wh, Image.LANCZOS)
                img.save(os.path.join(image_path,"%04d.png"%count))

                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1
            else:
                break

    else:
        images_path = os.listdir(image_path)
        images_path.sort()

        for path in images_path:
            img = Image.open(os.path.join(image_path,path))
            if downsample != 1.0:
                img = img.resize(img_wh, Image.LANCZOS)
                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1

    video_frames.release()
    print(f"Video {video_path} processed.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1] , img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs


def generate_novel_poses_infra(extrinsic_matrix, num_poses=30, translation_variation=0.1, rotation_variation=0.1):
    def perturb_matrix(matrix, translation_variation, rotation_variation):
        # Generate random small translations
        translation_perturbation = np.random.uniform(-translation_variation, translation_variation, size=(3,))

        # Generate random small rotations
        angle_x = np.random.uniform(-rotation_variation, rotation_variation)
        angle_y = np.random.uniform(-rotation_variation, rotation_variation)
        angle_z = np.random.uniform(-rotation_variation, rotation_variation)

        # Construct rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Combine rotations
        rotation_perturbation = Rz @ Ry @ Rx

        # Apply perturbations to the extrinsic matrix
        perturbed_matrix = np.copy(matrix)
        perturbed_matrix[:3, :3] = rotation_perturbation @ perturbed_matrix[:3, :3]
        perturbed_matrix[:3, 3] += translation_perturbation

        return perturbed_matrix

    novel_poses = [perturb_matrix(extrinsic_matrix, translation_variation, rotation_variation) for _ in
                   range(num_poses)]

    return novel_poses

def generate_translated_poses(extrinsic_matrix, translations):
    poses = []
    for translation in translations:
        perturbed_matrix = np.copy(extrinsic_matrix)
        perturbed_matrix[:3, 3] += translation
        poses.append(perturbed_matrix)
    return poses

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


def generate_spiral_poses_vehicle(extrinsic_matrix, num_poses=20, translation_step=0.05, rotation_step=2):
    poses = []
    for i in range(num_poses):
        angle = i * rotation_step * (np.pi / 180)  # Convert degrees to radians
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        translation_vector = np.array([
            [1, 0, 0, translation_step * i],
            [0, 1, 0, translation_step * i],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        perturbed_matrix = np.matmul(rotation_matrix, extrinsic_matrix)
        perturbed_matrix = np.matmul(translation_vector, perturbed_matrix)

        poses.append(perturbed_matrix)
    return poses

"""
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"Here we go again", i want to go to ECCV2025, Render novel poses for visualization -June 10 2024
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""


def generate_zoom_extrinsics(initial_extrinsic, steps=30, zoom_distance=1.0, vehicle=bool):
    # Extract the rotation matrix and translation vector
    rotation_matrix = initial_extrinsic[:3, :3]
    translation_vector = initial_extrinsic[:3, 3]

    # Get the camera viewing direction (negative z-axis in camera space)
    #viewing_direction = rotation_matrix[:, 2]  # Third column of the rotation matrix
    if vehicle:
        viewing_direction = rotation_matrix[:, 1]
    else:
        viewing_direction = rotation_matrix[:, 1]


    # Create a list to hold the new extrinsic matrices
    extrinsics_list = []

    # Generate new extrinsic matrices by moving the camera along the viewing direction
    for i in range(steps):
        # Calculate the new translation vector
        new_translation_vector = translation_vector + (i / steps) * zoom_distance * viewing_direction

        # Construct the new extrinsic matrix
        new_extrinsic = np.eye(4, dtype=np.float32)
        new_extrinsic[:3, :3] = rotation_matrix
        new_extrinsic[:3, 3] = new_translation_vector

        # Append the new extrinsic matrix to the list
        extrinsics_list.append(new_extrinsic)

    return extrinsics_list


def reorder_extrinsic_matrices(image_poses):
    # Split the list into the first 30 and the remaining elements

    first_30 = image_poses[:30]
    remaining = image_poses[30:]

    #first_30 = image_poses[:20]
    #remaining = image_poses[20:]

    # Reverse the first 30 elements
    first_30_reversed = first_30[::-1]
    #remaining = remaining[::-1]

    # Concatenate the reversed first 30 elements with the remaining elements
    reordered_poses = first_30_reversed + remaining
    #reordered_poses = first_30_reversed + remaining_reversed -> Not good

    return reordered_poses

def compute_intermediate_matrices(E1, E2, num_intermediates=20):
    """
    Compute intermediate extrinsic matrices between E1 and E2.

    Parameters:
    E1 (numpy.ndarray): The first extrinsic matrix (4x4).
    E2 (numpy.ndarray): The second extrinsic matrix (4x4).
    num_intermediates (int): The number of intermediate matrices to compute (default is 20).

    Returns:
    list: A list of intermediate extrinsic matrices.
    """
    intermediate_matrices = []
    for i in range(1, num_intermediates + 1):
        t = i / (num_intermediates + 1.0)
        E_i = E1 + t * (E2 - E1)
        intermediate_matrices.append(E_i)
    return intermediate_matrices

# Example usage
def compute_intermediate_matrices_novel(E1, E2, num_intermediates=8):
    """
    Compute intermediate extrinsic matrices between E1 and E2.

    Parameters:
    E1 (numpy.ndarray): The first extrinsic matrix (4x4).
    E2 (numpy.ndarray): The second extrinsic matrix (4x4).
    num_intermediates (int): The number of intermediate matrices to compute (default is 8).

    Returns:
    list: A list of intermediate extrinsic matrices including E1 and E2.
    """
    intermediate_matrices = [E1]
    for i in range(1, num_intermediates + 1):
        t = i / (num_intermediates + 1.0)
        E_i = E1 + t * (E2 - E1)
        intermediate_matrices.append(E_i)
    intermediate_matrices.append(E2)
    return intermediate_matrices

import numpy as np
from scipy.spatial.transform import Rotation as R

def slerp(q1, q2, alpha):
    """ Spherical linear interpolation (slerp) between two quaternions """
    dot = np.dot(q1, q2)

    # Clamp dot product to stay within the domain of acos
    dot = np.clip(dot, -1.0, 1.0)

    # If the dot product is very close to 1, use linear interpolation
    if dot > 0.9995:
        result = (1.0 - alpha) * q1 + alpha * q2
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2

def generate_intermediate_posesshortest(E1, E2, n_poses=12):
    # Extract rotation matrices and translation vectors
    R1 = E1[:3, :3]
    t1 = E1[:3, 3]

    R2 = E2[:3, :3]
    t2 = E2[:3, 3]

    # Convert rotation matrices to quaternions
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    poses = []

    for i in range(n_poses):
        alpha = i / (n_poses - 1)  # interpolation factor
        # Spherical linear interpolation (slerp) for quaternions
        q_interpolated = slerp(q1, q2, alpha)
        # Linear interpolation for translation vectors
        t_interpolated = (1 - alpha) * t1 + alpha * t2
        # Convert the quaternion back to a rotation matrix
        R_interpolated = R.from_quat(q_interpolated).as_matrix()
        # Form the transformation matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_interpolated
        pose[:3, 3] = t_interpolated
        poses.append(pose)

    return poses


class Neural3D_NDC_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        bd_factor=0.75,
        eval_step=1,
        eval_index=0,
        sphere_scale=1.0,
    ):
        self.img_wh = (
            int(1920 / downsample),
            int(1200 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.split = split
        self.downsample = 1 #2704 / self.img_wh[0]
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 1.0#0.0
        self.far = 100 #Please look into it later
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False

        self.load_meta(datadir)
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self, datadir):
        """
        Load meta data from the dataset N3D
        """
        """
        Render Novel val poses is still pending please dont use the below val poses for training
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        #poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        poses = poses_arr[:, :-4].reshape([-1, 3, 5])
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))
        videos = sorted(videos)
        # breakpoint()
        #assert len(videos) == poses_arr.shape[0]


        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = [focal, focal]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # poses, _ = center_poses(
        #     poses, self.blender2opencv
        # )  # Re-center poses so that the average is near the center.

        # near_original = self.near_fars.min()
        # scale_factor = near_original * 0.75
        # self.near_fars /= (
        #     scale_factor  # rescale nearest plane so that it is at z = 4/3.
        # )
        # poses[..., 3] /= scale_factor

        # Sample N_views poses for validation - NeRF-like camera trajectory.
        N_views = 300
        self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views)
        # self.val_poses = self.directions
        W, H = self.img_wh
        poses_i_train = []

        for i in range(len(poses)):
            if i != self.eval_index:
                poses_i_train.append(i)
        self.poses = poses[poses_i_train]
        self.poses_all = poses
        self.image_paths, self.image_poses, self.image_times, N_cam, N_time, self.depth_paths = self.load_images_path(datadir, self.split)

        """
        Render novel poses for validation 
        
        pose_south2 = self.image_poses[0]
        pose_vehicle = self.image_poses[-1]

        zoom_extrinsics_south = generate_zoom_extrinsics(extrinsic_south_2, steps=len(image_files), zoom_distance=3.0,
                                                   vehicle=False)
        """


        self.cam_number = N_cam
        self.time_number = N_time
    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times
    def load_images_path(self,datadir,split):
        image_paths = []
        image_poses = []
        image_times = []
        depth_paths = []
        N_cams = 0
        N_time = 0
        countss = 300
        #nb_images = 10

        for root, dirs, files in os.walk(datadir):
            for dir in dirs:
                if dir == "cam01":  # South 2
                    N_cams += 1
                    image_folders = os.path.join(root, dir)
                    image_files = sorted(os.listdir(image_folders))
                    this_count = 0
                    #image_files = image_files[:20] #hardcode to take only first 20 samples
                    #image_files = image_files[20:50]
                    #scen-1 divide by 30
                    #image_files = image_files[:20] #hardcode to take only first 20 samples
                    #image_files = image_files[40:50] #no nan

                    #image_files = image_files[50:]
                    #Benchmarking ge use madthini ivaga



                    for img_file in image_files:
                        if this_count >= countss: break
                        img_index = image_files.index(img_file)
                        images_path = os.path.join(image_folders, img_file)
                        if self.depth_data:
                            depth_folder = os.path.join(root, "cam01_depth")
                            depth_files = sorted(os.listdir(depth_folder))
                            depth_files = depth_files[:20]
                            assert len(depth_files) == len(image_files)
                            for depth_file in depth_files:
                                depth_path = os.path.join(depth_folder, depth_file)

                        intrinsic_south_2 = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                            [0.0, 1362.7757568359375, 580.6482296623581],
                                                            [0.0, 0.0, 1.0]], dtype=np.float32)
                        extrinsic_south_2 = np.asarray([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                                        [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                                        [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        south_2_proj = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                                   [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                                   [0.73326062, 0.59708904, -0.32528854, -1.30114325]],
                                                  dtype=np.float32)

                        """
                        Use this only for rendering novel poses
                        """

                        #novel_poses = generate_novel_poses_infra(extrinsic_south_2, num_poses= 30, translation_variation=0.05, rotation_variation=0.1)
                        #novel_poses = get_spiral(poses, self.near_fars, N_views=N_views)

                        #for i, pose in enumerate(novel_poses):
                         #   print(f"Pose {i + 1}:\n{pose}\n")



                        """
                        Use this for up and down poses
                        

                        translation_variations_up = [[0, i * 0.09, 0] for i in range(1, 11)]
                        translation_variations_down = [[0, -i * 0.05, 0] for i in range(1, 11)]

                        novel_poses_up = generate_translated_poses(extrinsic_south_2, translation_variations_up)
                        novel_poses_down = generate_translated_poses(extrinsic_south_2, translation_variations_down)

                        novel_poses = novel_poses_up + novel_poses_down
                        """

                        """
                        Use this for left and right
                        
                        translation_variations_left = [[-i * 0.1, 0, 0] for i in range(1, 11)]
                        translation_variations_right = [[i * 0.1, 0, 0] for i in range(1, 11)]

                        novel_poses_left = generate_translated_poses(extrinsic_south_2, translation_variations_left)
                        novel_poses_right = generate_translated_poses(extrinsic_south_2, translation_variations_right)

                        novel_poses = novel_poses_left + novel_poses_right
                        """

                        """
                        dumb hack
                        """

                        # Generate 30 extrinsic matrices with a zoom distance of 1.0 units
                        zoom_extrinsics = generate_zoom_extrinsics(extrinsic_south_2, steps=len(image_files), zoom_distance=3.0, vehicle=False)
                        #zoom_extrinsics = generate_zoom_extrinsics(extrinsic_south_2, steps=20, zoom_distance=4.0, vehicle=False)
                        update_extrinsics = zoom_extrinsics[img_index]
                        #print("I am using zoomed ones")

                        """
                        Lets try new one 
                        """
                        E2 = np.array([[-9.9610770e-01, 2.4993204e-02, -8.4530041e-02, 1.5214223e+01],
                                       [8.0638155e-02, -1.2893400e-01, -9.8836863e-01, 1.2008096e+00],
                                       [-3.5601299e-02, -9.9133861e-01, 1.2641662e-01, 5.5706013e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)

                        E1 = np.array([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                       [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                       [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                       [0., 0., 0., 1.]], dtype=np.float32)

                        E3 = np.array([[9.8894250e-01, -1.1535851e-02, -1.4785174e-01, -2.7661972e+01],
                               [1.4816706e-01, 1.1927225e-01, 9.8174328e-01, -1.7739400e+00],
                               [6.3093919e-03, -9.9279499e-01, 1.1966250e-01, 3.2480175e+01],
                               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
                              dtype=np.float32)

                        #intermediate_matrices = compute_intermediate_matrices(E1, E2, num_intermediates=len(image_files))
                        #intermediate_matrices = generate_intermediate_posesshortest(E1, E2, num_intermediates=len(image_files))

                        #intermediate_matrices = compute_intermediate_matrices_novel(E2, E1, num_intermediates=20)
                        #update_inter = intermediate_matrices[img_index]






                        image_pose_dict = {
                            'intrinsic_matrix': intrinsic_south_2,
                            'extrinsic_matrix': update_extrinsics, #extrinsic_south_2,#update_inter, # #novel_poses[img_index], # #
                            "projection_matrix": south_2_proj
                        }
                        N_time = len(image_files)
                        image_paths.append(os.path.join(images_path))
                        if self.depth_data:
                            depth_paths.append(os.path.join(depth_path))

                        image_times.append(float(img_index / len(image_files)))
                        image_poses.append(image_pose_dict)

                        this_count += 1

                if dir == "cam02":
                    N_cams += 1
                    image_folders = os.path.join(root, dir)
                    image_files = sorted(os.listdir(image_folders))
                    metadata_folders = os.path.join(root, "vehicle_meta_data")
                    meta_files = sorted(os.listdir(metadata_folders))
                    assert len(meta_files) == len(image_files)
                    #image_files = image_files[:20]
                    #meta_files = meta_files[:20]

                    #image_files = image_files[20:50]
                    #meta_files = meta_files[20:50]

                    #For scene1 night
                    #image_files = image_files[25:35]
                    #meta_files = meta_files[25:35]


                    #image_files = image_files[50:]
                    #meta_files = meta_files[50:]
                    for img_file, meta_file in zip(image_files, meta_files):
                        images_path = os.path.join(image_folders, img_file)
                        img_index = image_files.index(img_file)
                        metadata_path = os.path.join(metadata_folders, meta_file)
                        intrinsic_vehicle = np.asarray([[2726.55, 0.0, 685.235],
                                                        [0.0, 2676.64, 262.745],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
                        vehicle_cam_to_lidar = np.asarray([[0.12672871, 0.12377692, 0.9841849, 0.14573078],  # TBD
                                                           [-0.9912245, -0.02180046, 0.13037732, 0.19717109],
                                                           [0.03759337, -0.99207014, 0.11992808, -0.02214238],
                                                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        if self.depth_data:
                            depth_folder = os.path.join(root, "cam02_depth")
                            depth_files = sorted(os.listdir(depth_folder))
                            depth_files = depth_files[:20]
                            assert len(depth_files) == len(image_files)
                            for depth_file in depth_files:
                                depth_path = os.path.join(depth_folder, depth_file)

                        import json
                        with open(metadata_path, 'r') as file:
                            calib_data = json.load(file)
                            frames = calib_data.get("openlabel", {}).get("frames", {})

                            for frame_key, frame_data in frames.items():
                                transforms = frame_data.get("frame_properties", {}).get("transforms", {})

                                for transform_key, transform_data in transforms.items():
                                    matrix = transform_data.get("transform_src_to_dst", {}).get("matrix4x4")

                        vehicle_to_infra_transformation_matrix = np.array(matrix)
                        extrinsic_matrix_vehicle = np.matmul(np.linalg.inv(vehicle_cam_to_lidar),
                                                             np.linalg.inv(vehicle_to_infra_transformation_matrix))
                        vehicle_proj = np.matmul(np.hstack((intrinsic_vehicle, np.zeros((3, 1), dtype=np.float32))), extrinsic_matrix_vehicle)
                        #Vehicle transformation needs to be transposed inorder to maintaion consistentcy
                        extrinsic_v = np.eye(4).astype(np.float32)
                        extrinsic_v[:3, :3] = extrinsic_matrix_vehicle[:3, :3].transpose()
                        extrinsic_v[:3, 3] = extrinsic_matrix_vehicle[:3, 3]
                        if img_index==0:
                            save_first_pose = extrinsic_v



                        """
                        Use only for novel view rendering -> 
                        
                        translation_variations = [
                            [0.2, 0, 0],  # Right
                            [-0.2, 0, 0],  # Left
                            [0, 0.2, 0],  # Up
                            [0, -0.2, 0],  # Down
                            [0.2, 0.2, 0],  # Right and Up
                            [0.2, -0.2, 0],  # Right and Down
                            [-0.2, 0.2, 0],  # Left and Up
                            [-0.2, -0.2, 0],  # Left and Down
                            [0, 0, 0.2],  # Forward
                            [0, 0, -0.2],  # Backward
                            [0.1, 0, 0],  # Slight Right
                            [-0.1, 0, 0],  # Slight Left
                            [0, 0.1, 0],  # Slight Up
                            [0, -0.1, 0],  # Slight Down
                            [0.1, 0.1, 0],  # Slight Right and Up
                            [0.1, -0.1, 0],  # Slight Right and Down
                            [-0.1, 0.1, 0],  # Slight Left and Up
                            [-0.1, -0.1, 0],  # Slight Left and Down
                            [0, 0, 0.1],  # Slight Forward
                            [0, 0, -0.1]  # Slight Backward
                        ]
                        # novel_poses_vehicle = generate_translated_poses(extrinsic_v, translation_variations)
                        """
                        """
                        Use for up and down movments for vehicle
                        """


                        #translation_variations_up = [[0, i * 0.9, 0] for i in range(1, 21)]
                        #translation_variations_down = [[0, -i * 0.05, 0] for i in range(1, 16)]

                        #novel_poses_up = generate_translated_poses(extrinsic_v, translation_variations_up)
                        #novel_poses_down = generate_translated_poses(extrinsic_v, translation_variations_down)

                        #novel_poses_vehicle = novel_poses_up #+ novel_poses_down

                        zoom_extrinsics_ve= generate_zoom_extrinsics(save_first_pose, steps=len(image_files), zoom_distance=2.0, vehicle=True)
                        #zoom_extrinsics_ve = generate_zoom_extrinsics(save_first_pose, steps=20, zoom_distance=1.0, vehicle=True)
                        update_extrinsics_ve = zoom_extrinsics_ve[img_index]

                        """
                        Use for left and right movments for vehicle
                        

                        translation_variations_left = [[-i * 0.1, 0, 0] for i in range(1, 31)]
                        translation_variations_right = [[i * 0.1, 0, 0] for i in range(1, 31)]

                        #novel_poses_left = generate_translated_poses(extrinsic_v, translation_variations_left)
                        #novel_poses_right = generate_translated_poses(extrinsic_v, translation_variations_right)

                        novel_poses_vehicle = novel_poses_right #+novel_poses_left #
                        """

                        #novel_poses_vehicle = generate_novel_poses_infra(extrinsic_v, translation_variation=0.05, rotation_variation=0.05)

                        """
                        Lets try new one 
                        """
                        E1 = np.array([[-9.9610770e-01, 2.4993204e-02, -8.4530041e-02, 1.5214223e+01],
                                       [8.0638155e-02, -1.2893400e-01, -9.8836863e-01, 1.2008096e+00],
                                       [-3.5601299e-02, -9.9133861e-01, 1.2641662e-01, 5.5706013e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)

                        E2 = np.array([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                       [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                       [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                       [0., 0., 0., 1.]], dtype=np.float32)
                        #neww nexz_scene_20 ge e1
                        E1 = np.array([[-9.9896169e-01, 7.1184416e-03, -4.5003839e-02, 1.3543032e+01],
                                       [4.3314915e-02, -1.5806700e-01, -9.8647743e-01, -2.8491244e+00],
                                       [-1.4135795e-02, -9.8740315e-01, 1.5759440e-01, 2.6300457e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)
                        # New E1
                        E1 = np.array([[-9.9939364e-01, 3.4756299e-02, 2.2151275e-03, 1.3439852e+01],
                                       [-8.0836415e-03, -1.6962631e-01, -9.8547488e-01, -4.6583529e+00],
                                       [-3.3875708e-02, -9.8489583e-01, 1.6980423e-01, 1.5624603e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)
                        E3 = np.array([[9.8894250e-01, -1.1535851e-02, -1.4785174e-01, -2.7661972e+01],
                               [1.4816706e-01, 1.1927225e-01, 9.8174328e-01, -1.7739400e+00],
                               [6.3093919e-03, -9.9279499e-01, 1.1966250e-01, 3.2480175e+01],
                               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
                              dtype=np.float32)

                        intermediate_matrices = compute_intermediate_matrices(E3,E2, num_intermediates=len(image_files))
                        #intermediate_matrices = compute_intermediate_matrices_novel(E1, E2, num_intermediates=20)
                        update_inter_veh = intermediate_matrices[img_index]

                        image_pose_dict = {
                            'intrinsic_matrix': intrinsic_vehicle,
                            'extrinsic_matrix': update_extrinsics_ve, #update_inter_veh, #extrinsic_v, # , novel_poses_vehicle[img_index], # #
                            "projection_matrix": vehicle_proj
                        }
                        N_time = len(image_files)
                        image_paths.append(os.path.join(images_path))
                        if self.depth_data:
                            depth_paths.append(os.path.join(depth_path))
                        image_times.append(float(img_index / len(image_files)))
                        image_poses.append(image_pose_dict)

                        this_count += 1

                if dir == "cam03":  # South 2
                    N_cams += 1
                    image_folders = os.path.join(root, dir)
                    image_files = sorted(os.listdir(image_folders))
                    this_count = 0
                    image_files = image_files[:20] #hardcode to take only first 20 samples
                    for img_file in image_files:
                        if this_count >= countss: break
                        img_index = image_files.index(img_file)
                        images_path = os.path.join(image_folders, img_file)
                        if self.depth_data:
                            depth_folder = os.path.join(root, "cam01_depth")
                            depth_files = sorted(os.listdir(depth_folder))
                            depth_files = depth_files[:20]
                            assert len(depth_files) == len(image_files)
                            for depth_file in depth_files:
                                depth_path = os.path.join(depth_folder, depth_file)

                        intrinsic_south_2 = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                            [0.0, 1362.7757568359375, 580.6482296623581],
                                                            [0.0, 0.0, 1.0]], dtype=np.float32)
                        extrinsic_south_2 = np.asarray([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                                        [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                                        [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        south_2_proj = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                                   [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                                   [0.73326062, 0.59708904, -0.32528854, -1.30114325]],
                                                  dtype=np.float32)
                        image_pose_dict = {
                            'intrinsic_matrix': intrinsic_south_2,
                            'extrinsic_matrix': extrinsic_south_2,
                            "projection_matrix": south_2_proj
                        }
                        N_time = len(image_files)
                        image_paths.append(os.path.join(images_path))
                        if self.depth_data:
                            depth_paths.append(os.path.join(depth_path))

                        image_times.append(float(img_index / len(image_files)))
                        image_poses.append(image_pose_dict)

                        this_count += 1

                #image_poses = reorder_extrinsic_matrices(image_poses)
        return image_paths, image_poses,  image_times, N_cams, N_time, depth_paths #Use this for training gaussian spalatting on TUMTRAF image_poses,
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):
        img = Image.open(self.image_paths[index])
        img = img.resize(self.img_wh, Image.LANCZOS)
        if self.depth_data:
            depth_image = Image.open(self.depth_paths[index])
            depth_tensor = self.transform(depth_image).float()
            depth = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
            #np.array(cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED), dtype=np.float32))
        # depth = np.load(depth_path).astype(np.float32)
        else:
            depth = None

        img = self.transform(img)

        return img, self.image_poses[index], self.image_times[index], depth
    def load_pose(self,index):
        pose = self.image_poses[index] #
        extrinsic_matrix = pose['extrinsic_matrix']
        R = extrinsic_matrix[:3, :3] #np.transpose(extrinsic_matrix[:3, :3])
        T = extrinsic_matrix[:3, 3]
        return R, T #self.image_poses[index]

