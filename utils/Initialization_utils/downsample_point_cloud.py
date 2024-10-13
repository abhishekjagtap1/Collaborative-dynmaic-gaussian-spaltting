import open3d as o3d
import numpy as np
import argparse

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def downsample_point_cloud(pcd, num_points):
    return pcd.uniform_down_sample(int(len(pcd.points) / num_points))

def upsample_point_cloud(pcd, num_points):
    current_points = np.asarray(pcd.points)
    num_current_points = len(current_points)
    if num_points <= num_current_points:
        return pcd
    additional_points = num_points - num_current_points
    indices = np.random.choice(num_current_points, additional_points, replace=True)
    new_points = np.vstack((current_points, current_points[indices]))
    upsampled_pcd = o3d.geometry.PointCloud()
    upsampled_pcd.points = o3d.utility.Vector3dVector(new_points)
    return upsampled_pcd

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":


    # Load the point cloud
    pcd = load_point_cloud("/home/uchihadj/ECCV_workshop/4DGaussians/data/scene_overlapping_pedestrian/lidar_init/downsample_points_colouzred_distengled_scene.ply")

    pcd = downsample_point_cloud(pcd, 1000)
    pcd.voxel_down_sample(voxel_size=100)


    # Visualize the point cloud
    visualize_point_cloud(pcd)
