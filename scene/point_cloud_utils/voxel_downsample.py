import open3d as o3d
import numpy as np
import argparse

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def voxel_downsample_point_cloud(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, voxel downsample, and visualize a point cloud")
    #parser.add_argument("file_path", type=str, help="Path to the .pcd file")
    parser.add_argument("--voxel_size", type=float, default=1, help="Voxel size for downsampling")

    args = parser.parse_args()

    # Load the point cloud
    #pcd = load_point_cloud("/home/uchihadj/Uchiha_MODE_Oct/Scene_pedestrians/Scene_234/lidar_init/gaussian_init/gaussian_init_scene234.ply")
    #pcd = load_point_cloud("/home/uchihadj/Uchiha_MODE_Oct/Scene_pedestrians/resuöts/All_collab_static_gaussian_pc/point_cloud.ply")
    pcd = load_point_cloud(
        "/home/uchihadj/Failure_mode/Collaborative-dynmaic-gaussian-spaltting/data/Bicycle_occlusion/lidar_init/bicycle_vehicle_south_2_south_1_north_1_coloured.ply")
    # Voxel downsampling
    downsampled_pcd = voxel_downsample_point_cloud(pcd, args.voxel_size)

    min_bound = downsampled_pcd.get_min_bound()
    max_bound = downsampled_pcd.get_max_bound()

    print(f"Original bounds:\nMin bound: {min_bound}\nMax bound: {max_bound}")

    # Calculate the center of the point cloud
    center = (min_bound + max_bound) / 2

    print(f"Center of the point cloud: {center}")

    # Visualize the point cloud
    visualize_point_cloud(downsampled_pcd)
    output_file = "/home/uchihadj/Failure_mode/Collaborative-dynmaic-gaussian-spaltting/data/Bicycle_occlusion/lidar_init/downsampled_0.5_filtered_pc_both_infra_coloured_combined_point_cloud.ply"
    #o3d.io.write_point_cloud(output_file, downsampled_pcd)