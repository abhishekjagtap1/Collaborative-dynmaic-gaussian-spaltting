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
    parser.add_argument("file_path", type=str, help="Path to the .pcd file")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for downsampling")

    args = parser.parse_args()

    # Load the point cloud
    pcd = load_point_cloud(args.file_path)

    # Voxel downsampling
    pcd = voxel_downsample_point_cloud(pcd, args.voxel_size)

    # Visualize the point cloud
    visualize_point_cloud(pcd)
