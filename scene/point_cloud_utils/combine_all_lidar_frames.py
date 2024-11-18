import open3d as o3d
import os


def concatenate_pcd_files(input_folder, output_file):
    # Create an empty PointCloud object
    combined_pcd = o3d.geometry.PointCloud()

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pcd'):
            # Load the point cloud file
            pcd = o3d.io.read_point_cloud(os.path.join(input_folder, filename))
            # Concatenate the point cloud to the combined point cloud
            combined_pcd += pcd

    # Save the combined point cloud
    o3d.io.write_point_cloud(output_file, combined_pcd)
    print(f"Combined point cloud saved as {output_file}")


# Specify the folder containing .pcd files and the output file path
input_folder = "/home/uchihadj/Failure_mode/Collaborative-dynmaic-gaussian-spaltting/data/Bicycle_occlusion/lidar_init/combined_lidar_frames"
output_file = "/home/uchihadj/Failure_mode/Collaborative-dynmaic-gaussian-spaltting/data/Bicycle_occlusion/lidar_init/combined_raw_lidar_output.pcd"

concatenate_pcd_files(input_folder, output_file)
