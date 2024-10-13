import cv2
import os

# Paths to the three videos
video_gt = "/home/uchihadj/ECCV_workshop/4DGaussians/output/AM_I_going_CVPR_for_real/avoidong_nan/train/Same_pose_ours_50000/renders_same_pose_gt.mp4"
video_render = "/home/uchihadj/ECCV_workshop/4DGaussians/output/AM_I_going_CVPR_for_real/avoidong_nan/train/Same_pose_ours_50000/renders_same_pose.mp4"
video_depth = "/home/uchihadj/ECCV_workshop/4DGaussians/output/AM_I_going_CVPR_for_real/avoidong_nan/train/Same_pose_ours_50000/renders_same_pose_depth.mp4"

# Output video path
output_video_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/AM_I_going_CVPR_for_real/avoidong_nan/train/Same_pose_ours_50000/Benchmark_video.mp4"

# Open the three videos
cap_gt = cv2.VideoCapture(video_gt)
cap_render = cv2.VideoCapture(video_render)
cap_depth = cv2.VideoCapture(video_depth)

# Get video properties from one of the videos (assuming they all have the same FPS and dimensions)
fps = cap_gt.get(cv2.CAP_PROP_FPS)
width = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
frame_size = (width, height * 3)  # Stack height-wise (three videos vertically)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Loop to read frames from each video
while cap_gt.isOpened() and cap_render.isOpened() and cap_depth.isOpened():
    # Read frames from each video
    ret_gt, frame_gt = cap_gt.read()
    ret_render, frame_render = cap_render.read()
    ret_depth, frame_depth = cap_depth.read()

    # Break the loop if any video ends
    if not ret_gt or not ret_render or not ret_depth:
        break

    # Stack frames vertically (top to bottom: gt, render, depth)
    stacked_frame = cv2.vconcat([frame_gt, frame_depth,  frame_render])

    # Write the stacked frame to the output video
    out.write(stacked_frame)

# Release everything
cap_gt.release()
cap_render.release()
cap_depth.release()
out.release()

print(f"Video saved as {output_video_path}")
