import cv2
import os
from natsort import natsorted


# Path to the folder containing images
folder = "/home/uchihadj/ECCV_workshop/4DGaussians/output/AM_I_going_CVPR_for_real/avoidong_nan/train/ours_50000/renders"
output_folder = "/home/uchihadj/ECCV_workshop/4DGaussians/output/AM_I_going_CVPR_for_real/avoidong_nan/train/ours_50000/vis_alligner"  # Folder to save the concatenated frames

# Output video parameters
output_video_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/AM_I_going_CVPR_for_real/avoidong_nan/train/ours_50000/renders_same_pose.mp4"

"""# Path to the folder containing images
folder = "/home/uchihadj/ECCV_workshop/4DGaussians/output/CVPR_2025_Benchmark/20_40_3_AGENTS_Frame/train/ours_20000/renders"
output_folder = "/home/uchihadj/ECCV_workshop/4DGaussians/output/CVPR_2025_Benchmark/20_40_3_AGENTS_Frame/train/ours_20000/benchmark_renders_20_40_3"  # Folder to save the concatenated frames

# Output video parameters
output_video_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/CVPR_2025_Benchmark/20_40_3_AGENTS_Frame/train/ours_20000/renders.mp4
"""
fps = 5  # Frames per second for the output video
frame_size = None

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to sort images and ensure consistent order
def get_sorted_images(folder):
    # Get list of files
    files = os.listdir(folder)
    # Sort files using natsorted for natural sorting of filenames
    sorted_files = natsorted([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
    # Return full paths to the images
    return [os.path.join(folder, f) for f in sorted_files]

# Get sorted images from the folder
images = get_sorted_images(folder)

# Ensure there are exactly 60 images
num_images = len(images)
if num_images != 60:
    raise ValueError("The folder must contain exactly 60 images.")

# Reorder the images according to the desired sequence:
# - First: images from index 20 to 40
# - Second: images from index 0 to 20
# - Third: images from index 40 to 60
"""
Align according to emernerf bencmarking video
"""
left_images = images[20:40]  # Images from index 20 to 40
center_images = images[40:60]   # Images from index 0 to 20
right_images = images[0:20]  # Images from index 40 to 60

# Initialize video writer (use first image to set frame size)
first_img = cv2.imread(images[0])
h, w, _ = first_img.shape
frame_size = (w * 3, h)  # Since we are combining three images horizontally

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Process the images in the new sequence and combine them horizontally
for idx, (left_img_path, center_img_path, right_img_path) in enumerate(zip(left_images, center_images, right_images)):
    # Read the corresponding images
    img_left = cv2.imread(left_img_path)
    img_center = cv2.imread(center_img_path)
    img_right = cv2.imread(right_img_path)

    # Stack images horizontally (side-by-side: left, center, right)
    combined_img = cv2.hconcat([img_left, img_center, img_right])

    # Write the frame to the video file
    out.write(combined_img)

    # Save the concatenated frame as an image
    output_frame_path = os.path.join(output_folder, f"frame_{idx:03d}.png")
    cv2.imwrite(output_frame_path, combined_img)
    print(f"Saved: {output_frame_path}")

# Release the video writer object
out.release()
print(f"Video saved as {output_video_path}")
