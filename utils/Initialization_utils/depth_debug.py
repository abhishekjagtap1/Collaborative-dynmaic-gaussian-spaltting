import cv2
import numpy as np
import os

# Load the image
depth_image_path = '/home/uchihadj/CVPR_2025/4DGaussians/data/1688625741_252177946_vehicle_camera_basler_16mm.png'  # Replace with your depth image path
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# Convert to a NumPy array
depth_array = np.array(depth_image, dtype=np.float32)

# Generate the output .npy filename
npy_filename = os.path.splitext(depth_image_path)[0] + '.npy'

# Save the NumPy array as a .npy file
np.save(npy_filename, depth_array)

print(f"Depth image saved as: {npy_filename}")
