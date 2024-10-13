import os
import numpy as np
import cv2

"""# Define paths
rgb_folder = '/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/Semantic_20_3_sec_benchmarking_copy/cam01/images'  # Update with your RGB images folder path
mask_folder = '/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/Semantic_20_3_sec_benchmarking_copy/cam01/truck_masks'  # Update with your mask files folder path
output_folder = '/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/Semantic_20_3_sec_benchmarking_copy/cam01/images/new_rgb'  # Update with your output folder path
"""
# Define paths
rgb_folder = '/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_2_sec_benchmarking/cam01'  # Update with your RGB images folder path
mask_folder = '/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_2_sec_benchmarking/cam01/vehicle_pedestrian_masks'  # Update with your mask files folder path
output_folder = '/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_2_sec_benchmarking/cam01/new_rgb'  # Update with your output folder path


#output_folder = 'path/to/output/folder'  # Update with your output folder path

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through the RGB images
for filename in os.listdir(rgb_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
        # Load RGB image
        rgb_image_path = os.path.join(rgb_folder, filename)
        rgb_image = cv2.imread(rgb_image_path)

        # Create corresponding mask filename
        mask_filename = filename.split('.')[0] + "_vehicle_pedestrian_mask.npy" ##_truck_mask.npy'  # Get the base name and append '_truck_mask.npy'
        mask_file_path = os.path.join(mask_folder, mask_filename)

        # Load corresponding mask
        if os.path.exists(mask_file_path):  # Check if the mask file exists
            mask = np.load(mask_file_path)

            # Ensure mask is binary (0s and 1s)
            if mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)

            # Create an output image by applying the mask
            masked_image = rgb_image.copy()
            masked_image[mask == 0] = [0, 0, 0]  # Set background to black (or transparent)

            # Save the masked image
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, masked_image)
        else:
            print(f"Mask file not found for {filename}: {mask_filename}")

print("Masked images created and saved in the output folder.")

