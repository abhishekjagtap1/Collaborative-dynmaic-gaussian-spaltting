import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_3_sec_benchmarking/cam01/1688626668_130252863_s110_camera_basler_south2_8mm.jpg')

# Get the dimensions of the image
rows, cols, ch = image.shape


# Define the points for the perspective transformation
pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

# Slightly shift the points to simulate a change in perspective
pts2 = np.float32([[cols*0.1, rows*0.1], [cols*0.9, rows*0.05], [cols*0.05, rows*0.9], [cols*0.95, rows*0.95]])

# Get the perspective transformation matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the perspective transformation
dst = cv2.warpPerspective(image, M, (cols, rows))

# Add border to prevent black regions
# You can use different border types: cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, etc.
dst_filled = cv2.copyMakeBorder(dst, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

# Show the original and transformed image
cv2.imshow('Original Image', image)
cv2.imshow('Perspective Transformed Image (With Borders Filled)', dst_filled)

# Save the transformed image with filled borders
cv2.imwrite('perspective_transformed_image_filled.jpg', dst_filled)

cv2.waitKey(0)
cv2.destroyAllWindows()
