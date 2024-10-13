import torch
import clip
from PIL import Image
import cv2
from segment_anything import SamPredictor, sam_model_registry

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load SAM model
model_type = "vit_h"  # SAM architecture
sam_checkpoint_path = "/home/uchihadj/ECCV_workshop/4DGaussians/utils/benchmarking_utils/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device)
predictor = SamPredictor(sam)

# Load and preprocess image
image_path = "/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_3_sec_benchmarking/cam01/1688626668_130252863_s110_camera_basler_south2_8mm.jpg"
image = Image.open(image_path)
image_tensor = preprocess(image).unsqueeze(0).to(device)

# Define the prompt
text_prompt = ["truck"]

# Compute CLIP embeddings for image and text
with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(clip.tokenize(text_prompt).to(device))

# Find the most relevant region in the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Calculate similarity between text and image
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# The region with the highest similarity score will likely contain the "truck"
_, best_idx = similarity.max(dim=0)

# Use OpenCV to read the image in RGB
image_cv = cv2.imread(image_path)
image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# Use SAM to predict masks for the detected region
predictor.set_image(image_cv_rgb)

# Assuming that CLIP provided the coordinates of the truck, we can use a bounding box or points
# In a more sophisticated case, you would need a bounding box around the truck

input_box = [100, 200, 300, 400]  # Replace this with coordinates for the "truck"
masks, _, _ = predictor.predict(box=input_box)

# Visualize the mask
import matplotlib.pyplot as plt
plt.imshow(masks[0], cmap='gray')
plt.show()

# Save the mask or process further
masked_image = image_cv.copy()
masked_image[masks[0] == 0] = 0  # Keep only the truck
cv2.imwrite('truck_masked_image.png', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
