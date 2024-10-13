import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


# Load the DeepLabV3 model with pretrained weights
def load_deeplabv3_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model


# Preprocess the image before passing it to the model
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return input_tensor


# Get semantic segmentation output
def get_semantic_segmentation(model, input_image_path):
    input_tensor = preprocess_image(input_image_path)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]  # Output has shape [num_classes, H, W]

    output_predictions = output.argmax(0).cpu().numpy()  # Get the class with the highest score for each pixel
    return output_predictions


# Cityscapes color palette (19 classes, used for visualizing segmentation)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],  # Road
    [244, 35, 232],  # Sidewalk
    [70, 70, 70],  # Building
    [102, 102, 156],  # Wall
    [190, 153, 153],  # Fence
    [153, 153, 153],  # Pole
    [250, 170, 30],  # Traffic light
    [220, 220, 0],  # Traffic sign
    [107, 142, 35],  # Vegetation
    [152, 251, 152],  # Terrain
    [70, 130, 180],  # Sky
    [220, 20, 60],  # Person
    [255, 0, 0],  # Rider
    [0, 0, 142],  # Car
    [0, 0, 70],  # Truck
    [0, 60, 100],  # Bus
    [0, 80, 100],  # Train
    [0, 0, 230],  # Motorcycle
    [119, 11, 32]  # Bicycle
])


# Visualize the segmentation mask using the Cityscapes color palette
def visualize_cityscapes_segmentation(segmentation_mask):
    colored_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)

    for label in range(CITYSCAPES_PALETTE.shape[0]):
        colored_mask[segmentation_mask == label] = CITYSCAPES_PALETTE[label]

    return colored_mask


# Visualize the segmentation mask with random colors for each class
def visualize_segmentation(segmentation_mask, num_classes=21):
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colored_mask = colors[segmentation_mask]
    return colored_mask


# Example usage in a script
if __name__ == "__main__":
    # Load the DeepLabV3 model
    model = load_deeplabv3_model()

    # Path to the input image
    input_image_path = "/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_3_sec_benchmarking/cam01/1688626668_130252863_s110_camera_basler_south2_8mm.jpg"

    # Get the segmentation output
    segmentation_mask = get_semantic_segmentation(model, input_image_path)

    # Visualize the output
    colored_segmentation = visualize_cityscapes_segmentation(segmentation_mask)

    # Save or display the output
    cv2.imwrite("cityscape_segmentation_output.png", colored_segmentation)
