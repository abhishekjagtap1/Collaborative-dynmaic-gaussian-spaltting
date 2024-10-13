import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.projects.maskformer import add_maskformer2_config
import cv2
import numpy as np


def setup_mask2former(cityscapes_model_weights: str = "path/to/model_weights.pth"):
    """
    Set up the Mask2Former model with the Cityscapes pretrained model.

    :param cityscapes_model_weights: Path to the Mask2Former pretrained weights for Cityscapes.
    :return: DefaultPredictor object for inference.
    """
    # Load Mask2Former configuration
    cfg = get_cfg()
    add_deeplab_config(cfg)  # Add DeepLab configuration
    add_maskformer2_config(cfg)  # Add Mask2Former-specific config

    cfg.merge_from_file("path/to/mask2former_config.yaml")  # Path to the config file for Mask2Former
    cfg.MODEL.WEIGHTS = cityscapes_model_weights  # Set path to the trained weights for Cityscapes

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for post-processing (you can adjust this)

    # Create predictor instance
    predictor = DefaultPredictor(cfg)

    return predictor


def get_semantic_segmentation(predictor, input_image_path: str):
    """
    Given an input image, return the semantic segmentation using Mask2Former.

    :param predictor: DefaultPredictor object for running inference.
    :param input_image_path: Path to the input image.
    :return: The semantic segmentation mask.
    """
    # Read and preprocess the input image
    image = cv2.imread(input_image_path)
    outputs = predictor(image)  # Run the model to get predictions

    # Extract the semantic segmentation mask from outputs
    semantic_segmentation = outputs["sem_seg"].argmax(dim=0).cpu().numpy()  # Get the predicted class for each pixel

    return semantic_segmentation


def visualize_segmentation(segmentation_mask, metadata):
    """
    Visualizes the segmentation mask with Cityscapes labels.

    :param segmentation_mask: Output segmentation mask from Mask2Former.
    :param metadata: Cityscapes metadata for coloring.
    :return: Colored segmentation image.
    """
    # Get colors for Cityscapes labels
    label_colors = metadata.stuff_colors
    label_names = metadata.stuff_classes

    colored_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(label_colors):
        colored_mask[segmentation_mask == label] = color

    return colored_mask


# Example usage in train.py or any other script
if __name__ == "__main__":
    # Path to pre-trained Cityscapes weights for Mask2Former
    pretrained_weights = "path/to/cityscapes_weights.pth"
    image_path = "path/to/input_image.jpg"

    # Setup model
    predictor = setup_mask2former(pretrained_weights)

    # Perform semantic segmentation
    segmentation_mask = get_semantic_segmentation(predictor, image_path)

    # Get Cityscapes metadata (for visualization or further use)
    cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_sem_seg_val")

    # Optional: visualize the segmentation
    colored_segmentation = visualize_segmentation(segmentation_mask, cityscapes_metadata)

    # Save or display the segmentation result
    cv2.imwrite("segmentation_output.png", colored_segmentation)

    # You can now use segmentation_mask as priors for downstream tasks
