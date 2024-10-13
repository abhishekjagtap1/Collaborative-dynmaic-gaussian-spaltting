"""import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from yolov5 import YOLOv5
from argparse import ArgumentParser
# Initialize YOLOv5 for object detection
yolo = YOLOv5('/home/uchihadj/ECCV_workshop/4DGaussians/utils/benchmarking_utils/yolov5m.pt')  # You can use yolov5m or yolov5l for more accuracy

if __name__ == '__main__':
    parser = ArgumentParser(description="SAM with object detection for truck segmentation")

    parser.add_argument("--image_root", default='/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_2_sec_benchmarking/cam01', type=str)
    parser.add_argument("--sam_checkpoint_path", default='/home/uchihadj/ECCV_workshop/4DGaussians/utils/benchmarking_utils/sam_vit_h_4b8939.pth',
                        type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default=8, type=int)

    args = parser.parse_args()

    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    IMAGE_DIR = args.image_root
    assert os.path.exists(IMAGE_DIR), "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'truck_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Extracting truck masks using YOLOv5 and SAM...")

    for path in tqdm(sorted(os.listdir(IMAGE_DIR))):
        name = path.split('.')[0]
        img_path = os.path.join(IMAGE_DIR, path)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use YOLOv5 to detect objects, specifically trucks (class 7 in COCO dataset)
        results = yolo.predict(img_rgb)

        # Filter for 'truck' (COCO class ID for truck is 7)
        for detection in results.pred[0]:
            class_id = int(detection[5])
            if class_id == 7:  # Class ID for truck
                bbox = detection[:4].cpu().numpy()  # Extract the bounding box [x1, y1, x2, y2]
                print(f"Truck detected in {name} with bounding box: {bbox}")

                # Convert the bounding box format from YOLO to SAM (x1, y1, x2, y2 -> x, y, w, h)
                x1, y1, x2, y2 = bbox
                input_box = np.array([x1, y1, x2, y2])

                # Use SAM to predict the mask for the truck within the bounding box
                predictor.set_image(img_rgb)
                masks, _, _ = predictor.predict(box=input_box)

                # Visualize and save the mask
                mask_np = masks[0]  # Use the first mask if there are multiple
                plt.figure(figsize=(10, 10))
                plt.imshow(mask_np, cmap='gray')
                plt.title(f"Truck Mask - {name}")
                plt.axis('off')
                plt.show()

                # Save the mask as a .npy file
                save_path_npy = os.path.join(OUTPUT_DIR, f"{name}_truck_mask.npy")
                np.save(save_path_npy, mask_np)
                print(f"Truck mask saved as {save_path_npy}")

                # Optionally, save the mask as a PNG image
                mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))  # Convert to uint8
                save_path_png = os.path.join(OUTPUT_DIR, f"{name}_truck_mask.png")
                mask_image.save(save_path_png)
                print(f"Truck mask saved as {save_path_png}")

        print(f"No truck detected in {name}")

"""
import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from yolov5 import YOLOv5
from argparse import ArgumentParser

# Initialize YOLOv5 for object detection
yolo = YOLOv5('/home/uchihadj/ECCV_workshop/4DGaussians/utils/benchmarking_utils/yolov5m.pt')  # YOLOv5 model path

if __name__ == '__main__':
    parser = ArgumentParser(description="SAM with object detection for vehicles and pedestrian segmentation")

    parser.add_argument("--image_root", default='/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_2_sec_benchmarking/cam02', type=str)
    parser.add_argument("--sam_checkpoint_path", default='/home/uchihadj/ECCV_workshop/4DGaussians/utils/benchmarking_utils/sam_vit_h_4b8939.pth',
                        type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default=8, type=int)

    args = parser.parse_args()

    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    IMAGE_DIR = args.image_root
    assert os.path.exists(IMAGE_DIR), "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'vehicle_pedestrian_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Extracting vehicle and pedestrian masks using YOLOv5 and SAM...")

    for path in tqdm(sorted(os.listdir(IMAGE_DIR))):
        name = path.split('.')[0]
        img_path = os.path.join(IMAGE_DIR, path)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use YOLOv5 to detect objects: vehicles (cars, buses, trucks, motorcycles, bicycles) and pedestrians (person)
        results = yolo.predict(img_rgb)

        # Initialize an empty mask for vehicles and pedestrians
        combined_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)

        # Define COCO class IDs for vehicles and pedestrians
        vehicle_pedestrian_class_ids = [0, 1, 2, 3, 6, 7]  # 0: person, 1: bicycle, 2: car, 3: motorcycle, 6: bus, 7: truck

        # Process each detection by YOLO
        for detection in results.pred[0]:
            class_id = int(detection[5])
            if class_id in vehicle_pedestrian_class_ids:
                object_type = 'person' if class_id == 0 else 'vehicle'
                bbox = detection[:4].cpu().numpy()  # Extract the bounding box [x1, y1, x2, y2]
                print(f"{object_type.capitalize()} detected in {name} with bounding box: {bbox}")

                # Convert the bounding box format from YOLO to SAM (x1, y1, x2, y2 -> x, y, w, h)
                x1, y1, x2, y2 = bbox
                input_box = np.array([x1, y1, x2, y2])

                # Use SAM to predict the mask for the object within the bounding box
                predictor.set_image(img_rgb)

                # Use SAM to get the mask for this object
                masks, _, _ = predictor.predict(box=input_box)

                # Add the mask for the detected object to the combined mask
                mask_np = masks[0]  # Use the first mask if there are multiple
                combined_mask = np.maximum(combined_mask, mask_np.astype(np.uint8))  # Combine the masks

        # After processing all detections, save the combined mask as a .npy file
        save_path_npy = os.path.join(OUTPUT_DIR, f"{name}_vehicle_pedestrian_mask.npy")
        np.save(save_path_npy, combined_mask)
        print(f"Combined vehicle and pedestrian mask saved as {save_path_npy}")

        # Optionally, save the combined mask as a PNG image
        mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))  # Convert to uint8
        save_path_png = os.path.join(OUTPUT_DIR, f"{name}_vehicle_pedestrian_mask.png")
        mask_image.save(save_path_png)
        print(f"Combined vehicle and pedestrian mask saved as {save_path_png}")

        print(f"Finished processing {name}")
