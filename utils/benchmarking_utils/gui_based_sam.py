import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from segment_anything import SamPredictor, sam_model_registry

# Load SAM model
model_type = "vit_h"  # SAM architecture
sam_checkpoint_path = "/home/uchihadj/ECCV_workshop/4DGaussians/utils/benchmarking_utils/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to('cuda')
predictor = SamPredictor(sam)

# Initialize the GUI window
class MaskSelectorApp:
    def __init__(self, root, img_path):
        self.root = root
        self.root.title("Select Truck Mask")
        self.img_path = img_path

        self.image = cv2.imread(self.img_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Get masks from SAM
        predictor.set_image(self.image_rgb)
        self.masks, _, _ = predictor.predict()

        # Display the first mask
        self.current_mask_idx = 0
        self.show_mask()

        # Create button frame for better layout control
        button_frame = tk.Frame(root)
        button_frame.grid(row=1, column=0, pady=10)

        # Buttons for Next, Previous, Save
        next_btn = tk.Button(button_frame, text="Next Mask", command=self.next_mask, width=15)
        next_btn.grid(row=0, column=1, padx=5)

        prev_btn = tk.Button(button_frame, text="Previous Mask", command=self.prev_mask, width=15)
        prev_btn.grid(row=0, column=0, padx=5)

        save_btn = tk.Button(button_frame, text="Save Masked Image", command=self.save_masked_image, width=15)
        save_btn.grid(row=0, column=2, padx=5)

    def show_mask(self):
        """ Display current mask """
        mask_np = self.masks[self.current_mask_idx] #.numpy()

        # Convert mask to an image (grayscale)
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
        self.tk_mask_image = ImageTk.PhotoImage(mask_image)

        # Create a label to display the image and mask
        if hasattr(self, 'mask_label'):
            self.mask_label.config(image=self.tk_mask_image)
        else:
            self.mask_label = tk.Label(self.root, image=self.tk_mask_image)
            self.mask_label.grid(row=0, column=0)  # Place the image at the top of the window

    def next_mask(self):
        """ Show next mask """
        if self.current_mask_idx < len(self.masks) - 1:
            self.current_mask_idx += 1
        self.show_mask()

    def prev_mask(self):
        """ Show previous mask """
        if self.current_mask_idx > 0:
            self.current_mask_idx -= 1
        self.show_mask()

    def save_masked_image(self):
        """ Save entire image with only truck visible """
        mask_np = self.masks[self.current_mask_idx].cpu().numpy()

        # Create mask where truck is visible (values of 1) and other areas are blacked out (0)
        mask_np = mask_np.astype(np.uint8)  # Convert to uint8 for masking
        masked_image = self.image.copy()
        masked_image[mask_np == 0] = 0  # Set non-truck areas to black

        # Convert masked image to PIL format for saving
        masked_image_pil = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

        # Prompt user for save location
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            masked_image_pil.save(save_path)
            print(f"Masked image saved to {save_path}")

# Main function to run the application
if __name__ == '__main__':
    img_path = "/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_3_sec_benchmarking/cam01/1688626668_130252863_s110_camera_basler_south2_8mm.jpg"  # Image to segment

    # Initialize Tkinter root window
    root = tk.Tk()

    # Start the MaskSelectorApp
    app = MaskSelectorApp(root, img_path)

    # Start the Tkinter event loop
    root.mainloop()
