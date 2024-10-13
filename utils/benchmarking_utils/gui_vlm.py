import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, simpledialog
import clip
from segment_anything import SamPredictor, sam_model_registry

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load SAM model
model_type = "vit_h"  # SAM architecture
sam_checkpoint_path = "/home/uchihadj/ECCV_workshop/4DGaussians/utils/benchmarking_utils/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device)
predictor = SamPredictor(sam)

# Initialize the GUI window
class MaskSelectorApp:
    def __init__(self, root, img_path):
        self.root = root
        self.root.title("Select Mask Based on Prompt")
        self.img_path = img_path

        self.image = cv2.imread(self.img_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_pil = Image.fromarray(self.image_rgb)

        # Entry for text prompt
        self.prompt_label = tk.Label(root, text="Enter your prompt (e.g., 'truck'):")
        self.prompt_label.pack()

        self.prompt_entry = tk.Entry(root, width=50)
        self.prompt_entry.pack()

        self.submit_btn = tk.Button(root, text="Submit Prompt", command=self.get_clip_mask)
        self.submit_btn.pack()

        # Display the image (without mask first)
        self.display_image(self.image_pil)

        # Buttons for toggling masks
        self.next_btn = tk.Button(root, text="Next Mask", state=tk.DISABLED, command=self.next_mask)
        self.prev_btn = tk.Button(root, text="Previous Mask", state=tk.DISABLED, command=self.prev_mask)
        self.save_btn = tk.Button(root, text="Save Masked Image", state=tk.DISABLED, command=self.save_masked_image)

        self.next_btn.pack(side=tk.RIGHT)
        self.prev_btn.pack(side=tk.LEFT)
        self.save_btn.pack(side=tk.BOTTOM)

        self.masks = None
        self.current_mask_idx = 0

    def display_image(self, image):
        """ Display an image (PIL format) in the GUI. """
        self.tk_image = ImageTk.PhotoImage(image)
        if hasattr(self, 'img_label'):
            self.img_label.config(image=self.tk_image)
        else:
            self.img_label = tk.Label(self.root, image=self.tk_image)
            self.img_label.pack()

    def get_clip_mask(self):
        """ Get the mask corresponding to the prompt using CLIP. """
        prompt = self.prompt_entry.get()
        if not prompt:
            print("Please enter a prompt.")
            return

        # Preprocess the image for CLIP
        image_tensor = preprocess(self.image_pil).unsqueeze(0).to(device)
        text_tensor = clip.tokenize([prompt]).to(device)

        # CLIP embeddings
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            text_features = clip_model.encode_text(text_tensor)

        # Compute similarity between text and image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # For simplicity, assume we're working on the whole image
        # Normally, you'd want to crop or find a specific region based on similarity score
        input_box = [0, 0, self.image.shape[1], self.image.shape[0]]  # Using entire image
        input_box = np.array(input_box)  # Convert list to NumPy array

        # SAM segmentation on the selected region
        predictor.set_image(self.image_rgb)
        self.masks, _, _ = predictor.predict(box=input_box)

        # Initialize to show the first mask
        self.current_mask_idx = 0
        self.show_mask()

        # Enable buttons
        self.next_btn.config(state=tk.NORMAL)
        self.prev_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)

    def show_mask(self):
        """ Display current mask. """
        if self.masks is None or len(self.masks) == 0:
            print("No masks available.")
            return

        # Get the current mask and convert it to PIL format
        mask_np = self.masks[self.current_mask_idx] #.cpu().numpy()
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Masked image (image with only the selected region)
        masked_image = self.image_rgb.copy()
        masked_image[mask_np == 0] = 0  # Zero-out the background (non-mask)

        # Display masked image
        masked_image_pil = Image.fromarray(masked_image)
        self.display_image(masked_image_pil)

    def next_mask(self):
        """ Show next mask. """
        if self.masks is None:
            return
        if self.current_mask_idx < len(self.masks) - 1:
            self.current_mask_idx += 1
        self.show_mask()

    def prev_mask(self):
        """ Show previous mask. """
        if self.masks is None:
            return
        if self.current_mask_idx > 0:
            self.current_mask_idx -= 1
        self.show_mask()

    def save_masked_image(self):
        """ Save the masked image to a file. """
        if self.masks is None:
            return

        mask_np = self.masks[self.current_mask_idx].cpu().numpy()

        # Create the masked image (only truck visible)
        masked_image = self.image_rgb.copy()
        masked_image[mask_np == 0] = 0  # Zero-out the background

        # Convert to PIL and prompt user for save location
        masked_image_pil = Image.fromarray(masked_image)
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            masked_image_pil.save(save_path)
            print(f"Masked image saved to {save_path}")

# Main function to run the application
if __name__ == '__main__':
    img_path = "/home/uchihadj/ECCV_workshop/4DGaussians/data/s3_based_benchmarking/20_3_sec_benchmarking/cam01/1688626668_130252863_s110_camera_basler_south2_8mm.jpg" # Replace with the actual image path

    # Initialize Tkinter root window
    root = tk.Tk()

    # Start the MaskSelectorApp
    app = MaskSelectorApp(root, img_path)

    # Start the Tkinter event loop
    root.mainloop()
