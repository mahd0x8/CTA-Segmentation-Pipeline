from model import *  # Import the trained model and related setup

import os
import shutil
import numpy as np
import cv2

# ==========================
# Color Map for Classes
# ==========================
# Class 1 → Red, Class 2 → Green, Class 3 → Pinkish Purple
colors = {
    1: (0, 0, 255),     # Red
    2: (0, 255, 0),     # Green
    3: (205, 0, 115)    # Magenta-like
}

# ==========================
# Input & Output Paths
# ==========================
RUN = f"EXPERIMENTS/V{1}/"
save_to    = "EXPERIMENTS/V1/Evaluation"

# ==========================
# Prepare Output Directory
# ==========================
# Remove existing output folder (if exists)
try:
    shutil.rmtree(save_to)
except FileNotFoundError:
    pass

# Create output folder
os.makedirs(save_to)

# ==========================
# Gather Image File Paths
# ==========================
#images = [os.path.join(image_path, fname) for fname in os.listdir(image_path) if fname.endswith(".png")]
split_cache = {
        "val_images":   os.path.join(RUN, "val_images.npy"),
        "val_masks":    os.path.join(RUN, "val_masks.npy"),
    }

images   = np.load(split_cache["val_images"], allow_pickle=True).tolist()
masks    = np.load(split_cache["val_masks"], allow_pickle=True).tolist()

# ==========================
# Predict and Visualize
# ==========================
for img_path in images:
    # Read and normalize image
    xray = np.array([cv2.imread(img_path) / 255.0]).astype("float32")  # Shape: (1, H, W, 3)

    # Predict segmentation mask and get class with highest probability per pixel
    y_pred_argmax = np.argmax(LOADED_MODEL.predict(xray), axis=3)  # Shape: (1, H, W)

    # Convert input image back to 0–255 scale for visualization
    xray_vis = (xray[0] * 255.0).astype(np.uint8)

    # Draw contours for each predicted class
    for class_label in [1, 2, 3]:
        # Create binary mask for current class
        class_mask = np.where(y_pred_argmax == class_label, 255, 0).astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(class_mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the image
        cv2.drawContours(xray_vis, contours, -1, colors[class_label], thickness=1)

    # Save the final visualized image
    save_name = os.path.basename(img_path)
    save_path = os.path.join(save_to, save_name)
    cv2.imwrite(save_path, xray_vis)
