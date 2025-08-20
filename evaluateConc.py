import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm # type: ignore
from model import *  # Trained model is loaded from here

# =========================
# Paths & Constants
# =========================

DATA = "/home/xys-05/CT project/Dataset/CT_Training_Multi/23/Sabir Hussain/"
PNG_IMAGES_PATH = os.path.join(DATA, "PNG_IMAGES")
RESULT_MASKS = os.path.join(DATA, "PREDICTED_MASKS")

# Define BGR color codes for each class label
colors = {
    1: (0, 0, 255),    # Red for class 1
    2: (0, 255, 0),    # Green for class 2
    3: (205, 0, 115)   # Purple-pink for class 3
}

# =========================
# Prepare Output Directory
# =========================

# Delete previous results if any, then create a new directory
shutil.rmtree(RESULT_MASKS, ignore_errors=True)
os.makedirs(RESULT_MASKS, exist_ok=True)

# Get all PNG image paths from the source directory
PNG_IMAGES = [
    os.path.join(PNG_IMAGES_PATH, fname)
    for fname in os.listdir(PNG_IMAGES_PATH)
    if fname.endswith('.png')
]

# =========================
# Prediction and Visualization
# =========================

for img_path in tqdm(PNG_IMAGES, desc="Processing Images"):
    
    # Define path to save the final visualized image
    save_path = os.path.join(RESULT_MASKS, os.path.basename(img_path))

    # Load image and normalize to [0, 1] for model input
    input_img = cv2.imread(img_path).astype("float32") / 255.0

    # Expand dimensions for batch prediction
    input_batch = np.expand_dims(input_img, axis=0)

    # Predict the segmentation mask (shape: [H, W])
    pred = np.argmax(model.predict(input_batch), axis=3)[0]

    # Prepare original and overlay image (rescale back to 0-255)
    orig_img = (input_img * 255).astype("uint8")
    overlay_img = orig_img.copy()

    # Draw contours for each class label
    for class_label in [1, 2, 3]:
        # Create binary mask for current class
        class_mask = (pred == class_label).astype(np.uint8) * 255
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours on the overlay image
        cv2.drawContours(overlay_img, contours, -1, colors[class_label], 1)

    # =========================
    # Beautify the predicted mask
    # =========================

    # Add white borders to the mask
    border_thickness = 3
    pred[:, -border_thickness:] = 255
    pred[-border_thickness:, :] = 255
    pred[:, :border_thickness] = 255
    pred[:border_thickness, :] = 255

    # Convert class values to grayscale intensities
    mask_img = pred.copy()
    mask_img[mask_img == 1] = 80
    mask_img[mask_img == 2] = 180
    mask_img[mask_img == 3] = 255
    mask_img = mask_img.astype("uint8")

    # Stack grayscale mask into 3 channels for display
    mask_rgb = np.dstack([mask_img] * 3)

    # =========================
    # Save Combined Visualization
    # =========================

    # Concatenate original, contour, and mask images side-by-side
    final_output = np.concatenate((orig_img, overlay_img, mask_rgb), axis=1)

    # Save the visualization
    cv2.imwrite(save_path, final_output)