from pathlib import Path
import os, cv2, shutil
import numpy as np
from tqdm import tqdm

# ----------------- Config -----------------
SRC_ROOT = Path("DATASET")                  # expects: DATASET/<study>/{MASKS,PNG_IMAGES}/*
OUT_ROOT = Path("DATASET_MASKED_SELECTIVE") # output root
MASKS_SUBDIR = "MASKS"
IMAGES_SUBDIR = "PNG_IMAGES"
OVERLAY_THICKNESS = 1                       # change to taste

# Predefined BGR colors for known labels
LABEL_COLORS = {
    255: (0, 0, 255),    # red
    80:  (0, 255, 0),    # green
    180: (205, 0, 115),  # pink/purple
}

# ----------------- Setup -----------------
(OUT_ROOT / "ALL" / "MASKS").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "ALL" / "IMAGES").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "ALL" / "OVERLAYS").mkdir(parents=True, exist_ok=True)

(OUT_ROOT / "MULTICLASS" / "MASKS").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "MULTICLASS" / "IMAGES").mkdir(parents=True, exist_ok=True)

# ----------------- Helpers -----------------
def safe_imread(path: Path, flags=1):
    img = cv2.imread(str(path), flags)
    return img

def color_for_label(lbl: int) -> tuple:
    """
    Return BGR color for a label. Use predefined if present; otherwise generate
    a distinct HSV-based color deterministically.
    """
    if lbl in LABEL_COLORS:
        return LABEL_COLORS[lbl]
    hue = int((lbl * 35) % 180)  # spread hues
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def copy_pair(mask_path: Path, img_path: Path, study: str, group: str):
    """Copy mask+image to OUT_ROOT/group/{MASKS,IMAGES} with study-prefixed names."""
    mask_dst = OUT_ROOT / group / "MASKS" / f"{study}_{mask_path.name}"
    img_dst  = OUT_ROOT / group / "IMAGES" / f"{study}_{img_path.name}"
    # Use copy2 to preserve times/metadata (optional)
    shutil.copy2(mask_path, mask_dst)
    shutil.copy2(img_path, img_dst)

# ----------------- Gather -----------------
# Find all mask files under DATASET/*/MASKS/*
mask_paths = list(SRC_ROOT.glob(f"*/{MASKS_SUBDIR}/*"))

total_multiclasses = 0
missing_images = 0
failed_reads = 0
overlays_written = 0

# ----------------- Main Loop -----------------
for mask_path in tqdm(mask_paths, desc="Processing"):
    # Study is the immediate child of SRC_ROOT
    try:
        study = mask_path.relative_to(SRC_ROOT).parts[0]
    except Exception:
        # Unexpected layout; skip
        continue

    # Compute the paired image path:
    # Replace MASKS -> PNG_IMAGES and remove a trailing "_mask" from filename stem
    img_name = mask_path.name
    stem, ext = os.path.splitext(img_name)
    stem = stem.replace("_mask", "")
    image_path = mask_path.parent.parent / IMAGES_SUBDIR / f"{stem}{ext}"

    if not image_path.exists():
        missing_images += 1
        # Still copy mask to ALL/MASKS so you can inspect later, if you want:
        # shutil.copy2(mask_path, OUT_ROOT / "ALL" / "MASKS" / f"{study}_{mask_path.name}")
        continue

    # Read mask (grayscale) and image (color)
    mask = safe_imread(mask_path, flags=0)
    overlay = safe_imread(image_path, flags=1)
    if mask is None or overlay is None:
        failed_reads += 1
        continue

    # Copy originals into ALL/{MASKS,IMAGES}
    copy_pair(mask_path, image_path, study, "ALL")

    # Check multiclass: more than 2 unique values (background + >=2 labels)
    uniq = np.unique(mask)
    if len(uniq) > 2:
        copy_pair(mask_path, image_path, study, "MULTICLASS")
        total_multiclasses += 1

    # Draw boundaries per non-zero label
    labels = uniq[uniq != 0]
    if labels.size:
        for lbl in labels:
            lbl = int(lbl)
            lbl_mask = (mask == lbl).astype(np.uint8) * 255
            # findContours expects binary uint8 image
            contours, _ = cv2.findContours(lbl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(
                    overlay, contours, -1, color_for_label(lbl),
                    thickness=OVERLAY_THICKNESS, lineType=cv2.LINE_AA
                )

    # Save overlay
    overlay_name = f"{study}_OVERLAY_{image_path.name}"
    out_overlay_path = OUT_ROOT / "ALL" / "OVERLAYS" / overlay_name
    cv2.imwrite(str(out_overlay_path), overlay)
    overlays_written += 1

# ----------------- Summary -----------------
print(f"Total overlays written: {overlays_written}")
print(f"Total multiclass images: {total_multiclasses}")
print(f"Missing image pairs: {missing_images}")
print(f"Failed reads: {failed_reads}")
