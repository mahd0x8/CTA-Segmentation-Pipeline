# -*- coding: utf-8 -*-
"""
UNet (InceptionV3 encoder) fine-tuning + training loop Refined

Author: MAHD BIN NAEEM
"""

# =========================
# Standard Libraries
# =========================
import os
import sys
import glob
import shutil
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# =========================
# External Libraries
# =========================
import numpy as np
import cv2
import warnings
import wandb
#from wandb.keras import WandbCallback
from matplotlib import pyplot as plt  # (not required to run; kept for quick viz)
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, CSVLogger, TensorBoard,
                                        ReduceLROnPlateau, EarlyStopping)


# Segmentation Models (SM) - make sure to set framework before import
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm  # type: ignore

# Your local model definition / weights 
from model import *  # noqa: F401,F403
from Monitoring import *
# =========================
# Global Config
# =========================

# Reproducibility
GLOBAL_SEED = 42

# Data
DATA_ROOT = "DATASET_MASKED_SELECTIVE/ALL/"
MASK_DIR = os.path.join(DATA_ROOT, "MASKS")
IMG_DIR  = os.path.join(DATA_ROOT, "IMAGES")

# Experiment output folder
RUN = f"EXPERIMENTS/V{1}/"
os.makedirs(RUN, exist_ok=True)

# Image / Model config
SIZE_X, SIZE_Y = 512, 512
N_CLASSES = 4                      # hard cap; checks below will enforce this
BACKBONE = "inceptionv3"           # used for preprocessing only (encoder is InceptionV3)
FREEZE_ENCODER_FIRST = True        # decoder-only warm-up
FREEZE_BN_ALWAYS = True            # keep BatchNorms frozen for stability

# Training
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4               # conservative for fine-tuning

# Mixed precision (optional but recommended on RTX 30xx+)
USE_MIXED_PRECISION = True

# Optional: Known grayscale values in masks (stick to dataset’s palette).
# If None, we auto-infer from a sample of masks.
KNOWN_MASK_VALUES: Optional[List[int]] = None  # e.g., [0, 60, 120, 180]

# =========================
# Utilities
# =========================

def seed_everything(seed: int = 42) -> None:
    """Set seeds for Python/NumPy/TF to help reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def check_dependencies() -> bool:
    """Print environment info and return True if GPU is visible."""
    print("TensorFlow version:", tf.__version__)
    print("NumPy version:", np.__version__)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\nGPUs available:", gpus)
        # Optional: memory growth
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("Could not set memory growth:", e)
        return True
    else:
        print("No GPU available")
        return False

def safe_makedirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def list_dataset_pairs(mask_dir: str, img_dir: str) -> Tuple[List[str], List[str]]:
    """List masks and corresponding images by filename convention."""
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    images = [m.replace("/MASKS/", "/IMAGES/").replace("_mask.png", ".png") for m in masks]
    # Defensive: ensure each image exists
    images = [p for p in images if os.path.exists(p)]
    masks  = [m for m in masks if os.path.exists(m.replace("/MASKS/", "/IMAGES/").replace("_mask.png", ".png"))]
    return images, masks

def infer_mask_values(mask_paths: List[str], max_samples: int = 500) -> List[int]:
    """Scan up to `max_samples` masks and collect unique grayscale values."""
    uniques = set()
    for p in mask_paths[:max_samples]:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # If your dataset has files with "flow" in path: force >0 pixels to 180 (your rule)
        if "flow" in p:
            m = m.copy()
            m[m > 0] = 180
        vals = np.unique(m)
        uniques.update(vals.tolist())
    values = sorted(list(uniques))
    print(f"~ Detected mask values (sampled): {values}")
    return values

def build_value_to_class_map(values: List[int], n_classes: int) -> Dict[int, int]:
    """
    Build a fixed mapping from grayscale value -> class index 0..C-1.
    Ensures stable one-hot encodings across all batches.
    """
    if len(values) != n_classes:
        raise ValueError(
            f"Expected exactly {n_classes} unique mask values, got {len(values)}: {values}\n"
            "Adjust N_CLASSES or provide KNOWN_MASK_VALUES to match your dataset."
        )
    return {v: i for i, v in enumerate(values)}

def apply_value_map(mask: np.ndarray, value_to_class: Dict[int, int]) -> np.ndarray:
    """
    Convert grayscale mask values to contiguous class IDs 0..C-1.
    Unseen values will be clipped to 0 (background) as a safety net.
    """
    out = np.zeros_like(mask, dtype=np.int32)
    for v, c in value_to_class.items():
        out[mask == v] = c
    return out

def get_preproc_fn(backbone: str):
    """
    Segmentation Models provides preprocessing aligned to the backbone.
    For InceptionV3: scales to [-1, 1] and applies channel-wise normalization.
    """
    return sm.get_preprocessing(backbone)

# =========================
# Encoder Freezing Helpers
# =========================

def set_encoder_trainable(model: tf.keras.Model, trainable: bool, freeze_bn: bool = True) -> tf.keras.Model:
    """
    For SM-UNet + InceptionV3: encoder layers are before 'decoder_stage0_upsampling'.
    Keeps BN frozen when encoder is frozen (recommended for small batches).
    """
    in_decoder = False
    for layer in model.layers:
        if layer.name.startswith("decoder_stage0_upsampling"):
            in_decoder = True
        if not in_decoder:  # encoder (and bridge) side
            layer.trainable = trainable
            if freeze_bn and isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        else:
            # decoder stays trainable
            layer.trainable = True
    return model

# =========================
# Data Generator
# =========================

class SegmentationGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence-based generator:
    - Deterministic length (len = steps_per_epoch)
    - Thread/process-safe
    - Shuffles indices every epoch
    - Applies backbone preprocessing
    - Encodes masks to one-hot with a fixed value->class map
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        batch_size: int,
        value_to_class: Dict[int, int],
        image_size: Tuple[int, int] = (512, 512),
        preproc_fn=None,
        shuffle: bool = True,
    ):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.batch_size  = batch_size
        self.value_to_class = value_to_class
        self.n_classes = len(value_to_class)
        self.image_size = image_size
        self.preproc_fn = preproc_fn
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        # number of batches per epoch (drop last if not divisible)
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_ids = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for i in batch_ids:
            img = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)
            msk = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)

            if img is None or msk is None:
                # Skip corrupt pairs by substituting a random valid sample
                j = np.random.randint(0, len(self.indices))
                img = cv2.imread(self.image_paths[j], cv2.IMREAD_COLOR)
                msk = cv2.imread(self.mask_paths[j],  cv2.IMREAD_GRAYSCALE)

            # Resize if needed (kept simple; add interpolation control if you want)
            if (img.shape[1], img.shape[0]) != self.image_size:
                img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            if (msk.shape[1], msk.shape[0]) != self.image_size:
                msk = cv2.resize(msk, self.image_size, interpolation=cv2.INTER_NEAREST)

            # Your dataset-specific rule: maps any >0 to 180 for "flow" masks
            if "flow" in self.mask_paths[i]:
                msk = msk.copy()
                msk[msk > 0] = 180

            # Encode mask to class IDs 0..C-1 using a FIXED
            msk_cls = apply_value_map(msk, self.value_to_class)

            # One-hot encode
            msk_1h = tf.keras.utils.to_categorical(msk_cls, num_classes=self.n_classes).astype("float32")

            # Preprocess image for the backbone (e.g., InceptionV3)
            if self.preproc_fn is not None:
                img = self.preproc_fn(img.astype("float32"))
            else:
                # fallback; matches your original normalization
                img = (img.astype("float32") / 255.0)

            batch_x.append(img)
            batch_y.append(msk_1h)

        X = np.stack(batch_x, axis=0)
        Y = np.stack(batch_y, axis=0)
        return X, Y

# =========================
# Training
# =========================

def compile_model(model: tf.keras.Model, lr: float) -> None:
    """
    Compile with Dice loss and IoU metric from SM.
    (Both assume one-hot masks for multi-class.)
    """
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=sm.losses.DiceLoss(),
        metrics=[sm.metrics.iou_score],
    )

def train(
    loaded_model: tf.keras.Model,
    train_images: List[str],
    val_images: List[str],
    train_masks: List[str],
    val_masks: List[str],
    run_dir: str,
    n_classes: int = N_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    backbone: str = BACKBONE,
    freeze_encoder_first: bool = FREEZE_ENCODER_FIRST,
    freeze_bn_always: bool = FREEZE_BN_ALWAYS,
    lr: float = LEARNING_RATE,
):
    """Main training pipeline."""

    # ---------- Preprocessing ----------
    preproc_fn = get_preproc_fn(backbone)

    # Build mask palette mapping (stable across all batches)
    if KNOWN_MASK_VALUES is None:
        values = infer_mask_values(train_masks)
    else:
        values = sorted(KNOWN_MASK_VALUES)

    value_to_class = build_value_to_class_map(values, n_classes=n_classes)
    print("~ value_to_class:", value_to_class)

    # ---------- Generators ----------
    train_gen = SegmentationGenerator(
        image_paths=train_images,
        mask_paths=train_masks,
        batch_size=batch_size,
        value_to_class=value_to_class,
        image_size=(SIZE_X, SIZE_Y),
        preproc_fn=preproc_fn,
        shuffle=True,
    )

    val_gen = SegmentationGenerator(
        image_paths=val_images,
        mask_paths=val_masks,
        batch_size=batch_size,
        value_to_class=value_to_class,
        image_size=(SIZE_X, SIZE_Y),
        preproc_fn=preproc_fn,
        shuffle=False,
    )

    # ---------- Fine-tuning Phases ----------
    current_datetime = datetime.now()
    stamp = current_datetime.strftime("%Y%m%d_%H%M%S")

    # Logs/checkpoints
    safe_makedirs(run_dir)
    csv_logger = CSVLogger(os.path.join(run_dir, f"training_log_{stamp}.csv"), append=True)
    tb_log_dir = os.path.join(run_dir, "logs", "fit", current_datetime.strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

    ckpt_path = os.path.join(run_dir, f"Best_Weights_Model_{stamp}.keras")
    ckpt_cb = ModelCheckpoint(
        filepath=ckpt_path,
        save_best_only=True,
        monitor="val_iou_score",
        mode="max",
        verbose=1,
    )

    sched_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    early_cb = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1)
    
    # --- W&B init ---
    run_name = f"unet-inceptionv3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_config = {
        "backbone": backbone,
        "image_size": (SIZE_X, SIZE_Y),
        "n_classes": n_classes,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr_phase1": lr,
        "lr_phase2": min(lr, 5e-5),
        "freeze_encoder_first": freeze_encoder_first,
        "freeze_bn_always": freeze_bn_always,
        "train_count": len(train_images),
        "val_count": len(val_images),
        "value_to_class": value_to_class,  # logged for reproducibility
    }
    os.environ.setdefault("WANDB_CONSOLE", "wrap")
    wandb.init(
        project="CTA_Segmentation_R2_V2",
        name=run_name,
        config=wandb_config,
        dir=run_dir,                      # store run files under your experiment dir
    )
    wb_metrics = WandbMetricsLogger(log_freq="epoch")

    class_labels = {i: f"class_{i}" for i in range(len(value_to_class))}
    viz_cb = WandbSegViz(val_gen, class_labels=class_labels, every_n_epochs=1, max_items=4)
    iou_cb = PerClassIoU(val_gen, max_batches=3)

    # Phase 1: freeze encoder (decoder-only warm-up)
    if freeze_encoder_first:
        print("\n[Phase 1] Freezing encoder (decoder-only training)")
        set_encoder_trainable(loaded_model, trainable=False, freeze_bn=freeze_bn_always)
        compile_model(loaded_model, lr=lr)
        loaded_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=max(5, min(epochs // 5, 20)),
            callbacks=[ckpt_cb, csv_logger, tensorboard_cb, wb_metrics, viz_cb, iou_cb, sched_cb, early_cb],
            verbose=1,
        )


    # Phase 2: unfreeze entire encoder (keep BN frozen if requested) with smaller LR
    print("\n[Phase 2] Unfreezing encoder (full fine-tune)")
    set_encoder_trainable(loaded_model, trainable=True, freeze_bn=freeze_bn_always)
    
    
    # Generally go smaller when unfreezing more parameters
    compile_model(loaded_model, lr=min(lr, 5e-5))
    history = loaded_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[ckpt_cb, csv_logger, tensorboard_cb, wb_metrics, viz_cb, iou_cb, sched_cb, early_cb],
        verbose=1,
    )

    # Final save (SavedModel/keras format is preferred over legacy .h5)
    final_path = os.path.join(run_dir, f"MODEL_{stamp}.keras")
    loaded_model.save(final_path)
    # Upload your best checkpoint picked by Keras ModelCheckpoint
    artifact = wandb.Artifact("unet-inception-best", type="model")
    artifact.add_file(ckpt_path)   # your .keras best model path
    wandb.log_artifact(artifact)

    wandb.finish()

    print(f"\n✅ Training complete. Best weights: {ckpt_path}\n   Final model: {final_path}")

    return history

# =========================
# Main
# =========================

if __name__ == "__main__":

    # 1) Reproducibility + (optional) mixed precision
    seed_everything(GLOBAL_SEED)
    if USE_MIXED_PRECISION:
        try:
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (float16).")
        except Exception as e:
            print("Could not enable mixed precision:", e)

    # 2) Check runtime (GPU will print above; those cuDNN/cuBLAS 'already registered'
    #    warnings are noisy but harmless in most TF builds)
    _ = check_dependencies()

    # 3) Get pairs
    images, masks = list_dataset_pairs(MASK_DIR, IMG_DIR)
    print(f"\n~ Total Number of image/mask pairs found: {len(masks)}")

    # 4) Split (and optionally cache as .npy so future runs are identical)
    split_cache = {
        "train_images": os.path.join(RUN, "train_images.npy"),
        "val_images":   os.path.join(RUN, "val_images.npy"),
        "train_masks":  os.path.join(RUN, "train_masks.npy"),
        "val_masks":    os.path.join(RUN, "val_masks.npy"),
    }

    if all(os.path.exists(p) for p in split_cache.values()):
        train_images = np.load(split_cache["train_images"], allow_pickle=True).tolist()
        train_masks  = np.load(split_cache["train_masks"], allow_pickle=True).tolist()
        val_images   = np.load(split_cache["val_images"], allow_pickle=True).tolist()
        val_masks    = np.load(split_cache["val_masks"], allow_pickle=True).tolist()
        print("~ Loaded cached splits (.npy)")
    else:
        train_images, val_images, train_masks, val_masks = train_test_split(
            images, masks, test_size=0.1, random_state=GLOBAL_SEED, shuffle=True
        )
        # Cache splits for reproducibility
        np.save(split_cache["train_images"], np.array(train_images, dtype=object))
        np.save(split_cache["val_images"],   np.array(val_images,   dtype=object))
        np.save(split_cache["train_masks"],  np.array(train_masks,  dtype=object))
        np.save(split_cache["val_masks"],    np.array(val_masks,    dtype=object))
        print("~ Saved splits to .npy (reproducible)")

    print(
        f"~ TRAIN IMAGES:\t{len(train_images)}\tVALIDATION IMAGES:\t{len(val_images)}"
        f"\tTRAIN MASKS:\t{len(train_masks)}\tVALIDATION MASKS:\t{len(val_masks)}\n"
    )

    # 5) Load/receive model
    try:
        model_to_train = LOADED_MODEL  # from model.py
    except NameError as e:
        raise RuntimeError(
            "LOADED_MODEL not found. Ensure your model.py defines LOADED_MODEL or "
            "replace this section with keras.models.load_model(<path>, custom_objects=...)."
        ) from e

    # 6) Train
    train(
        loaded_model=model_to_train,
        train_images=train_images,
        val_images=val_images,
        train_masks=train_masks,
        val_masks=val_masks,
        run_dir=RUN,
        n_classes=N_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        backbone=BACKBONE,
        freeze_encoder_first=FREEZE_ENCODER_FIRST,
        freeze_bn_always=FREEZE_BN_ALWAYS,
        lr=LEARNING_RATE,
    )