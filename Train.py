# -*- coding: utf-8 -*-
"""
UNet (InceptionV3 encoder) fine-tuning + training loop Refined

Author: MAHD BIN NAEEM
"""

from __future__ import annotations

# =========================
# Standard Libraries
# =========================
import os
import sys
import glob
import shutil
import random
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# =========================
# Third-Party Libraries
# =========================
import numpy as np
import cv2
import warnings

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import callbacks as KCallbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, CSVLogger, TensorBoard,
                                        EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.utils import to_categorical

# Segmentation Models
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import IOUScore

# Sklearn
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split

# Weights & Biases
import wandb
from wandb.integration.keras import WandbMetricsLogger
from model import *

# =========================
# Configuration
# =========================
@dataclass
class TrainConfig:
    # Data
    dataset_root: str = "DATASET_MASKED_SELECTIVE/OVERFITING/"
    images_subdir: str = "IMAGES"
    masks_subdir: str = "MASKS"
    image_ext: str = ".png"
    mask_suffix: str = "_mask"
    mask_ext: str = ".png"

    # Experiment
    run_root: str = "EXPERIMENTS/V1/"
    log_subdir: str = "logs/fit"
    save_best_monitor: str = "val_loss"
    save_best_mode: str = "min"

    # Model / Training
    input_size: Tuple[int, int] = (512, 512)
    n_classes: int = 4
    batch_size: int = 2
    epochs: int = 150
    learning_rate: float = 1e-3
    backbone: str = "inceptionv3"
    encoder_weights: Optional[str] = "imagenet"
    encoder_freeze: bool = True
    activation: str = "softmax"

    # Mask handling
    known_mask_values: Optional[List[int]] = None  # e.g., [0, 85, 170, 255]
    flow_label_value: Optional[int] = 180         # if "flow" in filename, map >0 to this value

    # Reproducibility
    seed: int = 42

    # GPU management
    kill_large_gpu_procs: bool = False
    kill_threshold_mb: int = 1000

    # Logging
    use_wandb: bool = True
    wandb_project: str = "CTA_Segmentation_R2_V2"

    # Validation split (if not using pre-saved splits)
    val_split: float = 0.1

    def ensure_dirs(self) -> None:
        os.makedirs(self.run_root, exist_ok=True)
        os.makedirs(os.path.join(self.run_root, self.log_subdir), exist_ok=True)


# =========================
# Utilities
# =========================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_kill_large_gpu_processes(threshold_mb: int) -> None:
    """
    Kill GPU processes using > threshold_mb MB GPU memory via nvidia-smi.
    Runs safely; ignored if command not available.
    """
    cmd = rf"""nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | sed 's/,//g' \
        | awk '$2 > {threshold_mb} {{print $1}}' \
        | xargs -r -n1 kill -9"""
    try:
        subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"[WARN] Could not run GPU cleanup: {e}")


def check_dependencies() -> bool:
    print("TensorFlow:", tf.__version__)
    print("Segmentation Models:", sm.__version__)
    print("NumPy:", np.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs detected: {gpus}")
        return True
    else:
        print("No GPU detected. Training will use CPU.")
        return False


def list_image_mask_pairs(cfg: TrainConfig) -> Tuple[List[str], List[str]]:
    """
    Finds mask files; derives image paths by replacing subdir and suffix.
    """
    mask_dir = os.path.join(cfg.dataset_root, cfg.masks_subdir)
    masks = sorted(glob.glob(os.path.join(mask_dir, f"*{cfg.mask_ext}")))
    images = [
        m.replace(os.sep + cfg.masks_subdir + os.sep, os.sep + cfg.images_subdir + os.sep)\
         .replace(cfg.mask_suffix + cfg.mask_ext, cfg.image_ext)
        for m in masks
    ]
    assert len(images) == len(masks), "Images and masks count mismatch."
    return images, masks


def infer_mask_values(masks: Sequence[str], sample_k: int = 50) -> List[int]:
    """
    Infer unique grayscale values across a subset of masks (for speed).
    """
    vals = set()
    pick = masks if len(masks) <= sample_k else random.sample(masks, sample_k)
    for p in pick:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        vals.update(np.unique(m).tolist())
    return sorted(vals)


def build_value_to_class_map(values: List[int], n_classes: int) -> Dict[int, int]:
    """
    Build a fixed mapping from grayscale value -> class index 0..C-1.
    """
    if len(values) != n_classes:
        raise ValueError(
            f"Expected {n_classes} unique mask values, got {len(values)}: {values}.\n"
            "Adjust cfg.n_classes or cfg.known_mask_values to match your dataset."
        )
    return {v: i for i, v in enumerate(values)}


def apply_flow_rule(mask: np.ndarray, flow_value: Optional[int]) -> np.ndarray:
    """
    If flow_value is set and mask filename indicates 'flow', we map >0 to flow_value.
    The caller should decide when to apply (based on file path).
    """
    if flow_value is None:
        return mask
    out = mask.copy()
    out[out > 0] = flow_value
    return out


def preprocess_image(img: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    """
    Resize and scale image to [0,1], shape (H,W,3), float32.
    """
    w, h = target_wh
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    return img


def preprocess_mask(mask: np.ndarray,
                    target_wh: Tuple[int, int],
                    value_to_class: Dict[int, int]) -> np.ndarray:
    """
    Resize mask (nearest), then map grayscale values -> class indices, then one-hot encode.
    """
    w, h = target_wh
    if (mask.shape[1], mask.shape[0]) != (w, h):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Vectorized mapping
    class_map = np.zeros_like(mask, dtype=np.int32)
    for v, c in value_to_class.items():
        class_map[mask == v] = c

    one_hot = to_categorical(class_map, num_classes=len(value_to_class)).astype("float32")
    return one_hot


def data_generator(
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
    batch_size: int,
    target_wh: Tuple[int, int],
    value_to_class: Dict[int, int],
    flow_value: Optional[int],
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields (batch_images, batch_onehot_masks).
    """
    assert len(image_paths) == len(mask_paths)
    n = len(image_paths)
    idxs = np.arange(n)

    while True:
        np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start + batch_size]
            batch_imgs, batch_lbls = [], []

            for i in batch_idx:
                img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
                msk = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)

                if img is None or msk is None:
                    # Skip corrupt pairs by substituting a random valid sample
                    j = np.random.randint(0, n)
                    img = cv2.imread(image_paths[j], cv2.IMREAD_COLOR)
                    msk = cv2.imread(mask_paths[j], cv2.IMREAD_GRAYSCALE)

                # If special "flow" rule applies by filename:
                if ("flow" in os.path.basename(mask_paths[i]).lower()) and flow_value is not None:
                    msk = apply_flow_rule(msk, flow_value)

                img = preprocess_image(img, target_wh)
                one_hot = preprocess_mask(msk, target_wh, value_to_class)

                batch_imgs.append(img)
                batch_lbls.append(one_hot)

            yield np.stack(batch_imgs, axis=0), np.stack(batch_lbls, axis=0)


# =========================
# Model
# =========================
def build_model(cfg: TrainConfig) -> tf.keras.Model:
    """
    Build UNet with a selected backbone.
    """
    """model = Unet(
        backbone_name=cfg.backbone,
        classes=cfg.n_classes,
        activation=cfg.activation,
        encoder_weights=cfg.encoder_weights,
        input_shape=(cfg.input_size[1], cfg.input_size[0], 3),
        encoder_freeze=cfg.encoder_freeze,
    )"""
    return LOADED_MODEL


def compile_model(model: tf.keras.Model, cfg: TrainConfig) -> None:
    optimizer = Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=DiceLoss(),                   # can be changed to hybrid loss if desired
        metrics=[IOUScore(threshold=None)] # threshold=None => soft IoU
    )


# =========================
# Training
# =========================
def build_callbacks(cfg: TrainConfig, run_tag: str) -> List[KCallbacks.Callback]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = os.path.join(cfg.run_root, f"Best_Weights_Model_{ts}.keras")
    log_dir = os.path.join(cfg.run_root, cfg.log_subdir, ts)

    cbs: List[KCallbacks.Callback] = [
        ModelCheckpoint(
            filepath=best_path,
            save_best_only=True,
            monitor=cfg.save_best_monitor,
            mode=cfg.save_best_mode,
            verbose=1,
        ),
        CSVLogger(os.path.join(cfg.run_root, "training_log.csv"), append=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        #EarlyStopping(monitor=cfg.save_best_monitor, mode=cfg.save_best_mode, patience=20, verbose=1, restore_best_weights=True),
        #ReduceLROnPlateau(monitor=cfg.save_best_monitor, mode=cfg.save_best_mode, factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ]

    cbs.append(WandbMetricsLogger(log_freq="epoch"))

    return cbs


def maybe_init_wandb(cfg: TrainConfig, extras: Dict) -> None:

    run_name = f"unet-{cfg.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config={**asdict(cfg), **extras},
        dir=cfg.run_root
    )


def train(cfg: TrainConfig) -> None:
    cfg.ensure_dirs()
    set_global_seed(cfg.seed)

    if cfg.kill_large_gpu_procs:
        safe_kill_large_gpu_processes(cfg.kill_threshold_mb)

    _ = check_dependencies()

    # Discover data
    images, masks = list_image_mask_pairs(cfg)
    print(f"Total masks found: {len(masks)}")

    # Train/val split
    train_imgs, val_imgs, train_msks, val_msks = train_test_split(
        images, masks, test_size=cfg.val_split, random_state=cfg.seed, shuffle=True
    )

    # Determine mask values
    if cfg.known_mask_values is not None:
        mask_values = sorted(cfg.known_mask_values)
    else:
        mask_values = infer_mask_values(masks, sample_k=100)

    print(f"Detected/Using mask values: {mask_values}")
    value_to_class = build_value_to_class_map(mask_values, cfg.n_classes)
    print(f"value_to_class: {value_to_class}")

    # Generators
    train_gen = data_generator(
        train_imgs, train_msks, cfg.batch_size, cfg.input_size, value_to_class, cfg.flow_label_value
    )
    val_gen = data_generator(
        val_imgs.copy(), val_msks.copy(), cfg.batch_size, cfg.input_size, value_to_class, cfg.flow_label_value
    )

    # Build & compile model
    model = build_model(cfg)
    compile_model(model, cfg)
    model.summary()

    # W&B (optional)
    extras = {
        "train_count": len(train_imgs),
        "val_count": len(val_imgs),
        "value_to_class": value_to_class,
    }
    maybe_init_wandb(cfg, extras)

    # Train
    steps_per_epoch = max(1, len(train_imgs) // cfg.batch_size)
    val_steps = max(1, len(val_imgs) // cfg.batch_size)
    cbs = build_callbacks(cfg, run_tag="main")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=cfg.epochs,
        callbacks=cbs,
        verbose=1,
    )

    # Save final model
    final_path = os.path.join(cfg.run_root, "MODEL_final.keras")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

    # Log best model artifact to W&B if available
    # Find best checkpoint path from callbacks
    ckpt_cb = [cb for cb in cbs if isinstance(cb, ModelCheckpoint)]
    if ckpt_cb:
        ckpt_path = ckpt_cb[0].filepath  # type: ignore[attr-defined]
        artifact = wandb.Artifact("unet-best", type="model")
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)
    wandb.finish()


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    warnings.filterwarnings(action="ignore", category=DataConversionWarning)

    cfg = TrainConfig(
        dataset_root="DATASET_MASKED_SELECTIVE/ALL/",
        run_root="EXPERIMENTS/V4/",
        input_size=(512, 512),
        n_classes=4,
        batch_size=8,              # match your original
        epochs=100,                # match your original
        known_mask_values=None,    # set to a list like [0, 85, 170, 255] if you know them
        kill_large_gpu_procs=False,  # set True to auto-kill large GPU procs
        use_wandb=True,              # set False if you don't want W&B
    )

    train(cfg)
