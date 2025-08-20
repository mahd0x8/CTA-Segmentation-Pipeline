# =========================
# Standard Libraries
# =========================
import os
import sys
import glob
import shutil
import random
import warnings
from datetime import datetime

# =========================
# External Libraries
# =========================
import numpy as np
import cv2
from tqdm import tqdm                          # type: ignore # Progress bar
from matplotlib import pyplot as plt

# Suppress scikit-learn warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# =========================
# TensorFlow & Keras
# =========================
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore
import keras

# For segmentation loss & metrics (SM Framework)
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import IOUScore

# =========================
# Session Reset
# =========================
K.clear_session()
tf.compat.v1.reset_default_graph()

# =========================
# Custom Loss and Metric Wrappers
# =========================

class CustomDiceLoss(DiceLoss):
    """Custom Dice Loss wrapper to ensure compatibility with model loading."""
    def __init__(self, *args, **kwargs):
        super(CustomDiceLoss, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            return cls()
        return cls(**config)

class CustomIOUScore(IOUScore):
    """Custom IOU Score wrapper to ensure compatibility with model loading."""
    def __init__(self, *args, **kwargs):
        super(CustomIOUScore, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            return cls()
        return cls(**config)

# =========================
# Model Loading
# =========================

# Path to the best model weights (in .keras format)
best_weights_path_training = "MODELING/RegionOneModel/Best_Weights_Model_20250428_142432imgs.keras"
best_weights_path_testing = "EXPERIMENTS/V1/Best_Weights_Model_20250819_155956.keras"

# Load the trained model along with custom loss and metric
LOADED_MODEL = keras.models.load_model(
    # best_weights_path_training,
    best_weights_path_testing,
    custom_objects={
        'DiceLoss': CustomDiceLoss,
        'IOUScore': CustomIOUScore
    }
)

print("✅ Model Loaded Successfully.")
