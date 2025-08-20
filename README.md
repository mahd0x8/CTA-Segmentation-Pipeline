# CTA Segmentation Pipeline

This repository contains a complete **end-to-end training and evaluation pipeline** for **CTA (Computed Tomography Angiography) segmentation** using **UNet with an InceptionV3 encoder**. It supports fine-tuning, transfer learning, and integration with **Weights & Biases (W\&B)** for experiment monitoring.

---

## 📂 Project Structure

### Data Preparation

* **`MODELING/dicom_to_png.py`**
  Converts CTA **DICOM** files to **PNG images** in bone window for model training and evaluation.

* **`MODELING/creating_masks.py`**
  Converts **JSON annotation files** into **segmentation masks**, handling multiple studies at once based on contrast information.

* **`MODELING/image_selection.py`**
  Selection engine that:

  * Identifies studies with available masks.
  * Copies valid images and corresponding masks into a unified dataset folder.
  * Separates images/masks containing multiple labels into a different folder for experimental testing.

---

### Modeling & Training

* **`MODELING/model.py`**
  Defines and loads the trained segmentation model (`LOADED_MODEL`) for testing and evaluation.

* **`MODELING/Train.py`**
  Training script with features:

  * UNet (InceptionV3 backbone) architecture.
  * Encoder freezing/unfreezing for fine-tuning.
  * Mixed precision support (faster training on RTX GPUs).
  * Automatic dataset splits and reproducibility caching.
  * Integration with **Weights & Biases (W\&B)** for experiment tracking, metrics logging, and visualization.

---

### Evaluation

* **`MODELING/evaluate.py`**
  Uses the trained model (`model.py`) to generate predictions and overlays masks on original images for evaluation.

* **`MODELING/evaluateConc.py`**
  Same functionality as `evaluate.py` but provides **stacked result visualizations**, enabling easier side-by-side comparison of predictions.

---

## ⚙️ Key Features

* **Automated Dataset Handling**: Converts and organizes DICOM, JSON, and PNG data.
* **Robust Training**: Supports fine-tuning, encoder freezing, learning rate scheduling, and early stopping.
* **Reproducibility**: Split caching ensures identical train/validation splits across runs.
* **Monitoring & Logging**: Integrated with **W\&B** for loss/IoU tracking, visualizations, and model artifact storage.
* **Evaluation Tools**: Overlay predictions or generate stacked results for detailed visual inspection.

---

## 🚀 How to Use

1. **Preprocess Data**

   ```bash
   python MODELING/dicom_to_png.py
   python MODELING/creating_masks.py
   python MODELING/image_selection.py
   ```

2. **Train Model**

   ```bash
   python MODELING/Train.py
   ```

3. **Evaluate Model**

   ```bash
   python MODELING/evaluate.py
   # OR
   python MODELING/evaluateConc.py
   ```

---

## 🧑‍💻 Author

**MAHD BIN NAEEM**

---
