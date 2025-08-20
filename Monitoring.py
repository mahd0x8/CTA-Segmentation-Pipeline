import wandb
import tensorflow as tf
import numpy as np
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# --- Visualize predictions vs GT every epoch ---
class WandbSegViz(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, class_labels=None, every_n_epochs=1, max_items=4, input_is_inception_preproc=True):
        super().__init__()
        self.val_gen = val_gen
        self.every_n = every_n_epochs
        self.max_items = max_items
        self.n_classes = val_gen.n_classes
        self.class_labels = class_labels or {i: f"class_{i}" for i in range(self.n_classes)}
        self.input_is_inception_preproc = input_is_inception_preproc  # you use inceptionv3 preprocessing -> [-1,1]

    @staticmethod
    def _deprocess_inception(x):
        x = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return x

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n != 0:
            return
        x, y_onehot = self.val_gen[0]
        x = x[: self.max_items]
        y_onehot = y_onehot[: self.max_items]
        y_true = np.argmax(y_onehot, axis=-1)

        preds = self.model.predict(x, verbose=0)
        y_pred = np.argmax(preds, axis=-1)

        panels = []
        for i in range(len(x)):
            img_disp = self._deprocess_inception(x[i]) if self.input_is_inception_preproc else (x[i] * 255).astype(np.uint8)
            panels.append(
                wandb.Image(
                    img_disp,
                    masks={
                        "ground_truth": {"mask_data": y_true[i].astype(np.uint8), "class_labels": self.class_labels},
                        "prediction":   {"mask_data": y_pred[i].astype(np.uint8), "class_labels": self.class_labels},
                    },
                )
            )
        wandb.log({"val_examples": panels}, commit=False)

# --- Log current learning rate each epoch ---
class WandbLogLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            lr_t = self.model.optimizer.learning_rate
            lr = float(tf.keras.backend.get_value(lr_t))
            wandb.log({"lr": lr}, commit=False)
        except Exception:
            pass

# --- Compute & log per-class IoU on a small val subset each epoch ---
class PerClassIoU(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, max_batches=3):
        super().__init__()
        self.val_gen = val_gen
        self.max_batches = max_batches
        self.n_classes = val_gen.n_classes

    def on_epoch_end(self, epoch, logs=None):
        cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)  # rows=gt, cols=pred
        batches = min(len(self.val_gen), self.max_batches)
        for b in range(batches):
            x, y_onehot = self.val_gen[b]
            y_true = np.argmax(y_onehot, axis=-1).reshape(-1)
            y_pred = np.argmax(self.model.predict(x, verbose=0), axis=-1).reshape(-1)
            # update confusion matrix
            idx = self.n_classes * y_true + y_pred
            binc = np.bincount(idx, minlength=self.n_classes**2)
            cm += binc.reshape(self.n_classes, self.n_classes)

        TP = np.diag(cm).astype(np.float64)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        denom = (TP + FP + FN + 1e-7)
        iou = TP / denom
        log_dict = {f"iou_class_{i}": float(iou_i) for i, iou_i in enumerate(iou)}
        log_dict["mean_iou_eval"] = float(np.nanmean(iou))
        wandb.log(log_dict, commit=False)
