"""Image-level features shared by the prediction log and drift detection.

Object-detection drift can't use raw pixels, so we monitor cheap image
statistics (brightness, contrast, size) plus prediction outputs (count,
confidence). A shift in these is the observable drift signal.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Input-image features: the basis for *data* drift (independent of the model).
INPUT_FEATURE_COLS = ["brightness", "contrast", "width", "height"]
# Prediction-output features: logged for monitoring, but NOT used as the data-drift
# signal (the training reference has no predictions, so they'd always "drift").
OUTPUT_FEATURE_COLS = ["n_detections", "mean_confidence"]
FEATURE_COLS = INPUT_FEATURE_COLS + OUTPUT_FEATURE_COLS


def image_stats(img: Image.Image) -> dict[str, float]:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return {
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
        "width": float(img.width),
        "height": float(img.height),
    }


def features_from(img: Image.Image, n_detections: int, mean_confidence: float) -> dict[str, float]:
    return {
        **image_stats(img),
        "n_detections": float(n_detections),
        "mean_confidence": float(mean_confidence),
    }
