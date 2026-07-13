import numpy as np
import pandas as pd
from PIL import Image

from cvmlops.monitor import logging_store
from cvmlops.monitor.drift import compute_drift
from cvmlops.monitor.features import FEATURE_COLS, features_from


def test_features_from_image():
    img = Image.new("RGB", (128, 96), (10, 80, 40))
    f = features_from(img, n_detections=3, mean_confidence=0.7)
    assert set(FEATURE_COLS) <= set(f)
    assert f["width"] == 128 and f["height"] == 96
    assert f["n_detections"] == 3 and f["mean_confidence"] == 0.7


def test_prediction_log_roundtrip():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    logging_store.log_prediction("req1", "local:best.pt", features_from(img, 2, 0.5))
    logging_store.log_prediction("req2", "local:best.pt", features_from(img, 0, 0.0))
    df = logging_store.load_predictions()
    assert len(df) == 2
    assert set(FEATURE_COLS) <= set(df.columns)
    assert logging_store.count() == 2


def _frame(brightness, contrast, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "brightness": rng.normal(brightness, 5, n),
        "contrast": rng.normal(contrast, 3, n),
        "width": np.full(n, 128.0), "height": np.full(n, 128.0),
        "n_detections": rng.poisson(2, n).astype(float),
        "mean_confidence": rng.uniform(0.6, 0.9, n),
    })


def test_drift_detects_shift():
    ref = _frame(50, 20)
    drifted = _frame(130, 50, seed=1)  # brightness + contrast shifted hard
    r = compute_drift(ref, drifted, threshold=0.25)
    assert r.n_drifted >= 2
    assert r.drifted is True


def test_no_drift_on_same_distribution():
    ref = _frame(50, 20)
    r = compute_drift(ref, _frame(50, 20, seed=2), threshold=0.5)
    assert r.drifted is False
