"""Data-drift detection with Evidently 0.7.

Compares logged production image features against a reference sampled from the
training set. Returns the share of drifted columns and a boolean that the
scheduled retrain workflow acts on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from cvmlops.config import REPO_ROOT, load_params
from cvmlops.monitor.features import FEATURE_COLS, image_stats


@dataclass
class DriftResult:
    drift_share: float          # fraction of monitored columns that drifted
    drifted: bool               # drift_share >= configured threshold
    n_drifted: int
    n_columns: int


def _to_dataset(df: pd.DataFrame) -> Dataset:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    definition = DataDefinition(numerical_columns=cols)
    return Dataset.from_pandas(df[cols], data_definition=definition)


def _extract_share(result: dict) -> tuple[int, int, float]:
    """Pull (n_drifted, n_columns, drift_share) from the DataDriftPreset snapshot."""
    metrics = result.get("metrics", [])
    n_columns = sum("ValueDrift(" in str(m.get("metric_name", "")) for m in metrics)
    for metric in metrics:
        if "DriftedColumnsCount" in str(metric.get("metric_name", "")):
            val = metric["value"]
            return int(val["count"]), n_columns, float(val["share"])
    raise ValueError("DriftedColumnsCount not found in Evidently report")


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  threshold: float | None = None,
                  html_out: str | Path | None = None) -> DriftResult:
    if threshold is None:
        threshold = load_params()["monitor"]["drift_threshold"]

    report = Report([DataDriftPreset()])
    snapshot = report.run(_to_dataset(current), _to_dataset(reference))
    n_drifted, n_cols, share = _extract_share(snapshot.dict())

    if html_out:
        Path(html_out).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_out))

    return DriftResult(drift_share=share, drifted=share >= threshold,
                       n_drifted=n_drifted, n_columns=n_cols)


def reference_from_training() -> pd.DataFrame:
    """Build a reference feature frame from the prepared training images."""
    params = load_params()
    root = REPO_ROOT / params["dataset"]["root"]
    n = params["monitor"]["reference_sample"]
    from PIL import Image

    imgs = sorted((root / "images" / "train").glob("*"))[:n]
    if not imgs:
        raise FileNotFoundError(f"No training images under {root}; run data prepare first")
    rows = []
    for p in imgs:
        with Image.open(p) as im:
            rows.append({**image_stats(im), "n_detections": 0.0, "mean_confidence": 0.0})
    return pd.DataFrame(rows)
