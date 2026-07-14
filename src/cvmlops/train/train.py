"""Train YOLO on PCB defects, log to MLflow, register the model version.

Run:  python -m cvmlops.train.train            # full run from params.yaml
      python -m cvmlops.train.train --smoke    # 1 epoch, tiny — CI / self-check
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlflow

from cvmlops.config import REPO_ROOT, load_params
from cvmlops.data import prepare, tile
from cvmlops.mlflow_utils import init_mlflow, register_weights

ARTIFACTS = REPO_ROOT / "artifacts"
# Ultralytics metric keys we surface to MLflow (val set, box detection).
_METRIC_KEYS = [
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
]


def _clean_metrics(results_dict: dict) -> dict[str, float]:
    out = {}
    for k, v in results_dict.items():
        if k in _METRIC_KEYS and v is not None:
            out[k.replace("metrics/", "").replace("(B)", "")] = float(v)
    return out


def train(smoke: bool = False, resume: bool = False) -> dict:
    from ultralytics import YOLO  # local import: keeps torch out of light imports

    params = load_params()
    tp, rp = params["train"], params["registry"]
    tiling = params.get("tiling", {})

    epochs = 1 if smoke else tp["epochs"]
    imgsz = 64 if smoke else tp["imgsz"]
    batch = 4 if smoke else tp["batch"]

    init_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_params({"base_model": tp["model"], "smoke": smoke, "resumed": resume})

        if resume:
            # Continue an interrupted run from its last checkpoint to the original
            # epoch count. Ultralytics reads all other args from the checkpoint, so
            # log the *effective* args (not the possibly-changed params.yaml values).
            last = ARTIFACTS / "runs" / "train" / "weights" / "last.pt"
            model = YOLO(str(last))
            results = model.train(resume=True)
            a = model.trainer.args
            mlflow.log_params({"epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
                               "patience": a.patience})
        else:
            mlflow.log_params({"epochs": epochs, "imgsz": imgsz, "batch": batch,
                               "patience": tp["patience"], "tiled": tiling.get("enabled", False)})
            prepared = prepare.main()
            # Tiling keeps tiny defects full-size; train on tiles when enabled.
            data_yaml = tile.main() if (tiling.get("enabled") and not smoke) else prepared
            model = YOLO(tp["model"])
            results = model.train(
                data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                patience=tp["patience"], device=tp["device"] or None,
                project=str(ARTIFACTS / "runs"), name="train", exist_ok=True, verbose=False,
            )

        metrics = _clean_metrics(results.results_dict)
        mlflow.log_metrics(metrics)

        best = Path(results.save_dir) / "weights" / "best.pt"
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, ARTIFACTS / "best.pt")  # local serving fallback

        mlflow.log_artifact(str(best), artifact_path="model")
        version = register_weights(run.info.run_id, rp["model_name"])
        mlflow.set_tag("registered_version", version)

        print(f"Run {run.info.run_id} | metrics={metrics} | "
              f"registered {rp['model_name']} v{version}")
        return {"run_id": run.info.run_id, "version": version, "metrics": metrics}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny run for CI")
    ap.add_argument("--resume", action="store_true", help="resume from last.pt checkpoint")
    args = ap.parse_args()
    train(smoke=args.smoke, resume=args.resume)
