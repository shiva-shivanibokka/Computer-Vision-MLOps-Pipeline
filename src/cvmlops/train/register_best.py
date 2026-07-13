"""Register the current best.pt checkpoint as a model version, independent of a
training run completing.

Long trainings can be interrupted (host limits, spot preemption). This evaluates
whatever best.pt exists on the val set, logs its real metrics to MLflow, and
registers it — so a good checkpoint is never lost to an incomplete run.

Run:  python -m cvmlops.train.register_best
"""

from __future__ import annotations

import shutil

import mlflow

from cvmlops.config import REPO_ROOT, load_params
from cvmlops.mlflow_utils import init_mlflow, register_weights
from cvmlops.train.train import ARTIFACTS, _clean_metrics

BEST = ARTIFACTS / "runs" / "train" / "weights" / "best.pt"


def register_best() -> dict:
    from ultralytics import YOLO

    params = load_params()
    tp, rp = params["train"], params["registry"]
    if not BEST.exists():
        raise FileNotFoundError(f"No checkpoint at {BEST}; train first")

    data_yaml = REPO_ROOT / params["dataset"]["root"] / "data.yaml"
    model = YOLO(str(BEST))
    metrics_obj = model.val(data=str(data_yaml), imgsz=tp["imgsz"], verbose=False)
    metrics = _clean_metrics(metrics_obj.results_dict)

    init_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_params({"base_model": tp["model"], "imgsz": tp["imgsz"],
                           "source": "register_best"})
        mlflow.log_metrics(metrics)
        shutil.copy2(BEST, ARTIFACTS / "best.pt")
        mlflow.log_artifact(str(BEST), artifact_path="model")
        version = register_weights(run.info.run_id, rp["model_name"])
        mlflow.set_tag("registered_version", version)

    print(f"Registered {rp['model_name']} v{version} | metrics={metrics}")
    return {"run_id": run.info.run_id, "version": version, "metrics": metrics}


if __name__ == "__main__":
    register_best()
