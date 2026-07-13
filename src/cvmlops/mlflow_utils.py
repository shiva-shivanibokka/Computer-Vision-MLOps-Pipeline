"""MLflow wiring shared by training, registry, and serving.

The tracking URI + auth come entirely from env (config.Settings), so pointing
at DagsHub later is just setting MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD.
"""

from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient

from cvmlops.config import get_settings


def init_mlflow(experiment: str = "pcb-defect-detection") -> None:
    s = get_settings()
    if s.mlflow_tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = s.mlflow_tracking_username
    if s.mlflow_tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = s.mlflow_tracking_password
    mlflow.set_tracking_uri(s.resolved_tracking_uri())
    mlflow.set_experiment(experiment)


def register_weights(run_id: str, name: str, artifact_path: str = "model") -> str:
    """Register a raw weights artifact as a new model version.

    YOLO .pt files have no native MLflow flavor, so we register the artifact
    directory directly instead of via mlflow.register_model (which requires a
    logged-model flavor). Returns the new version number.
    """
    client = MlflowClient()
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists
    source = f"{client.get_run(run_id).info.artifact_uri}/{artifact_path}"
    return client.create_model_version(name=name, source=source, run_id=run_id).version
