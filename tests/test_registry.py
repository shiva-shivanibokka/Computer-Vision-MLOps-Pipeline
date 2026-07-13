"""Promotion-gate tests against a real (temp SQLite) MLflow registry — no training."""

import mlflow

from cvmlops.config import load_params
from cvmlops.mlflow_utils import init_mlflow, register_weights
from cvmlops.registry.promote import promote


def _register(map_metric: float, tmp_path) -> str:
    name = load_params()["registry"]["model_name"]
    with mlflow.start_run() as run:
        mlflow.log_metric("mAP50-95", map_metric)
        art = tmp_path / "best.pt"
        art.write_bytes(b"stub-weights")
        mlflow.log_artifact(str(art), artifact_path="model")
        version = register_weights(run.info.run_id, name)
    return version


def test_promotion_lifecycle(tmp_path):
    init_mlflow()

    v1 = _register(0.30, tmp_path)
    d1 = promote(v1)
    assert d1.promoted and "no incumbent" in d1.reason

    v2 = _register(0.50, tmp_path)
    d2 = promote(v2)
    assert d2.promoted
    assert d2.candidate_metric == 0.50 and d2.incumbent_metric == 0.30

    # a worse model must be rejected — the gate protects production
    v3 = _register(0.40, tmp_path)
    d3 = promote(v3)
    assert not d3.promoted
    assert d3.candidate_metric == 0.40 and d3.incumbent_metric == 0.50


def test_promote_same_version_is_noop(tmp_path):
    init_mlflow()
    v1 = _register(0.30, tmp_path)
    promote(v1)
    again = promote(v1)
    assert not again.promoted and again.reason == "already production"
