"""Test isolation: point every stateful path (SQLite log, MLflow store) at a
temp dir so tests never touch the real repo artifacts."""

from __future__ import annotations

import pytest

from cvmlops import config


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    db = tmp_path / "predictions.db"
    mlflow_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("PREDICTION_DB", str(db))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlartifacts"))
    # caches captured the pre-env values at import time — clear them.
    config.get_settings.cache_clear()
    config.load_params.cache_clear()
    yield
    config.get_settings.cache_clear()
