"""Config: static pipeline params (params.yaml) + runtime env (accounts/secrets).

Account wiring lives entirely in env vars so the whole stack runs locally with
zero accounts, and "connecting the accounts later" is just setting variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"


class Settings(BaseSettings):
    """Runtime/account settings. All optional — sensible local defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MLflow: unset -> local SQLite store (see resolved_tracking_uri). Set to a
    # DagsHub URL to go remote.
    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""

    # Where the serving layer looks for a model when the registry is unreachable.
    local_weights: str = "artifacts/best.pt"

    # SQLite prediction log (drift + monitoring source of truth).
    prediction_db: str = "artifacts/predictions.db"

    def resolved_tracking_uri(self) -> str:
        """DagsHub URL if configured, else a local SQLite store under the repo.

        SQLite (not file://) so the MLflow Model Registry works locally too —
        the registry is unsupported on a file store.
        """
        return self.mlflow_tracking_uri or f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_params(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else PARAMS_PATH
    with p.open() as f:
        return yaml.safe_load(f)
