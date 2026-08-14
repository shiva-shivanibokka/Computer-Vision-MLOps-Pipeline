"""Config: static pipeline params (params.yaml) + runtime env (accounts/secrets).

Account wiring lives entirely in env vars so the whole stack runs locally with
zero accounts, and "connecting the accounts later" is just setting variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_root() -> Path:
    """Locate the directory holding params.yaml, web/ and artifacts/.

    Walking up from __file__ only works in a source checkout. In the container
    the package is pip-installed, so __file__ lives in site-packages and
    parents[2] is /usr/local/lib/python3.11 — where params.yaml has never
    existed. That failure is invisible locally and total in the image: the model
    cannot resolve its weights and /ui is never mounted.

    Candidates in order of authority: an explicit override, the source-checkout
    layout, then the working directory (WORKDIR /app in the container).
    """
    candidates = []
    if env := os.environ.get("CVMLOPS_ROOT"):
        candidates.append(Path(env))
    candidates += [Path(__file__).resolve().parents[2], Path.cwd()]
    for c in candidates:
        if (c / "params.yaml").is_file():
            return c
    return candidates[-1]


REPO_ROOT = _find_root()
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
