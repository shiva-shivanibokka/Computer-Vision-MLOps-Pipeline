"""Model promotion gate: a new version only becomes `production` if it beats
the current production model on the promotion metric by the required margin.

This is the automated validation gate in the closed MLOps loop. Uses MLflow
registry aliases (the modern replacement for stages).

Run:  python -m cvmlops.registry.promote            # consider latest version
      python -m cvmlops.registry.promote --version 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from mlflow import MlflowClient

from cvmlops.config import load_params
from cvmlops.mlflow_utils import init_mlflow


@dataclass
class PromotionDecision:
    promoted: bool
    version: str
    reason: str
    candidate_metric: float | None = None
    incumbent_metric: float | None = None


def _metric_for_version(client: MlflowClient, name: str, version: str, metric: str) -> float | None:
    mv = client.get_model_version(name, version)
    if not mv.run_id:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(metric)


def _latest_version(client: MlflowClient, name: str) -> str | None:
    versions = client.search_model_versions(f"name = '{name}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))


def promote(version: str | None = None) -> PromotionDecision:
    rp = load_params()["registry"]
    name, alias, metric, margin = (
        rp["model_name"], rp["alias"], rp["promotion_metric"], rp["promotion_margin"],
    )
    init_mlflow()
    client = MlflowClient()

    version = version or _latest_version(client, name)
    if version is None:
        return PromotionDecision(False, "", "no model versions registered")

    candidate = _metric_for_version(client, name, version, metric)
    if candidate is None:
        return PromotionDecision(False, version, f"candidate has no metric '{metric}'")

    try:
        current = client.get_model_version_by_alias(name, alias)
    except Exception:
        current = None

    if current is None:
        client.set_registered_model_alias(name, alias, version)
        return PromotionDecision(True, version, "no incumbent — promoted", candidate, None)

    if current.version == version:
        return PromotionDecision(False, version, "already production", candidate, candidate)

    incumbent = _metric_for_version(client, name, current.version, metric) or 0.0
    if candidate >= incumbent + margin:
        client.set_registered_model_alias(name, alias, version)
        return PromotionDecision(
            True, version, f"{candidate:.4f} >= {incumbent:.4f}+{margin}", candidate, incumbent)
    return PromotionDecision(
        False, version, f"{candidate:.4f} < {incumbent:.4f}+{margin}", candidate, incumbent)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    d = promote(ap.parse_args().version)
    print(f"{'PROMOTED' if d.promoted else 'REJECTED'} v{d.version}: {d.reason}")
