"""Drift check entrypoint for the scheduled retrain workflow.

Compares recent production predictions against the training reference. Writes an
Evidently HTML report and prints a machine-readable verdict. Exit code 10 =
drift detected (workflow triggers retrain), 0 = no drift, 2 = not enough data.

Run:  python -m cvmlops.monitor.check
"""

from __future__ import annotations

import sys
from pathlib import Path

from cvmlops.config import REPO_ROOT, load_params
from cvmlops.monitor import logging_store
from cvmlops.monitor.drift import compute_drift, reference_from_training

REPORT = REPO_ROOT / "artifacts" / "drift_report.html"
DRIFT_EXIT = 10


def main(min_current: int = 30) -> int:
    # Compare only the most-recent window so a real recent shift isn't diluted
    # by all historical predictions.
    window = load_params()["monitor"]["current_window"]
    current = logging_store.load_predictions(limit=window)
    if len(current) < min_current:
        print(f"drift=skip reason=only {len(current)} predictions (< {min_current})")
        return 2

    reference = reference_from_training()
    result = compute_drift(reference, current, html_out=REPORT)
    print(f"drift={str(result.drifted).lower()} share={result.drift_share:.3f} "
          f"n_drifted={result.n_drifted}/{result.n_columns} report={REPORT}")
    return DRIFT_EXIT if result.drifted else 0


if __name__ == "__main__":
    Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
    sys.exit(main())
