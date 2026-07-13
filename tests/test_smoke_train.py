"""End-to-end smoke: train 1 epoch on synthetic data -> registered version with
metrics -> promotion gate promotes it. Heavy (needs ultralytics/torch)."""

import pytest

pytest.importorskip("ultralytics")

from cvmlops.registry.promote import promote  # noqa: E402
from cvmlops.train.train import train  # noqa: E402


@pytest.mark.heavy
def test_smoke_train_registers_and_promotes():
    result = train(smoke=True)
    assert result["version"]
    assert "mAP50-95" in result["metrics"]

    decision = promote(result["version"])
    assert decision.promoted  # first model -> becomes production
