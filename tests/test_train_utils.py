"""Light tests for monitoring/registration entrypoints (non-heavy)."""

import pandas as pd
import pytest

from cvmlops.monitor import check


def test_check_returns_2_on_insufficient_data():
    # empty DB (conftest isolates it) -> not enough predictions -> exit code 2
    assert check.main() == 2


def test_check_requests_bounded_window(monkeypatch):
    """check.main must query only the most-recent `current_window` predictions,
    not the whole history."""
    captured = {}

    def spy(limit=None):
        captured["limit"] = limit
        return pd.DataFrame()  # empty -> main returns 2 before the heavy path

    monkeypatch.setattr(check.logging_store, "load_predictions", spy)
    assert check.main() == 2
    assert captured["limit"] == 200  # params.yaml monitor.current_window


def test_register_best_raises_without_checkpoint(monkeypatch, tmp_path):
    from cvmlops.train import register_best as rb

    monkeypatch.setattr(rb, "BEST", tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError):
        rb.register_best()
