"""API tests with a stubbed model — no weights download, fully offline."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cvmlops.serve import model as model_mod
from cvmlops.serve.model import Detection


class FakeModel:
    version = "test:stub"

    def predict(self, img, conf=0.25):
        return [Detection(label="short", confidence=0.9, box=[1.0, 2.0, 3.0, 4.0])]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(model_mod.ModelService, "_instance", FakeModel())
    from cvmlops.serve.app import app
    with TestClient(app) as c:
        yield c


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test:stub"}


def test_predict_returns_detections_and_logs(client):
    r = client.post("/predict", files={"file": ("pcb.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test:stub"
    assert body["detections"][0]["label"] == "short"
    assert body["detections"][0]["confidence"] == 0.9

    # prediction was logged -> monitor summary sees it
    summary = client.get("/monitor/summary").json()
    assert summary["n"] == 1
    assert summary["model_versions"] == {"test:stub": 1}
