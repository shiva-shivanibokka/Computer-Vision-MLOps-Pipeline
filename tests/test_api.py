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


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_predict_reports_image_size_so_the_panel_can_place_boxes(client):
    r = client.post("/predict", files={"file": ("pcb.png", _png_bytes(), "image/png")})
    assert r.json()["image_size"] == [32, 32]


def test_model_info_describes_the_serving_config(client):
    body = client.get("/model/info").json()
    assert body["model_version"] == "test:stub"
    assert body["inference_imgsz"] == 1280  # must match what the model trained at
    assert len(body["classes"]) == 6


def test_sample_endpoint_refuses_unknown_ids(client):
    assert client.post("/predict/sample", params={"sample_id": "nope"}).status_code == 404


def test_sample_endpoint_refuses_path_traversal(client):
    """The id indexes the manifest; it must never reach the filesystem as a path."""
    for evil in ["../../artifacts/best", "..\\..\\params", "/etc/passwd"]:
        assert client.post("/predict/sample", params={"sample_id": evil}).status_code == 404


def test_shipped_samples_are_scorable(client):
    from cvmlops.serve.app import _sample_manifest

    manifest = _sample_manifest()
    assert manifest, "no sample boards shipped"
    r = client.post("/predict/sample", params={"sample_id": manifest[0]["id"]})
    assert r.status_code == 200
    assert r.json()["detections"][0]["label"] == "short"  # stubbed model


def test_every_sample_has_ground_truth_labels(client):
    from cvmlops.serve.app import _sample_manifest

    for s in _sample_manifest():
        assert s["truth"], f"{s['id']} ships with no labels, so misses cannot be shown"
        for t in s["truth"]:
            assert len(t["box"]) == 4
            assert t["box"][0] < t["box"][2] and t["box"][1] < t["box"][3]
