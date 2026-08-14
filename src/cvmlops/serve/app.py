"""FastAPI inference service.

Loads the production model at startup, serves detections, and logs every
prediction (image features + outputs) to SQLite for monitoring/drift.
"""

from __future__ import annotations

import io
import json
import uuid
from contextlib import asynccontextmanager
from threading import Thread
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from cvmlops.config import REPO_ROOT, load_params
from cvmlops.monitor import logging_store
from cvmlops.monitor.features import features_from
from cvmlops.serve.model import Detection, ModelService

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Bound the threshold at the edge. Ultralytics asserts on conf outside [0, 1]
# deep inside predict(), which surfaces to the caller as a 500 — a validation
# error reported as a server fault. Constraining it here turns that into a 422
# naming the offending value, and keeps the two entry points consistent.
ConfThreshold = Annotated[float, Query(ge=0.0, le=1.0)]
WEB_DIR = REPO_ROOT / "web"
SAMPLES_DIR = WEB_DIR / "samples"


class DetectionOut(BaseModel):
    label: str
    confidence: float
    box: list[float]


class PredictResponse(BaseModel):
    # field starts with "model_" — opt out of pydantic's protected namespace.
    model_config = ConfigDict(protected_namespaces=())
    request_id: str
    model_version: str
    image_size: list[int]  # [w, h] — the panel scales boxes to its rendered size
    detections: list[DetectionOut]


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model off the event loop. Uvicorn binds the socket only after
    # lifespan startup returns, so loading torch + YOLO inline would leave the
    # port closed for the whole warm-up — the panel is served by this same
    # process, so the page itself would be unreachable rather than merely
    # showing "still loading".
    Thread(target=ModelService.instance, daemon=True, name="warm-model").start()
    yield


app = FastAPI(title="PCB Defect Detector", version="0.1.0", lifespan=lifespan)


def _require_model() -> ModelService:
    """Block until the model is loaded, rather than 503-ing while it warms.

    Counter-intuitive on Cloud Run, but correct there: with default CPU
    throttling the container is only given CPU *while a request is in flight*.
    A warm-up thread therefore makes almost no progress between requests, and a
    poll loop that returns 503 in 50ms buys the loader 50ms of CPU per poll —
    warm-up effectively never finishes. Holding this request open instead gives
    the loader a full core until it is done. The load is ~15s and the platform
    request timeout is 300s.

    The thread started in lifespan is still worth keeping: it uses the CPU of
    whatever request happens to arrive first, and instance() is lock-guarded, so
    a caller that arrives mid-load waits for that same load rather than starting
    a second one.
    """
    return ModelService.instance()


@app.get("/health", response_model=Health)
def health() -> Health:
    """Always 200 — liveness. Readiness is the model_version field."""
    ready = ModelService.ready()
    return Health(
        status="ok" if ready else "loading",
        model_version=ModelService.instance().version if ready else "loading",
    )


async def _detect(img: Image.Image, conf: float) -> PredictResponse:
    """Shared path for uploads and shipped samples: infer, log, respond."""
    svc = _require_model()
    # Inference and the SQLite write are blocking — keep them off the event loop.
    detections: list[Detection] = await run_in_threadpool(svc.predict, img, conf)

    request_id = uuid.uuid4().hex
    mean_conf = sum(d.confidence for d in detections) / len(detections) if detections else 0.0
    await run_in_threadpool(
        logging_store.log_prediction,
        request_id, svc.version, features_from(img, len(detections), mean_conf))

    return PredictResponse(
        request_id=request_id,
        model_version=svc.version,
        image_size=list(img.size),
        detections=[DetectionOut(**d.__dict__) for d in detections],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), conf: ConfThreshold = 0.25) -> PredictResponse:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(400, "invalid or unreadable image file") from e
    return await _detect(img, conf)


@app.post("/predict/sample", response_model=PredictResponse)
async def predict_sample(sample_id: str, conf: ConfThreshold = 0.25) -> PredictResponse:
    """Score one of the shipped boards, read from disk at full resolution.

    The browser only ever displays these images; sending them back up as an
    upload would re-encode a ~3000px photograph for no reason, and the defects
    this model looks for are about 70px across.
    """
    # Resolve against the known sample set rather than trusting the id — an
    # unchecked join here would read any file on the container.
    known = {s["id"] for s in _sample_manifest()}
    if sample_id not in known:
        raise HTTPException(404, f"unknown sample {sample_id!r}")
    path = SAMPLES_DIR / f"{sample_id}.jpg"
    if not path.exists():
        raise HTTPException(404, f"sample {sample_id!r} is listed but its image is missing")
    return await _detect(Image.open(path).convert("RGB"), conf)


def _sample_manifest() -> list[dict]:
    mf = SAMPLES_DIR / "manifest.json"
    if not mf.exists():
        return []
    return json.loads(mf.read_text(encoding="utf8"))


@app.get("/model/info")
def model_info() -> dict:
    """What the serving layer is actually running — the panel reads this."""
    params = load_params()
    return {
        "model_version": _require_model().version,
        "classes": params["dataset"]["classes"],
        "inference_imgsz": params["train"]["imgsz"],
        "gate_metric": params["registry"]["promotion_metric"],
        "n_samples": len(_sample_manifest()),
    }


@app.get("/monitor/summary")
def monitor_summary(limit: int = 500) -> dict:
    df = logging_store.load_predictions(limit=limit)
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "avg_detections": float(df["n_detections"].mean()),
        "avg_confidence": float(df["mean_confidence"].mean()),
        "avg_brightness": float(df["brightness"].mean()),
        "model_versions": df["model_version"].value_counts().to_dict(),
    }
