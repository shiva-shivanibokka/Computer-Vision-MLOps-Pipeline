"""FastAPI inference service.

Loads the production model at startup, serves detections, and logs every
prediction (image features + outputs) to SQLite for monitoring/drift.
"""

from __future__ import annotations

import io
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from cvmlops.monitor import logging_store
from cvmlops.monitor.features import features_from
from cvmlops.serve.model import Detection, ModelService

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class DetectionOut(BaseModel):
    label: str
    confidence: float
    box: list[float]


class PredictResponse(BaseModel):
    # field starts with "model_" — opt out of pydantic's protected namespace.
    model_config = ConfigDict(protected_namespaces=())
    request_id: str
    model_version: str
    detections: list[DetectionOut]


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelService.instance()  # warm the model on boot
    yield


app = FastAPI(title="PCB Defect Detector", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", model_version=ModelService.instance().version)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), conf: float = 0.25) -> PredictResponse:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(400, "invalid or unreadable image file") from e

    svc = ModelService.instance()
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
        detections=[DetectionOut(**d.__dict__) for d in detections],
    )


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
