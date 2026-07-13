"""FastAPI inference service.

Loads the production model at startup, serves detections, and logs every
prediction (image features + outputs) to SQLite for monitoring/drift.
"""

from __future__ import annotations

import io
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from cvmlops.monitor import logging_store
from cvmlops.monitor.features import features_from
from cvmlops.serve.model import Detection, ModelService


class DetectionOut(BaseModel):
    label: str
    confidence: float
    box: list[float]


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    detections: list[DetectionOut]


class Health(BaseModel):
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
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    svc = ModelService.instance()
    detections: list[Detection] = svc.predict(img, conf=conf)

    request_id = uuid.uuid4().hex
    mean_conf = sum(d.confidence for d in detections) / len(detections) if detections else 0.0
    logging_store.log_prediction(
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
