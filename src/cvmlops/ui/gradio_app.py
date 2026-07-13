"""Gradio demo: upload a PCB image -> detections; plus an MLOps monitor tab.

Runs the model in-process (no HTTP hop) but reuses the exact serving code path
(ModelService + prediction logging), so the demo and the API stay consistent.
"""

from __future__ import annotations

import uuid

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw

from cvmlops.monitor import logging_store
from cvmlops.monitor.features import features_from
from cvmlops.serve.model import ModelService


def _detect(img: Image.Image, conf: float):
    if img is None:
        return None, pd.DataFrame(columns=["label", "confidence"])
    svc = ModelService.instance()
    detections = svc.predict(img, conf=conf)

    drawn = img.convert("RGB").copy()
    draw = ImageDraw.Draw(drawn)
    for d in detections:
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 10)), f"{d.label} {d.confidence:.2f}", fill=(255, 0, 0))

    mean_conf = sum(d.confidence for d in detections) / len(detections) if detections else 0.0
    logging_store.log_prediction(
        uuid.uuid4().hex, svc.version, features_from(img, len(detections), mean_conf))

    table = pd.DataFrame(
        [{"label": d.label, "confidence": round(d.confidence, 3)} for d in detections]
    )
    return drawn, table


def _monitor():
    df = logging_store.load_predictions(limit=500)
    if df.empty:
        return "No predictions logged yet.", pd.DataFrame()
    summary = (
        f"**Predictions logged:** {len(df)}  \n"
        f"**Model versions seen:** {', '.join(df['model_version'].unique())}  \n"
        f"**Avg detections/image:** {df['n_detections'].mean():.2f}  \n"
        f"**Avg confidence:** {df['mean_confidence'].mean():.3f}"
    )
    return summary, df.head(50)


def build() -> gr.Blocks:
    with gr.Blocks(title="PCB Defect Detector") as demo:
        gr.Markdown("# PCB Defect Detector\nUpload a PCB image to detect defects.")
        with gr.Tab("Detect"):
            with gr.Row():
                inp = gr.Image(type="pil", label="PCB image")
                out = gr.Image(type="pil", label="Detections")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            tbl = gr.Dataframe(headers=["label", "confidence"], label="Detected defects")
            gr.Button("Detect", variant="primary").click(_detect, [inp, conf], [out, tbl])
        with gr.Tab("MLOps Monitor"):
            md = gr.Markdown()
            hist = gr.Dataframe(label="Recent predictions")
            gr.Button("Refresh").click(_monitor, None, [md, hist])
            demo.load(_monitor, None, [md, hist])
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860)
