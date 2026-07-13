# PCB Defect Detector — API + Gradio UI in one container (HF Spaces Docker SDK).
FROM python:3.11-slim

# System libs ultralytics/opencv need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml ./
COPY src ./src
COPY params.yaml ./
RUN pip install --upgrade pip && pip install .

# Weights baked in if present (produced by training / DVC pull); otherwise the
# app falls back to base yolov8n.pt at runtime. Copy the dir so build never
# fails when best.pt is absent (only .gitkeep present).
COPY artifacts ./artifacts

EXPOSE 7860
CMD ["uvicorn", "cvmlops.serve.asgi:app", "--host", "0.0.0.0", "--port", "7860"]
