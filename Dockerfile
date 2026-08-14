# PCB Defect Detector — API + control panel in one container (HF Spaces Docker SDK).
FROM python:3.11-slim

# System libs ultralytics/opencv need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as a non-root user with only /tmp writable, so
# every library that wants a cache dir gets pointed there.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/mpl \
    PREDICTION_DB=/tmp/predictions.db

WORKDIR /app

# CPU-only torch first, from PyTorch's CPU index. Installing it via the default
# index instead pulls ~2.5GB of nvidia-* CUDA wheels into an image that will
# only ever run on a CPU Space.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Then the serving deps (pyproject's base set — no mlflow/dvc/evidently).
COPY pyproject.toml ./
COPY src ./src
COPY params.yaml ./
RUN pip install .

# Trained weights, baked in. The app falls back to base yolov8n.pt if absent,
# which boots but knows nothing about PCBs — so this COPY is what makes the
# demo real.
COPY artifacts ./artifacts

# The control panel: three static files plus the six sample boards.
COPY web ./web

EXPOSE 7860
CMD ["uvicorn", "cvmlops.serve.asgi:app", "--host", "0.0.0.0", "--port", "7860"]
