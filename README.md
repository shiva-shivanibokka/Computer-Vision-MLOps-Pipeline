---
title: PCB Defect Detector
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Computer Vision MLOps Pipeline — PCB Defect Detection

An end-to-end, production-style MLOps system for detecting manufacturing defects
on printed circuit boards with a YOLO object detector. The point of this project
is not the model — it's the **closed MLOps loop** around it: versioned data,
tracked experiments, a gated model registry, containerized serving, drift
monitoring, and drift-triggered automated retraining. Everything runs on **free
tiers, no credit card**.

## The closed MLOps loop

```
 data/pcb ──DVC──▶ train (YOLO) ──▶ MLflow: log run + register version
     ▲                                          │
     │                                   promotion gate  (new model must beat
     │                                          │         current production)
  retrain  ◀── drift? ──┐                       ▼
  (GitHub Actions)      │                set alias: production
     ▲                  │                        │
     │             Evidently                     ▼
     │            drift report          FastAPI loads production model
     └────────────  SQLite  ◀── log every prediction ◀── /predict, Gradio UI
```

1. **Version** — `dvc repro` regenerates the dataset and trains; data + weights tracked by DVC.
2. **Track + register** — every training run is logged to MLflow and registered as a new model version.
3. **Gate** — `registry/promote.py` only moves a version to the `production` alias if it beats the incumbent on mAP@50-95.
4. **Serve** — FastAPI loads the `production` model from the registry (falls back to local weights, then base weights, so it always boots).
5. **Monitor** — every prediction's image features + outputs are logged to SQLite; Evidently computes data drift.
6. **Auto-retrain** — a scheduled GitHub Action checks drift and, if detected, retrains → re-gates → the next CI run redeploys.

## Tech stack (all free)

| Concern | Tool | Free tier |
|---|---|---|
| Detector | PyTorch + Ultralytics YOLO | open source |
| Data + model versioning | DVC → **DagsHub** remote | free storage |
| Experiment tracking + registry | **MLflow** on **DagsHub** | free managed server |
| Serving API | FastAPI + Uvicorn | open source |
| Demo UI | Gradio | open source |
| Containers | Docker + docker-compose | open source |
| CI/CD | GitHub Actions | free (public repo) |
| Deploy | **Hugging Face Spaces** (Docker SDK) | free |
| Drift monitoring | Evidently + SQLite | open source |
| Quality | pytest, ruff, pre-commit | open source |

## Quickstart (fully local, no accounts)

```bash
pip install -e .[dev]

python -m cvmlops.data.prepare        # build dataset (synthetic fallback if no raw data)
python -m cvmlops.train.train         # train, log to local MLflow, register v1
python -m cvmlops.registry.promote    # promote v1 to `production`

uvicorn cvmlops.serve.asgi:app --port 7860
#   API docs: http://localhost:7860/docs
#   Gradio UI: http://localhost:7860/ui
```

Or the whole stack in Docker:

```bash
docker compose up --build
```

Run the tests:

```bash
pytest -m "not heavy"     # fast: data, monitor, api, registry
pytest tests/test_smoke_train.py   # end-to-end: train -> register -> promote
```

## Using real PCB data

Drop a YOLO-format PCB defect dataset (e.g. [DeepPCB](https://github.com/tangsanli5201/DeepPCB)
or a Roboflow export) into `data/pcb_raw/` as `images/*.jpg` + `labels/*.txt`.
`python -m cvmlops.data.prepare` splits it into train/val and writes `data.yaml`.
With no raw data present it generates a synthetic dataset so the pipeline always runs.

## Connecting the free accounts (do this last)

Everything above works with **zero accounts** (local SQLite MLflow, local weights).
To go to the hosted, production configuration, set these and nothing else changes:

1. **DagsHub** — create a repo, copy `.env.example` → `.env`, fill in
   `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`.
   Add the same as GitHub repo secrets for the retrain workflow. Configure the
   DVC remote: `dvc remote add origin s3://... ` (DagsHub gives the exact command).
2. **Hugging Face** — create a Docker Space, add `HF_TOKEN` and `HF_SPACE`
   (`user/space-name`) as GitHub secrets; `ci.yml` pushes to it on merge to `main`.

## Layout

```
src/cvmlops/
  data/       prepare + synthetic dataset (YOLO format)
  train/      YOLO training + MLflow logging + registration
  registry/   promotion gate (alias `production`)
  serve/      FastAPI app, model loader, combined ASGI (API + UI)
  monitor/    prediction log (SQLite), Evidently drift, drift-check entrypoint
  ui/         Gradio app
tests/        data, monitor, api, registry, end-to-end smoke train
.github/workflows/   ci.yml (lint/test/build/deploy), retrain.yml (drift loop)
dvc.yaml · params.yaml · Dockerfile · docker-compose.yml
```
