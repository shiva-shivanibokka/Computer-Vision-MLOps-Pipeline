# Computer Vision MLOps Pipeline — PCB Defect Detection

> [!IMPORTANT]
> **The hosted demo is temporary.** This project's backend runs on Google Cloud
> Run under a Google Cloud free trial that ends **around 19 September 2026**.
> When the trial closes the service is stopped, and every `run.app` link below
> stops responding.
>
> Nothing in this repository depends on that. The code, tests and results are
> complete, and the instructions below run the whole thing locally.


An end-to-end, production-style MLOps system for detecting manufacturing defects
on printed circuit boards with a YOLO object detector. The point of this project
is not the model — it's the **closed MLOps loop** around it: versioned data,
tracked experiments, a gated model registry, containerized serving, drift
monitoring, and drift-triggered automated retraining. Everything runs on **free
tiers, no credit card**.

### ▶ [Open the live demo](https://pcb-defect-detector-548930096299.us-central1.run.app)

Running on Cloud Run — **control panel** at
[`/ui`](https://pcb-defect-detector-548930096299.us-central1.run.app/ui/),
**API docs** at
[`/docs`](https://pcb-defect-detector-548930096299.us-central1.run.app/docs).
Real weights, real inference, nothing pre-recorded. Start on the **Manual** tab.
It scales to zero, so the first request after an idle spell waits about 10
seconds for the container to wake and load the model.

## What the demo is honest about

Three things the panel deliberately shows rather than hides, because each one is
a question an interviewer will ask:

- **It finds about a third of the defects.** mAP@50 is 0.336 and recall is 0.355.
  The six sample boards come from the labelled validation set, so the panel draws
  a dashed box around **every defect the model missed** and the counter reads
  `found 3 of 6`, not `3 detections`.
- **A model version has to earn its way into production.** The gate compares
  `mAP50-95` against the incumbent. v1, v2 and v3 cleared it. **v4 did not** — the
  tiling experiment — and was never served.
- **The ceiling is resolution, not effort.** v2 and v3 are the same model at the
  same image size; v3 had 3× the epochs and bought 0.311 → 0.336. The Limits tab
  shows that evidence and explains the tiling failure that came from 32% empty
  tiles teaching the model to predict nothing.

## The control panel

Three static files (`web/`) served by the same FastAPI process that runs the
model — no build step, no second origin, no CORS. Six tabs: **Manual · Inspect ·
Registry · Drift · Limits · API**, themed off the artefact itself (FR4 board
green, copper trace, gold pad, silkscreen white).

Uploaded images are scored and discarded. Only derived statistics — brightness,
contrast, sharpness, detection count, mean confidence — reach the prediction log,
which is what drift monitoring actually needs.

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
     └────────────  SQLite  ◀── log every prediction ◀── /predict, control panel
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
| Control panel | Hand-built static HTML/CSS/JS | open source |
| Containers | Docker + docker-compose | open source |
| CI/CD | GitHub Actions | free (public repo) |
| Deploy | **Google Cloud Run** (scale to zero) | free tier |
| Drift monitoring | Evidently + SQLite | open source |
| Quality | pytest, ruff, pre-commit | open source |

## Quickstart (fully local, no accounts)

```bash
pip install -e ".[dev,mlops]"   # mlops extra = mlflow + dvc + evidently

python -m cvmlops.data.prepare        # build dataset (synthetic fallback if no raw data)
python -m cvmlops.train.train         # train, log to local MLflow, register v1
python -m cvmlops.registry.promote    # promote v1 to `production`

uvicorn cvmlops.serve.asgi:app --port 7860
#   Control panel: http://localhost:7860/ui
#   API docs:      http://localhost:7860/docs
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

## Training on real PCB data (one GPU run)

The default dataset is **HRIPCB / PKU-Market-PCB** — its 6 defect classes already
match `params.yaml`. Download it (Kaggle `akhatova/pcb-defects` or GitHub
[PCB-DATASET](https://github.com/Ironbrotherstyle/PCB-DATASET)), extract, then:

```bash
python -m cvmlops.data.convert_hripcb --src /path/to/PCB_DATASET   # VOC XML -> YOLO
python -m cvmlops.data.prepare                                     # split train/val
python -m cvmlops.train.train                                      # trains on GPU if present
python -m cvmlops.registry.promote                                 # gate -> production
```

`train.py` auto-uses the GPU (`device: ""` in `params.yaml` = auto). To host the
run in DagsHub's MLflow UI, set the `.env` vars **before** training (local runs
don't migrate). Any other YOLO-format dataset also works — just drop
`images/*.jpg` + `labels/*.txt` into `data/pcb_raw/` and skip the convert step.
With no raw data present, `prepare.py` generates a synthetic dataset so the
pipeline always runs.

## Connecting the free accounts (do this last)

Everything above works with **zero accounts** (local SQLite MLflow, local weights).
To go to the hosted, production configuration, set these and nothing else changes:

1. **DagsHub** — create a repo, copy `.env.example` → `.env`, fill in
   `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`.
   Add the same as GitHub repo secrets for the retrain workflow. Configure the
   DVC remote: `dvc remote add origin s3://... ` (DagsHub gives the exact command).
2. **Deploy** — one command, no secrets required:
   ```bash
   gcloud run deploy pcb-defect-detector --source . --region us-central1      --port 7860 --memory 2Gi --cpu 2 --max-instances 1 --cpu-boost
   ```
   `--max-instances 1` is deliberate: the prediction log is SQLite inside the
   container, so a second instance would be a second, disagreeing log.

## Layout

```
src/cvmlops/
  data/       prepare + synthetic dataset (YOLO format)
  train/      YOLO training + MLflow logging + registration
  registry/   promotion gate (alias `production`)
  serve/      FastAPI app, model loader, combined ASGI (API + panel)
  monitor/    prediction log (SQLite), Evidently drift, drift-check entrypoint
web/          the control panel: index.html, styles.css, app.js, sample boards
tests/        data, monitor, api, registry, end-to-end smoke train
.github/workflows/   ci.yml (lint/test/build/deploy), retrain.yml (drift loop)
dvc.yaml · params.yaml · Dockerfile · docker-compose.yml
```
