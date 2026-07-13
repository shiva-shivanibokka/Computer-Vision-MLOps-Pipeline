# Repo Audit Report — Computer-Vision-MLOps-Pipeline

**Date:** 2026-07-13
**Stack detected:** Python 3.12 · FastAPI · Ultralytics YOLO / PyTorch · MLflow · DVC · Evidently · Gradio
**Scope:** all of `src/cvmlops/**` + `tests/**`. Excluded: generated artifacts, `data/`, model weights.

## Summary

- Total findings: 10 (1 auto-fixed, 9 need review)
- Auto-fixed (trivial-safe): 1
- Needs review (see PLAN.md): 9
- Critical: 0 | Major: 3 | Minor: 4 | Notes: 2

**Bottom line:** the **training path is clean** — none of the findings block the next
training run. The real issues cluster in the **promotion gate**, **drift monitoring**,
and **serving** layers: a safety-gate bypass on error, a drift baseline that always
reports two columns as drifted, and a drift window that never reflects recent data.

## Production-readiness scorecard

| Category | Status | Notes |
|---|---|---|
| Correctness | ⚠️ | Drift baseline + drift window logic wrong (F2, F3) |
| Silent failures | ❌ | Promotion gate bypassed on registry error (F1) |
| Security | ⚠️ | Public inference endpoint: no upload validation / rate limit (F5) |
| Concurrency | ⚠️ | Blocking inference + SQLite writes on the async event loop (F4, F6) |
| Performance | ✅ | No N+1 / unbounded-growth issues in the hot path |
| Architecture | ✅ | Clean layering (data/train/registry/serve/monitor/ui), no cycles |
| Production-readiness | ⚠️ | Good fallbacks & logging; hardening gaps above |
| Test coverage | ⚠️ | Loop-critical paths (gate bypass, drift baseline) untested (F10) |

## Auto-fixed (trivial-safe)

- `src/cvmlops/config.py:25` — stale comment said MLflow defaults to a `./mlruns` file
  store; the code now defaults to SQLite. Corrected the comment. No behavior change.

## Findings requiring review

### Silent failures

**F1 — `src/cvmlops/registry/promote.py:62-65` · Major**
Broad `except Exception` when fetching the current `production` alias treats *any*
error (network blip, auth failure on DagsHub) as "no incumbent" and then promotes the
candidate unconditionally.
*Why it matters:* in the scheduled auto-retrain loop, a transient registry error would
bypass the entire validation gate and push an unvalidated model to production — the
exact failure the gate exists to prevent. Should catch only the "alias not found" case
and re-raise everything else.

### Correctness

**F2 — `src/cvmlops/monitor/drift.py:77` · Major**
The drift *reference* is built with `n_detections=0.0` and `mean_confidence=0.0` for
every row, but live predictions have real non-zero values. Those two of six monitored
columns therefore drift essentially always, pinning `drift_share` at a ~0.33 floor.
*Why it matters:* drift is a showcased feature and the retrain trigger. A baseline that
always shows 2/6 columns drifted makes the detector over-sensitive (only one more real
drift trips a retrain) and semantically wrong. Monitor input-image features for data
drift and treat prediction outputs as a separate signal, or compute a real baseline.

**F3 — `src/cvmlops/monitor/check.py:24` · Major**
`load_predictions()` pulls the entire prediction history as the "current" window,
though the docstring says "recent." Drift is computed against lifetime data.
*Why it matters:* a recent distribution shift gets diluted by all past data, so the
detector won't fire when it should. Use a bounded recent window (configurable N).

### Concurrency

**F4 — `src/cvmlops/serve/app.py:53-62` · Minor**
`async def predict` runs blocking model inference and a synchronous SQLite write
directly on the event loop.
*Why it matters:* under concurrent requests, one inference blocks all others (defeats
async). Offload to a threadpool (`starlette.concurrency.run_in_threadpool`) or make the
route a plain `def`.

**F6 — `src/cvmlops/monitor/logging_store.py:33-36` · Minor**
SQLite connections open without WAL mode or a busy timeout.
*Why it matters:* concurrent prediction writes from the serving endpoint can raise
`database is locked`. Set `PRAGMA journal_mode=WAL` and a `busy_timeout` on connect.

### Security

**F5 — `src/cvmlops/serve/app.py:55` · Minor**
No validation or size limit on the uploaded file; a non-image or very large upload
raises an unhandled 500 / risks OOM on a public endpoint.
*Why it matters:* on a public HF Space, malformed input crashes the request path and
large uploads can exhaust memory. Wrap `Image.open` → HTTP 400 and cap upload size.
(Also: the endpoint has no auth/rate limit — acceptable for a demo, but note it.)

### Consistency / correctness

**F7 — `src/cvmlops/registry/promote.py:71` · Minor**
`current.version == version` compares values that can be `str` vs `int` (MLflow returns
version as a string in some calls, and `_latest_version` stringifies). The
"already production" no-op branch can be skipped, causing a harmless but incorrect
re-promotion and misleading output. Compare `str(...) == str(...)`.

### Notes

**F8 — `src/cvmlops/serve/app.py:28,34` · Note**
Pydantic field `model_version` collides with the protected `model_` namespace, emitting
a warning on every import. Add `model_config = ConfigDict(protected_namespaces=())`.

**F9 — `src/cvmlops/train/train.py:46-49` · Note**
On `--resume`, logged MLflow params (epochs/imgsz/batch) come from the *current*
`params.yaml`, but the resumed run uses the checkpoint's original args. If params
changed between runs, the logged params mislead. Tag resumed runs or read effective args.

### Test coverage (F10 · Note)

No tests cover the loop-critical failure paths: promotion-gate behavior on a registry
error (F1), the drift-baseline column issue (F2), `check.py` exit codes (F3), or
`register_best`. These should get tests alongside their fixes (test-first).

## Clean areas

- **Training pipeline** (`train/train.py`, `data/*`, `convert_hripcb.py`) — correct, no findings; verified end-to-end with real HRIPCB data.
- **Architecture** — clean layer separation, no circular imports, no god objects.
- **Secrets** — none hardcoded; all via env (`.env` gitignored, `.env.example` documents).
- **Model loading fallback** (`serve/model.py`) — the broad `except` there is intentional, logged graceful degradation for boot; acceptable by design.
- **DVC / CI / Docker** config — sound.
