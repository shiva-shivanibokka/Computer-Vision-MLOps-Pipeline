# Fix Plan — Computer-Vision-MLOps-Pipeline

Generated from repo-bug-audit on 2026-07-13. 11 tasks, ordered by severity (Major first).
None of these block the next training run — the training path is clean.

**STATUS: all 11 tasks applied on 2026-07-13** (commit follows). 17 non-heavy tests green.
Kept as the record of what was changed and why.

Convention: test-first where a behavior changes. Run `pytest -m "not heavy" -q` after each
task; it must stay green. Do NOT run the heavy smoke test while a real checkpoint exists in
`artifacts/runs/train/` (it overwrites it with synthetic data).

---

## Task 1: Test — promotion gate must NOT bypass on registry error (F1)

- **File:** `tests/test_registry.py`
- **Category:** Silent failure / test coverage
- **Severity:** Major
- **Finding:** No test proves the gate re-raises non-"not-found" errors instead of promoting.
- **Proposed change:** add a test that monkeypatches `get_model_version_by_alias` to raise a
  generic error and asserts `promote()` raises (not silently promotes):
  ```python
  def test_promote_reraises_on_registry_error(tmp_path, monkeypatch):
      from cvmlops.registry import promote as promo
      init_mlflow()
      v1 = _register(0.30, tmp_path)
      import mlflow
      def boom(*a, **k): raise RuntimeError("registry down")
      monkeypatch.setattr(mlflow.MlflowClient, "get_model_version_by_alias", boom)
      import pytest
      with pytest.raises(RuntimeError):
          promo.promote(v1)
  ```
- **Verification:** `pytest tests/test_registry.py -k reraises -q` — fails before Task 2, passes after.

## Task 2: Fix — narrow the exception in the promotion gate (F1)

- **File:** `src/cvmlops/registry/promote.py:62-65`
- **Category:** Silent failure
- **Severity:** Major
- **Finding:** `except Exception` treats any error as "no incumbent" → unconditional promote.
- **Proposed change:**
  ```python
  from mlflow.exceptions import MlflowException
  ...
  try:
      current = client.get_model_version_by_alias(name, alias)
  except MlflowException as e:
      # Only "alias/model not found" means there's no incumbent; anything else
      # (network, auth) must NOT silently bypass the gate.
      if getattr(e, "error_code", "") in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND") \
              or "not found" in str(e).lower() or "does not exist" in str(e).lower():
          current = None
      else:
          raise
  ```
- **Verification:** `pytest tests/test_registry.py -q` (all pass, incl. Task 1's test).
- **Depends on:** Task 1.

## Task 3: Fix — drift reference must not always-drift on prediction outputs (F2)

- **File:** `src/cvmlops/monitor/features.py`, `src/cvmlops/monitor/drift.py`
- **Category:** Correctness
- **Severity:** Major
- **Finding:** reference sets `n_detections=0`/`mean_confidence=0`, so those two columns always
  drift vs live values, pinning `drift_share` at ~0.33.
- **Proposed change:** split feature sets — monitor **data drift** on input-image features only:
  ```python
  # features.py
  INPUT_FEATURE_COLS = ["brightness", "contrast", "width", "height"]
  # (keep FEATURE_COLS = INPUT_FEATURE_COLS + ["n_detections", "mean_confidence"] for logging)
  ```
  In `drift.py`, use `INPUT_FEATURE_COLS` in `_to_dataset` and drop the zeroed output columns
  from `reference_from_training`. Prediction-output drift can be a separate future metric.
- **Verification:** add/extend a test in `tests/test_monitor.py`: identical input features across
  ref/current → `drifted is False` and `n_columns == 4`. `pytest tests/test_monitor.py -q`.

## Task 4: Fix — drift uses a recent window, not all history (F3)

- **File:** `src/cvmlops/monitor/check.py:24`, `params.yaml`
- **Category:** Correctness
- **Severity:** Major
- **Finding:** `load_predictions()` (unbounded) makes drift reflect lifetime, not recent, data.
- **Proposed change:** add `monitor.current_window: 200` to `params.yaml`; in `check.py`:
  ```python
  window = load_params()["monitor"]["current_window"]
  current = logging_store.load_predictions(limit=window)
  ```
- **Verification:** `python -m cvmlops.monitor.check` still returns exit 2 on <30 predictions;
  add a unit test that seeds >window rows and asserts only `window` are compared.

## Task 5: Fix — offload blocking inference off the event loop (F4)

- **File:** `src/cvmlops/serve/app.py:53-62`
- **Category:** Concurrency
- **Severity:** Minor
- **Proposed change:**
  ```python
  from starlette.concurrency import run_in_threadpool
  ...
  detections = await run_in_threadpool(svc.predict, img, conf)
  await run_in_threadpool(logging_store.log_prediction, request_id, svc.version,
                          features_from(img, len(detections), mean_conf))
  ```
- **Verification:** `pytest tests/test_api.py -q` stays green.

## Task 6: Fix — validate uploaded file + size cap (F5)

- **File:** `src/cvmlops/serve/app.py:55`
- **Category:** Security
- **Severity:** Minor
- **Proposed change:** cap read size (e.g. 10 MB) and wrap decode:
  ```python
  raw = await file.read()
  if len(raw) > 10 * 1024 * 1024:
      raise HTTPException(413, "image too large (max 10MB)")
  try:
      img = Image.open(io.BytesIO(raw)).convert("RGB")
  except Exception:
      raise HTTPException(400, "invalid image file")
  ```
- **Verification:** add `tests/test_api.py::test_predict_rejects_non_image` (post text → 400).

## Task 7: Fix — SQLite WAL + busy timeout (F6)

- **File:** `src/cvmlops/monitor/logging_store.py:33-36`
- **Category:** Concurrency
- **Severity:** Minor
- **Proposed change:**
  ```python
  def _connect():
      conn = sqlite3.connect(_db_path(), timeout=5.0)
      conn.execute("PRAGMA journal_mode=WAL")
      conn.execute("PRAGMA busy_timeout=5000")
      conn.execute(_SCHEMA)
      return conn
  ```
- **Verification:** `pytest tests/test_monitor.py -q` green; roundtrip test unaffected.

## Task 8: Fix — type-safe version comparison in gate (F7)

- **File:** `src/cvmlops/registry/promote.py:71`
- **Category:** Consistency
- **Severity:** Minor
- **Proposed change:** `if str(current.version) == str(version):`
- **Verification:** `pytest tests/test_registry.py::test_promote_same_version_is_noop -q`.

## Task 9: Fix — silence pydantic protected-namespace warning (F8)

- **File:** `src/cvmlops/serve/app.py`
- **Category:** Note
- **Severity:** Note
- **Proposed change:** on `PredictResponse` and `Health`:
  ```python
  from pydantic import ConfigDict
  model_config = ConfigDict(protected_namespaces=())
  ```
- **Verification:** `pytest -q` shows no "protected namespace" warnings.

## Task 10: Fix — accurate MLflow params on resume (F9)

- **File:** `src/cvmlops/train/train.py:46-49`
- **Category:** Note
- **Severity:** Note
- **Proposed change:** when `resume=True`, tag the run `resumed_from=last.pt` and read effective
  args from `model.trainer.args` after training, logging those instead of the params.yaml values.
- **Verification:** manual — inspect the resumed run's params in MLflow after a resume.

## Task 11: Backfill tests for register_best + check exit codes (F10)

- **File:** `tests/test_train_utils.py` (new)
- **Category:** Test coverage
- **Severity:** Note
- **Proposed change:** light tests: `check.main()` returns 2 with an empty DB;
  `register_best` raises `FileNotFoundError` when no checkpoint exists (both non-heavy).
- **Verification:** `pytest tests/test_train_utils.py -q`.
