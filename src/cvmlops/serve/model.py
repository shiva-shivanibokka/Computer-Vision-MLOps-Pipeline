"""Model loading + inference for serving.

Resolution order (production first, always-bootable last):
  1. MLflow registry alias `production`  (the real production path)
  2. local artifacts/best.pt             (last trained model, offline)
  3. base yolov8n.pt                      (untrained fallback so the API boots)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from PIL import Image

from cvmlops.config import REPO_ROOT, get_settings, load_params


@dataclass
class Detection:
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2] in pixels


class ModelService:
    """Lazily-loaded, thread-safe singleton wrapping a YOLO model."""

    _instance: ModelService | None = None
    _lock = Lock()

    def __init__(self) -> None:
        self.version = "unknown"
        self._model = None

    @classmethod
    def instance(cls) -> ModelService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._instance._load()
            return cls._instance

    def _load(self) -> None:
        from ultralytics import YOLO

        weights, version = self._resolve_weights()
        self._model = YOLO(weights)
        self.version = version

    def _resolve_weights(self) -> tuple[str, str]:
        rp = load_params()["registry"]
        # 1) MLflow registry production alias
        try:
            path, ver = self._download_production(rp["model_name"], rp["alias"])
            if path:
                return path, f"{rp['model_name']}@{rp['alias']}:v{ver}"
        except Exception as e:  # registry unreachable / no alias yet
            print(f"[model] registry unavailable ({e!r}); falling back to local weights")

        # 2) local best.pt
        local = Path(get_settings().local_weights)
        if not local.is_absolute():
            local = REPO_ROOT / local
        if local.exists():
            return str(local), f"local:{local.name}"

        # 3) base pretrained — API still boots, just not PCB-trained
        print("[model] no trained weights found; using base yolov8n.pt")
        return "yolov8n.pt", "base:yolov8n"

    @staticmethod
    def _download_production(name: str, alias: str) -> tuple[str | None, str | None]:
        import mlflow
        from mlflow import MlflowClient

        from cvmlops.mlflow_utils import init_mlflow

        init_mlflow()
        client = MlflowClient()
        mv = client.get_model_version_by_alias(name, alias)
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="model")
        pt = next(Path(local_dir).rglob("*.pt"), None)
        return (str(pt) if pt else None), mv.version

    def predict(self, img: Image.Image, conf: float = 0.25) -> list[Detection]:
        results = self._model.predict(img, conf=conf, verbose=False)
        names = self._model.names
        out: list[Detection] = []
        for r in results:
            for b in r.boxes:
                out.append(Detection(
                    label=names[int(b.cls)],
                    confidence=float(b.conf),
                    box=[float(x) for x in b.xyxy[0].tolist()],
                ))
        return out
