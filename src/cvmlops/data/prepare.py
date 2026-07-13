"""Prepare the PCB dataset into YOLO format with a train/val split.

Real data: drop a YOLO-format PCB defect dataset (e.g. DeepPCB or a Roboflow
export) into `data/pcb_raw/` as flat `images/*.jpg` + `labels/*.txt`. This splits
it deterministically and writes `data.yaml`.

No raw data present -> falls back to a synthetic dataset so the pipeline (train,
serve, monitor, CI) always runs. `dvc repro prepare` calls main().
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from cvmlops.config import REPO_ROOT, load_params
from cvmlops.data import synthetic

RAW_DIR = REPO_ROOT / "data" / "pcb_raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _split_raw(raw: Path, root: Path, val_fraction: float, seed: int, classes: list[str]) -> Path:
    images = sorted(p for p in (raw / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"{raw}/images has no images")

    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_fraction))
    splits = {"val": images[:n_val], "train": images[n_val:]}

    if root.exists():
        shutil.rmtree(root)
    for split, files in splits.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, root / "images" / split / img.name)
            label = raw / "labels" / f"{img.stem}.txt"
            dst = root / "labels" / split / f"{img.stem}.txt"
            # Missing label file = a legit negative (no defects) -> empty file.
            dst.write_text(label.read_text() if label.exists() else "")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    return data_yaml


def main() -> Path:
    params = load_params()
    ds = params["dataset"]
    root = REPO_ROOT / ds["root"]

    if (RAW_DIR / "images").is_dir():
        yml = _split_raw(RAW_DIR, root, ds["val_fraction"], ds["seed"], ds["classes"])
        print(f"Prepared real dataset -> {yml}")
    else:
        yml = synthetic.generate(root, seed=ds["seed"])
        print(f"No raw data at {RAW_DIR}; generated synthetic dataset -> {yml}")
    return yml


if __name__ == "__main__":
    main()
