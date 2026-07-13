"""Generate a tiny synthetic PCB-defect dataset in YOLO format.

Real training uses DeepPCB / PKU-Market-PCB (see prepare.py). This keeps the
whole pipeline — train, serve, monitor, CI — runnable offline with no download
and no accounts. Deterministic given a seed.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

from cvmlops.config import load_params


def _draw_board(rng: random.Random, w: int, h: int, n_classes: int):
    """A greenish board with a few colored 'defect' boxes. Returns (img, labels)."""
    img = Image.new("RGB", (w, h), (14, 82, 45))
    draw = ImageDraw.Draw(img)
    # copper traces so images aren't uniform (drift signal has something to move).
    for _ in range(rng.randint(6, 14)):
        y = rng.randint(0, h)
        draw.line([(0, y), (w, y)], fill=(184, 134, 11), width=rng.randint(1, 3))

    labels = []
    for _ in range(rng.randint(1, 4)):
        cls = rng.randint(0, n_classes - 1)
        bw, bh = rng.randint(10, 40), rng.randint(10, 40)
        cx, cy = rng.randint(bw, w - bw), rng.randint(bh, h - bh)
        x0, y0, x1, y1 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2
        color = [(220, 20, 60), (255, 215, 0), (30, 144, 255)][cls % 3]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        # YOLO: class cx cy w h, all normalized.
        labels.append(f"{cls} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return img, labels


def generate(root: str | Path, n_train: int = 24, n_val: int = 8, imgsz: int = 128,
             seed: int = 0, tint: int = 0) -> Path:
    """Write images/{train,val} + labels/{train,val} + data.yaml under root.

    `tint` shifts brightness — used by tests to fabricate a drifted batch.
    Returns the path to data.yaml.
    """
    params = load_params()
    classes = params["dataset"]["classes"]
    root = Path(root)
    rng = random.Random(seed)

    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = _draw_board(rng, imgsz, imgsz, len(classes))
            if tint:
                img = Image.eval(img, lambda px, t=tint: min(255, max(0, px + t)))
            img.save(img_dir / f"{split}_{i:04d}.jpg", quality=85)
            (lbl_dir / f"{split}_{i:04d}.txt").write_text("\n".join(labels))

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    return data_yaml


if __name__ == "__main__":  # ponytail: self-check — dataset is well-formed YOLO
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        yml = generate(d, n_train=4, n_val=2, seed=1)
        assert yml.exists()
        imgs = list((Path(d) / "images" / "train").glob("*.jpg"))
        lbls = list((Path(d) / "labels" / "train").glob("*.txt"))
        assert len(imgs) == 4 and len(lbls) == 4, (len(imgs), len(lbls))
        # every label line is "cls cx cy w h" with normalized coords in [0,1]
        for line in lbls[0].read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5, parts
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:]), line
        print("synthetic.py self-check OK")
