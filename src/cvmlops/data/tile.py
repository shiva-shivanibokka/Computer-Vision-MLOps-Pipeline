"""Slice the full-board YOLO dataset into tiles so tiny defects stay full-size.

HRIPCB boards are ~3000px with ~70px defects. Training on whole boards downscaled
to 1280 shrinks defects to ~30px (near-undetectable). Tiling keeps each defect at
native resolution inside a small tile, which is what lets the same yolov8s model
actually learn them. Inference re-slices the board (see serve/tiled_inference.py).

Run:  python -m cvmlops.data.tile      # data/pcb -> data/pcb_tiled
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from cvmlops.config import REPO_ROOT, load_params


def tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Top-left offsets covering `size`, last tile clamped inside the edge."""
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, stride))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


def remap_boxes(boxes: list[tuple[int, float, float, float, float]], img_w: int, img_h: int,
                tx: int, ty: int, tile: int, min_visibility: float = 0.3
                ) -> list[tuple[int, float, float, float, float]]:
    """Remap full-image YOLO boxes into one tile's local normalized coords.

    A box is kept only if at least `min_visibility` of its area falls in the tile;
    it is clipped to the tile edge. Returns YOLO tuples (cls, cx, cy, w, h) in
    tile-normalized coordinates.
    """
    out = []
    for cls, cx, cy, bw, bh in boxes:
        bx1, by1 = (cx - bw / 2) * img_w, (cy - bh / 2) * img_h
        bx2, by2 = (cx + bw / 2) * img_w, (cy + bh / 2) * img_h
        ix1, iy1 = max(bx1, tx), max(by1, ty)
        ix2, iy2 = min(bx2, tx + tile), min(by2, ty + tile)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        box_area = (bx2 - bx1) * (by2 - by1)
        if box_area <= 0 or (ix2 - ix1) * (iy2 - iy1) / box_area < min_visibility:
            continue
        lx1, ly1, lx2, ly2 = ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty
        out.append((cls, (lx1 + lx2) / 2 / tile, (ly1 + ly2) / 2 / tile,
                    (lx2 - lx1) / tile, (ly2 - ly1) / tile))
    return out


def _read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            boxes.append((int(p[0]), *map(float, p[1:])))
    return boxes


def tile_split(src_root: Path, dst_root: Path, split: str, tile: int, overlap: float,
               keep_background: float, rng: random.Random) -> tuple[int, int]:
    stride = max(1, int(tile * (1 - overlap)))
    img_dir = src_root / "images" / split
    lbl_dir = src_root / "labels" / split
    out_img = dst_root / "images" / split
    out_lbl = dst_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_tiles = n_boxes = 0
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            boxes = _read_labels(lbl_dir / f"{img_path.stem}.txt")
            for ty in tile_origins(H, tile, stride):
                for tx in tile_origins(W, tile, stride):
                    local = remap_boxes(boxes, W, H, tx, ty, tile)
                    if not local and rng.random() > keep_background:
                        continue  # drop most empty tiles, keep a few as negatives
                    name = f"{img_path.stem}_{tx}_{ty}"
                    im.crop((tx, ty, tx + tile, ty + tile)).save(
                        out_img / f"{name}.jpg", quality=90)
                    (out_lbl / f"{name}.txt").write_text(
                        "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                                  for c, x, y, w, h in local))
                    n_tiles += 1
                    n_boxes += len(local)
    return n_tiles, n_boxes


def main(force: bool = False) -> Path:
    params = load_params()
    ds, tl = params["dataset"], params["tiling"]
    src = REPO_ROOT / ds["root"]
    dst = REPO_ROOT / tl["root"]
    data_yaml = dst / "data.yaml"

    # Tiling is deterministic (seeded); skip the expensive re-slice if already built.
    if not force and data_yaml.exists() and any((dst / "images" / "train").glob("*")):
        print(f"Tiled dataset already present at {dst}; skipping (force=True to rebuild)")
        return data_yaml

    rng = random.Random(ds["seed"])

    for split in ("train", "val"):
        if (src / "images" / split).is_dir():
            n_t, n_b = tile_split(src, dst, split, tl["tile_size"], tl["overlap"],
                                   tl["keep_background"], rng)
            print(f"{split}: {n_t} tiles, {n_b} boxes")

    data_yaml.write_text(
        f"path: {dst.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(ds['classes'])}\n"
        f"names: {ds['classes']}\n"
    )
    return data_yaml


if __name__ == "__main__":
    main()
