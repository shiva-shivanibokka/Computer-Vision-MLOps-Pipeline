"""Convert the HRIPCB / PKU-Market-PCB dataset (VOC XML) to flat YOLO format.

HRIPCB ships PASCAL VOC annotations (one .xml per image, absolute pixel boxes).
This walks the annotations, converts every box to normalized YOLO, and writes
`data/pcb_raw/{images,labels}` — which `prepare.py` then splits into train/val.

Download (choose one, extract, point --src at the extracted PCB_DATASET folder):
  - Kaggle:  akhatova/pcb-defects
  - GitHub:  https://github.com/Ironbrotherstyle/PCB-DATASET

Run:  python -m cvmlops.data.convert_hripcb --src /path/to/PCB_DATASET
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from cvmlops.config import REPO_ROOT, load_params

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _yolo_lines(xml_path: Path, class_to_idx: dict[str, int]) -> tuple[str, list[str]]:
    """Return (image_filename, yolo_label_lines) for one VOC annotation."""
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
    size = root.find("size")
    w, h = float(size.findtext("width")), float(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in class_to_idx:
            continue  # unknown label — skip, don't guess
        b = obj.find("bndbox")
        x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
        x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{class_to_idx[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return filename, lines


def convert(src: Path, out: Path | None = None) -> Path:
    classes = load_params()["dataset"]["classes"]
    class_to_idx = {c.lower(): i for i, c in enumerate(classes)}
    out = out or (REPO_ROOT / "data" / "pcb_raw")

    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No .xml annotations found under {src}")
    # Index every image by stem so we can locate it regardless of subfolder.
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS}

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    n_boxes = n_imgs = 0
    for xml in xmls:
        filename, lines = _yolo_lines(xml, class_to_idx)
        img = images.get(Path(filename).stem) or images.get(xml.stem)
        if img is None:
            print(f"  warn: no image for {xml.name}; skipping")
            continue
        shutil.copy2(img, out / "images" / f"{xml.stem}{img.suffix.lower()}")
        (out / "labels" / f"{xml.stem}.txt").write_text("\n".join(lines))
        n_imgs += 1
        n_boxes += len(lines)

    print(f"Converted {n_imgs} images, {n_boxes} boxes -> {out}")
    print("Next: python -m cvmlops.data.prepare  (splits into train/val)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to extracted PCB_DATASET folder")
    ap.add_argument("--out", default=None, help="output dir (default data/pcb_raw)")
    args = ap.parse_args()
    convert(Path(args.src), Path(args.out) if args.out else None)
