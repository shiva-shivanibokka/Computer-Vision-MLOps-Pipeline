"""Sliced inference (SAHI-style) for full boards.

The model is trained on tiles, so at serve time we slice the full image the same
way, run the model per tile, offset each detection back to full-image coordinates,
and merge overlaps with class-wise NMS. Pure torchvision — no extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.ops import nms

from cvmlops.data.tile import tile_origins


@dataclass
class RawDet:
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2] full-image pixels
    cls: int


def sliced_predict(model, img: Image.Image, tile: int, overlap: float,
                   conf: float = 0.25, iou: float = 0.5) -> list[RawDet]:
    stride = max(1, int(tile * (1 - overlap)))
    W, H = img.size
    names = model.names

    dets: list[RawDet] = []
    for ty in tile_origins(H, tile, stride):
        for tx in tile_origins(W, tile, stride):
            crop = img.crop((tx, ty, tx + tile, ty + tile))
            for r in model.predict(crop, conf=conf, verbose=False):
                for b in r.boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                    cls = int(b.cls)
                    dets.append(RawDet(
                        label=names[cls], confidence=float(b.conf),
                        box=[x1 + tx, y1 + ty, x2 + tx, y2 + ty], cls=cls))

    return _merge(dets, iou)


def _merge(dets: list[RawDet], iou: float) -> list[RawDet]:
    """Class-wise NMS to dedupe detections that span tile overlaps."""
    if not dets:
        return []
    kept: list[RawDet] = []
    for cls in {d.cls for d in dets}:
        group = [d for d in dets if d.cls == cls]
        boxes = torch.tensor([d.box for d in group], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in group], dtype=torch.float32)
        for i in nms(boxes, scores, iou).tolist():
            kept.append(group[i])
    kept.sort(key=lambda d: d.confidence, reverse=True)
    return kept
