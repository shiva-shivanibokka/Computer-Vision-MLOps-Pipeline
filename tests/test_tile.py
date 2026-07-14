from cvmlops.data.tile import remap_boxes, tile_origins
from cvmlops.serve.tiled_inference import RawDet, _merge


def test_tile_origins_cover_and_clamp():
    assert tile_origins(100, 640, 512) == [0]           # smaller than tile
    o = tile_origins(2000, 640, 512)
    assert o[0] == 0 and o[-1] == 2000 - 640            # last tile clamped to edge
    assert all(0 <= x <= 2000 - 640 for x in o)


def test_remap_box_fully_inside_tile():
    # 1000x1000 image, one 100x100 box centered at (500,500)
    boxes = [(3, 0.5, 0.5, 0.1, 0.1)]
    out = remap_boxes(boxes, 1000, 1000, tx=400, ty=400, tile=200)
    assert len(out) == 1
    cls, cx, cy, w, h = out[0]
    assert cls == 3
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0.5, 0.5, 0.5, 0.5)


def test_remap_box_outside_tile_is_dropped():
    boxes = [(0, 0.5, 0.5, 0.1, 0.1)]        # box at (450-550)
    assert remap_boxes(boxes, 1000, 1000, tx=0, ty=0, tile=200) == []


def test_remap_box_barely_overlapping_dropped_by_min_visibility():
    # box (450-550); tile (500-700) -> only 25% of box visible < 0.3 default
    boxes = [(0, 0.5, 0.5, 0.1, 0.1)]
    assert remap_boxes(boxes, 1000, 1000, tx=500, ty=500, tile=200) == []


def test_remap_box_clipped_when_mostly_visible():
    # box (450-550); tile (460-660) -> 90% visible in x, full in y -> kept, clipped
    boxes = [(1, 0.5, 0.5, 0.1, 0.1)]
    out = remap_boxes(boxes, 1000, 1000, tx=460, ty=440, tile=200)
    assert len(out) == 1 and out[0][0] == 1


def test_merge_dedupes_same_class_overlap():
    a = RawDet("short", 0.9, [10, 10, 50, 50], cls=3)
    b = RawDet("short", 0.6, [12, 12, 52, 52], cls=3)   # heavy overlap, same class
    kept = _merge([a, b], iou=0.5)
    assert len(kept) == 1 and kept[0].confidence == 0.9  # higher score survives


def test_merge_keeps_different_classes():
    a = RawDet("short", 0.9, [10, 10, 50, 50], cls=3)
    b = RawDet("spur", 0.8, [12, 12, 52, 52], cls=4)     # same area, different class
    assert len(_merge([a, b], iou=0.5)) == 2
