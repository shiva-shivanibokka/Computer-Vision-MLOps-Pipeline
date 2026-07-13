from pathlib import Path

from cvmlops.data import synthetic


def test_synthetic_dataset_is_valid_yolo(tmp_path):
    yml = synthetic.generate(tmp_path, n_train=6, n_val=3, seed=1)
    assert yml.exists()

    imgs = list((tmp_path / "images" / "train").glob("*.jpg"))
    lbls = list((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(imgs) == 6 and len(lbls) == 6

    # every label row is "cls cx cy w h" with normalized coords
    for line in Path(lbls[0]).read_text().splitlines():
        parts = line.split()
        assert len(parts) == 5
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_generate_is_deterministic(tmp_path):
    a = synthetic.generate(tmp_path / "a", n_train=3, n_val=1, seed=7)
    b = synthetic.generate(tmp_path / "b", n_train=3, n_val=1, seed=7)
    ia = (a.parent / "images" / "train" / "train_0000.jpg").read_bytes()
    ib = (b.parent / "images" / "train" / "train_0000.jpg").read_bytes()
    assert ia == ib
