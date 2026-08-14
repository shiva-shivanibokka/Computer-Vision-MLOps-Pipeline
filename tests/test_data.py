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


def test_root_resolves_when_the_package_is_installed_not_checked_out(tmp_path, monkeypatch):
    """Regression: the deployed container found no params.yaml at all.

    `pip install .` puts cvmlops in site-packages, so walking up from __file__
    reaches /usr/local/lib/python3.11 instead of the app directory. The model
    loader and the /ui mount both silently resolved against that. Reproduced by
    pointing the __file__-derived candidate at a directory with no params.yaml.
    """
    import cvmlops.config as cfg

    fake_site_packages = tmp_path / "site-packages" / "cvmlops"
    fake_site_packages.mkdir(parents=True)
    monkeypatch.setattr(cfg, "__file__", str(fake_site_packages / "config.py"))

    app_dir = tmp_path / "app"
    (app_dir).mkdir()
    (app_dir / "params.yaml").write_text("dataset: {}\n", encoding="utf8")

    # explicit override wins
    monkeypatch.setenv("CVMLOPS_ROOT", str(app_dir))
    assert cfg._find_root() == app_dir

    # and without it, the working directory is the fallback
    monkeypatch.delenv("CVMLOPS_ROOT")
    monkeypatch.chdir(app_dir)
    assert cfg._find_root() == app_dir


def test_the_real_root_actually_holds_params():
    """Whatever _find_root picked, params.yaml must be under it."""
    from cvmlops.config import PARAMS_PATH, REPO_ROOT

    assert PARAMS_PATH.is_file(), f"params.yaml not under resolved root {REPO_ROOT}"
