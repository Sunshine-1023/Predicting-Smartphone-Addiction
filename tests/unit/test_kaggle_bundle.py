"""Tests for the offline Kaggle packaging helpers."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path

from smartphone_addiction.kaggle_bundle import MANIFEST_NAME, package_kaggle_bundle, sha256_file
from smartphone_addiction.paths import ROOT_MARKER


def _fake_build(wheel_bytes: bytes = b"fake-wheel-content"):
    def fake_build(_root: Path, dist_dir: Path) -> Path:
        dist_dir.mkdir(parents=True, exist_ok=True)
        target = dist_dir / "smartphone_addiction-0.1.0-py3-none-any.whl"
        target.write_bytes(wheel_bytes)
        return target

    return fake_build


def test_bundle_is_deterministic_and_clean(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = root / "configs" / "experiments" / "catboost_domain_v1.yaml"
    dist_a = tmp_path / "dist_a"
    dist_b = tmp_path / "dist_b"

    first = package_kaggle_bundle(
        config_path=config,
        root=root,
        dist_dir=dist_a,
        build_wheel_fn=_fake_build(),
        git_sha_fn=lambda _root: "deadbeef",
        created_at="1970-01-01T00:00:00Z",
    )
    time.sleep(1.1)
    second = package_kaggle_bundle(
        config_path=config,
        root=root,
        dist_dir=dist_b,
        build_wheel_fn=_fake_build(),
        git_sha_fn=lambda _root: "deadbeef",
        created_at="1970-01-01T00:00:00Z",
    )

    man_a = json.loads(first["manifest"].read_text(encoding="utf-8"))
    man_b = json.loads(second["manifest"].read_text(encoding="utf-8"))
    assert man_a == man_b
    assert man_a["git_sha"] == "deadbeef"
    assert man_a["created_at"] == "1970-01-01T00:00:00Z"
    assert man_a["config"] == "configs/experiments/catboost_domain_v1.yaml"
    assert man_a["base"] == "configs/base.yaml"
    assert man_a["profile"] == "configs/profiles/final.yaml"
    assert man_a["model_config"] == "configs/models/catboost.yaml"
    paths = {item["path"] for item in man_a["files"]}
    assert man_a["wheel"] in paths
    assert "run_offline.sh" in paths
    assert "verify_environment.py" in paths
    assert MANIFEST_NAME not in paths  # avoid self-referential digest
    assert ROOT_MARKER in paths
    assert "configs/base.yaml" in paths
    assert "configs/profiles/final.yaml" in paths
    assert "configs/models/catboost.yaml" in paths
    assert "configs/experiments/catboost_domain_v1.yaml" in paths
    assert first["zip"].read_bytes() == second["zip"].read_bytes()

    with zipfile.ZipFile(first["zip"]) as archive:
        names = set(archive.namelist())
        launcher = archive.read("run_offline.sh").decode("utf-8")
        staged = tmp_path / "extract"
        archive.extractall(staged)
    assert "smartphone-addiction train" in launcher
    assert "SMARTPHONE_ADDICTION_ROOT" in launcher
    assert "DATA_DIR" in launcher
    assert "--train-only" in launcher
    assert "--base" in launcher
    for forbidden in (".git", "kaggle.json", "artifacts/", "submissions/", "train.csv"):
        assert not any(forbidden in name for name in names)

    # Every listed digest must match the extracted file on disk.
    for item in man_a["files"]:
        path = staged / item["path"]
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]
        assert path.stat().st_size == item["bytes"]

    # Manifest itself is present in the zip and matches the outer copy.
    assert MANIFEST_NAME in names
    assert (
        hashlib.sha256((staged / MANIFEST_NAME).read_bytes()).hexdigest()
        == hashlib.sha256(first["manifest"].read_bytes()).hexdigest()
    )
