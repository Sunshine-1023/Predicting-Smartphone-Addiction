"""Tests for the offline Kaggle packaging helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from smartphone_addiction.kaggle_bundle import package_kaggle_bundle


def test_bundle_is_deterministic_and_clean(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = root / "configs" / "experiments" / "catboost_domain_v1.yaml"
    dist_a = tmp_path / "dist_a"
    dist_b = tmp_path / "dist_b"
    wheel_bytes = b"fake-wheel-content"

    def fake_build(_root: Path, dist_dir: Path) -> Path:
        dist_dir.mkdir(parents=True, exist_ok=True)
        target = dist_dir / "smartphone_addiction-0.1.0-py3-none-any.whl"
        target.write_bytes(wheel_bytes)
        return target

    first = package_kaggle_bundle(
        config_path=config,
        root=root,
        dist_dir=dist_a,
        build_wheel_fn=fake_build,
        git_sha_fn=lambda _root: "deadbeef",
    )
    second = package_kaggle_bundle(
        config_path=config,
        root=root,
        dist_dir=dist_b,
        build_wheel_fn=fake_build,
        git_sha_fn=lambda _root: "deadbeef",
    )

    man_a = json.loads(first["manifest"].read_text(encoding="utf-8"))
    man_b = json.loads(second["manifest"].read_text(encoding="utf-8"))
    assert man_a["git_sha"] == man_b["git_sha"] == "deadbeef"
    assert man_a["wheel"] == man_b["wheel"]
    assert man_a["config"] == config.name
    assert {item["path"] for item in man_a["files"]} == {
        man_a["wheel"],
        config.name,
        "run_offline.sh",
        "verify_environment.py",
        "bundle_manifest.json",
    }
    assert first["zip"].read_bytes() == second["zip"].read_bytes()

    with zipfile.ZipFile(first["zip"]) as archive:
        names = set(archive.namelist())
    for forbidden in (".git", "kaggle.json", "artifacts/", "submissions/", "train.csv"):
        assert not any(forbidden in name for name in names)
