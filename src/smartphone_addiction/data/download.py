"""Secure Kaggle competition data download into data/raw."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.errors import DataValidationError

OFFICIAL_FILES = ("train.csv", "test.csv", "sample_submission.csv")
DEFAULT_COMPETITION = "playground-series-s6e8"

Runner = Callable[..., CompletedProcess[str]]


def fingerprint_files(directory: Path) -> dict[str, str]:
    """Return SHA-256 digests for each official competition CSV."""
    directory = Path(directory)
    digests: dict[str, str] = {}
    for name in OFFICIAL_FILES:
        path = directory / name
        if not path.is_file():
            raise DataValidationError(f"missing competition file for fingerprint: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[name] = digest.hexdigest()
    return digests


def check_kaggle_credentials(*, credential_check: bool = True) -> Path | None:
    """Ensure Kaggle credentials exist with safe permissions; never read secrets."""
    if not credential_check:
        return None
    kaggle_dir = Path.home() / ".kaggle"
    json_path = kaggle_dir / "kaggle.json"
    token_path = kaggle_dir / "access_token"

    if json_path.is_file():
        _assert_private_file(json_path)
        return json_path
    if token_path.is_file():
        _assert_private_file(token_path)
        return token_path
    raise DataValidationError(
        "Kaggle credentials not found. Create ~/.kaggle/kaggle.json "
        "(chmod 600) or ~/.kaggle/access_token (chmod 600), then accept the competition rules."
    )


def _assert_private_file(path: Path) -> None:
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise DataValidationError(
            f"{path} must not be group/world accessible; run: chmod 600 {path}"
        )


def _require_kaggle_on_path() -> None:
    if shutil.which("kaggle") is None:
        raise DataValidationError(
            "kaggle CLI not found on PATH; install the kaggle package and retry"
        )


def _translate_kaggle_failure(stderr: str, stdout: str) -> str:
    text = f"{stderr}\n{stdout}".lower()
    if "401" in text or "unauthorized" in text or (
        "invalid" in text and "credential" in text
    ):
        return (
            "Kaggle authentication failed. Check ~/.kaggle/kaggle.json "
            "(chmod 600) or access_token and regenerate the API token if needed."
        )
    if "403" in text or "rules" in text or "accept" in text:
        return (
            "Kaggle refused the download. Open the competition page and accept the rules, "
            "then retry."
        )
    detail = (stderr or stdout or "unknown kaggle error").strip()
    return f"kaggle download failed: {detail}"


def _extract_official_csvs(archive_dir: Path, extract_dir: Path) -> None:
    """Extract only the three official CSVs from any zip under archive_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(archive_dir.glob("*.zip"))
    if not zips:
        missing = [name for name in OFFICIAL_FILES if not (archive_dir / name).is_file()]
        if missing:
            raise DataValidationError(
                f"download produced neither zip nor official CSVs; missing={missing}"
            )
        for name in OFFICIAL_FILES:
            shutil.copy2(archive_dir / name, extract_dir / name)
        return

    found: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(zips[0]) as archive:
        for info in archive.infolist():
            base = Path(info.filename).name
            if base in OFFICIAL_FILES and base not in found:
                found[base] = info
        missing = [name for name in OFFICIAL_FILES if name not in found]
        if missing:
            raise DataValidationError(f"zip is missing official files: {missing}")
        for name, info in found.items():
            target = extract_dir / name
            with archive.open(info) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)


def _atomic_publish(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_FILES:
        src = source_dir / name
        dest = destination_dir / name
        tmp = destination_dir / f".{name}.tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)


def download_competition(
    competition: str,
    destination: Path,
    *,
    runner: Runner | None = None,
    credential_check: bool = True,
    extract_and_validate: bool = True,
) -> dict[str, Any]:
    """Download official CSVs into destination via a temp workspace.

    Never logs credential contents. Always cleans temporary files.
    """
    destination = Path(destination)
    _require_kaggle_on_path()
    check_kaggle_credentials(credential_check=credential_check)

    run = runner or _default_runner
    temp_root = Path(tempfile.mkdtemp(prefix="smartphone-addiction-download-"))
    try:
        download_dir = temp_root / "download"
        extract_dir = temp_root / "extract"
        download_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition,
            "-p",
            str(download_dir),
            "--force",
        ]
        completed = run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise DataValidationError(
                _translate_kaggle_failure(completed.stderr or "", completed.stdout or "")
            )

        if not extract_and_validate:
            return {"command": command, "destination": destination}

        _extract_official_csvs(download_dir, extract_dir)
        frames = load_competition_frames(extract_dir)
        _atomic_publish(extract_dir, destination)
        digests = fingerprint_files(destination)
        return {
            "command": command,
            "destination": destination,
            "fingerprints": digests,
            "n_train": len(frames.train),
            "n_test": len(frames.test),
            "n_sample": len(frames.sample_submission),
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _default_runner(command: list[str], **kwargs: Any) -> CompletedProcess[str]:
    return subprocess.run(command, **kwargs)
