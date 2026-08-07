"""Unit tests for secure competition download helpers."""

from __future__ import annotations

import stat
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from smartphone_addiction.data.download import (
    check_kaggle_credentials,
    download_competition,
    fingerprint_files,
)
from smartphone_addiction.errors import DataValidationError


def test_download_uses_expected_competition_and_directory(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    download_competition(
        "playground-series-s6e8",
        tmp_path,
        runner=fake_run,
        credential_check=False,
        extract_and_validate=False,
    )
    assert len(calls) == 1
    assert calls[0][:5] == [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "playground-series-s6e8",
    ]
    assert calls[0][5] == "-p"
    assert calls[0][7] == "--force"
    assert Path(calls[0][6]).name == "download"


def test_download_extracts_validates_and_publishes(
    tmp_path: Path,
    competition_frames,
) -> None:
    train, test, sample = competition_frames
    dest = tmp_path / "raw"

    def fake_run(command: list[str], **_: object) -> CompletedProcess[str]:
        download_dir = Path(command[6])
        zip_path = download_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("train.csv", train.to_csv(index=False))
            archive.writestr("nested/test.csv", test.to_csv(index=False))
            archive.writestr("sample_submission.csv", sample.to_csv(index=False))
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    result = download_competition(
        "playground-series-s6e8",
        dest,
        runner=fake_run,
        credential_check=False,
        extract_and_validate=True,
    )
    assert (dest / "train.csv").is_file()
    assert (dest / "test.csv").is_file()
    assert (dest / "sample_submission.csv").is_file()
    assert result["n_train"] == len(train)
    digests = fingerprint_files(dest)
    assert set(digests) == {"train.csv", "test.csv", "sample_submission.csv"}
    assert all(len(value) == 64 for value in digests.values())


def test_credential_rejects_world_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    cred = kaggle_dir / "kaggle.json"
    cred.write_text("{}", encoding="utf-8")
    cred.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(DataValidationError, match="chmod 600"):
        check_kaggle_credentials(credential_check=True)


def test_fingerprint_requires_files(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="missing competition file"):
        fingerprint_files(tmp_path)
