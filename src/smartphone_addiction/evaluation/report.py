"""Public experiment summary and final report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from smartphone_addiction.errors import ArtifactError

SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "feature_groups",
    "profile",
    "seeds",
    "folds",
    "oof_auc_mean",
    "oof_auc_std",
    "duration_seconds",
    "git_sha",
    "status",
    "notes",
]

FINAL_REPORT_SECTIONS = [
    "## Data facts",
    "## Validation design",
    "## Baselines",
    "## Feature ablations",
    "## Tuning candidates",
    "## Multi-seed stability",
    "## Model correlation",
    "## Blend decision",
    "## Submission history",
    "## Limitations",
    "## Next steps",
]


def empty_summary_frame() -> pd.DataFrame:
    """Return an empty summary table with the fixed public columns."""
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def row_from_run_dir(
    run_dir: Path | str,
    *,
    feature_groups: str = "all",
    profile: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build one public summary row from a completed ArtifactStore run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"missing manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    status = manifest.get("status", "unknown")
    if status != "completed":
        raise ArtifactError(f"refusing to publish non-completed run {run_dir.name}: {status}")

    seeds = manifest.get("seeds") or metrics.get("seeds") or []
    folds = manifest.get("n_splits") or metrics.get("n_splits")
    started = manifest.get("started_at")
    ended = manifest.get("ended_at")
    duration = manifest.get("duration_seconds")
    if duration is None and started and ended:
        # Best-effort; leave blank if timestamps are not comparable.
        duration = ""

    return {
        "run_id": run_dir.name,
        "model": metrics.get("model_name") or manifest.get("slug", ""),
        "feature_groups": feature_groups,
        "profile": profile,
        "seeds": ",".join(str(seed) for seed in seeds) if isinstance(seeds, list) else seeds,
        "folds": folds,
        "oof_auc_mean": metrics.get("oof_auc"),
        "oof_auc_std": metrics.get("oof_auc_std", metrics.get("seed_auc_std")),
        "duration_seconds": duration if duration is not None else "",
        "git_sha": manifest.get("git_sha", ""),
        "status": status,
        "notes": notes,
    }


def write_experiment_summary(
    rows: list[dict[str, Any]],
    path: Path | str,
) -> Path:
    """Overwrite the public summary CSV with only explicitly selected rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[SUMMARY_COLUMNS]
    # Never publish model/prediction paths in the public summary.
    path_tokens = ("artifacts/", "models/", ".cbm", ".joblib", "predictions/", "submissions/")
    for column in ("notes", "profile", "feature_groups", "model"):
        for value in frame[column].astype(str):
            lowered = value.lower()
            if any(token in lowered for token in path_tokens) or "/" in value or "\\" in value:
                raise ArtifactError(
                    "public experiment summary must not include model or prediction paths"
                )
    frame.to_csv(path, index=False)
    return path


def append_runs_to_summary(
    run_dirs: list[Path | str],
    summary_path: Path | str,
    *,
    feature_groups: str = "all",
    profile: str = "",
    notes: str = "",
) -> Path:
    """Append selected completed runs to reports/experiment_summary.csv."""
    summary_path = Path(summary_path)
    existing = (
        pd.read_csv(summary_path)
        if summary_path.is_file() and summary_path.stat().st_size > 0
        else empty_summary_frame()
    )
    rows = existing.to_dict(orient="records")
    known = {str(row.get("run_id")) for row in rows}
    for run_dir in run_dirs:
        row = row_from_run_dir(
            run_dir,
            feature_groups=feature_groups,
            profile=profile,
            notes=notes,
        )
        if row["run_id"] in known:
            continue
        rows.append(row)
        known.add(row["run_id"])
    return write_experiment_summary(rows, summary_path)


def write_final_report_scaffold(path: Path | str) -> Path:
    """Create final_report.md with required section headings if missing/empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing.strip():
        return path
    body = [
        "# Final report: Predicting Smartphone Addiction",
        "",
        "Numerical results must be filled only after source runs complete.",
        "Do not tune on Public Leaderboard feedback.",
        "",
    ]
    for section in FINAL_REPORT_SECTIONS:
        body.extend([section, "", "_Pending selected completed runs._", ""])
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
    return path
