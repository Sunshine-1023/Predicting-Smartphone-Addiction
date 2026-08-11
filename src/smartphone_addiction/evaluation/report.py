"""Public experiment summary and submission ledger helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from smartphone_addiction.errors import ArtifactError

SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "feature_groups",
    "profile",
    "config_hash",
    "feature_code_digest",
    "seeds",
    "folds",
    "oof_auc",
    "seed_auc_mean",
    "seed_auc_std",
    "fold_auc_mean",
    "fold_auc_std",
    "duration_seconds",
    "git_sha",
    "status",
    "oof_path",
    "test_pred_path",
    "blend_method",
    "blend_weights",
    "source_runs",
    "has_submission",
    "submission_csv",
    "public_lb",
    "private_lb",
    "notes",
]

SUBMISSION_COLUMNS = [
    "utc_time",
    "run_id",
    "local_oof_auc",
    "public_lb",
    "private_lb",
    "submission_csv",
    "notes",
]

TEXT_COLUMNS_NO_PATHS = ("notes", "profile", "feature_groups", "model", "blend_method")
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


def empty_submission_frame() -> pd.DataFrame:
    """Return an empty submissions ledger with the fixed columns."""
    return pd.DataFrame(columns=SUBMISSION_COLUMNS)


def row_from_run_dir(
    run_dir: Path | str,
    *,
    root: Path | str | None = None,
    feature_groups: str | None = None,
    profile: str | None = None,
    feature_code_digest: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build one summary row from a completed training or blend directory."""
    run_dir = Path(run_dir).resolve()
    root_path = Path(root).resolve() if root is not None else _infer_root(run_dir)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"missing manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    status = manifest.get("status", "unknown")
    if status != "completed":
        raise ArtifactError(f"refusing to publish non-completed run {run_dir.name}: {status}")

    resolved = _load_resolved_config(run_dir)
    seeds = manifest.get("seeds") or metrics.get("seeds") or []
    folds = manifest.get("n_splits") or metrics.get("n_splits")
    duration = manifest.get("duration_seconds")
    fold_mean, fold_std = _fold_auc_stats(run_dir)

    groups = feature_groups
    if groups is None:
        groups = _feature_groups_text(resolved) or "all"
    profile_value = profile if profile is not None else _profile_from_slug(manifest.get("slug", ""))
    digest = feature_code_digest if feature_code_digest is not None else ""

    oof_rel = _relative_if_exists(run_dir / "oof_predictions.parquet", root_path)
    test_rel = _relative_if_exists(run_dir / "test_predictions.parquet", root_path)
    submission_csv = _find_submission_csv(run_dir.name, root_path)
    blend_method, blend_weights, source_runs = _blend_fields(manifest, metrics, root_path)

    return {
        "run_id": run_dir.name,
        "model": metrics.get("model_name") or manifest.get("slug", ""),
        "feature_groups": groups,
        "profile": profile_value,
        "config_hash": manifest.get("config_hash", ""),
        "feature_code_digest": digest,
        "seeds": ",".join(str(seed) for seed in seeds) if isinstance(seeds, list) else seeds,
        "folds": folds if folds is not None else "",
        "oof_auc": metrics.get("oof_auc", ""),
        "seed_auc_mean": metrics.get("seed_auc_mean", ""),
        "seed_auc_std": metrics.get("seed_auc_std", ""),
        "fold_auc_mean": fold_mean,
        "fold_auc_std": fold_std,
        "duration_seconds": duration if duration is not None else "",
        "git_sha": manifest.get("git_sha", ""),
        "status": status,
        "oof_path": oof_rel,
        "test_pred_path": test_rel,
        "blend_method": blend_method,
        "blend_weights": blend_weights,
        "source_runs": source_runs,
        "has_submission": "yes" if submission_csv else "no",
        "submission_csv": submission_csv,
        "public_lb": "",
        "private_lb": "",
        "notes": notes,
    }


def write_experiment_summary(
    rows: list[dict[str, Any]],
    path: Path | str,
) -> Path:
    """Overwrite the public summary CSV with selected rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[SUMMARY_COLUMNS]
    _assert_safe_summary_text(frame)
    frame.to_csv(path, index=False)
    return path


def upsert_run_to_summary(
    run_dir: Path | str,
    summary_path: Path | str,
    *,
    root: Path | str | None = None,
    feature_groups: str | None = None,
    profile: str | None = None,
    feature_code_digest: str | None = None,
    notes: str = "",
    preserve_leaderboard: bool = True,
) -> Path:
    """Insert or update one completed run in the experiment summary CSV."""
    summary_path = Path(summary_path)
    existing = _read_summary(summary_path)
    row = row_from_run_dir(
        run_dir,
        root=root,
        feature_groups=feature_groups,
        profile=profile,
        feature_code_digest=feature_code_digest,
        notes=notes,
    )
    rows = existing.to_dict(orient="records")
    replaced = False
    for index, old in enumerate(rows):
        if str(old.get("run_id")) != row["run_id"]:
            continue
        if preserve_leaderboard:
            for key in ("public_lb", "private_lb"):
                if old.get(key) not in (None, ""):
                    row[key] = old.get(key)
            if not notes and old.get("notes") not in (None, ""):
                row["notes"] = old.get("notes")
            if old.get("feature_code_digest") not in (None, "") and not feature_code_digest:
                row["feature_code_digest"] = old.get("feature_code_digest")
        rows[index] = {column: row.get(column, "") for column in SUMMARY_COLUMNS}
        replaced = True
        break
    if not replaced:
        rows.append(row)
    return write_experiment_summary(rows, summary_path)


def append_runs_to_summary(
    run_dirs: list[Path | str],
    summary_path: Path | str,
    *,
    root: Path | str | None = None,
    feature_groups: str | None = None,
    profile: str | None = None,
    feature_code_digest: str | None = None,
    notes: str = "",
) -> Path:
    """Append or refresh selected completed runs in reports/experiment_summary.csv."""
    path = Path(summary_path)
    for run_dir in run_dirs:
        path = upsert_run_to_summary(
            run_dir,
            path,
            root=root,
            feature_groups=feature_groups,
            profile=profile,
            feature_code_digest=feature_code_digest,
            notes=notes,
        )
    return path


def sync_artifact_runs_to_summary(
    *,
    root: Path | str,
    summary_path: Path | str,
    runs_glob: str = "artifacts/runs/*",
    blends_glob: str = "artifacts/blends/*",
) -> Path:
    """Scan completed run/blend directories and upsert them into the summary CSV."""
    root_path = Path(root).resolve()
    paths: list[Path] = []
    for pattern in (runs_glob, blends_glob):
        paths.extend(sorted(p for p in root_path.glob(pattern) if p.is_dir()))
    summary = Path(summary_path)
    for run_dir in paths:
        manifest = run_dir / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "completed":
            continue
        summary = upsert_run_to_summary(run_dir, summary, root=root_path)
    return summary


def mark_submission_built(
    *,
    run_dir: Path | str,
    submission_csv: Path | str,
    summary_path: Path | str,
    submissions_path: Path | str,
    root: Path | str | None = None,
    notes: str = "",
) -> dict[str, Path]:
    """Record that a local submission CSV was built from a completed run."""
    run_dir = Path(run_dir).resolve()
    root_path = Path(root).resolve() if root is not None else _infer_root(run_dir)
    submission_csv = Path(submission_csv).resolve()
    summary_path = Path(summary_path)
    submissions_path = Path(submissions_path)

    upsert_run_to_summary(run_dir, summary_path, root=root_path)
    existing = _read_summary(summary_path)
    rows = existing.to_dict(orient="records")
    rel_csv = _relative_to_root(submission_csv, root_path)
    oof_auc = ""
    for row in rows:
        if str(row.get("run_id")) != run_dir.name:
            continue
        row["has_submission"] = "yes"
        row["submission_csv"] = rel_csv
        oof_auc = row.get("oof_auc", "")
        break
    write_experiment_summary(rows, summary_path)

    ledger = _read_submissions(submissions_path)
    ledger_rows = ledger.to_dict(orient="records")
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    updated = False
    for row in ledger_rows:
        if str(row.get("run_id")) == run_dir.name and str(row.get("submission_csv")) == rel_csv:
            row["local_oof_auc"] = oof_auc
            if notes:
                row["notes"] = notes
            updated = True
            break
    if not updated:
        ledger_rows.append(
            {
                "utc_time": stamp,
                "run_id": run_dir.name,
                "local_oof_auc": oof_auc,
                "public_lb": "",
                "private_lb": "",
                "submission_csv": rel_csv,
                "notes": notes,
            }
        )
    write_submissions_ledger(ledger_rows, submissions_path)
    return {"summary": summary_path, "submissions": submissions_path}


def record_leaderboard_score(
    *,
    run_id: str,
    public_lb: float | None = None,
    private_lb: float | None = None,
    summary_path: Path | str,
    submissions_path: Path | str,
    notes: str = "",
) -> dict[str, Path]:
    """Write Public/Private LB scores into both summary and submissions ledgers."""
    if public_lb is None and private_lb is None:
        raise ArtifactError("provide at least one of public_lb or private_lb")
    summary_path = Path(summary_path)
    submissions_path = Path(submissions_path)

    summary = _read_summary(summary_path)
    rows = summary.to_dict(orient="records")
    found = False
    for row in rows:
        if str(row.get("run_id")) != run_id:
            continue
        if public_lb is not None:
            row["public_lb"] = public_lb
        if private_lb is not None:
            row["private_lb"] = private_lb
        if notes:
            row["notes"] = notes
        found = True
        break
    if not found:
        raise ArtifactError(f"run_id not found in experiment summary: {run_id}")
    write_experiment_summary(rows, summary_path)

    ledger = _read_submissions(submissions_path)
    ledger_rows = ledger.to_dict(orient="records")
    matched = False
    for row in ledger_rows:
        if str(row.get("run_id")) != run_id:
            continue
        if public_lb is not None:
            row["public_lb"] = public_lb
        if private_lb is not None:
            row["private_lb"] = private_lb
        if notes:
            row["notes"] = notes
        matched = True
    if not matched:
        stamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        oof = next((row.get("oof_auc", "") for row in rows if str(row.get("run_id")) == run_id), "")
        ledger_rows.append(
            {
                "utc_time": stamp,
                "run_id": run_id,
                "local_oof_auc": oof,
                "public_lb": "" if public_lb is None else public_lb,
                "private_lb": "" if private_lb is None else private_lb,
                "submission_csv": next(
                    (
                        row.get("submission_csv", "")
                        for row in rows
                        if str(row.get("run_id")) == run_id
                    ),
                    "",
                ),
                "notes": notes,
            }
        )
    write_submissions_ledger(ledger_rows, submissions_path)
    return {"summary": summary_path, "submissions": submissions_path}


def write_submissions_ledger(rows: list[dict[str, Any]], path: Path | str) -> Path:
    """Overwrite reports/submissions.csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    for column in SUBMISSION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[SUBMISSION_COLUMNS]
    frame.to_csv(path, index=False)
    return path


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


def _read_summary(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return empty_summary_frame()
    existing = pd.read_csv(path)
    rename = {}
    if "oof_auc_mean" in existing.columns and "oof_auc" not in existing.columns:
        rename["oof_auc_mean"] = "oof_auc"
    if "oof_auc_std" in existing.columns and "seed_auc_std" not in existing.columns:
        rename["oof_auc_std"] = "seed_auc_std"
    if rename:
        existing = existing.rename(columns=rename)
    for column in SUMMARY_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
    return existing[SUMMARY_COLUMNS]


def _read_submissions(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return empty_submission_frame()
    existing = pd.read_csv(path)
    for column in SUBMISSION_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
    return existing[SUBMISSION_COLUMNS]


def _assert_safe_summary_text(frame: pd.DataFrame) -> None:
    path_tokens = ("artifacts/", "models/", ".cbm", ".joblib", "predictions/", "submissions/")
    for column in TEXT_COLUMNS_NO_PATHS:
        for value in frame[column].astype(str):
            if value in {"", "nan", "None"}:
                continue
            lowered = value.lower()
            if any(token in lowered for token in path_tokens) or "\\" in value:
                raise ArtifactError(
                    "public experiment summary text fields must not include "
                    "model or prediction paths"
                )


def _load_resolved_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _feature_groups_text(resolved: dict[str, Any]) -> str:
    groups = ((resolved.get("features") or {}).get("groups")) if resolved else None
    if isinstance(groups, list) and groups:
        return ",".join(str(item) for item in groups)
    return ""


def _profile_from_slug(slug: str) -> str:
    if not slug:
        return ""
    if slug == "blend" or slug.startswith("blend-"):
        return "blend"
    parts = slug.rsplit("-", 1)
    if len(parts) == 2 and parts[1] in {"smoke", "dev", "final", "base"}:
        return parts[1]
    return ""


def _fold_auc_stats(run_dir: Path) -> tuple[Any, Any]:
    path = run_dir / "fold_metrics.csv"
    if not path.is_file():
        return "", ""
    frame = pd.read_csv(path)
    if "auc" not in frame.columns or frame.empty:
        return "", ""
    values = pd.to_numeric(frame["auc"], errors="coerce").dropna()
    if values.empty:
        return "", ""
    return float(values.mean()), float(values.std(ddof=0))


def _blend_fields(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    root: Path,
) -> tuple[str, str, str]:
    method = metrics.get("method") or ""
    if not method and str(metrics.get("model_name", "")).startswith("blend-"):
        method = str(metrics["model_name"]).removeprefix("blend-")
    first_w = metrics.get("first_weight")
    second_w = metrics.get("second_weight")
    weights = ""
    if first_w is not None and second_w is not None:
        weights = f"{first_w},{second_w}"
    sources = manifest.get("source_runs") or []
    source_text = ""
    if isinstance(sources, list) and sources:
        names: list[str] = []
        for item in sources:
            path = Path(str(item))
            try:
                names.append(path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                names.append(path.name)
        source_text = ";".join(names)
    return str(method), weights, source_text


def _find_submission_csv(run_name: str, root: Path) -> str:
    candidate = root / "submissions" / f"{run_name}.csv"
    if candidate.is_file():
        return _relative_to_root(candidate, root)
    return ""


def _relative_if_exists(path: Path, root: Path) -> str:
    if not path.is_file():
        return ""
    return _relative_to_root(path, root)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _infer_root(run_dir: Path) -> Path:
    for candidate in (run_dir, *run_dir.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return run_dir.parent.parent if run_dir.parent.name in {"runs", "blends"} else run_dir.parent
