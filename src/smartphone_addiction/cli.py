"""Typer CLI: data, features, train, and submission commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import StratifiedShuffleSplit

from smartphone_addiction import __version__
from smartphone_addiction.config import RunConfig, load_config
from smartphone_addiction.data.load import CompetitionFrames, load_competition_frames
from smartphone_addiction.data.schema import TARGET_COLUMN
from smartphone_addiction.errors import (
    ArtifactError,
    ConfigurationError,
    DataValidationError,
    SubmissionValidationError,
    TrainingError,
)
from smartphone_addiction.evaluation.blend import blend_run_predictions
from smartphone_addiction.evaluation.report import (
    append_runs_to_summary,
    write_final_report_scaffold,
)
from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.features.io import write_processed_dataset
from smartphone_addiction.kaggle_bundle import package_kaggle_bundle
from smartphone_addiction.paths import project_root, resolve_path
from smartphone_addiction.submission import build_submission_from_run
from smartphone_addiction.training.runner import run_training
from smartphone_addiction.training.tuning import (
    TuningBudget,
    make_tuning_objective,
    run_tuning,
)

app = typer.Typer(
    name="smartphone-addiction",
    help="Reproducible pipeline for Predicting Smartphone Addiction.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Download and validate official competition data.")
features_app = typer.Typer(help="Build processed feature tables.")
submission_app = typer.Typer(help="Build and validate submission CSV files.")
report_app = typer.Typer(help="Publish selected experiment summaries.")
package_app = typer.Typer(help="Build offline packages.")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(submission_app, name="submission")
app.add_typer(report_app, name="report")
app.add_typer(package_app, name="package")

DOMAIN_ERRORS = (
    ConfigurationError,
    DataValidationError,
    TrainingError,
    ArtifactError,
    SubmissionValidationError,
)


def _fail(message: str, code: int = 1) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=project_root(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError:
        pass
    return "nogit"


def _collect_config_paths(
    *,
    base: Path,
    profile: Path | None,
    model_config: Path | None,
    experiment: Path | None,
) -> list[Path]:
    root = project_root()
    paths = [resolve_path(base, root)]
    if profile is not None:
        paths.append(resolve_path(profile, root))
    if model_config is not None:
        paths.append(resolve_path(model_config, root))
    if experiment is not None:
        paths.append(resolve_path(experiment, root))
    return paths


def _load_run_config(
    *,
    base: Path,
    profile: Path | None,
    model_config: Path | None,
    experiment: Path | None,
    override: list[str],
) -> RunConfig:
    return load_config(
        _collect_config_paths(
            base=base,
            profile=profile,
            model_config=model_config,
            experiment=experiment,
        ),
        override or None,
        resolve=True,
    )


def _maybe_sample(train: pd.DataFrame, sample_rows: int | None) -> pd.DataFrame:
    if sample_rows is None or sample_rows >= len(train):
        return train
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_rows,
        random_state=42,
    )
    idx, _ = next(splitter.split(train, train[TARGET_COLUMN]))
    return train.iloc[idx].reset_index(drop=True)


@app.callback()
def main() -> None:
    """Predicting Smartphone Addiction CLI."""


@app.command("version")
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@data_app.command("download")
def data_download(
    competition: str = typer.Option("playground-series-s6e8", help="Kaggle competition slug"),
    output_dir: Path = typer.Option(Path("data/raw"), "--output-dir", "-o"),
) -> None:
    """Download official competition files via Kaggle CLI."""
    root = project_root()
    target = resolve_path(output_dir, root)
    target.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(target),
        "--force",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        _fail("kaggle CLI not found on PATH; install kaggle and configure credentials")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        _fail(f"kaggle download failed\n{detail}")
    typer.echo(f"Downloaded competition files into {target}")
    typer.echo("If a zip was written, unzip it so train/test/sample_submission CSV exist.")


@data_app.command("validate")
def data_validate(
    data_dir: Path = typer.Option(Path("data/raw"), "--data-dir", "-d"),
) -> None:
    """Validate official train/test/sample_submission CSV files."""
    try:
        frames = load_competition_frames(resolve_path(data_dir, project_root()))
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(
        f"OK train={len(frames.train)} test={len(frames.test)} "
        f"sample={len(frames.sample_submission)}"
    )


@features_app.command("build")
def features_build(
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    out_dir: Path = typer.Option(Path("data/processed"), "--out-dir"),
    version: str = typer.Option("v1", "--version"),
) -> None:
    """Transform official CSV files into processed parquet features."""
    root = project_root()
    try:
        frames = load_competition_frames(resolve_path(raw_dir, root))
        transformed = transform_competition_frames(frames.train, frames.test)
        paths = write_processed_dataset(
            transformed,
            resolve_path(out_dir, root),
            version=version,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(
        f"features={len(transformed.feature_columns)} "
        f"train={len(transformed.train)} test={len(transformed.test)}"
    )
    for key, path in paths.items():
        typer.echo(f"{key}={path}")


@app.command("train")
def train(
    base: Path = typer.Option(Path("configs/base.yaml"), "--base"),
    profile: Path | None = typer.Option(None, "--profile", "-p"),
    model_config: Path | None = typer.Option(None, "--model-config", "-m"),
    experiment: Path | None = typer.Option(None, "--experiment", "-e"),
    override: list[str] = typer.Option([], "--override", "-o"),
    processed: bool = typer.Option(
        True,
        "--processed/--raw",
        help="Use data/processed parquet (default) or transform from raw CSV.",
    ),
    resume_run_dir: Path | None = typer.Option(None, "--resume"),
) -> None:
    """Run OOF training for CatBoost or LightGBM using merged YAML config."""
    try:
        config = _load_run_config(
            base=base,
            profile=profile,
            model_config=model_config,
            experiment=experiment,
            override=override,
        )
        model_params = dict(config.model.params)
        # Align thread settings unless explicitly set in model YAML.
        if config.model.name == "catboost":
            model_params.setdefault("thread_count", config.runtime.threads)
        else:
            model_params.setdefault("n_jobs", config.runtime.threads)

        if processed:
            processed_dir = Path(config.data.processed_directory)
            train_path = processed_dir / "train_features.parquet"
            test_path = processed_dir / "test_features.parquet"
            manifest_path = processed_dir / "feature_manifest.json"
            if not train_path.is_file() or not test_path.is_file():
                _fail(
                    f"processed features missing under {processed_dir}; "
                    "run: smartphone-addiction features build"
                )
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            feature_columns = None
            categorical_columns = None
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                feature_columns = list(manifest["feature_columns"])
                categorical_columns = list(manifest["categorical_columns"])
            train_df = _maybe_sample(train_df, config.data.sample_rows)
            result = run_training(
                train=train_df,
                test=test_df,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                model_name=config.model.name,
                model_params=model_params,
                n_splits=config.cv.n_splits,
                seeds=list(config.cv.seeds),
                artifact_root=Path(config.artifacts.directory),
                git_sha=_git_sha(),
                resume_run_dir=resolve_path(resume_run_dir) if resume_run_dir else None,
                slug=f"{config.model.name}-{Path(profile).stem if profile else 'base'}",
            )
        else:
            frames = load_competition_frames(Path(config.data.directory))
            if config.data.sample_rows is not None:
                frames = CompetitionFrames(
                    train=_maybe_sample(frames.train, config.data.sample_rows),
                    test=frames.test,
                    sample_submission=frames.sample_submission,
                )
            result = run_training(
                frames=frames,
                model_name=config.model.name,
                model_params=model_params,
                n_splits=config.cv.n_splits,
                seeds=list(config.cv.seeds),
                artifact_root=Path(config.artifacts.directory),
                git_sha=_git_sha(),
                resume_run_dir=resolve_path(resume_run_dir) if resume_run_dir else None,
                slug=f"{config.model.name}-{Path(profile).stem if profile else 'base'}",
            )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"train failed: {exc}")

    typer.echo(f"run_dir={result.run_dir}")
    typer.echo(f"status={result.store.manifest().status}")
    typer.echo(f"oof_auc={result.metrics.get('oof_auc')}")
    typer.echo(f"oof_coverage={result.metrics.get('oof_coverage')}")


@submission_app.command("build")
def submission_build(
    run_dir: Path = typer.Option(..., "--run", help="Training run directory"),
    sample: Path = typer.Option(
        Path("data/raw/sample_submission.csv"),
        "--sample",
        help="Official sample_submission.csv",
    ),
    output: Path = typer.Option(
        Path("submissions/submission.csv"),
        "--output",
        "-o",
    ),
) -> None:
    """Build a validated submission CSV from a completed run (does not upload)."""
    root = project_root()
    try:
        sample_df = pd.read_csv(resolve_path(sample, root))
        paths = build_submission_from_run(
            run_dir=resolve_path(run_dir, root),
            sample=sample_df,
            output_csv=resolve_path(output, root),
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"csv={paths['csv']}")
    typer.echo(f"meta={paths['meta']}")
    typer.echo("Submission ready. Upload manually on Kaggle; this CLI never auto-uploads.")


@app.command("tune")
def tune(
    base: Path = typer.Option(Path("configs/base.yaml"), "--base"),
    profile: Path | None = typer.Option(None, "--profile", "-p"),
    model_config: Path | None = typer.Option(None, "--model-config", "-m"),
    experiment: Path | None = typer.Option(None, "--experiment", "-e"),
    override: list[str] = typer.Option([], "--override", "-o"),
    output_dir: Path = typer.Option(Path("artifacts/tuning"), "--output-dir"),
    n_trials: int = typer.Option(20, "--n-trials"),
) -> None:
    """Run bounded Optuna search (50% sample / 3-fold / seed 42 by default)."""
    root = project_root()
    try:
        config = _load_run_config(
            base=base,
            profile=profile,
            model_config=model_config,
            experiment=experiment,
            override=override,
        )
        processed_dir = Path(config.data.processed_directory)
        train_path = processed_dir / "train_features.parquet"
        manifest_path = processed_dir / "feature_manifest.json"
        if not train_path.is_file() or not manifest_path.is_file():
            _fail("processed train features missing; run features build first")
        train_df = pd.read_parquet(train_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        feature_columns = list(manifest["feature_columns"])
        categorical_columns = list(manifest["categorical_columns"])
        budget = TuningBudget(
            sample_fraction=0.5,
            n_splits=3,
            seed=42,
            n_trials=n_trials,
            n_candidates=3,
        )
        objective = make_tuning_objective(
            model_name=config.model.name,
            train=train_df,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            budget=budget,
        )
        out = resolve_path(output_dir, root) / config.model.name
        result = run_tuning(
            model_name=config.model.name,
            objective=objective,
            output_dir=out,
            budget=budget,
            study_name=f"{config.model.name}-tune",
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"tune failed: {exc}")
    typer.echo(f"trials_csv={result.trials_csv}")
    for path in result.candidate_yamls:
        typer.echo(f"candidate={path}")
    typer.echo("Optuna-stage scores are not final; re-run top candidates on full 5-fold CV.")


@app.command("blend")
def blend(
    runs: list[Path] = typer.Option(
        ...,
        "--runs",
        help="Two completed training run directories",
    ),
    output_dir: Path = typer.Option(Path("artifacts/blends"), "--output-dir"),
    step: float = typer.Option(0.05, "--step"),
) -> None:
    """Search probability/rank blend weights on two OOF runs."""
    if len(runs) != 2:
        _fail("blend requires exactly two --runs directories")
    root = project_root()
    try:
        first = resolve_path(runs[0], root)
        second = resolve_path(runs[1], root)
        out = resolve_path(output_dir, root) / f"{first.name}__{second.name}"
        payload = blend_run_predictions(
            first_run_dir=first,
            second_run_dir=second,
            output_dir=out,
            step=step,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"blend failed: {exc}")
    typer.echo(f"output_dir={out}")
    typer.echo(f"method={payload['method']}")
    typer.echo(f"first_weight={payload['first_weight']}")
    typer.echo(f"oof_auc={payload['auc']}")


@report_app.command("experiments")
def report_experiments(
    run: list[Path] = typer.Option(..., "--run", help="Completed run directory to publish"),
    summary: Path = typer.Option(Path("reports/experiment_summary.csv"), "--summary"),
    profile: str = typer.Option("", "--profile"),
    feature_groups: str = typer.Option("all", "--feature-groups"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    """Append explicitly selected completed runs to the public summary CSV."""
    root = project_root()
    try:
        path = append_runs_to_summary(
            [resolve_path(item, root) for item in run],
            resolve_path(summary, root),
            feature_groups=feature_groups,
            profile=profile,
            notes=notes,
        )
        write_final_report_scaffold(resolve_path(Path("reports/final_report.md"), root))
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"summary={path}")


@package_app.command("kaggle")
def package_kaggle(
    config: Path = typer.Option(
        ...,
        "--config",
        help="Experiment YAML included in the offline bundle",
    ),
    dist_dir: Path = typer.Option(Path("dist"), "--dist-dir"),
) -> None:
    """Build wheel + deterministic zip for offline Kaggle execution."""
    root = project_root()
    try:
        paths = package_kaggle_bundle(
            config_path=resolve_path(config, root),
            root=root,
            dist_dir=resolve_path(dist_dir, root),
        )
    except Exception as exc:
        _fail(f"package failed: {exc}")
    for key, path in paths.items():
        typer.echo(f"{key}={path}")


if __name__ == "__main__":
    app()
