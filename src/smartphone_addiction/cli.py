"""Typer CLI: data, features, train, and submission commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import StratifiedShuffleSplit

from smartphone_addiction import __version__
from smartphone_addiction.config import RunConfig, load_config
from smartphone_addiction.data.download import download_competition, fingerprint_files
from smartphone_addiction.data.load import CompetitionFrames, load_competition_frames
from smartphone_addiction.data.schema import TARGET_COLUMN
from smartphone_addiction.errors import (
    AlignmentError,
    ArtifactError,
    ConfigurationError,
    DataValidationError,
    SubmissionValidationError,
    TrainingError,
)
from smartphone_addiction.evaluation.blend import blend_run_predictions
from smartphone_addiction.evaluation.importance import compute_run_importance
from smartphone_addiction.evaluation.report import (
    append_runs_to_summary,
    mark_submission_built,
    record_leaderboard_score,
    sync_artifact_runs_to_summary,
    upsert_run_to_summary,
    write_final_report_scaffold,
)
from smartphone_addiction.features.base import (
    select_feature_columns_from_groups,
    transform_competition_frames,
)
from smartphone_addiction.features.domain import ALL_FEATURE_GROUPS
from smartphone_addiction.features.io import (
    feature_code_fingerprint,
    validate_processed_manifest,
    write_processed_dataset,
)
from smartphone_addiction.git_info import git_is_dirty, git_sha
from smartphone_addiction.kaggle_bundle import package_kaggle_bundle
from smartphone_addiction.paths import project_root, resolve_path
from smartphone_addiction.submission import build_submission_from_run, default_submission_csv
from smartphone_addiction.training.runner import run_training
from smartphone_addiction.training.tuning import (
    TuningBudget,
    evaluate_candidates,
    make_tuning_objective,
    promote_candidate,
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
    AlignmentError,
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
    return git_sha()


def _git_dirty() -> bool:
    return git_is_dirty()


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
    """Securely download, extract, validate, and publish official CSVs."""
    root = project_root()
    target = resolve_path(output_dir, root)
    try:
        result = download_competition(competition, target)
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"destination={result['destination']}")
    typer.echo(
        f"rows train={result['n_train']} test={result['n_test']} sample={result['n_sample']}"
    )
    for name, digest in result["fingerprints"].items():
        typer.echo(f"sha256 {name}={digest}")


@data_app.command("validate")
def data_validate(
    data_dir: Path = typer.Option(Path("data/raw"), "--data-dir", "-d"),
) -> None:
    """Validate official train/test/sample_submission CSV files."""
    root = project_root()
    directory = resolve_path(data_dir, root)
    try:
        frames = load_competition_frames(directory)
        digests = fingerprint_files(directory)
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(
        f"OK train={len(frames.train)} test={len(frames.test)} "
        f"sample={len(frames.sample_submission)}"
    )
    for name, digest in digests.items():
        typer.echo(f"sha256 {name}={digest}")


@features_app.command("build")
def features_build(
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    out_dir: Path = typer.Option(Path("data/processed"), "--out-dir"),
    version: str = typer.Option("v1", "--version"),
    groups: list[str] = typer.Option(
        [],
        "--group",
        "-g",
        help="Feature group to include (repeatable). Default: full production set.",
    ),
) -> None:
    """Transform official CSV files into processed parquet features."""
    root = project_root()
    selected = groups or list(ALL_FEATURE_GROUPS)
    try:
        raw_path = resolve_path(raw_dir, root)
        frames = load_competition_frames(raw_path)
        transformed = transform_competition_frames(
            frames.train,
            frames.test,
            groups=selected,
        )
        paths = write_processed_dataset(
            transformed,
            resolve_path(out_dir, root),
            version=version,
            raw_directory=raw_path,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"groups={','.join(transformed.feature_groups)}")
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
    allow_dirty: bool = typer.Option(
        False,
        "--allow-dirty",
        help="Allow final-profile training when the git tree is dirty",
    ),
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

        profile_name = Path(profile).stem if profile else ""
        if profile_name == "final" and _git_dirty() and not allow_dirty:
            _fail(
                "refusing final-profile training on a dirty git tree; "
                "commit changes or pass --allow-dirty"
            )

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
            if not manifest_path.is_file():
                _fail(
                    f"processed feature_manifest.json missing under {processed_dir}; "
                    "run: smartphone-addiction features build"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_processed_manifest(
                manifest,
                raw_directory=Path(config.data.directory),
                train=train_df,
                test=test_df,
                train_path=train_path,
                test_path=test_path,
            )
            feature_columns = list(manifest["feature_columns"])
            categorical_columns = list(manifest["categorical_columns"])
            train_df = _maybe_sample(train_df, config.data.sample_rows)
            result = run_training(
                train=train_df,
                test=test_df,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                feature_groups=list(config.features.groups),
                model_name=config.model.name,
                model_params=model_params,
                n_splits=config.cv.n_splits,
                seeds=list(config.cv.seeds),
                artifact_root=Path(config.artifacts.directory),
                git_sha=_git_sha(),
                git_dirty=_git_dirty(),
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
                feature_groups=list(config.features.groups),
                model_name=config.model.name,
                model_params=model_params,
                n_splits=config.cv.n_splits,
                seeds=list(config.cv.seeds),
                artifact_root=Path(config.artifacts.directory),
                git_sha=_git_sha(),
                git_dirty=_git_dirty(),
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
    try:
        digest = feature_code_fingerprint()["digest"] if processed else ""
        summary = upsert_run_to_summary(
            result.run_dir,
            resolve_path(Path("reports/experiment_summary.csv"), project_root()),
            root=project_root(),
            feature_groups=",".join(config.features.groups),
            profile=Path(profile).stem if profile else "",
            feature_code_digest=digest,
        )
        typer.echo(f"summary={summary}")
    except DOMAIN_ERRORS as exc:
        typer.echo(f"summary_warning={exc}")


@submission_app.command("build")
def submission_build(
    run_dir: Path = typer.Option(..., "--run", help="Training run directory"),
    sample: Path = typer.Option(
        Path("data/raw/sample_submission.csv"),
        "--sample",
        help="Official sample_submission.csv",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (default: submissions/<run_dir_name>.csv)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing submission CSV/meta pair",
    ),
) -> None:
    """Build a validated submission CSV from a completed run (does not upload)."""
    root = project_root()
    try:
        resolved_run = resolve_path(run_dir, root)
        output_csv = (
            resolve_path(output, root)
            if output is not None
            else resolve_path(default_submission_csv(resolved_run), root)
        )
        sample_df = pd.read_csv(resolve_path(sample, root))
        paths = build_submission_from_run(
            run_dir=resolved_run,
            sample=sample_df,
            output_csv=output_csv,
            force=force,
        )
        ledgers = mark_submission_built(
            run_dir=resolved_run,
            submission_csv=paths["csv"],
            summary_path=resolve_path(Path("reports/experiment_summary.csv"), root),
            submissions_path=resolve_path(Path("reports/submissions.csv"), root),
            root=root,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"csv={paths['csv']}")
    typer.echo(f"meta={paths['meta']}")
    typer.echo(f"summary={ledgers['summary']}")
    typer.echo(f"submissions={ledgers['submissions']}")
    typer.echo("Submission ready. Upload manually on Kaggle; this CLI never auto-uploads.")


@app.command("tune")
def tune(
    base: Path = typer.Option(Path("configs/base.yaml"), "--base"),
    profile: Path | None = typer.Option(None, "--profile", "-p"),
    model_config: Path | None = typer.Option(None, "--model-config", "-m"),
    experiment: Path | None = typer.Option(None, "--experiment", "-e"),
    override: list[str] = typer.Option([], "--override", "-o"),
    output_dir: Path = typer.Option(Path("artifacts/tuning"), "--output-dir"),
    n_trials: int | None = typer.Option(
        None,
        "--n-trials",
        help="Override tuning.n_trials from merged YAML (default: use config).",
    ),
    fresh_study: bool = typer.Option(
        False,
        "--fresh-study",
        help="Start a new Optuna study instead of resuming an existing sqlite db",
    ),
) -> None:
    """Run bounded Optuna search using merged YAML (cv + tuning budget)."""
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
        test_path = processed_dir / "test_features.parquet"
        manifest_path = processed_dir / "feature_manifest.json"
        if not train_path.is_file() or not test_path.is_file() or not manifest_path.is_file():
            _fail("processed features missing; run features build first")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_processed_manifest(
            manifest,
            raw_directory=Path(config.data.directory),
            train=train_df,
            test=test_df,
            train_path=train_path,
            test_path=test_path,
        )
        feature_columns = select_feature_columns_from_groups(
            list(manifest["feature_columns"]),
            list(config.features.groups),
        )
        if not feature_columns:
            _fail("feature groups selected zero columns from the processed manifest")
        categorical_columns = [
            column for column in list(manifest["categorical_columns"]) if column in feature_columns
        ]
        budget = TuningBudget(
            sample_fraction=config.tuning.sample_fraction,
            n_splits=config.cv.n_splits,
            seed=int(config.cv.seeds[0]),
            n_trials=n_trials if n_trials is not None else config.tuning.n_trials,
            n_candidates=config.tuning.n_candidates,
        )
        base_params = dict(config.model.params)
        objective = make_tuning_objective(
            model_name=config.model.name,
            train=train_df,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            budget=budget,
            base_params=base_params,
        )
        study_suffix = hashlib.sha256(
            json.dumps(
                {
                    "feature_columns": feature_columns,
                    "sample_fraction": budget.sample_fraction,
                    "n_splits": budget.n_splits,
                    "seed": budget.seed,
                    "model_params": base_params,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        out = resolve_path(output_dir, root) / config.model.name
        result = run_tuning(
            model_name=config.model.name,
            objective=objective,
            output_dir=out,
            budget=budget,
            study_name=f"{config.model.name}-tune",
            study_suffix=study_suffix,
            fresh_study=fresh_study,
            base_params=base_params,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"tune failed: {exc}")
    typer.echo(f"trials_csv={result.trials_csv}")
    for path in result.candidate_yamls:
        typer.echo(f"candidate={path}")
    typer.echo("Optuna-stage scores are not final; re-run top candidates on full 5-fold CV.")


@app.command("evaluate-candidates")
def evaluate_candidates_cmd(
    candidates: list[Path] = typer.Option(
        ...,
        "--candidate",
        "-c",
        help="Candidate YAML from tune (repeatable)",
    ),
    base: Path = typer.Option(Path("configs/base.yaml"), "--base"),
    profile: Path | None = typer.Option(Path("configs/profiles/dev.yaml"), "--profile", "-p"),
    model_config: Path | None = typer.Option(None, "--model-config", "-m"),
    experiment: Path | None = typer.Option(None, "--experiment", "-e"),
    override: list[str] = typer.Option([], "--override", "-o"),
    output_dir: Path = typer.Option(Path("artifacts/tuning/evaluation"), "--output-dir"),
) -> None:
    """Re-evaluate Optuna candidates with full stratified OOF CV and rank them."""
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
        test_path = processed_dir / "test_features.parquet"
        manifest_path = processed_dir / "feature_manifest.json"
        if not train_path.is_file() or not test_path.is_file() or not manifest_path.is_file():
            _fail("processed features missing; run features build first")
        train_df = _maybe_sample(pd.read_parquet(train_path), config.data.sample_rows)
        test_df = pd.read_parquet(test_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_processed_manifest(
            manifest,
            raw_directory=Path(config.data.directory),
            train=pd.read_parquet(train_path),
            test=test_df,
            train_path=train_path,
            test_path=test_path,
        )
        feature_columns = select_feature_columns_from_groups(
            list(manifest["feature_columns"]),
            list(config.features.groups),
        )
        categorical_columns = [
            column for column in list(manifest["categorical_columns"]) if column in feature_columns
        ]
        result = evaluate_candidates(
            candidate_yamls=[resolve_path(path, root) for path in candidates],
            train=train_df,
            test=test_df,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            feature_groups=list(config.features.groups),
            artifact_root=Path(config.artifacts.directory),
            output_dir=resolve_path(output_dir, root),
            n_splits=config.cv.n_splits,
            seeds=list(config.cv.seeds),
            git_sha=_git_sha(),
            git_dirty=_git_dirty(),
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"evaluate-candidates failed: {exc}")
    typer.echo(f"selection={result.selection_json}")
    typer.echo(f"ranking={result.ranking_csv}")
    typer.echo(f"selected_yaml={result.selected_yaml}")
    typer.echo(f"best_oof_auc={result.rows[0]['oof_auc']}")


@app.command("promote")
def promote_cmd(
    selection: Path = typer.Option(
        ..., "--selection", help="selection.json from evaluate-candidates"
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination experiment YAML (e.g. configs/experiments/catboost_final_v2.yaml)",
    ),
    template: Path | None = typer.Option(
        None,
        "--template",
        "-t",
        help="Optional template experiment YAML (features/groups preserved)",
    ),
) -> None:
    """Write a train-ready experiment YAML from an evaluate-candidates selection."""
    root = project_root()
    try:
        path = promote_candidate(
            selection_json=resolve_path(selection, root),
            output_yaml=resolve_path(output, root),
            template_yaml=resolve_path(template, root) if template else None,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"promote failed: {exc}")
    typer.echo(f"experiment={path}")
    typer.echo(f"meta={path.with_suffix('.meta.json')}")


@app.command("importance")
def importance_cmd(
    run_dir: Path = typer.Option(..., "--run", help="Completed training run directory"),
    processed_dir: Path = typer.Option(
        Path("data/processed"),
        "--processed-dir",
        help="Directory with train_features.parquet",
    ),
    fold_key: str | None = typer.Option(
        None,
        "--fold-key",
        help="Optional single fold (default: all saved folds, validation rows only)",
    ),
    sample_rows: int | None = typer.Option(5_000, "--sample-rows"),
    n_repeats: int = typer.Option(5, "--n-repeats"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Compute fold-local permutation importance for a completed training run."""
    root = project_root()
    try:
        train_path = resolve_path(processed_dir, root) / "train_features.parquet"
        test_path = resolve_path(processed_dir, root) / "test_features.parquet"
        if not train_path.is_file() or not test_path.is_file():
            _fail(f"missing processed features: {train_path} / {test_path}")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        summary = compute_run_importance(
            run_dir=resolve_path(run_dir, root),
            train=train_df,
            test=test_df,
            fold_key=fold_key,
            n_repeats=n_repeats,
            sample_rows=sample_rows,
            seed=seed,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"importance failed: {exc}")
    typer.echo(f"rows={len(summary)}")
    if not summary.empty:
        top = summary.iloc[0]
        typer.echo(f"top_feature={top['feature']} importance={top['importance_mean']:.6f}")
    typer.echo(f"summary={resolve_path(run_dir, root) / 'importance' / 'summary.csv'}")


@app.command("blend")
def blend(
    runs: list[Path] = typer.Option(
        ...,
        "--runs",
        help="Two completed training run directories",
    ),
    output_dir: Path = typer.Option(Path("artifacts/blends"), "--output-dir"),
    step: float = typer.Option(0.05, "--step"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace an existing blend output directory",
    ),
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
            force=force,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"blend failed: {exc}")
    typer.echo(f"output_dir={out}")
    typer.echo(f"method={payload['method']}")
    typer.echo(f"first_weight={payload['first_weight']}")
    typer.echo(f"oof_auc={payload['auc']}")
    try:
        summary = upsert_run_to_summary(
            out,
            resolve_path(Path("reports/experiment_summary.csv"), root),
            root=root,
            profile="blend",
            notes=(
                f"method={payload['method']} "
                f"weights={payload['first_weight']},{payload['second_weight']}"
            ),
        )
        typer.echo(f"summary={summary}")
    except DOMAIN_ERRORS as exc:
        typer.echo(f"summary_warning={exc}")


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
            root=root,
            feature_groups=feature_groups,
            profile=profile,
            notes=notes,
        )
        write_final_report_scaffold(resolve_path(Path("reports/final_report.md"), root))
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"summary={path}")


@report_app.command("sync")
def report_sync(
    summary: Path = typer.Option(Path("reports/experiment_summary.csv"), "--summary"),
) -> None:
    """Scan artifacts/runs and artifacts/blends and upsert completed runs into the summary."""
    root = project_root()
    try:
        path = sync_artifact_runs_to_summary(
            root=root,
            summary_path=resolve_path(summary, root),
        )
        write_final_report_scaffold(resolve_path(Path("reports/final_report.md"), root))
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"summary={path}")


@report_app.command("lb")
def report_lb(
    run_id: str = typer.Option(..., "--run-id", help="Run or blend directory name"),
    public_lb: float | None = typer.Option(None, "--public-lb"),
    private_lb: float | None = typer.Option(None, "--private-lb"),
    notes: str = typer.Option("", "--notes"),
    summary: Path = typer.Option(Path("reports/experiment_summary.csv"), "--summary"),
    submissions: Path = typer.Option(Path("reports/submissions.csv"), "--submissions"),
) -> None:
    """Record Public/Private leaderboard scores for an existing run_id."""
    root = project_root()
    try:
        paths = record_leaderboard_score(
            run_id=run_id,
            public_lb=public_lb,
            private_lb=private_lb,
            summary_path=resolve_path(summary, root),
            submissions_path=resolve_path(submissions, root),
            notes=notes,
        )
    except DOMAIN_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"summary={paths['summary']}")
    typer.echo(f"submissions={paths['submissions']}")


@package_app.command("kaggle")
def package_kaggle(
    config: Path = typer.Option(
        ...,
        "--config",
        help="Experiment YAML included in the offline bundle",
    ),
    base: Path = typer.Option(Path("configs/base.yaml"), "--base"),
    profile: Path = typer.Option(Path("configs/profiles/final.yaml"), "--profile", "-p"),
    model_config: Path | None = typer.Option(
        None,
        "--model-config",
        "-m",
        help="Model YAML; defaults to configs/models/<experiment model.name>.yaml",
    ),
    dist_dir: Path = typer.Option(Path("dist"), "--dist-dir"),
) -> None:
    """Build wheel + configs + deterministic zip for offline Kaggle execution."""
    root = project_root()
    try:
        paths = package_kaggle_bundle(
            config_path=resolve_path(config, root),
            root=root,
            dist_dir=resolve_path(dist_dir, root),
            base_path=resolve_path(base, root),
            profile_path=resolve_path(profile, root),
            model_config_path=resolve_path(model_config, root) if model_config else None,
        )
    except Exception as exc:
        _fail(f"package failed: {exc}")
    for key, path in paths.items():
        typer.echo(f"{key}={path}")


if __name__ == "__main__":
    app()
