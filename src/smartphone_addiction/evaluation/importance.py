"""Sampled permutation importance using ROC-AUC drop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.models.catboost import CatBoostAdapter
from smartphone_addiction.models.lightgbm import LightGBMAdapter
from smartphone_addiction.training.runner import compute_training_data_hashes

_FOLD_KEY_RE = re.compile(r"^seed(?P<seed>\d+)-fold(?P<fold>\d+)$")


class PredictProbaModel(Protocol):
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


def stratified_feature_sample(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    sample_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Optionally draw a stratified subset for faster importance estimates."""
    labels = np.asarray(y)
    if sample_rows is None or sample_rows >= len(x):
        return x.reset_index(drop=True), labels
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_rows,
        random_state=seed,
    )
    idx, _ = next(splitter.split(x, labels))
    return x.iloc[idx].reset_index(drop=True), labels[idx]


def permutation_importance_auc(
    model: PredictProbaModel,
    x: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str] | None = None,
    *,
    n_repeats: int = 5,
    sample_rows: int | None = 5_000,
    seed: int = 42,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Compute mean AUC drop when each feature is shuffled (no SHAP)."""
    if n_repeats < 1:
        raise TrainingError("n_repeats must be >= 1")
    columns = list(feature_columns or x.columns)
    missing = [column for column in columns if column not in x.columns]
    if missing:
        raise TrainingError(f"missing feature columns for importance: {missing}")

    x_sample, y_sample = stratified_feature_sample(
        x[columns],
        y,
        sample_rows=sample_rows,
        seed=seed,
    )
    baseline = float(roc_auc_score(y_sample, model.predict_proba(x_sample)))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for column in columns:
        drops: list[float] = []
        for _ in range(n_repeats):
            shuffled = x_sample.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = float(roc_auc_score(y_sample, model.predict_proba(shuffled)))
            drops.append(baseline - score)
        rows.append(
            {
                "feature": column,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops, ddof=0)),
                "n_repeats": n_repeats,
                "sample_rows": len(x_sample),
                "baseline_auc": baseline,
                "run_id": run_id,
                "seed": seed,
            }
        )
    frame = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    return frame.reset_index(drop=True)


def load_fold_model(
    run_dir: Path | str,
    *,
    fold_key: str = "seed42-fold0",
) -> tuple[PredictProbaModel, list[str], list[str]]:
    """Load one saved fold adapter plus feature metadata from a run directory."""
    run_dir = Path(run_dir)
    names = json.loads((run_dir / "feature_names.json").read_text(encoding="utf-8"))
    feature_columns = list(names["feature_columns"])
    categorical_columns = list(names["categorical_columns"])

    cbm = run_dir / "models" / f"{fold_key}.cbm"
    joblib_path = run_dir / "models" / f"{fold_key}.joblib"
    if cbm.is_file():
        adapter = CatBoostAdapter(categorical_columns=categorical_columns)
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(cbm))
        adapter._model = model
        adapter._feature_columns = feature_columns
        return adapter, feature_columns, categorical_columns
    if joblib_path.is_file():
        import joblib

        payload = joblib.load(joblib_path)
        adapter = LightGBMAdapter(categorical_columns=payload["categorical_columns"])
        adapter._model = payload["model"]
        adapter._mapper = payload["mapper"]
        adapter._feature_columns = payload["feature_columns"]
        adapter._best_iteration = payload.get("best_iteration")
        return adapter, feature_columns, categorical_columns
    raise TrainingError(f"no saved model found for fold_key={fold_key!r} under {run_dir}")


def list_fold_keys(run_dir: Path | str) -> list[str]:
    """Return fold keys that have a saved model under the run directory."""
    models_dir = Path(run_dir) / "models"
    if not models_dir.is_dir():
        return []
    keys: list[str] = []
    for path in sorted(models_dir.iterdir()):
        if path.suffix in {".cbm", ".joblib"}:
            keys.append(path.stem)
    return keys


def validation_frame_for_fold(
    train: pd.DataFrame,
    run_dir: Path | str,
    fold_key: str,
) -> pd.DataFrame:
    """Return the out-of-fold validation rows for a saved fold model."""
    match = _FOLD_KEY_RE.match(fold_key)
    if match is None:
        raise TrainingError(f"invalid fold_key={fold_key!r}; expected seedN-foldM")
    seed = int(match.group("seed"))
    fold_id = int(match.group("fold"))
    run_dir = Path(run_dir)
    fold_path = run_dir / f"folds_seed{seed}.parquet"
    if not fold_path.is_file():
        raise TrainingError(f"missing fold assignment file: {fold_path}")
    folds = pd.read_parquet(fold_path)
    if ID_COLUMN not in folds.columns or "fold" not in folds.columns:
        raise TrainingError(f"folds file must contain {ID_COLUMN} and fold columns")
    if ID_COLUMN not in train.columns:
        raise TrainingError("train frame must include id for fold alignment")
    valid_ids = set(folds.loc[folds["fold"] == fold_id, ID_COLUMN].tolist())
    if not valid_ids:
        raise TrainingError(f"no validation rows for {fold_key}")
    subset = train.loc[train[ID_COLUMN].isin(valid_ids)].reset_index(drop=True)
    if len(subset) != len(valid_ids):
        raise TrainingError(
            f"train ids do not cover validation fold {fold_key}: "
            f"expected {len(valid_ids)} rows, got {len(subset)}"
        )
    return subset


def assert_run_matches_processed_data(
    run_dir: Path | str,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> None:
    """Refuse importance when current processed frames differ from the training run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise TrainingError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("data_hashes")
    if not isinstance(expected, dict) or not expected:
        raise TrainingError(f"run missing data_hashes provenance: {run_dir}")
    computed = compute_training_data_hashes(train, test, feature_columns, categorical_columns)
    if computed != expected:
        raise TrainingError(
            "processed data does not match run provenance; "
            "rebuild features or select a run trained on the current parquet"
        )


def compute_run_importance(
    *,
    run_dir: Path | str,
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    fold_key: str | None = None,
    n_repeats: int = 5,
    sample_rows: int | None = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute fold-local permutation importance and write CSV under importance/.

    When ``fold_key`` is None, every saved fold model is evaluated on its own
    validation rows and results are averaged into a summary CSV.
    """
    run_dir = Path(run_dir)
    if TARGET_COLUMN not in train.columns:
        raise TrainingError("train frame must include the target column")
    if test is None:
        raise TrainingError("test frame is required for processed-data provenance checks")
    feature_path = run_dir / "feature_names.json"
    if not feature_path.is_file():
        raise TrainingError(f"missing feature_names.json in {run_dir}")
    feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_columns = list(
        feature_names.get("base_feature_columns") or feature_names.get("feature_columns") or []
    )
    categorical_columns = list(feature_names.get("categorical_columns") or [])
    if not feature_columns:
        raise TrainingError(f"feature_names.json missing feature_columns in {run_dir}")
    assert_run_matches_processed_data(
        run_dir,
        train=train,
        test=test,
        feature_columns=feature_columns,
        categorical_columns=[column for column in categorical_columns if column in feature_columns],
    )

    fold_keys = [fold_key] if fold_key is not None else list_fold_keys(run_dir)
    if not fold_keys:
        raise TrainingError(f"no fold models found under {run_dir / 'models'}")

    out_dir = run_dir / "importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_frames: list[pd.DataFrame] = []

    for key in fold_keys:
        model, model_feature_columns, _ = load_fold_model(run_dir, fold_key=key)
        valid = validation_frame_for_fold(train, run_dir, key)
        model_feature_columns = [
            column for column in model_feature_columns if column in valid.columns
        ]
        frame = permutation_importance_auc(
            model,
            valid,
            valid[TARGET_COLUMN].to_numpy(),
            model_feature_columns,
            n_repeats=n_repeats,
            sample_rows=sample_rows,
            seed=seed,
            run_id=run_dir.name,
        )
        frame = frame.copy()
        frame["fold_key"] = key
        per_fold_path = out_dir / f"permutation_{key}.csv"
        frame.to_csv(per_fold_path, index=False)
        meta = {
            "run_id": run_dir.name,
            "fold_key": key,
            "n_repeats": n_repeats,
            "sample_rows": sample_rows,
            "seed": seed,
            "n_valid_rows": len(valid),
            "output": str(per_fold_path),
            "scope": "validation_fold_only",
        }
        (out_dir / f"permutation_{key}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )
        per_fold_frames.append(frame)

    combined = pd.concat(per_fold_frames, ignore_index=True)
    combined.to_csv(out_dir / "per_fold.csv", index=False)
    summary = (
        combined.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            folds_evaluated=("fold_key", "nunique"),
            baseline_auc_mean=("baseline_auc", "mean"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary["importance_std"] = summary["importance_std"].fillna(0.0)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "fold_keys": fold_keys,
                "n_repeats": n_repeats,
                "sample_rows": sample_rows,
                "seed": seed,
                "scope": "validation_fold_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


# Re-export for tests and CLI helpers.
__all__ = [
    "assert_run_matches_processed_data",
    "compute_run_importance",
    "list_fold_keys",
    "load_fold_model",
    "permutation_importance_auc",
    "stratified_feature_sample",
    "validation_frame_for_fold",
]
