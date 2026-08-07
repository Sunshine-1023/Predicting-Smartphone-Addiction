"""Sampled permutation importance using ROC-AUC drop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from smartphone_addiction.data.schema import TARGET_COLUMN
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.models.catboost import CatBoostAdapter
from smartphone_addiction.models.lightgbm import LightGBMAdapter


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


def compute_run_importance(
    *,
    run_dir: Path | str,
    train: pd.DataFrame,
    fold_key: str = "seed42-fold0",
    n_repeats: int = 5,
    sample_rows: int | None = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance for one run and write CSV under importance/."""
    run_dir = Path(run_dir)
    model, feature_columns, _ = load_fold_model(run_dir, fold_key=fold_key)
    if TARGET_COLUMN not in train.columns:
        raise TrainingError("train frame must include the target column")
    frame = permutation_importance_auc(
        model,
        train,
        train[TARGET_COLUMN].to_numpy(),
        feature_columns,
        n_repeats=n_repeats,
        sample_rows=sample_rows,
        seed=seed,
        run_id=run_dir.name,
    )
    out_dir = run_dir / "importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"permutation_{fold_key}.csv"
    frame.to_csv(out_path, index=False)
    meta = {
        "run_id": run_dir.name,
        "fold_key": fold_key,
        "n_repeats": n_repeats,
        "sample_rows": sample_rows,
        "seed": seed,
        "output": str(out_path),
    }
    (out_dir / f"permutation_{fold_key}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    return frame


# Re-export for tests and CLI helpers.
__all__ = [
    "compute_run_importance",
    "load_fold_model",
    "permutation_importance_auc",
    "stratified_feature_sample",
]
