"""Bounded Optuna hyperparameter search for CatBoost and LightGBM."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedShuffleSplit

from smartphone_addiction.data.schema import TARGET_COLUMN
from smartphone_addiction.errors import ConfigurationError, TrainingError
from smartphone_addiction.evaluation.metrics import summarize_oof
from smartphone_addiction.models.catboost import build_catboost
from smartphone_addiction.models.lightgbm import build_lightgbm
from smartphone_addiction.training.cv import make_folds

SUPPORTED_MODELS = frozenset({"catboost", "lightgbm"})

DEFAULT_BUDGET = {
    "sample_fraction": 0.5,
    "n_splits": 3,
    "seed": 42,
    "n_trials": 20,
    "n_candidates": 3,
}


@dataclass(frozen=True)
class TuningBudget:
    """Hard budget for the Optuna stage (not full final CV)."""

    sample_fraction: float = 0.5
    n_splits: int = 3
    seed: int = 42
    n_trials: int = 20
    n_candidates: int = 3


@dataclass(frozen=True)
class TuningResult:
    """Artifacts produced by a completed (or resumed) Optuna study."""

    study_db: Path
    trials_csv: Path
    candidate_yamls: list[Path]
    top_params: list[dict[str, Any]]
    study_name: str


def suggest_catboost_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest CatBoost hyperparameters (no seeds / folds / feature groups)."""
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "iterations": trial.suggest_int("iterations", 200, 2000),
        "early_stopping_rounds": 50,
    }


def suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest LightGBM hyperparameters (no seeds / folds / feature groups)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "early_stopping_rounds": 50,
    }


def suggest_params(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Dispatch search space by model name."""
    name = model_name.lower().strip()
    if name == "catboost":
        return suggest_catboost_params(trial)
    if name == "lightgbm":
        return suggest_lightgbm_params(trial)
    raise ConfigurationError(f"unsupported tuning model: {model_name!r}")


def stratified_sample(
    train: pd.DataFrame,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Return a stratified subsample for the Optuna stage."""
    if not 0.0 < sample_fraction <= 1.0:
        raise ConfigurationError("sample_fraction must be in (0, 1]")
    if sample_fraction >= 1.0:
        return train.reset_index(drop=True)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_fraction,
        random_state=seed,
    )
    idx, _ = next(splitter.split(train, train[TARGET_COLUMN]))
    return train.iloc[idx].reset_index(drop=True)


def evaluate_params_oof(
    *,
    model_name: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    n_splits: int,
    seed: int,
) -> float:
    """Fit one parameter set with stratified OOF and return ROC-AUC."""
    y = train[TARGET_COLUMN].to_numpy()
    x = train[feature_columns]
    fold_ids = make_folds(y, n_splits=n_splits, seed=seed)
    oof = np.full(len(train), np.nan, dtype=float)
    for fold_id in range(n_splits):
        train_mask = fold_ids != fold_id
        valid_mask = fold_ids == fold_id
        model = _build_model(model_name, categorical_columns, params, seed=seed)
        model.fit(
            x.loc[train_mask],
            y[train_mask],
            x.loc[valid_mask],
            y[valid_mask],
        )
        oof[valid_mask] = model.predict_proba(x.loc[valid_mask])
        del model
    return float(summarize_oof(y, oof).auc)


def make_tuning_objective(
    *,
    model_name: str,
    train: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    budget: TuningBudget | None = None,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective that never persists trial models."""
    budget = budget or TuningBudget()
    sample = stratified_sample(train, budget.sample_fraction, budget.seed)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial)
        return evaluate_params_oof(
            model_name=model_name,
            params=params,
            train=sample,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            n_splits=budget.n_splits,
            seed=budget.seed,
        )

    return objective


def run_tuning(
    *,
    model_name: str,
    objective: Callable[[optuna.Trial], float],
    output_dir: Path | str,
    budget: TuningBudget | None = None,
    study_name: str | None = None,
    storage: str | None = None,
) -> TuningResult:
    """Run a bounded Optuna study, export trials.csv and top candidate YAMLs."""
    model_name = model_name.lower().strip()
    if model_name not in SUPPORTED_MODELS:
        raise TrainingError(
            f"unsupported model_name={model_name!r}; expected one of {sorted(SUPPORTED_MODELS)}"
        )
    budget = budget or TuningBudget()
    if budget.n_trials < 1:
        raise ConfigurationError("n_trials must be >= 1")
    if budget.n_candidates < 1:
        raise ConfigurationError("n_candidates must be >= 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = study_name or f"{model_name}-tune"
    if storage is None:
        db_path = output_dir / "optuna.db"
        storage = f"sqlite:///{db_path.resolve()}"
    else:
        db_path = Path(str(storage).removeprefix("sqlite:///"))

    sampler = TPESampler(seed=budget.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    remaining = max(0, budget.n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    trials_csv = output_dir / "trials.csv"
    _write_trials_csv(study, trials_csv)
    top_params, candidate_yamls = export_top_candidates(
        study=study,
        model_name=model_name,
        output_dir=output_dir,
        n_candidates=budget.n_candidates,
    )
    (output_dir / "tuning_meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "study_name": study_name,
                "n_trials": len(study.trials),
                "budget": {
                    "sample_fraction": budget.sample_fraction,
                    "n_splits": budget.n_splits,
                    "seed": budget.seed,
                    "n_trials": budget.n_trials,
                    "n_candidates": budget.n_candidates,
                },
                "best_value": study.best_value if study.trials else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TuningResult(
        study_db=db_path,
        trials_csv=trials_csv,
        candidate_yamls=candidate_yamls,
        top_params=top_params,
        study_name=study_name,
    )


def export_top_candidates(
    *,
    study: optuna.Study,
    model_name: str,
    output_dir: Path,
    n_candidates: int = 3,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Export the best complete trials as ranked YAML configs."""
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    selected = completed[:n_candidates]
    top_params: list[dict[str, Any]] = []
    paths: list[Path] = []
    for rank, trial in enumerate(selected, start=1):
        params = dict(trial.params)
        # Restore early_stopping default used during suggest_* helpers.
        if model_name == "catboost":
            params.setdefault("early_stopping_rounds", 50)
        else:
            params.setdefault("early_stopping_rounds", 50)
        payload = {
            "model": {"name": model_name, "params": params},
            "tuning": {
                "trial_number": trial.number,
                "optuna_value": float(trial.value),
                "rank": rank,
                "note": "Optuna-stage score only; re-evaluate on full 5-fold CV.",
            },
        }
        path = output_dir / f"candidate_{rank}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        top_params.append(params)
        paths.append(path)
    return top_params, paths


def _write_trials_csv(study: optuna.Study, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("value", ascending=False, na_position="last")
    frame.to_csv(path, index=False)


def _build_model(
    model_name: str,
    categorical_columns: list[str],
    params: dict[str, Any],
    *,
    seed: int,
) -> Any:
    clean = dict(params)
    if model_name == "catboost":
        clean["random_seed"] = seed
        return build_catboost(categorical_columns=categorical_columns, **clean)
    clean["random_state"] = seed
    return build_lightgbm(categorical_columns=categorical_columns, **clean)
