"""Unit tests for bounded Optuna tuning helpers."""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd
import yaml

from smartphone_addiction.training.tuning import (
    TuningBudget,
    export_top_candidates,
    run_tuning,
    suggest_catboost_params,
    suggest_lightgbm_params,
)


def test_deterministic_fake_study(tmp_path: Path) -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_float("y", 0.0, 1.0)
        return x + 0.1 * y

    result = run_tuning(
        model_name="catboost",
        objective=objective,
        output_dir=tmp_path,
        budget=TuningBudget(n_trials=8, n_candidates=3, seed=7),
        study_name="fake-study",
    )
    trials = pd.read_csv(result.trials_csv)
    assert len(trials) == 8
    assert list(trials["value"]) == sorted(trials["value"], reverse=True)
    assert len(result.candidate_yamls) == 3
    assert len(result.top_params) == 3
    values = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in result.candidate_yamls]
    scores = [item["tuning"]["optuna_value"] for item in values]
    assert scores == sorted(scores, reverse=True)
    assert values[0]["model"]["name"] == "catboost"


def test_search_spaces_do_not_include_seed_or_folds() -> None:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))

    def cat_obj(trial: optuna.Trial) -> float:
        params = suggest_catboost_params(trial)
        assert "random_seed" not in params
        assert "seed" not in params
        assert "n_splits" not in params
        return float(params["depth"])

    def lgb_obj(trial: optuna.Trial) -> float:
        params = suggest_lightgbm_params(trial)
        assert "random_state" not in params
        assert "seed" not in params
        assert "feature_groups" not in params
        return float(params["num_leaves"])

    study.optimize(cat_obj, n_trials=1)
    study.optimize(lgb_obj, n_trials=1)


def test_export_top_candidates_ranks_by_score(tmp_path: Path) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("learning_rate", 0.01, 0.2)

    study.optimize(objective, n_trials=5)
    top_params, paths = export_top_candidates(
        study=study,
        model_name="lightgbm",
        output_dir=tmp_path,
        n_candidates=3,
    )
    assert len(top_params) == 3
    assert all(path.is_file() for path in paths)
