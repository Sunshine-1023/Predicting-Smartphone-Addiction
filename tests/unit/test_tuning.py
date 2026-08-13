"""Unit tests for bounded Optuna tuning helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

from smartphone_addiction.data.schema import TARGET_COLUMN
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.features.domain import RAW_COLUMNS, add_missingness_features
from smartphone_addiction.training.masking import MaskingSettings
from smartphone_addiction.training.tuning import (
    TuningBudget,
    evaluate_candidates,
    evaluate_params_oof,
    export_top_candidates,
    merge_tuning_params,
    promote_candidate,
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
    scores = []
    for path in result.candidate_yamls:
        meta = json.loads(path.with_name(path.stem + ".meta.json").read_text(encoding="utf-8"))
        scores.append(meta["optuna_value"])
    assert scores == sorted(scores, reverse=True)
    assert values[0]["model"]["name"] == "catboost"
    assert "tuning" not in values[0]

    # Candidate YAML must be mergeable into a strict RunConfig.
    from smartphone_addiction.config import load_config
    from smartphone_addiction.paths import project_root

    root = project_root()
    config = load_config(
        [root / "configs/base.yaml", result.candidate_yamls[0]],
        resolve=False,
    )
    assert config.model.name == "catboost"
    assert "x" in config.model.params


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
    assert all(path.with_name(path.stem + ".meta.json").is_file() for path in paths)


def test_merge_tuning_params_keeps_fixed_yaml_values() -> None:
    merged = merge_tuning_params(
        {"thread_count": 4, "depth": 6, "learning_rate": 0.05},
        {"depth": 8, "learning_rate": 0.02},
    )
    assert merged["thread_count"] == 4
    assert merged["depth"] == 8
    assert merged["learning_rate"] == 0.02


def test_fresh_study_replaces_existing_sqlite_study(tmp_path: Path) -> None:
    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    first = run_tuning(
        model_name="catboost",
        objective=objective,
        output_dir=tmp_path,
        budget=TuningBudget(n_trials=3, n_candidates=1, seed=1),
        study_name="fresh-demo",
    )
    assert first.study_db.is_file()
    assert len(pd.read_csv(first.trials_csv)) == 3

    second = run_tuning(
        model_name="catboost",
        objective=objective,
        output_dir=tmp_path,
        budget=TuningBudget(n_trials=2, n_candidates=1, seed=2),
        study_name="fresh-demo",
        fresh_study=True,
    )
    assert len(pd.read_csv(second.trials_csv)) == 2


def test_export_includes_base_params(tmp_path: Path) -> None:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("learning_rate", 0.01, 0.2)

    study.optimize(objective, n_trials=2)
    _, paths = export_top_candidates(
        study=study,
        model_name="lightgbm",
        output_dir=tmp_path,
        n_candidates=1,
        base_params={"n_jobs": 2, "verbose": -1},
    )
    payload = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    assert payload["model"]["params"]["n_jobs"] == 2
    assert "learning_rate" in payload["model"]["params"]


def test_promote_candidate_writes_train_ready_yaml(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate_1.yaml"
    candidate.write_text(
        "model:\n  name: catboost\n  params:\n    depth: 7\n    learning_rate: 0.03\n",
        encoding="utf-8",
    )
    selected = tmp_path / "selected_final.yaml"
    selected.write_text(
        "features:\n"
        "  groups: [raw, missingness]\n"
        "  exclude_columns: [missing_pattern]\n"
        "  masking:\n    enabled: true\n    fraction: 0.2\n"
        "model:\n  name: catboost\n  params:\n    depth: 7\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "best": {
                    "candidate_yaml": str(candidate),
                    "oof_auc": 0.9,
                    "run_dir": str(tmp_path / "run"),
                },
                "selected_yaml": str(selected),
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "catboost_final_v2.yaml"
    path = promote_candidate(selection_json=selection, output_yaml=out)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["model"]["params"]["depth"] == 7
    assert payload["features"]["groups"] == ["raw", "missingness"]
    assert payload["features"]["exclude_columns"] == ["missing_pattern"]
    assert payload["features"]["masking"]["enabled"] is True
    assert payload["features"]["masking"]["fraction"] == 0.2
    assert "selection" not in payload
    assert path.with_suffix(".meta.json").is_file()


class _FitRecorder:
    n_train: ClassVar[list[int]] = []

    def __init__(self) -> None:
        self.best_iteration = 1

    def fit(self, x, y, x_valid=None, y_valid=None, **kwargs):
        _FitRecorder.n_train.append(len(x))
        return self

    def predict_proba(self, x):
        return np.full(len(x), 0.6)


def _oof_frames(n_train: int = 40, n_test: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(0)
    train = add_missingness_features(
        pd.DataFrame(
            {
                "age": rng.normal(30, 5, size=n_train),
                "daily_screen_time_hours": rng.normal(5, 1, size=n_train),
                "social_media_hours": rng.normal(2, 0.5, size=n_train),
                "gaming_hours": rng.normal(1, 0.4, size=n_train),
                "work_study_hours": rng.normal(2, 0.5, size=n_train),
                "sleep_hours": rng.normal(7, 1, size=n_train),
                "notifications_per_day": rng.normal(50, 10, size=n_train),
                "app_opens_per_day": rng.normal(80, 15, size=n_train),
                "weekend_screen_time": rng.normal(6, 1, size=n_train),
                "gender": rng.choice(["Male", "Female"], size=n_train),
                "stress_level": rng.choice(["Low", "Medium", "High"], size=n_train),
                "academic_work_impact": rng.choice(["None", "Mild", "Severe"], size=n_train),
            }
        ),
        RAW_COLUMNS,
    )
    train[TARGET_COLUMN] = np.array([0, 1] * (n_train // 2))
    test = pd.DataFrame(
        {
            "age": rng.normal(30, 5, size=n_test),
            "daily_screen_time_hours": rng.normal(5, 1, size=n_test),
            "social_media_hours": rng.normal(2, 0.5, size=n_test),
            "gaming_hours": rng.normal(1, 0.4, size=n_test),
            "work_study_hours": rng.normal(2, 0.5, size=n_test),
            "sleep_hours": rng.normal(7, 1, size=n_test),
            "notifications_per_day": rng.normal(50, 10, size=n_test),
            "app_opens_per_day": rng.normal(80, 15, size=n_test),
            "weekend_screen_time": rng.normal(6, 1, size=n_test),
            "gender": rng.choice(["Male", "Female"], size=n_test),
            "stress_level": rng.choice(["Low", "Medium", "High"], size=n_test),
            "academic_work_impact": rng.choice(["None", "Mild", "Severe"], size=n_test),
        }
    )
    test.loc[0, "daily_screen_time_hours"] = np.nan
    test.loc[1, ["weekend_screen_time", "social_media_hours"]] = np.nan
    test = add_missingness_features(test, RAW_COLUMNS)
    return train, test


def test_evaluate_params_oof_applies_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    train, test = _oof_frames()
    feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    monkeypatch.setattr(
        "smartphone_addiction.training.tuning._build_model",
        lambda *args, **kwargs: _FitRecorder(),
    )

    _FitRecorder.n_train = []
    evaluate_params_oof(
        model_name="lightgbm",
        params={},
        train=train,
        feature_columns=feature_columns,
        categorical_columns=["gender", "stress_level", "academic_work_impact"],
        n_splits=2,
        seed=42,
        test=test,
        masking={"enabled": False, "fraction": 0.20},
    )
    unmasked_sizes = list(_FitRecorder.n_train)

    _FitRecorder.n_train = []
    evaluate_params_oof(
        model_name="lightgbm",
        params={},
        train=train,
        feature_columns=feature_columns,
        categorical_columns=["gender", "stress_level", "academic_work_impact"],
        n_splits=2,
        seed=42,
        test=test,
        masking=MaskingSettings(enabled=True, fraction=0.20),
    )
    masked_sizes = list(_FitRecorder.n_train)

    assert len(unmasked_sizes) == 2
    assert len(masked_sizes) == 2
    assert masked_sizes[0] > unmasked_sizes[0]
    assert masked_sizes[1] > unmasked_sizes[1]


def test_evaluate_params_oof_requires_test_when_masking_enabled() -> None:
    train, _ = _oof_frames()
    feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    with pytest.raises(TrainingError, match="test features"):
        evaluate_params_oof(
            model_name="lightgbm",
            params={},
            train=train,
            feature_columns=feature_columns,
            categorical_columns=["gender"],
            n_splits=2,
            seed=42,
            masking={"enabled": True, "fraction": 0.20},
        )


def test_evaluate_candidates_forwards_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            run_dir=tmp_path / "run",
            metrics={
                "oof_auc": 0.91,
                "seed_auc_mean": 0.91,
                "seed_auc_std": 0.0,
                "n_features": 3,
            },
        )

    monkeypatch.setattr("smartphone_addiction.training.tuning.run_training", fake_run_training)
    candidate = tmp_path / "candidate_1.yaml"
    candidate.write_text(
        "model:\n  name: lightgbm\n  params:\n    num_leaves: 31\n",
        encoding="utf-8",
    )
    train, test = _oof_frames(n_train=8, n_test=4)
    result = evaluate_candidates(
        candidate_yamls=[candidate],
        train=train,
        test=test,
        feature_columns=["age"],
        categorical_columns=[],
        feature_groups=["raw"],
        exclude_columns=["missing_pattern"],
        masking={"enabled": True, "fraction": 0.20},
        artifact_root=tmp_path / "artifacts",
        output_dir=tmp_path / "eval",
        n_splits=2,
        seeds=[42],
    )
    masking = captured["masking"]
    assert isinstance(masking, MaskingSettings)
    assert masking.enabled is True
    assert masking.fraction == 0.20
    selected = yaml.safe_load(result.selected_yaml.read_text(encoding="utf-8"))
    assert selected["features"]["masking"]["enabled"] is True
    assert selected["features"]["masking"]["fraction"] == 0.2
    assert selected["features"]["masking"]["fields"] == "core5"
    assert selected["features"]["masking"]["compatible_sources"] is False
    assert selected["features"]["masking"]["sample_weight"] is False
    assert selected["features"]["exclude_columns"] == ["missing_pattern"]
    selection = json.loads(result.selection_json.read_text(encoding="utf-8"))
    assert selection["masking"]["enabled"] is True
    assert selection["masking"]["fields"] == "core5"
    assert selection["masking"]["version"] == 3
