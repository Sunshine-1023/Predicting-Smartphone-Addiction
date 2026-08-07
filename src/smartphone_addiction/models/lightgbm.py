"""LightGBM model adapter with fold-local categorical encoding."""

from __future__ import annotations

import inspect
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.models.base import prepare_categorical_frame
from smartphone_addiction.models.progress import (
    close_bar,
    make_iteration_bar,
    make_lightgbm_tqdm_callback,
)


class CategoryMapper:
    """Map categorical strings to integers fitted on the training fold only."""

    def __init__(self, categorical_columns: list[str]) -> None:
        self.categorical_columns = list(categorical_columns)
        self.mappings: dict[str, dict[str, int]] = {}
        self.unknown_codes: dict[str, int] = {}

    def fit(self, frame: pd.DataFrame) -> CategoryMapper:
        prepared = prepare_categorical_frame(frame, self.categorical_columns)
        for column in self.categorical_columns:
            if column not in prepared.columns:
                continue
            levels = sorted(prepared[column].astype(str).unique().tolist())
            mapping = {level: index for index, level in enumerate(levels)}
            self.mappings[column] = mapping
            self.unknown_codes[column] = len(mapping)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = prepare_categorical_frame(frame, self.categorical_columns)
        out = prepared.copy()
        for column in self.categorical_columns:
            if column not in out.columns:
                continue
            mapping = self.mappings[column]
            unknown = self.unknown_codes[column]
            out[column] = (
                out[column]
                .astype(str)
                .map(lambda value, m=mapping, u=unknown: m.get(value, u))
                .astype("int32")
            )
        return out


class LightGBMAdapter:
    """LightGBM binary classifier adapter for fold-wise training."""

    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        n_estimators: int = 1000,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_jobs: int = 4,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
        min_child_samples: int = 20,
        **params: object,
    ) -> None:
        self.categorical_columns = list(categorical_columns or [])
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_samples = min_child_samples
        self.extra_params = params
        self._model: lgb.LGBMClassifier | None = None
        self._mapper: CategoryMapper | None = None
        self._feature_columns: list[str] | None = None
        self._best_iteration: int | None = None

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray,
        x_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        *,
        show_progress: bool = False,
        progress_desc: str = "lightgbm",
    ) -> LightGBMAdapter:
        self._feature_columns = list(x.columns)
        cat_cols = [c for c in self.categorical_columns if c in x.columns]
        self._mapper = CategoryMapper(cat_cols).fit(x)
        x_train = self._mapper.transform(x)
        y_train = np.asarray(y)

        params = dict(self.extra_params)
        params.pop("show_progress", None)
        params.pop("progress_desc", None)

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=-1,
            **params,
        )

        bar = make_iteration_bar(self.n_estimators, progress_desc) if show_progress else None
        try:
            if x_valid is not None and y_valid is not None:
                x_eval = self._mapper.transform(x_valid[self._feature_columns])
                y_eval = np.asarray(y_valid)
                callbacks = [
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
                if bar is not None:
                    callbacks.append(make_lightgbm_tqdm_callback(bar))
                fit_kwargs: dict = {
                    "eval_metric": "auc",
                    "categorical_feature": cat_cols,
                    "callbacks": callbacks,
                }
                # Prefer eval_X/eval_y when available (LightGBM 4.5+); avoid try/except.
                if "eval_X" in inspect.signature(model.fit).parameters:
                    fit_kwargs["eval_X"] = x_eval
                    fit_kwargs["eval_y"] = y_eval
                else:
                    fit_kwargs["eval_set"] = [(x_eval, y_eval)]
                model.fit(x_train, y_train, **fit_kwargs)
            else:
                fit_kwargs = {"categorical_feature": cat_cols}
                if bar is not None:
                    fit_kwargs["callbacks"] = [make_lightgbm_tqdm_callback(bar)]
                model.fit(x_train, y_train, **fit_kwargs)
        finally:
            close_bar(bar)

        self._model = model
        best = getattr(model, "best_iteration_", None)
        if best is None or best == 0:
            best = int(getattr(model, "n_estimators", self.n_estimators))
        self._best_iteration = int(best)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        model = self._require_model()
        if self._mapper is None or self._feature_columns is None:
            raise TrainingError("LightGBMAdapter must be fitted before predict")
        frame = self._mapper.transform(x[self._feature_columns])
        probs = model.predict_proba(frame)
        positive = np.asarray(probs[:, 1], dtype=float)
        if not np.isfinite(positive).all():
            raise TrainingError("LightGBM produced non-finite probabilities")
        return positive

    def save(self, path: Path | str) -> Path:
        model = self._require_model()
        if self._mapper is None or self._feature_columns is None:
            raise TrainingError("LightGBMAdapter must be fitted before save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "mapper": self._mapper,
            "feature_columns": self._feature_columns,
            "categorical_columns": self.categorical_columns,
            "best_iteration": self._best_iteration,
        }
        joblib.dump(payload, target)
        return target

    @property
    def best_iteration(self) -> int | None:
        return self._best_iteration

    def _require_model(self) -> lgb.LGBMClassifier:
        if self._model is None:
            raise TrainingError("LightGBMAdapter is not fitted")
        return self._model


def build_lightgbm(
    categorical_columns: list[str] | None = None,
    **params: object,
) -> LightGBMAdapter:
    """Factory used by tests and the training runner."""
    return LightGBMAdapter(categorical_columns=categorical_columns, **params)
