"""CatBoost model adapter with native categorical support."""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.models.base import prepare_categorical_frame
from smartphone_addiction.models.progress import (
    CatBoostProgressStdout,
    close_bar,
    make_iteration_bar,
)


class CatBoostAdapter:
    """CatBoost binary classifier adapter for fold-wise training."""

    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.05,
        thread_count: int = 4,
        random_seed: int = 42,
        early_stopping_rounds: int = 50,
        l2_leaf_reg: float = 3.0,
        **params: object,
    ) -> None:
        self.categorical_columns = list(categorical_columns or [])
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.thread_count = thread_count
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.l2_leaf_reg = l2_leaf_reg
        self.extra_params = params
        self._model: CatBoostClassifier | None = None
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
        progress_desc: str = "catboost",
    ) -> CatBoostAdapter:
        self._feature_columns = list(x.columns)
        cat_cols = [c for c in self.categorical_columns if c in x.columns]
        x_train = prepare_categorical_frame(x, cat_cols)
        y_train = np.asarray(y)

        train_pool = Pool(x_train, y_train, cat_features=cat_cols)
        eval_set = None
        if x_valid is not None and y_valid is not None:
            x_eval = prepare_categorical_frame(x_valid[self._feature_columns], cat_cols)
            eval_set = Pool(x_eval, np.asarray(y_valid), cat_features=cat_cols)

        params = dict(self.extra_params)
        params.pop("show_progress", None)
        params.pop("progress_desc", None)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            task_type="CPU",
            allow_writing_files=False,
            verbose=False,
            **params,
        )
        fit_kwargs: dict[str, object] = {
            "verbose": 1 if show_progress else False,
        }
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            fit_kwargs["use_best_model"] = True

        bar = make_iteration_bar(self.iterations, progress_desc) if show_progress else None
        try:
            if bar is None:
                model.fit(train_pool, **fit_kwargs)
            else:
                with redirect_stdout(CatBoostProgressStdout(bar)):
                    model.fit(train_pool, **fit_kwargs)
        finally:
            close_bar(bar)

        self._model = model
        # CatBoost best_iteration_ is 0-based; tree_count_ is the used tree count.
        best = getattr(model, "best_iteration_", None)
        tree_count = getattr(model, "tree_count_", None)
        if tree_count is not None and int(tree_count) > 0:
            self._best_iteration = int(tree_count)
        elif best is not None:
            self._best_iteration = int(best) + 1
        else:
            self._best_iteration = int(self.iterations)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        model = self._require_model()
        if self._feature_columns is None:
            raise TrainingError("CatBoostAdapter must be fitted before predict")
        cat_cols = [c for c in self.categorical_columns if c in self._feature_columns]
        frame = prepare_categorical_frame(x[self._feature_columns], cat_cols)
        probs = model.predict_proba(frame)
        positive = np.asarray(probs[:, 1], dtype=float)
        if not np.isfinite(positive).all():
            raise TrainingError("CatBoost produced non-finite probabilities")
        return positive

    def save(self, path: Path | str) -> Path:
        model = self._require_model()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(target))
        return target

    @property
    def best_iteration(self) -> int | None:
        return self._best_iteration

    def _require_model(self) -> CatBoostClassifier:
        if self._model is None:
            raise TrainingError("CatBoostAdapter is not fitted")
        return self._model


def build_catboost(
    categorical_columns: list[str] | None = None,
    **params: object,
) -> CatBoostAdapter:
    """Factory used by tests and the training runner."""
    return CatBoostAdapter(categorical_columns=categorical_columns, **params)
