"""End-to-end OOF training runner for CatBoost and LightGBM."""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from smartphone_addiction.alignment import align_predictions_to_ids, assert_unique_ids
from smartphone_addiction.artifacts.store import ArtifactStore
from smartphone_addiction.data.load import CompetitionFrames
from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import (
    AlignmentError,
    ArtifactError,
    DataValidationError,
    TrainingError,
)
from smartphone_addiction.evaluation.metrics import summarize_oof
from smartphone_addiction.evaluation.slices import compute_slice_metrics
from smartphone_addiction.features.base import (
    exclude_feature_columns,
    select_feature_columns_from_groups,
    transform_competition_frames,
)
from smartphone_addiction.models.catboost import build_catboost
from smartphone_addiction.models.lightgbm import build_lightgbm
from smartphone_addiction.training.cv import make_folds
from smartphone_addiction.training.masking import MaskingSettings, augment_training_fold

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = frozenset({"catboost", "lightgbm"})


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    metrics: dict[str, Any]
    store: ArtifactStore


def compute_training_data_hashes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, str]:
    """SHA-256 fingerprints for train, test, and the feature column contract."""
    train_cols = [ID_COLUMN, TARGET_COLUMN, *feature_columns]
    test_cols = [ID_COLUMN, *feature_columns]
    return {
        "train": _fingerprint_frame(train, train_cols),
        "test": _fingerprint_frame(test, test_cols),
        "feature_manifest": _fingerprint_json(
            {
                "feature_columns": list(feature_columns),
                "categorical_columns": list(categorical_columns),
            }
        ),
    }


def run_training(
    *,
    artifact_root: Path,
    model_name: str,
    frames: CompetitionFrames | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    model_params: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    feature_groups: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    masking: dict[str, Any] | MaskingSettings | None = None,
    n_splits: int = 5,
    seeds: list[int] | None = None,
    git_sha: str = "localdev",
    git_dirty: bool = False,
    resume_run_dir: Path | None = None,
    data_hashes: dict[str, str] | None = None,
    slug: str | None = None,
    allow_resume_completed: bool = False,
) -> TrainingResult:
    """Run multi-seed stratified OOF training and persist artifacts.

    Provide either raw ``frames`` (train/test/sample) which will be transformed,
    or already-processed ``train`` / ``test`` frames with features and label.
    Only ``catboost`` and ``lightgbm`` are supported.

    Data identity is always derived from SHA-256 fingerprints of the resolved
    frames. Placeholder hashes such as ``{"source": "in-memory"}`` are rejected.
    """
    model_name = model_name.lower().strip()
    if model_name not in SUPPORTED_MODELS:
        raise TrainingError(
            f"unsupported model_name={model_name!r}; expected one of {sorted(SUPPORTED_MODELS)}"
        )

    seeds = list(seeds or [42])
    model_params = dict(model_params or {})
    masking_settings = (
        masking if isinstance(masking, MaskingSettings) else MaskingSettings.from_mapping(masking)
    )

    train_df, test_df, feature_cols, cat_cols = _resolve_frames(
        frames=frames,
        train=train,
        test=test,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        feature_groups=feature_groups,
        exclude_columns=exclude_columns,
    )

    computed_hashes = compute_training_data_hashes(train_df, test_df, feature_cols, cat_cols)
    if data_hashes is not None:
        provided = dict(data_hashes)
        if provided != computed_hashes:
            raise TrainingError("provided data_hashes do not match the resolved train/test frames")
    data_hashes = computed_hashes

    config = {
        "model": {"name": model_name, "params": model_params},
        "cv": {"n_splits": n_splits, "seeds": seeds},
        "features": {
            "feature_columns": feature_cols,
            "categorical_columns": cat_cols,
            "groups": list(feature_groups) if feature_groups is not None else None,
            "exclude_columns": list(exclude_columns or []),
            "masking": masking_settings.to_dict(),
        },
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
    }
    expected_fold_keys = [_fold_key(seed, fold_id) for seed in seeds for fold_id in range(n_splits)]

    y = train_df[TARGET_COLUMN].to_numpy()
    x_all = train_df[feature_cols]
    x_test = test_df[feature_cols]
    try:
        ids = assert_unique_ids(train_df[ID_COLUMN], label="train id").to_numpy()
        test_ids = assert_unique_ids(test_df[ID_COLUMN], label="test id").to_numpy()
    except AlignmentError as exc:
        raise TrainingError(str(exc)) from exc

    if resume_run_dir is not None:
        store = ArtifactStore.open(resume_run_dir)
        pending_keys = store.resume_missing_folds(
            config=config,
            data_hashes=data_hashes,
            expected_fold_keys=expected_fold_keys,
            allow_completed=allow_resume_completed,
        )
        _validate_resume_consistency(
            store=store,
            ids=ids,
            test_ids=test_ids,
            y=y,
            feature_cols=feature_cols,
            seeds=seeds,
            n_splits=n_splits,
            masking_settings=masking_settings,
            completed_keys=[key for key in expected_fold_keys if key not in pending_keys],
        )
    else:
        store = ArtifactStore.create(
            artifact_root=Path(artifact_root),
            slug=slug or f"{model_name}-oof",
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        store.start(
            config=config,
            data_hashes=data_hashes,
            n_train_rows=len(train_df),
            n_features=len(feature_cols),
            seeds=seeds,
            n_splits=n_splits,
        )
        pending_keys = list(expected_fold_keys)

    _setup_run_logging(store.run_dir)
    pred_dir = store.run_dir / "fold_predictions"
    pred_dir.mkdir(exist_ok=True)
    store.write_json(
        "feature_names.json",
        {
            "feature_columns": feature_cols,
            "categorical_columns": cat_cols,
            "masking": masking_settings.to_dict(),
        },
    )

    try:
        fold_ids_by_seed: dict[int, np.ndarray] = {}
        for seed in seeds:
            fold_ids = make_folds(y, n_splits=n_splits, seed=seed)
            fold_ids_by_seed[seed] = fold_ids
            store.write_frame(
                f"folds_seed{seed}.parquet",
                pd.DataFrame({ID_COLUMN: ids, "fold": fold_ids}),
            )

        fold_jobs = [
            (seed, fold_id)
            for seed in seeds
            for fold_id in range(n_splits)
            if _fold_key(seed, fold_id) in pending_keys
        ]
        progress = tqdm(
            fold_jobs,
            desc=f"{model_name} OOF",
            unit="fold",
            leave=True,
        )
        model_feature_cols = list(feature_cols)
        if not fold_jobs:
            existing_names = store.run_dir / "feature_names.json"
            if existing_names.is_file():
                saved_names = json.loads(existing_names.read_text(encoding="utf-8"))
                model_feature_cols = list(saved_names.get("feature_columns") or feature_cols)
        for seed, fold_id in progress:
            key = _fold_key(seed, fold_id)
            fold_ids = fold_ids_by_seed[seed]
            progress.set_postfix_str(key)

            train_mask = fold_ids != fold_id
            valid_mask = fold_ids == fold_id
            valid_index = np.flatnonzero(valid_mask)

            x_train = x_all.loc[train_mask]
            x_valid = x_all.loc[valid_mask]
            y_train = y[train_mask]
            x_masked, y_masked = augment_training_fold(
                x_train,
                y_train,
                test_features=x_test,
                settings=masking_settings,
                seed=seed,
                fold_id=fold_id,
            )
            if len(x_masked) > 0:
                x_train_fit = pd.concat([x_train, x_masked], axis=0, ignore_index=True)
                y_train_fit = np.concatenate([y_train, y_masked])
            else:
                x_train_fit = x_train
                y_train_fit = y_train
            model_feature_cols = list(x_train_fit.columns)

            model = _build_model(
                model_name,
                cat_cols,
                model_params,
                seed=seed,
            )
            model.fit(
                x_train_fit,
                y_train_fit,
                x_valid,
                y[valid_mask],
                show_progress=True,
                progress_desc=key,
            )
            valid_pred = model.predict_proba(x_valid)
            test_pred = model.predict_proba(x_test)
            valid_ids = ids[valid_index]

            np.savez_compressed(
                pred_dir / f"{key}.npz",
                valid_index=valid_index.astype(np.int64),
                valid_ids=valid_ids,
                valid_pred=valid_pred.astype(np.float64),
                test_ids=test_ids,
                test_pred=test_pred.astype(np.float64),
            )

            fold_auc = float(summarize_oof(y[valid_mask], valid_pred).auc)
            fold_row = {
                "seed": int(seed),
                "fold": int(fold_id),
                "fold_key": key,
                "auc": fold_auc,
                "n_train": int(train_mask.sum()),
                "n_train_masked": len(x_masked),
                "n_valid": int(valid_mask.sum()),
                "best_iteration": model.best_iteration,
                "model_seed": int(seed),
                "n_features": len(model_feature_cols),
            }
            suffix = ".cbm" if model_name == "catboost" else ".joblib"
            model.save(store.run_dir / "models" / f"{key}{suffix}")
            store.mark_fold_complete(key, fold_row)
            progress.set_postfix_str(f"{key} auc={fold_auc:.4f}")
            logger.info("completed %s auc=%.6f model_seed=%s", key, fold_auc, seed)
            del model

        store.write_json(
            "feature_names.json",
            {
                "feature_columns": model_feature_cols,
                "categorical_columns": cat_cols,
                "masking": masking_settings.to_dict(),
            },
        )

        metrics = _aggregate_and_write(
            store=store,
            pred_dir=pred_dir,
            expected_fold_keys=expected_fold_keys,
            seeds=seeds,
            n_splits=n_splits,
            model_name=model_name,
            ids=ids,
            test_ids=test_ids,
            y=y,
            feature_cols=model_feature_cols,
            train_features=train_df,
            test_features=test_df,
        )
        if git_dirty:
            metrics["git_dirty"] = True
            metrics["provenance_warning"] = (
                "run produced from a dirty git tree; git_sha alone cannot reproduce this run"
            )
        store.complete(metrics=metrics)
        return TrainingResult(run_dir=store.run_dir, metrics=metrics, store=store)

    except KeyboardInterrupt:
        store.interrupt("KeyboardInterrupt")
        raise
    except Exception as exc:
        store.fail(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        raise


def _validate_resume_consistency(
    *,
    store: ArtifactStore,
    ids: np.ndarray,
    test_ids: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    seeds: list[int],
    n_splits: int,
    masking_settings: MaskingSettings,
    completed_keys: list[str],
) -> None:
    """Ensure saved folds/predictions still match the current frames."""
    feature_path = store.run_dir / "feature_names.json"
    if feature_path.is_file():
        saved = json.loads(feature_path.read_text(encoding="utf-8"))
        saved_cols = list(saved.get("feature_columns", []))
        if saved_cols != list(feature_cols):
            raise ArtifactError("feature column order/content mismatch; refusing to resume")
        saved_masking = saved.get("masking") or MaskingSettings().to_dict()
        if saved_masking != masking_settings.to_dict():
            raise ArtifactError("masking config mismatch; refusing to resume")

    pred_dir = store.run_dir / "fold_predictions"
    for seed in seeds:
        fold_path = store.run_dir / f"folds_seed{seed}.parquet"
        expected_folds = make_folds(y, n_splits=n_splits, seed=seed)
        if fold_path.is_file():
            saved_folds = pd.read_parquet(fold_path)
            if not np.array_equal(saved_folds[ID_COLUMN].to_numpy(), ids):
                raise ArtifactError(f"id mismatch in folds_seed{seed}.parquet; refusing to resume")
            if not np.array_equal(saved_folds["fold"].to_numpy(), expected_folds):
                raise ArtifactError(f"fold assignment mismatch for seed={seed}; refusing to resume")

        for fold_id in range(n_splits):
            key = _fold_key(seed, fold_id)
            if key not in completed_keys:
                continue
            pred_path = pred_dir / f"{key}.npz"
            if not pred_path.is_file():
                raise ArtifactError(f"missing prediction file for completed fold {key}")
            try:
                _load_fold_predictions(
                    pred_path,
                    ids=ids,
                    test_ids=test_ids,
                    expected_valid_index=np.flatnonzero(expected_folds == fold_id),
                    label=key,
                )
            except AlignmentError as exc:
                raise ArtifactError(f"{key}: {exc}; refusing to resume") from exc


def _load_fold_predictions(
    path: Path,
    *,
    ids: np.ndarray,
    test_ids: np.ndarray,
    expected_valid_index: np.ndarray | None = None,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one fold payload and align OOF/test predictions by ID when available."""
    payload = np.load(path)
    valid_index = np.asarray(payload["valid_index"])
    valid_pred = np.asarray(payload["valid_pred"], dtype=float)
    test_pred = np.asarray(payload["test_pred"], dtype=float)

    if expected_valid_index is not None and not np.array_equal(valid_index, expected_valid_index):
        raise ArtifactError(f"prediction indices mismatch for {label}")
    if len(valid_pred) != len(valid_index):
        raise ArtifactError(f"prediction length mismatch for {label}")
    if not np.isfinite(valid_pred).all():
        raise ArtifactError(f"non-finite OOF predictions in {label}")

    if "valid_ids" in payload.files:
        valid_ids = np.asarray(payload["valid_ids"])
        if not np.array_equal(valid_ids, ids[valid_index]):
            raise AlignmentError(f"valid ids disagree with train id order for {label}")

    if "test_ids" in payload.files:
        test_pred = align_predictions_to_ids(
            test_ids,
            payload["test_ids"],
            test_pred,
            label=f"{label} test",
        )
    elif len(test_pred) != len(test_ids):
        raise AlignmentError(
            f"{label} test prediction length {len(test_pred)} != test rows {len(test_ids)}"
        )
    elif not np.isfinite(test_pred).all():
        raise AlignmentError(f"non-finite test predictions in {label}")

    return valid_index, valid_pred, test_pred


def _aggregate_and_write(
    *,
    store: ArtifactStore,
    pred_dir: Path,
    expected_fold_keys: list[str],
    seeds: list[int],
    n_splits: int,
    model_name: str,
    ids: np.ndarray,
    test_ids: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> dict[str, Any]:
    missing = [key for key in expected_fold_keys if not (pred_dir / f"{key}.npz").is_file()]
    if missing:
        raise TrainingError(f"missing fold prediction files: {missing}")

    oof_sum = np.zeros(len(ids), dtype=float)
    oof_count = np.zeros(len(ids), dtype=float)
    test_sum = np.zeros(len(test_ids), dtype=float)
    test_count = 0
    fold_rows: list[dict[str, Any]] = []
    seed_oof: dict[int, np.ndarray] = {
        seed: np.full(len(ids), np.nan, dtype=float) for seed in seeds
    }

    metrics_path = store.run_dir / "fold_metrics.csv"
    recorded: dict[str, dict[str, Any]] = {}
    if metrics_path.is_file():
        for row in pd.read_csv(metrics_path).to_dict(orient="records"):
            recorded[str(row["fold_key"])] = row

    for seed in seeds:
        for fold_id in range(n_splits):
            key = _fold_key(seed, fold_id)
            try:
                valid_index, valid_pred, test_pred = _load_fold_predictions(
                    pred_dir / f"{key}.npz",
                    ids=ids,
                    test_ids=test_ids,
                    label=key,
                )
            except AlignmentError as exc:
                raise TrainingError(str(exc)) from exc

            oof_sum[valid_index] += valid_pred
            oof_count[valid_index] += 1.0
            seed_oof[seed][valid_index] = valid_pred
            test_sum += test_pred
            test_count += 1

            if key in recorded:
                fold_rows.append(recorded[key])
            else:
                fold_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_id,
                        "fold_key": key,
                        "auc": float(summarize_oof(y[valid_index], valid_pred).auc),
                    }
                )

    if np.any(oof_count == 0):
        raise TrainingError("OOF coverage incomplete after aggregation")

    seed_aucs = {seed: float(summarize_oof(y, seed_oof[seed]).auc) for seed in seeds}
    oof_stack = np.vstack([seed_oof[seed] for seed in seeds])
    oof_mean = np.nanmean(oof_stack, axis=0)
    test_mean = test_sum / float(test_count)
    overall = summarize_oof(y, oof_mean)

    slice_metrics = compute_slice_metrics(
        train_features,
        y,
        oof_mean,
        test_features=test_features,
    )
    store.write_json("slice_metrics.json", slice_metrics)

    metrics: dict[str, Any] = {
        "oof_auc": overall.auc,
        "oof_coverage": overall.coverage,
        "oof_pred_min": overall.min,
        "oof_pred_max": overall.max,
        "oof_pred_mean": overall.mean,
        "oof_pred_std": overall.std,
        "core_complete_auc": slice_metrics.get("core_complete_auc"),
        "core_incomplete_auc": slice_metrics.get("core_incomplete_auc"),
        "top3_incomplete_auc": slice_metrics.get("top3_incomplete_auc"),
        "test_pattern_weighted_auc": slice_metrics.get("test_pattern_weighted_auc"),
        "n_core_complete": slice_metrics.get("n_core_complete"),
        "n_core_incomplete": slice_metrics.get("n_core_incomplete"),
        "seed_aucs": {str(seed): seed_aucs[seed] for seed in seeds},
        "seed_auc_mean": float(np.mean(list(seed_aucs.values()))),
        "seed_auc_std": float(np.std(list(seed_aucs.values()), ddof=0)),
        "n_splits": n_splits,
        "seeds": seeds,
        "model_name": model_name,
        "n_train_rows": len(ids),
        "n_test_rows": len(test_ids),
        "n_features": len(feature_cols),
    }

    store.write_frame(
        "oof_predictions.parquet",
        pd.DataFrame(
            {
                ID_COLUMN: ids,
                TARGET_COLUMN: y,
                "prediction": oof_mean,
            }
        ),
    )
    store.write_frame(
        "test_predictions.parquet",
        pd.DataFrame({ID_COLUMN: test_ids, "prediction": test_mean}),
    )
    store.write_frame("fold_metrics.csv", pd.DataFrame(fold_rows))
    return metrics


def _resolve_frames(
    *,
    frames: CompetitionFrames | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    train: pd.DataFrame | None,
    test: pd.DataFrame | None,
    feature_columns: list[str] | None,
    categorical_columns: list[str] | None,
    feature_groups: list[str] | None,
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if frames is not None:
        if isinstance(frames, CompetitionFrames):
            raw_train, raw_test = frames.train, frames.test
        else:
            raw_train, raw_test, _ = frames
        transformed = transform_competition_frames(
            raw_train,
            raw_test,
            groups=feature_groups,
        )
        feature_columns = list(transformed.feature_columns)
        categorical_columns = list(transformed.categorical_columns)
        try:
            feature_columns = exclude_feature_columns(feature_columns, exclude_columns)
        except DataValidationError as exc:
            raise TrainingError(str(exc)) from exc
        categorical_columns = [
            column for column in categorical_columns if column in feature_columns
        ]
        return (
            transformed.train,
            transformed.test,
            feature_columns,
            categorical_columns,
        )

    if train is None or test is None:
        raise TrainingError("provide either frames=... or both train= and test=")

    if feature_columns is None:
        feature_columns = [
            column for column in train.columns if column not in {ID_COLUMN, TARGET_COLUMN}
        ]
    if feature_groups is not None:
        try:
            feature_columns = select_feature_columns_from_groups(
                feature_columns,
                feature_groups,
                require_all=True,
            )
        except DataValidationError as exc:
            raise TrainingError(str(exc)) from exc
        if not feature_columns:
            raise TrainingError("feature_groups selected zero columns from the processed frame")
    try:
        feature_columns = exclude_feature_columns(feature_columns, exclude_columns)
    except DataValidationError as exc:
        raise TrainingError(str(exc)) from exc
    if categorical_columns is None:
        categorical_columns = [
            column
            for column in feature_columns
            if (
                train[column].dtype == object
                or str(train[column].dtype) == "string"
                or isinstance(train[column].dtype, pd.CategoricalDtype)
            )
        ]
    else:
        categorical_columns = [
            column for column in categorical_columns if column in feature_columns
        ]
    missing = [column for column in feature_columns if column not in train.columns]
    if missing:
        raise TrainingError(f"train missing feature columns: {missing}")
    if TARGET_COLUMN not in train.columns:
        raise TrainingError("processed train must contain addicted_label")
    if ID_COLUMN not in train.columns or ID_COLUMN not in test.columns:
        raise TrainingError("train/test must contain id")
    return train, test, list(feature_columns), list(categorical_columns)


def _build_model(
    model_name: str,
    categorical_columns: list[str],
    model_params: dict[str, Any],
    *,
    seed: int,
) -> Any:
    """Build a fresh adapter and inject the current CV seed into model RNG."""
    params = dict(model_params)
    if model_name == "catboost":
        params["random_seed"] = seed
        return build_catboost(categorical_columns=categorical_columns, **params)
    if model_name == "lightgbm":
        params["random_state"] = seed
        return build_lightgbm(categorical_columns=categorical_columns, **params)
    raise TrainingError(f"unsupported model_name={model_name!r}")


def _fold_key(seed: int, fold_id: int) -> str:
    return f"seed{seed}-fold{fold_id}"


def _fingerprint_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(columns).encode("utf-8"))
    for column in columns:
        series = frame[column]
        digest.update(column.encode("utf-8"))
        if pd.api.types.is_numeric_dtype(series):
            values = np.ascontiguousarray(series.to_numpy(dtype=np.float64, copy=False))
            digest.update(values.tobytes())
            digest.update(np.isnan(values).tobytes())
        else:
            as_text = series.astype("string").fillna("__NULL__").astype(str).to_numpy()
            digest.update("\0".join(as_text).encode("utf-8"))
    return digest.hexdigest()


def _setup_run_logging(run_dir: Path) -> None:
    log_path = run_dir / "training.log"
    root = logging.getLogger("smartphone_addiction.training.runner")
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
