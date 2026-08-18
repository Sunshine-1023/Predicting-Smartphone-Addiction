"""Five-fold Lookup Transformer classification runner and artifact writer."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from smartphone_addiction.alignment import align_predictions_to_ids, assert_unique_ids
from smartphone_addiction.artifacts.manifest import build_run_id
from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import ArtifactError, TrainingError
from smartphone_addiction.evaluation.metrics import summarize_oof
from smartphone_addiction.evaluation.slices import compute_slice_metrics
from smartphone_addiction.git_info import git_is_dirty, git_sha
from smartphone_addiction.neural.classification_config import (
    ClassificationGateConfig,
    NeuralClassificationConfig,
    load_classification_config,
)
from smartphone_addiction.neural.classification_features import (
    FoldClassificationEncoder,
    subset_encoded_table,
)
from smartphone_addiction.neural.classification_trainer import (
    fit_classification_fixed_epochs,
    predict_classifier,
    train_classifier,
)
from smartphone_addiction.neural.device import (
    environment_info,
    require_torch,
    resolve_device,
    seed_everything,
)
from smartphone_addiction.neural.lookup_transformer import build_lookup_transformer
from smartphone_addiction.training.cv import make_folds

SMOKE_ROWS = 20_000
SMOKE_EPOCHS = 2


@dataclass(frozen=True)
class ClassificationGateDecision:
    passed: bool
    reasons: list[str]
    used_for_official_gate: bool


def _utc_stamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_rows(train: pd.DataFrame, sample_rows: int | None, seed: int) -> pd.DataFrame:
    if sample_rows is None or sample_rows >= len(train):
        return train.reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_rows, random_state=seed)
    idx, _ = next(splitter.split(train, train[TARGET_COLUMN]))
    return train.iloc[idx].reset_index(drop=True)


def _subset_frame(frame: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
    return frame.iloc[index].reset_index(drop=True)


def _build_classification_model(config: NeuralClassificationConfig, cardinalities: list[int]):
    return build_lookup_transformer(cardinalities, config.model)


def stratified_holdout(
    labels: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split outer-train into fit/holdout indices for early stopping."""
    y = np.asarray(labels)
    n_rows = len(y)
    if n_rows < 4:
        raise TrainingError("need at least 4 rows for a classification holdout split")
    n_holdout = max(1, round(n_rows * fraction))
    n_holdout = min(n_holdout, n_rows - 2)
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=n_holdout,
            random_state=seed,
        )
        fit_idx, hold_idx = next(splitter.split(np.zeros((n_rows, 1)), y))
    except ValueError:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_rows)
        hold_idx = order[:n_holdout]
        fit_idx = order[n_holdout:]
    return np.sort(fit_idx), np.sort(hold_idx)


def evaluate_classification_gate(
    metrics: dict[str, Any],
    gate: ClassificationGateConfig,
    *,
    smoke: bool,
    n_splits: int,
    probe: bool = False,
) -> ClassificationGateDecision:
    reasons: list[str] = []
    if smoke:
        return ClassificationGateDecision(
            passed=False,
            reasons=["smoke runs are not used for the official gate"],
            used_for_official_gate=False,
        )
    auc = float(metrics.get("oof_auc", 0.0))
    if probe:
        if auc + 1e-12 < gate.probe_auc_min:
            reasons.append(f"probe AUC {auc:.6f} is below the minimum {gate.probe_auc_min:.6f}")
        return ClassificationGateDecision(
            passed=not reasons,
            reasons=reasons or ["one-fold probe passed; full 5-fold gate is still required"],
            used_for_official_gate=False,
        )
    coverage = float(metrics.get("oof_coverage", 0.0))
    if coverage + 1e-12 < gate.coverage_min:
        reasons.append(f"OOF coverage {coverage:.6f} is below the minimum {gate.coverage_min:.6f}")
    if auc + 1e-12 < gate.oof_auc_min:
        reasons.append(f"OOF AUC {auc:.6f} is below the minimum {gate.oof_auc_min:.6f}")
    if n_splits < gate.min_folds:
        reasons.append(f"completed {n_splits} folds but the gate requires {gate.min_folds}")
    incomplete = metrics.get("core_incomplete_auc")
    if gate.incomplete_auc_min is not None:
        if incomplete is None:
            reasons.append("core_incomplete_auc is missing")
        elif float(incomplete) + 1e-12 < gate.incomplete_auc_min:
            reasons.append(
                "core_incomplete_auc "
                f"{float(incomplete):.6f} is below the minimum {gate.incomplete_auc_min:.6f}"
            )
    return ClassificationGateDecision(
        passed=not reasons,
        reasons=reasons or ["all official classification gates passed"],
        used_for_official_gate=True,
    )


def _write_summary_md(
    path: Path,
    decision: ClassificationGateDecision,
    metrics: dict[str, Any],
) -> None:
    lines = [
        "# Lookup Transformer classification summary",
        "",
        f"- gate passed: **{'yes' if decision.passed else 'no'}**",
        f"- OOF AUC: {metrics.get('oof_auc')}",
        f"- incomplete AUC: {metrics.get('core_incomplete_auc')}",
        f"- coverage: {metrics.get('oof_coverage')}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision.reasons)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_classification_cv(
    config: NeuralClassificationConfig,
    *,
    smoke: bool = False,
    only_fold: int | None = None,
) -> Path:
    torch = require_torch()
    seed_everything(config.training.seed, torch_module=torch)
    device = resolve_device(config.device, torch_module=torch)
    frames = load_competition_frames(Path(config.data.directory))
    sample_rows = SMOKE_ROWS if smoke else config.data.sample_rows
    max_epochs = SMOKE_EPOCHS if smoke else config.training.max_epochs
    patience = 2 if smoke else config.training.early_stopping_patience
    train = _select_rows(frames.train, sample_rows, config.cv.seed)
    test = frames.test.reset_index(drop=True)
    n_splits = 2 if smoke else config.cv.n_splits
    labels = train[TARGET_COLUMN].to_numpy()
    fold_ids = make_folds(labels, n_splits=n_splits, seed=config.cv.seed)
    ids = assert_unique_ids(train[ID_COLUMN], label="train id")
    test_ids = assert_unique_ids(test[ID_COLUMN], label="test id")

    slug = "lookup-transformer-smoke" if smoke else "lookup-transformer"
    if only_fold is not None and not smoke:
        slug = f"{slug}-fold{only_fold}"
    run_id = build_run_id(slug, git_sha())
    run_dir = Path(config.artifacts.directory) / run_id
    if run_dir.exists():
        raise ArtifactError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()

    resolved = config.model_dump()
    resolved["smoke"] = smoke
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
    fingerprints = {
        "train.csv": _sha256_file(Path(config.data.directory) / "train.csv"),
        "test.csv": _sha256_file(Path(config.data.directory) / "test.csv"),
        "n_train_used": len(train),
        "smoke": smoke,
    }
    (run_dir / "data_fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], "fold": fold_ids}).to_parquet(
        run_dir / "fold_assignments.parquet",
        index=False,
    )

    oof = np.full(len(train), np.nan, dtype=np.float64)
    test_sum = np.zeros(len(test), dtype=np.float64)
    test_count = 0
    history_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    actual_batch = config.training.batch_size
    folds_to_run = [only_fold] if only_fold is not None else list(range(n_splits))
    if smoke and only_fold is None:
        folds_to_run = [0]

    extra_vocab = test if config.encoding.include_unlabeled_vocab else None
    for fold in folds_to_run:
        print(
            f"[{config.model.name}] fold {fold} start "
            f"n_train={int((fold_ids != fold).sum())} device={device}",
            flush=True,
        )
        checkpoint = run_dir / "checkpoints" / f"fold_{fold}.pt"
        valid_idx = np.flatnonzero(fold_ids == fold)
        train_idx = np.flatnonzero(fold_ids != fold)
        outer_train = _subset_frame(train, train_idx)
        outer_valid = _subset_frame(train, valid_idx)
        fit_idx, hold_idx = stratified_holdout(
            outer_train[TARGET_COLUMN].to_numpy(),
            config.training.holdout_fraction,
            config.training.seed + fold,
        )
        print(f"[{config.model.name}] encoding exact-value tokens", flush=True)
        encoder = FoldClassificationEncoder().fit(
            outer_train,
            extra_vocab_frame=extra_vocab,
        )
        outer_table = encoder.transform(
            outer_train,
            outer_train[TARGET_COLUMN].to_numpy(),
        )
        train_table = subset_encoded_table(outer_table, fit_idx)
        hold_table = subset_encoded_table(outer_table, hold_idx)
        valid_table = encoder.transform_eval(outer_valid)
        test_table = encoder.transform_eval(test)
        cardinalities = encoder.cardinalities()
        model = _build_classification_model(config, cardinalities)
        print(
            f"[{config.model.name}] selecting epoch max={max_epochs} "
            f"batch={config.training.batch_size}",
            flush=True,
        )
        result = train_classifier(
            model=model,
            train_table=train_table,
            holdout_table=hold_table,
            device=device,
            batch_size=config.training.batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            clip_norm=config.training.gradient_clip_norm,
            seed=config.training.seed + fold,
            checkpoint_path=None,
            encoder_state=encoder.to_state(),
            show_progress=True,
        )
        actual_batch = result.batch_size
        for row in result.history:
            history_rows.append({"fold": fold, "phase": "selection", **row})
        print(
            f"[lookup_transformer] refitting full outer-train epochs={result.best_epoch}",
            flush=True,
        )
        model = _build_classification_model(config, cardinalities)
        refit = fit_classification_fixed_epochs(
            model=model,
            train_table=outer_table,
            device=device,
            batch_size=config.training.batch_size,
            epochs=result.best_epoch,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            clip_norm=config.training.gradient_clip_norm,
            seed=config.training.seed + 10_000 + fold,
            checkpoint_path=checkpoint,
            encoder_state=encoder.to_state(),
            show_progress=True,
        )
        actual_batch = refit.batch_size
        for row in refit.history:
            history_rows.append({"fold": fold, "phase": "full_refit", **row})
        valid_pred = predict_classifier(
            model,
            valid_table,
            device=device,
            batch_size=actual_batch,
        )
        test_pred = predict_classifier(
            model,
            test_table,
            device=device,
            batch_size=actual_batch,
        )
        oof[valid_idx] = valid_pred
        aligned_test = align_predictions_to_ids(
            test_ids,
            test_table.row_ids,
            test_pred,
            label=f"fold {fold} test",
        )
        test_sum += aligned_test
        test_count += 1
        fold_auc = float(summarize_oof(outer_valid[TARGET_COLUMN].to_numpy(), valid_pred).auc)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(fit_idx),
                "n_holdout": len(hold_idx),
                "n_valid": len(valid_idx),
                "best_epoch": result.best_epoch,
                "best_holdout_auc": result.best_holdout_auc,
                "best_holdout_loss": result.best_holdout_loss,
                "batch_size": result.batch_size,
                "auc": fold_auc,
            }
        )
        print(
            f"[{config.model.name}] fold {fold} done auc={fold_auc:.6f} "
            f"best_epoch={result.best_epoch} holdout_auc={result.best_holdout_auc:.6f}",
            flush=True,
        )
        if only_fold is not None:
            probe_passed = fold_auc + 1e-12 >= config.gate.probe_auc_min
            print(
                f"[{config.model.name}] probe_auc={fold_auc:.6f} "
                f"threshold={config.gate.probe_auc_min:.6f} passed={probe_passed}",
                flush=True,
            )

    if test_count == 0:
        raise TrainingError("classification runner produced no test folds")
    test_mean = test_sum / float(test_count)
    partial = smoke or only_fold is not None
    finite = np.isfinite(oof)
    if partial:
        covered = train.loc[np.asarray(finite)].reset_index(drop=True)
        oof_pred = oof[finite]
        overall = summarize_oof(covered[TARGET_COLUMN].to_numpy(), oof_pred)
        slice_metrics = compute_slice_metrics(
            covered,
            covered[TARGET_COLUMN].to_numpy(),
            oof_pred,
            test_features=test,
        )
        oof_ids = covered[ID_COLUMN].to_numpy()
        oof_labels = covered[TARGET_COLUMN].to_numpy()
    else:
        if not bool(finite.all()):
            raise TrainingError("OOF coverage incomplete after classification aggregation")
        overall = summarize_oof(labels, oof)
        slice_metrics = compute_slice_metrics(
            train,
            labels,
            oof,
            test_features=test,
        )
        oof_ids = ids.to_numpy()
        oof_labels = labels
        oof_pred = oof

    metrics: dict[str, Any] = {
        "oof_auc": overall.auc,
        "oof_coverage": float(finite.mean()) if partial else overall.coverage,
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
        "n_splits": n_splits,
        "folds_run": folds_to_run,
        "seeds": [config.training.seed],
        "model_name": "lookup_transformer",
        "n_train_rows": len(ids),
        "n_test_rows": len(test_ids),
        "smoke": smoke,
        "only_fold": only_fold,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "slice_metrics.json").write_text(
        json.dumps(slice_metrics, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            ID_COLUMN: oof_ids,
            TARGET_COLUMN: oof_labels,
            "prediction": oof_pred,
        }
    ).to_parquet(run_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame({ID_COLUMN: test_ids.to_numpy(), "prediction": test_mean}).to_parquet(
        run_dir / "test_predictions.parquet",
        index=False,
    )
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
    decision = evaluate_classification_gate(
        metrics,
        config.gate,
        smoke=smoke,
        n_splits=len(folds_to_run) if partial else n_splits,
        probe=only_fold is not None and not smoke,
    )
    gate_payload = {
        "passed": decision.passed,
        "smoke": smoke,
        "only_fold": only_fold,
        "probe": only_fold is not None and not smoke,
        "probe_auc_min": config.gate.probe_auc_min,
        "reasons": list(decision.reasons),
        "used_for_official_gate": decision.used_for_official_gate,
        "oof_auc": metrics["oof_auc"],
        "oof_auc_min": config.gate.oof_auc_min,
    }
    (run_dir / "gate_decision.json").write_text(
        json.dumps(gate_payload, indent=2), encoding="utf-8"
    )
    _write_summary_md(run_dir / "summary.md", decision, metrics)
    env = environment_info(
        device_mode=config.device,
        device=device,
        dtype=config.training.dtype,
        batch_size=actual_batch,
        torch_module=torch,
    )
    env.update(
        {
            "created_at": _utc_stamp(),
            "git_sha": git_sha(),
            "git_dirty": git_is_dirty(),
            "macos": platform.mac_ver()[0] or None,
            "smoke": smoke,
        }
    )
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return run_dir


def run_classification_from_yaml(
    config_paths: list[Path],
    *,
    smoke: bool = False,
    device: str | None = None,
    only_fold: int | None = None,
) -> Path:
    config = load_classification_config(config_paths, resolve=True)
    payload = config.model_dump()
    if device is not None:
        payload["device"] = device
    config = NeuralClassificationConfig.model_validate(payload)
    return run_classification_cv(config, smoke=smoke, only_fold=only_fold)
