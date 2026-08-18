"""Five-fold core-field reconstruction runner and artifact writer."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from smartphone_addiction.artifacts.manifest import build_run_id
from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.data.schema import ID_COLUMN, TARGET_COLUMN
from smartphone_addiction.errors import ArtifactError, TrainingError
from smartphone_addiction.evaluation.reconstruction import (
    GateDecision,
    compute_field_metrics,
    evaluate_reconstruction_gate,
)
from smartphone_addiction.git_info import git_is_dirty, git_sha
from smartphone_addiction.neural.autoencoder import build_mlp_autoencoder
from smartphone_addiction.neural.config import (
    CORE5_FIELDS,
    NeuralReconstructionConfig,
    load_neural_config,
)
from smartphone_addiction.neural.device import (
    environment_info,
    require_torch,
    resolve_device,
    seed_everything,
)
from smartphone_addiction.neural.masking import (
    ValidationMaskBank,
    build_fixed_validation_mask_bank,
    move_mask_batch,
    pattern_distribution_from_test,
    subset_mask_batch,
)
from smartphone_addiction.neural.preprocessing import FoldTensorizer, TensorizedFrame
from smartphone_addiction.neural.trainer import train_reconstruction_model
from smartphone_addiction.training.cv import make_folds

SMOKE_ROWS = 20_000
SMOKE_EPOCHS = 2


def _utc_stamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    import hashlib

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


def _holdout_split(n_rows: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_holdout = max(1, round(n_rows * fraction))
    n_holdout = min(n_holdout, n_rows - 1)
    return np.sort(order[n_holdout:]), np.sort(order[:n_holdout])


def _subset_frame(frame: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
    return frame.iloc[index].reset_index(drop=True)


def _subset_tensorized(frame: TensorizedFrame, index: np.ndarray) -> TensorizedFrame:
    return TensorizedFrame(
        numeric=frame.numeric[index],
        categorical=frame.categorical[index],
        natural_observed=frame.natural_observed[index],
        row_ids=np.asarray(frame.row_ids)[index],
        core_raw=frame.core_raw[index],
        core_observed=frame.core_observed[index],
    )


def _build_model(config: NeuralReconstructionConfig, vocab_sizes: list[int]):
    return build_mlp_autoencoder(vocab_sizes, config.model)


def _predict_bank(
    model,
    bank: ValidationMaskBank,
    *,
    device: object,
    batch_size: int,
    tensorizer: FoldTensorizer,
) -> pd.DataFrame:
    torch = require_torch()
    model.eval()
    n_rows = int(bank.batch.loss_mask.shape[0])
    pred_std: list[np.ndarray] = []
    member_std: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            index = np.arange(start, min(start + batch_size, n_rows))
            mini = move_mask_batch(subset_mask_batch(bank.batch, index), device)
            output = model(mini)
            pred_std.append(output.mean_prediction.detach().cpu().numpy())
            if output.member_std is None:
                member_std.append(
                    np.full((len(index), len(CORE5_FIELDS)), np.nan, dtype=np.float64)
                )
            else:
                raw_std = tensorizer.inverse_core(
                    output.mean_prediction.detach().cpu().numpy()
                    + output.member_std.detach().cpu().numpy()
                ) - tensorizer.inverse_core(output.mean_prediction.detach().cpu().numpy())
                member_std.append(np.abs(raw_std))
    pred_raw = tensorizer.inverse_core(np.concatenate(pred_std, axis=0))
    std_raw = np.concatenate(member_std, axis=0)
    y_true_raw = tensorizer.inverse_core(bank.batch.targets.cpu().numpy())
    loss_mask = bank.batch.loss_mask.cpu().numpy()
    artificial = bank.batch.artificial_mask.cpu().numpy()
    rows: list[dict[str, Any]] = []
    for row_i in range(n_rows):
        pattern = "".join("0" if flag else "1" for flag in artificial[row_i].tolist())
        for field_i, field in enumerate(CORE5_FIELDS):
            if not bool(loss_mask[row_i, field_i]):
                continue
            member = std_raw[row_i, field_i]
            rows.append(
                {
                    ID_COLUMN: bank.batch.row_ids[row_i],
                    "repeat": int(bank.repeats[row_i]),
                    "field": field,
                    "y_true": float(y_true_raw[row_i, field_i]),
                    "y_pred": float(pred_raw[row_i, field_i]),
                    "median_baseline": float(tensorizer.core_median_[field_i]),
                    "mask_pattern": pattern,
                    "member_std": None if not np.isfinite(member) else float(member),
                }
            )
    return pd.DataFrame(rows)


def _write_summary_md(path: Path, decision: GateDecision, field_metrics: pd.DataFrame) -> None:
    oof = field_metrics.loc[field_metrics["fold"] == "oof"]
    lines = [
        "# Core-field reconstruction summary",
        "",
        f"- gate passed: **{'yes' if decision.passed else 'no'}**",
        f"- passing fields: {', '.join(decision.passing_fields) or '(none)'}",
        f"- top3 passing: {', '.join(decision.top3_passing) or '(none)'}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision.reasons)
    lines.extend(["", "## OOF field metrics", ""])
    for _, row in oof.iterrows():
        lines.append(
            f"- `{row['field']}`: R2={row['r2']:.4f}, Spearman={row['spearman']:.4f}, "
            f"RMSE improve={row['rmse_improvement']:.4f}, n={int(row['n_eval'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reconstruction_cv(
    config: NeuralReconstructionConfig,
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
    min_eval = 1 if smoke else config.masking.min_eval_per_field
    train = _select_rows(frames.train, sample_rows, config.cv.seed)
    n_splits = 2 if smoke else config.cv.n_splits
    fold_ids = make_folds(train[TARGET_COLUMN].to_numpy(), n_splits=n_splits, seed=config.cv.seed)
    distribution = pattern_distribution_from_test(
        frames.test,
        min_missing=config.masking.min_fields,
        max_missing=config.masking.max_fields,
    )

    slug = f"{config.model.name}-mae-smoke" if smoke else f"{config.model.name}-mae"
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
    (run_dir / "mask_distribution.json").write_text(
        json.dumps(
            {
                "patterns": list(distribution.patterns),
                "probabilities": distribution.probabilities.tolist(),
                "counts": distribution.counts,
                "n_test_rows": distribution.n_test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    oof_parts: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    actual_batch = config.training.batch_size
    folds_to_run = [only_fold] if only_fold is not None else list(range(n_splits))
    if smoke and only_fold is None:
        folds_to_run = [0]

    for fold in folds_to_run:
        checkpoint = run_dir / "checkpoints" / f"fold_{fold}.pt"
        valid_idx = np.flatnonzero(fold_ids == fold)
        train_idx = np.flatnonzero(fold_ids != fold)
        outer_train = _subset_frame(train, train_idx)
        outer_valid = _subset_frame(train, valid_idx)
        tensorizer = FoldTensorizer().fit(outer_train)
        train_tensor = tensorizer.transform(outer_train)
        valid_tensor = tensorizer.transform(outer_valid)
        fit_idx, hold_idx = _holdout_split(
            len(outer_train),
            config.training.holdout_fraction,
            config.training.seed + fold,
        )
        fit_tensor = _subset_tensorized(train_tensor, fit_idx)
        hold_tensor = _subset_tensorized(train_tensor, hold_idx)
        hold_bank = build_fixed_validation_mask_bank(
            hold_tensor,
            tensorizer,
            seed=config.training.seed + 1000 + fold,
            repeats=config.masking.valid_repeats,
            pattern_distribution=distribution,
            min_fields=config.masking.min_fields,
            max_fields=config.masking.max_fields,
            field_balance_prob=config.masking.field_balance_prob,
            min_eval_per_field=min_eval,
        )
        valid_bank = build_fixed_validation_mask_bank(
            valid_tensor,
            tensorizer,
            seed=config.training.seed + 2000 + fold,
            repeats=config.masking.valid_repeats,
            pattern_distribution=distribution,
            min_fields=config.masking.min_fields,
            max_fields=config.masking.max_fields,
            field_balance_prob=0.0,
            min_eval_per_field=min_eval,
        )
        model = _build_model(config, tensorizer.vocab_sizes())
        result = train_reconstruction_model(
            model=model,
            train_frame=fit_tensor,
            tensorizer=tensorizer,
            holdout_bank=hold_bank,
            pattern_distribution=distribution,
            device=device,
            batch_size=config.training.batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            huber_delta=config.training.huber_delta,
            clip_norm=config.training.gradient_clip_norm,
            seed=config.training.seed + fold,
            min_fields=config.masking.min_fields,
            max_fields=config.masking.max_fields,
            field_balance_prob=config.masking.field_balance_prob,
            checkpoint_path=checkpoint,
        )
        actual_batch = result.batch_size
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        payload["model_name"] = config.model.name
        payload["fold"] = fold
        torch.save(payload, checkpoint)
        pred = _predict_bank(
            model,
            valid_bank,
            device=device,
            batch_size=result.batch_size,
            tensorizer=tensorizer,
        )
        pred["fold"] = fold
        oof_parts.append(pred)
        for row in result.history:
            history_rows.append({"fold": fold, **row})
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(fit_idx),
                "n_holdout": len(hold_idx),
                "n_valid": len(valid_idx),
                "best_epoch": result.best_epoch,
                "best_holdout_loss": result.best_holdout_loss,
                "batch_size": result.batch_size,
                "n_eval": len(pred),
            }
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_parquet(run_dir / "reconstruction_oof.parquet", index=False)
    field_metrics = compute_field_metrics(oof, n_splits=n_splits)
    field_metrics.to_csv(run_dir / "field_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
    decision = evaluate_reconstruction_gate(
        field_metrics,
        config.gate,
        n_splits=n_splits if not smoke else max(len(folds_to_run), 2),
    )
    gate_payload = {
        "passed": decision.passed,
        "smoke": smoke,
        "passing_fields": list(decision.passing_fields),
        "top3_passing": list(decision.top3_passing),
        "n_passing_fields": decision.n_passing_fields,
        "n_top3_passing": decision.n_top3_passing,
        "reasons": list(decision.reasons),
        "field_summary": decision.field_summary,
        "used_for_official_gate": not smoke,
    }
    (run_dir / "gate_decision.json").write_text(
        json.dumps(gate_payload, indent=2), encoding="utf-8"
    )
    _write_summary_md(run_dir / "summary.md", decision, field_metrics)
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


def run_reconstruction_from_yaml(
    config_paths: list[Path],
    *,
    model_name: str | None = None,
    smoke: bool = False,
    device: str | None = None,
) -> Path:
    config = load_neural_config(config_paths, resolve=True)
    payload = config.model_dump()
    if model_name is not None:
        payload["model"]["name"] = model_name
    if device is not None:
        payload["device"] = device
    config = NeuralReconstructionConfig.model_validate(payload)
    return run_reconstruction_cv(config, smoke=smoke)
