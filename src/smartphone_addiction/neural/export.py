"""Fold-local latent export. Only used after the reconstruction gate passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.errors import ArtifactError, TrainingError
from smartphone_addiction.features.latent import TEST_LATENT_NAME, TRAIN_LATENT_NAME
from smartphone_addiction.neural.autoencoder import build_mlp_autoencoder
from smartphone_addiction.neural.config import CORE5_FIELDS, NeuralModelArchConfig
from smartphone_addiction.neural.device import require_torch, resolve_device
from smartphone_addiction.neural.masking import MaskBatch, move_mask_batch
from smartphone_addiction.neural.preprocessing import FoldTensorizer, TensorizedFrame
from smartphone_addiction.neural.tabm import build_tabm_autoencoder


def unmasked_batch(frame: TensorizedFrame, tensorizer: FoldTensorizer) -> MaskBatch:
    torch = require_torch()
    n_rows = int(frame.numeric.shape[0])
    n_core = len(CORE5_FIELDS)
    core_std = np.nan_to_num(
        (frame.core_raw - tensorizer.core_mean_) / tensorizer.core_std_,
        nan=0.0,
    ).astype(np.float32)
    return MaskBatch(
        masked_numeric=frame.numeric,
        natural_observed=frame.natural_observed,
        artificial_mask=torch.zeros((n_rows, n_core), dtype=torch.bool),
        categorical=frame.categorical,
        targets=torch.as_tensor(core_std, dtype=torch.float32),
        loss_mask=torch.zeros((n_rows, n_core), dtype=torch.bool),
        row_ids=np.asarray(frame.row_ids),
    )


def encode_latent(
    model,
    frame: TensorizedFrame,
    tensorizer: FoldTensorizer,
    *,
    device: object,
    batch_size: int,
) -> pd.DataFrame:
    torch = require_torch()
    model.eval()
    batch = unmasked_batch(frame, tensorizer)
    n_rows = int(frame.numeric.shape[0])
    latents: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    member_stds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            index = np.arange(start, min(start + batch_size, n_rows))
            mini = move_mask_batch(
                MaskBatch(
                    masked_numeric=batch.masked_numeric[index],
                    natural_observed=batch.natural_observed[index],
                    artificial_mask=batch.artificial_mask[index],
                    categorical=batch.categorical[index],
                    targets=batch.targets[index],
                    loss_mask=batch.loss_mask[index],
                    row_ids=np.asarray(batch.row_ids)[index],
                ),
                device,
            )
            output = model(mini)
            latents.append(output.mean_latent.detach().cpu().numpy())
            mean_pred = output.mean_prediction.detach().cpu().numpy()
            reconstructions.append(tensorizer.inverse_core(mean_pred))
            if output.member_std is None:
                member_stds.append(np.full((len(index), len(CORE5_FIELDS)), np.nan))
            else:
                high = tensorizer.inverse_core(mean_pred + output.member_std.detach().cpu().numpy())
                low = tensorizer.inverse_core(mean_pred)
                member_stds.append(np.abs(high - low))
    latent = np.concatenate(latents, axis=0)
    recon = np.concatenate(reconstructions, axis=0)
    std = np.concatenate(member_stds, axis=0)
    payload: dict[str, object] = {ID_COLUMN: np.asarray(frame.row_ids)}
    for dim in range(latent.shape[1]):
        payload[f"latent_{dim:02d}"] = latent[:, dim]
    for field_i, field in enumerate(CORE5_FIELDS):
        payload[f"recon_{field}"] = recon[:, field_i]
    if np.isfinite(std).any():
        for field_i, field in enumerate(CORE5_FIELDS):
            payload[f"recon_std_{field}"] = std[:, field_i]
    return pd.DataFrame(payload)


def write_latent_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_model(model_name: str, vocab_sizes: list[int], arch: NeuralModelArchConfig):
    if model_name == "mlp":
        return build_mlp_autoencoder(vocab_sizes, arch)
    if model_name == "tabm":
        return build_tabm_autoencoder(vocab_sizes, arch)
    raise TrainingError(f"unsupported reconstruction model for latent export: {model_name}")


def export_latents_from_run(
    run_dir: Path | str,
    *,
    device: str | None = None,
    batch_size: int | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Export fold-local OOF train latent and mean test latent from a gated run."""
    torch = require_torch()
    run_dir = Path(run_dir)
    gate_path = run_dir / "gate_decision.json"
    if not gate_path.is_file():
        raise ArtifactError(f"missing gate_decision.json under {run_dir}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("invalid"):
        raise ArtifactError(
            f"refusing latent export from invalid run: {gate.get('invalid_reason')}"
        )
    if not gate.get("passed"):
        raise ArtifactError("refusing latent export: reconstruction gate did not pass")
    if gate.get("smoke"):
        raise ArtifactError("refusing latent export from a smoke run")

    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise ArtifactError(f"missing config_resolved.yaml under {run_dir}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_name = str(resolved.get("model", {}).get("name", "mlp"))
    arch = NeuralModelArchConfig.model_validate(resolved.get("model", {}))
    data_dir = Path(resolved["data"]["directory"])
    training = resolved.get("training", {})
    actual_batch = int(batch_size or training.get("batch_size") or 4096)
    device_obj = resolve_device(device or resolved.get("device") or "auto", torch_module=torch)

    folds_path = run_dir / "fold_assignments.parquet"
    if not folds_path.is_file():
        raise ArtifactError(f"missing fold_assignments.parquet under {run_dir}")
    fold_frame = pd.read_parquet(folds_path)
    if list(fold_frame.columns) != [ID_COLUMN, "fold"]:
        raise ArtifactError("fold_assignments.parquet must contain columns [id, fold]")

    frames = load_competition_frames(data_dir)
    train = frames.train.merge(fold_frame, on=ID_COLUMN, how="left", validate="1:1")
    if train["fold"].isna().any():
        raise TrainingError("fold assignments do not cover the full train set")
    train["fold"] = train["fold"].astype(int)

    out_dir = Path(output_dir) if output_dir is not None else run_dir / "latent"
    if out_dir.exists():
        raise ArtifactError(f"latent output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "per_fold").mkdir()

    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    fold_meta: list[dict[str, Any]] = []
    n_splits = int(train["fold"].nunique())

    for fold in range(n_splits):
        checkpoint = run_dir / "checkpoints" / f"fold_{fold}.pt"
        if not checkpoint.is_file():
            raise ArtifactError(f"missing checkpoint: {checkpoint}")
        payload = _load_checkpoint(checkpoint)
        tensorizer = FoldTensorizer.from_state(payload["tensorizer"])
        model = _build_model(model_name, tensorizer.vocab_sizes(), arch)
        model.load_state_dict(payload["model_state"])
        model.to(device_obj)
        model.eval()

        valid_frame = train.loc[train["fold"] == fold].reset_index(drop=True)
        valid_tensor = tensorizer.transform(valid_frame)
        valid_latent = encode_latent(
            model,
            valid_tensor,
            tensorizer,
            device=device_obj,
            batch_size=actual_batch,
        )
        valid_latent["encoder_fold"] = fold
        oof_parts.append(valid_latent)

        test_tensor = tensorizer.transform(frames.test)
        test_latent = encode_latent(
            model,
            test_tensor,
            tensorizer,
            device=device_obj,
            batch_size=actual_batch,
        )
        test_latent["encoder_fold"] = fold
        test_parts.append(test_latent)
        test_latent.to_parquet(
            out_dir / "per_fold" / f"test_fold_{fold}_latent.parquet", index=False
        )

        fold_meta.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "best_epoch": payload.get("epoch"),
                "holdout_loss": payload.get("holdout_loss"),
                "n_valid": len(valid_latent),
                "n_test": len(test_latent),
            }
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof = oof.drop(columns=["encoder_fold"])
    if oof[ID_COLUMN].duplicated().any():
        raise TrainingError("OOF latent contains duplicate ids")
    if set(oof[ID_COLUMN]) != set(train[ID_COLUMN]):
        raise TrainingError("OOF latent id set does not match train")
    # Restore train id order.
    oof = train[[ID_COLUMN]].merge(oof, on=ID_COLUMN, how="left", validate="1:1")
    if oof.isna().any().any():
        raise TrainingError("OOF latent reorder introduced nulls")
    oof.to_parquet(out_dir / TRAIN_LATENT_NAME, index=False)

    test_stack = pd.concat(test_parts, ignore_index=True)
    value_cols = [c for c in test_stack.columns if c not in {ID_COLUMN, "encoder_fold"}]
    test_mean = test_stack.groupby(ID_COLUMN, sort=False)[value_cols].mean().reset_index()
    test_mean = frames.test[[ID_COLUMN]].merge(test_mean, on=ID_COLUMN, how="left", validate="1:1")
    if test_mean[value_cols].isna().any().any():
        raise TrainingError("test latent mean has nulls; coverage must be 1.0")
    if len(test_mean) != len(frames.test):
        raise TrainingError("test latent mean row count mismatch")
    test_mean.to_parquet(out_dir / TEST_LATENT_NAME, index=False)

    latent_cols = [c for c in oof.columns if c.startswith("latent_")]
    recon_cols = [
        c for c in oof.columns if c.startswith("recon_") and not c.startswith("recon_std_")
    ]
    std_cols = [c for c in oof.columns if c.startswith("recon_std_")]
    manifest = {
        "source_run": str(run_dir),
        "model_name": model_name,
        "n_splits": n_splits,
        "device": str(device_obj),
        "batch_size": actual_batch,
        "train_rows": len(oof),
        "test_rows": len(test_mean),
        "latent_columns": latent_cols,
        "recon_columns": recon_cols,
        "recon_std_columns": std_cols,
        "folds": fold_meta,
        "files": {
            "train_oof": TRAIN_LATENT_NAME,
            "test_mean": TEST_LATENT_NAME,
            "per_fold": "per_fold/test_fold_{fold}_latent.parquet",
        },
    }
    write_latent_manifest(out_dir / "latent_manifest.json", manifest)
    return out_dir
