"""Fold-native encoder features for LightGBM (same encoder for train/mask/valid/test)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from smartphone_addiction.data.schema import FEATURE_COLUMNS, ID_COLUMN
from smartphone_addiction.errors import ArtifactError, TrainingError
from smartphone_addiction.neural.autoencoder import build_mlp_autoencoder
from smartphone_addiction.neural.config import CORE5_FIELDS, NeuralModelArchConfig
from smartphone_addiction.neural.device import require_torch, resolve_device
from smartphone_addiction.neural.export import encode_latent
from smartphone_addiction.neural.preprocessing import FoldTensorizer

IncludeKind = Literal["imputed_core", "latent"]


@dataclass
class FoldEncoder:
    fold: int
    model: Any
    tensorizer: FoldTensorizer
    device: object
    batch_size: int
    model_name: str

    def encode_frame(
        self,
        frame: pd.DataFrame,
        *,
        include: list[IncludeKind],
    ) -> pd.DataFrame:
        if ID_COLUMN not in frame.columns:
            raise TrainingError("frame must contain id for fold-native encoding")
        missing_raw = [name for name in FEATURE_COLUMNS if name not in frame.columns]
        if missing_raw:
            raise TrainingError(f"frame missing raw columns required by encoder: {missing_raw}")
        raw = frame.loc[:, [ID_COLUMN, *FEATURE_COLUMNS]].copy()
        encoded = self.tensorizer.transform(raw)
        payload = encode_latent(
            self.model,
            encoded,
            self.tensorizer,
            device=self.device,
            batch_size=self.batch_size,
        )
        if list(payload[ID_COLUMN]) != list(raw[ID_COLUMN]):
            raise TrainingError("encoder output id order diverged from input frame")
        out = pd.DataFrame({ID_COLUMN: payload[ID_COLUMN].to_numpy()})
        if "imputed_core" in include:
            for field in CORE5_FIELDS:
                raw_values = raw[field].to_numpy(dtype=np.float64)
                missing = ~np.isfinite(raw_values)
                recon = payload[f"recon_{field}"].to_numpy(dtype=np.float64)
                out[f"imputed_{field}"] = np.where(missing, recon, raw_values)
                out[f"{field}_is_imputed"] = missing.astype(np.int8)
        if "latent" in include:
            for column in payload.columns:
                if column.startswith("latent_"):
                    out[column] = payload[column].to_numpy()
        if out.columns.tolist() == [ID_COLUMN]:
            raise TrainingError(f"fold-native include={include!r} produced no feature columns")
        return out


@dataclass
class FoldEncoderBank:
    run_dir: Path
    folds: dict[int, FoldEncoder]
    include: list[IncludeKind]
    n_splits: int

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if "imputed_core" in self.include:
            for field in CORE5_FIELDS:
                names.append(f"imputed_{field}")
                names.append(f"{field}_is_imputed")
        if "latent" in self.include:
            dim = int(
                yaml.safe_load((self.run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))[
                    "model"
                ]["latent_dim"]
            )
            names.extend(f"latent_{index:02d}" for index in range(dim))
        return names

    def for_fold(self, fold: int) -> FoldEncoder:
        if fold not in self.folds:
            raise TrainingError(f"encoder bank missing fold {fold}")
        return self.folds[fold]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_fold_encoder_bank(
    reconstruction_run: Path | str,
    *,
    include: list[IncludeKind] | None = None,
    device: str = "auto",
    batch_size: int | None = None,
) -> FoldEncoderBank:
    run_dir = Path(reconstruction_run)
    gate_path = run_dir / "gate_decision.json"
    if not gate_path.is_file():
        raise ArtifactError(f"missing gate_decision.json under {run_dir}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("invalid"):
        raise ArtifactError(f"refusing invalid reconstruction run: {gate.get('invalid_reason')}")
    if not gate.get("passed") or gate.get("smoke"):
        raise ArtifactError("reconstruction run must be a non-smoke gated success")

    resolved = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    model_name = str(resolved["model"]["name"])
    arch = NeuralModelArchConfig.model_validate(resolved["model"])
    actual_batch = int(batch_size or resolved.get("training", {}).get("batch_size") or 4096)
    torch = require_torch()
    device_obj = resolve_device(device or resolved.get("device") or "auto", torch_module=torch)

    folds: dict[int, FoldEncoder] = {}
    checkpoint_dir = run_dir / "checkpoints"
    for checkpoint in sorted(checkpoint_dir.glob("fold_*.pt")):
        fold = int(checkpoint.stem.split("_")[1])
        payload = _load_checkpoint(checkpoint)
        tensorizer = FoldTensorizer.from_state(payload["tensorizer"])
        if model_name != "mlp":
            raise TrainingError(f"unsupported encoder model: {model_name}")
        model = build_mlp_autoencoder(tensorizer.vocab_sizes(), arch)
        model.load_state_dict(payload["model_state"])
        model.to(device_obj)
        model.eval()
        folds[fold] = FoldEncoder(
            fold=fold,
            model=model,
            tensorizer=tensorizer,
            device=device_obj,
            batch_size=actual_batch,
            model_name=model_name,
        )
    if not folds:
        raise ArtifactError(f"no fold checkpoints under {checkpoint_dir}")
    return FoldEncoderBank(
        run_dir=run_dir,
        folds=folds,
        include=list(include or ["imputed_core"]),
        n_splits=len(folds),
    )


def attach_encoder_features(
    frame: pd.DataFrame,
    encoder: FoldEncoder,
    *,
    include: list[IncludeKind],
) -> pd.DataFrame:
    """Append fold-native encoder features aligned by row position.

    Row-order alignment is required because LightGBM masking copies reuse source
    ids, so an id-based 1:1 join would be invalid.
    """
    encoded = encoder.encode_frame(frame, include=include)
    if len(encoded) != len(frame):
        raise TrainingError("encoder feature row count diverged from input frame")
    # Allow duplicate ids, but positions must still correspond.
    if not encoded[ID_COLUMN].equals(
        frame[ID_COLUMN].reset_index(drop=True)
    ) and not np.array_equal(
        encoded[ID_COLUMN].to_numpy(),
        frame[ID_COLUMN].to_numpy(),
    ):
        raise TrainingError("encoder feature id order diverged from input frame")
    feature_cols = [column for column in encoded.columns if column != ID_COLUMN]
    overlap = [column for column in feature_cols if column in frame.columns]
    if overlap:
        raise TrainingError(f"encoder features already present on frame: {overlap}")
    if encoded[feature_cols].isna().any().any():
        raise TrainingError("encoder features produced nulls")
    base = frame.reset_index(drop=True)
    extra = encoded.loc[:, feature_cols].reset_index(drop=True)
    return pd.concat([base, extra], axis=1)
