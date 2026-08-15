"""Fold-local latent export. Only used after the reconstruction gate passes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smartphone_addiction.data.schema import ID_COLUMN
from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.config import CORE5_FIELDS
from smartphone_addiction.neural.device import require_torch
from smartphone_addiction.neural.masking import MaskBatch, move_mask_batch
from smartphone_addiction.neural.preprocessing import FoldTensorizer, TensorizedFrame


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
            reconstructions.append(
                tensorizer.inverse_core(output.mean_prediction.detach().cpu().numpy())
            )
    latent = np.concatenate(latents, axis=0)
    recon = np.concatenate(reconstructions, axis=0)
    payload: dict[str, object] = {ID_COLUMN: np.asarray(frame.row_ids)}
    for dim in range(latent.shape[1]):
        payload[f"latent_{dim:02d}"] = latent[:, dim]
    for field_i, field in enumerate(CORE5_FIELDS):
        payload[f"recon_{field}"] = recon[:, field_i]
    if ID_COLUMN in payload and pd.Series(payload[ID_COLUMN]).duplicated().any():
        raise TrainingError("latent export produced duplicate ids")
    return pd.DataFrame(payload)


def write_latent_manifest(path: Path, payload: dict) -> None:
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
