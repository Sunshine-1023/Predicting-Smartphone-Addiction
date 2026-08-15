"""MLP masked autoencoder control model."""

from __future__ import annotations

from smartphone_addiction.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from smartphone_addiction.neural.config import CORE5_FIELDS, NeuralModelArchConfig
from smartphone_addiction.neural.device import require_torch
from smartphone_addiction.neural.masking import MaskBatch
from smartphone_addiction.neural.outputs import (
    ReconstructionOutput,
    encode_reconstruction_inputs,
    input_dim,
)


def build_mlp_autoencoder(
    vocab_sizes: list[int],
    config: NeuralModelArchConfig,
    *,
    n_numeric: int | None = None,
):
    torch = require_torch()
    nn = torch.nn
    n_numeric = len(NUMERIC_COLUMNS) if n_numeric is None else n_numeric
    n_core = len(CORE5_FIELDS)
    n_categorical = len(CATEGORICAL_COLUMNS)
    in_dim = input_dim(n_numeric, n_core, n_categorical, config.embedding_dim)

    class MLPMaskedAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(int(size), config.embedding_dim) for size in vocab_sizes]
            )
            blocks: list = []
            last = in_dim
            for _ in range(config.n_blocks):
                blocks.extend(
                    [
                        nn.Linear(last, config.hidden_dim),
                        nn.LayerNorm(config.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                    ]
                )
                last = config.hidden_dim
            self.encoder = nn.Sequential(*blocks)
            self.to_latent = nn.Linear(last, config.latent_dim)
            self.decoder = nn.Linear(config.latent_dim, n_core)

        def forward(self, batch: MaskBatch) -> ReconstructionOutput:
            hidden = encode_reconstruction_inputs(batch, list(self.embeddings))
            latent = self.to_latent(self.encoder(hidden))
            prediction = self.decoder(latent)
            return ReconstructionOutput(mean_prediction=prediction, mean_latent=latent)

    return MLPMaskedAutoencoder()


MaskedAutoencoder = build_mlp_autoencoder
