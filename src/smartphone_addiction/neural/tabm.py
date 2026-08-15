"""TabM-style BatchEnsemble masked autoencoder implemented with core PyTorch ops."""

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


def build_tabm_autoencoder(
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
    ensemble_size = int(config.ensemble_size)
    in_dim = input_dim(n_numeric, n_core, n_categorical, config.embedding_dim)

    class BatchEnsembleLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.r = nn.Parameter(torch.empty(ensemble_size, in_features))
            self.s = nn.Parameter(torch.empty(ensemble_size, out_features))
            self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            nn.init.normal_(self.r, mean=1.0, std=0.5)
            nn.init.normal_(self.s, mean=1.0, std=0.5)

        def forward(self, x):
            # x: [batch, ensemble, in]
            scaled = x * self.r.unsqueeze(0)
            out = torch.einsum("bei,oi->beo", scaled, self.weight)
            return out * self.s.unsqueeze(0) + self.bias.unsqueeze(0)

    class TabMBlock(nn.Module):
        def __init__(self, in_features: int, hidden_dim: int) -> None:
            super().__init__()
            self.linear = BatchEnsembleLinear(in_features, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            return self.dropout(self.activation(self.norm(self.linear(x))))

    class TabMReconstructionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ensemble_size = ensemble_size
            self.embeddings = nn.ModuleList(
                [nn.Embedding(int(size), config.embedding_dim) for size in vocab_sizes]
            )
            blocks = []
            last = in_dim
            for _ in range(config.n_blocks):
                blocks.append(TabMBlock(last, config.hidden_dim))
                last = config.hidden_dim
            self.blocks = nn.ModuleList(blocks)
            self.to_latent = BatchEnsembleLinear(last, config.latent_dim)
            self.decoder = BatchEnsembleLinear(config.latent_dim, n_core)

        def forward(self, batch: MaskBatch) -> ReconstructionOutput:
            hidden = encode_reconstruction_inputs(batch, list(self.embeddings))
            members = hidden.unsqueeze(1).expand(-1, ensemble_size, -1).contiguous()
            for block in self.blocks:
                members = block(members)
            latents = self.to_latent(members)
            predictions = self.decoder(latents)
            return ReconstructionOutput(
                mean_prediction=predictions.mean(dim=1),
                mean_latent=latents.mean(dim=1),
                member_predictions=predictions,
                member_latents=latents,
                member_std=predictions.std(dim=1, unbiased=False),
            )

    return TabMReconstructionModel()
