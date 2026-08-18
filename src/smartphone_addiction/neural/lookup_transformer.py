"""Exact-value lookup embeddings followed by a small feature-token Transformer."""

from __future__ import annotations

from dataclasses import dataclass

from smartphone_addiction.errors import TrainingError
from smartphone_addiction.neural.classification_config import LookupTransformerArchConfig
from smartphone_addiction.neural.device import require_torch


@dataclass
class LookupOutput:
    logits: object
    probability: object


def build_lookup_transformer(
    cardinalities: list[int],
    config: LookupTransformerArchConfig,
):
    """Build the 128-d, four-layer Lookup Transformer used by the S6E8 probe."""
    torch = require_torch()
    nn = torch.nn
    if not cardinalities:
        raise TrainingError("lookup transformer requires at least one exact-value feature")
    if config.hidden_dim % config.n_heads != 0:
        raise TrainingError("lookup hidden_dim must be divisible by n_heads")
    d_model = int(config.hidden_dim)
    n_features = len(cardinalities)

    class LookupTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_embeddings = nn.ModuleList(
                [
                    nn.Embedding(
                        int(cardinality),
                        d_model,
                        padding_idx=None,
                    )
                    for cardinality in cardinalities
                ]
            )
            self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
            self.feature_offsets = nn.Parameter(torch.empty(1, n_features, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(config.n_heads),
                dim_feedforward=int(config.feedforward_dim),
                dropout=float(config.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=int(config.n_blocks),
                enable_nested_tensor=False,
            )
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            nn.init.normal_(self.feature_offsets, mean=0.0, std=0.02)

        def forward(self, cat_indices) -> LookupOutput:
            if cat_indices.ndim != 2 or cat_indices.shape[1] != n_features:
                raise TrainingError(
                    "lookup indices must have shape "
                    f"[batch, {n_features}], got {tuple(cat_indices.shape)}"
                )
            tokens = torch.stack(
                [
                    embedding(cat_indices[:, feature_i])
                    for feature_i, embedding in enumerate(self.value_embeddings)
                ],
                dim=1,
            )
            tokens = tokens + self.feature_offsets
            cls = self.cls_token.expand(len(cat_indices), -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            logits = self.head(encoded[:, 0]).squeeze(-1)
            return LookupOutput(
                logits=logits,
                probability=torch.sigmoid(logits),
            )

    return LookupTransformer()
