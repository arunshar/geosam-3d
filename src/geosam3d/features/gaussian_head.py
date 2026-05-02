"""Per-Gaussian feature head trained by SAM 2 mask consistency.

Each Gaussian gets a 32-d embedding produced by a small transformer over
its position + appearance attributes (mean color, opacity, scale norm). The
head is trained with a contrastive loss: features inside the same SAM 2 video
mask should be close in cosine distance, features in different masks should
be far.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GaussianFeatureHead(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 256, out_dim: int = 32, depth: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, gauss_attrs: Tensor) -> Tensor:
        h = self.in_proj(gauss_attrs)
        h = self.encoder(h.unsqueeze(0)).squeeze(0)
        return F.normalize(self.out(h), dim=-1)


def contrastive_loss(features: Tensor, mask_id: Tensor, tau: float = 0.07) -> Tensor:
    """InfoNCE-style contrastive loss: same mask_id => positive."""
    sim = features @ features.t() / tau
    labels = (mask_id.unsqueeze(0) == mask_id.unsqueeze(1)).float()
    labels.fill_diagonal_(0)
    log_p = sim - sim.logsumexp(dim=-1, keepdim=True)
    return -(labels * log_p).sum(-1).mean() / labels.sum(-1).clamp_min(1.0).mean()
