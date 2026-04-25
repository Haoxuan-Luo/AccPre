"""Frozen τ-conditioned committed-length predictor.

Takes a flat feature vector (from
`accpre.collect.features.extract_features`), normalizes the features,
concatenates the scalar τ, and emits logits over `{0, 1, ..., γ}`.

Inference:
    L̂(τ) = argmax_k P(L = k | features, τ)

Architecture (v2 — fixed τ routing):

  features (feature_dim)
    → LayerNorm(feature_dim)              ← ONLY over features; τ not included
    → concat with τ (1 scalar)
    → Linear(feature_dim+1, hidden_dim), ReLU, Dropout
    → Linear(hidden_dim, hidden_dim), ReLU, Dropout
    → Linear(hidden_dim, γ+1)

Why: the previous architecture applied `LayerNorm(feature_dim + 1)`
over `[features; τ]`. For `hidden_numeric` (772 dims total) τ is
0.13 % of the coordinates and LayerNorm effectively whitens it out;
the network learned to ignore τ completely (logit std across τ on
the same record was ≈ 2e-4 — see debug report). Normalizing features
first and concatenating τ afterwards keeps τ on an absolute scale
the Linear can latch onto.

`kind = "committed_length"`.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from accpre.collect.features import feature_dim
from accpre.predictors.base import PredictorBase


class LengthMLP(PredictorBase):
    kind: ClassVar[str] = "committed_length"
    # `family` / `deploy_mode` are per-instance attributes set by
    # PredictorBase.__init__ from our kwargs (no class-attribute mutation).

    def __init__(
        self,
        family: str,
        gamma: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        drafter_hidden_dim: int = 768,
        verifier_hidden_dim: int = 1600,
    ) -> None:
        deploy_mode = "V-free" if family == "hidden" else "V-prefix"
        super().__init__(family=family, deploy_mode=deploy_mode, gamma=gamma)

        feat_dim = feature_dim(
            family,
            drafter_hidden_dim=drafter_hidden_dim,
            verifier_hidden_dim=verifier_hidden_dim,
        )
        self.feat_dim = feat_dim
        self.in_dim = feat_dim + 1  # +1 for τ, after feature normalization
        self.hidden_dim = int(hidden_dim)
        self.n_classes = self.gamma + 1

        # Normalize features ONLY (not τ).
        self.feat_norm = nn.LayerNorm(feat_dim)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.n_classes),
        )

    def _pack_input(
        self, features: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Normalize features, then concat τ. Handles both (D,) single
        features and (B, D) batches.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)  # (1, D)
        features = self.feat_norm(features)   # <-- fix: normalize features only

        if tau.dim() == 0:
            tau = tau.unsqueeze(0)  # (1,)
        tau_col = tau.to(features.dtype).reshape(-1, 1)
        if tau_col.shape[0] != features.shape[0]:
            raise ValueError(
                f"features batch {features.shape[0]} != tau batch {tau_col.shape[0]}"
            )
        return torch.cat([features, tau_col], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        tau: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        """Training-time forward.

        Returns raw logits of shape `(batch, γ+1)`. The trainer applies
        cross-entropy on these directly (see `accpre.train.losses.ce_length`).
        """
        x = self._pack_input(features, tau)
        return self.net(x)

    def predict_L(
        self, features: torch.Tensor, tau: float, **_kwargs
    ) -> int:
        """Inference: argmax over categorical logits for the given τ."""
        training_flag = self.training
        self.eval()
        try:
            tau_t = torch.tensor(float(tau), dtype=features.dtype)
            logits = self.forward(features, tau_t)  # (1, γ+1)
            if logits.dim() == 2:
                logits = logits.squeeze(0)
            return int(logits.argmax().item())
        finally:
            if training_flag:
                self.train()
