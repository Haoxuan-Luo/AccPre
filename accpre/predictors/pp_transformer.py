"""Phase 9 — per-position transformer predictor family.

A shared lightweight transformer encoder over the gamma draft positions
feeds two heads:

  - AcceptanceTx : one sigmoid scalar per position → Q_hat_j in [0, 1]^gamma
                   (kind = "acceptance")
  - LengthTx     : mean-pool over gamma → LayerNorm → concat tau → classifier
                   over {0, ..., gamma}
                   (kind = "committed_length")

Both heads are paired with the four Phase 9 feature families:

  hidden_per_pos_v1   (F_in = H)               drafter hidden only
  hidden_per_pos_v2   (F_in = H + 5)           + [q, log q, entropy, margin, top1]
  hidden_per_pos_v3   (F_in = H + 3)           + verifier prefix numeric broadcast
  hidden_per_pos_v4   (F_in = H + 5 + H_v)     v2 + raw verifier_hidden broadcast

v1 and v2 are V-free; v3 and v4 are V-prefix (enforced by
PredictorBase._V_FREE_FAMILIES).

Encoder hyperparameters are fixed for the Phase 9 family (do NOT sweep):
  d_model=128, num_layers=2, num_heads=4, dropout=0.1.

Position is NOT part of the offline feature tensor. The encoder adds a
learned positional embedding at its input. This keeps the feature tensor
interface identical to the existing per-position MLP heads so dataset
collation is unchanged.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from accpre.collect.features import feature_dim
from accpre.predictors.base import PredictorBase, _V_FREE_FAMILIES


# Families this module handles. Kept separately from the global
# _VALID_FAMILIES tuple so an auditor can see the Phase 9 scope at a
# glance; the PredictorBase constructor still enforces _VALID_FAMILIES.
_TX_FAMILIES: tuple = (
    "hidden_per_pos_v1",
    "hidden_per_pos_v2",
    "hidden_per_pos_v3",
    "hidden_per_pos_v4",
)


def _deploy_mode_for_family(family: str) -> str:
    """Derive deploy_mode from family using the base-module invariant.

    Keeping a single call site (rather than hard-coding per-class
    if/else) means the PredictorBase V-free guard remains the source of
    truth; this helper just reads it.
    """
    return "V-free" if family in _V_FREE_FAMILIES else "V-prefix"


class PerPositionTransformerEncoder(nn.Module):
    """Lightweight per-position transformer encoder shared by all Phase 9 heads.

    Input : (B, gamma, in_dim)
    Output: (B, gamma, d_model)

    Pipeline:
      LayerNorm(in_dim)                                 # per-position input norm
        -> Linear(in_dim, d_model)
        + learned Embedding(gamma, d_model)             # added, not concatenated
        -> TransformerEncoder(n_layers, pre-norm)

    Uses `norm_first=True` (pre-norm) because small d_model + small
    n_layers models tend to be more stable with pre-norm.

    The leading `LayerNorm(in_dim)` mirrors AcceptanceMLPPerPos /
    AcceptanceMLPPerPosWide. Raw drafter hidden states have std ~55 and
    are not pre-normalized; without an input LN the residual stream
    dominates the transformer's attention+FFN contributions, collapsing
    the head to effectively `sigmoid(Linear(raw_hidden))`.
    """

    def __init__(
        self,
        in_dim: int,
        gamma: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.gamma = int(gamma)
        self.d_model = int(d_model)

        self.input_norm = nn.LayerNorm(self.in_dim)
        self.input_proj = nn.Linear(self.in_dim, self.d_model)
        self.pos_emb = nn.Embedding(self.gamma, self.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=self.d_model * 2,
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"PerPositionTransformerEncoder expects (B, gamma, in_dim); "
                f"got shape {tuple(x.shape)}"
            )
        B, gamma, F = x.shape
        if F != self.in_dim:
            raise ValueError(
                f"input feature dim {F} != encoder in_dim {self.in_dim}"
            )
        if gamma > self.gamma:
            raise ValueError(
                f"input gamma={gamma} exceeds encoder's trained gamma="
                f"{self.gamma}; pos_emb undefined beyond index {self.gamma - 1}"
            )
        h = self.input_norm(x)                                  # (B, gamma, in_dim)
        h = self.input_proj(h)                                  # (B, gamma, d)
        positions = torch.arange(gamma, device=x.device)        # (gamma,)
        h = h + self.pos_emb(positions).unsqueeze(0)            # (B, gamma, d)
        return self.encoder(h)                                  # (B, gamma, d)


class AcceptanceTx(PredictorBase):
    """Per-position transformer acceptance predictor.

    Wraps PerPositionTransformerEncoder + a per-position sigmoid head.
    Produces Q_hat_j in [0, 1]^gamma for each draft position.
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        family: str,
        gamma: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        drafter_hidden_dim: int = 768,
        verifier_hidden_dim: int = 1600,
    ) -> None:
        if family not in _TX_FAMILIES:
            raise ValueError(
                f"AcceptanceTx supports Phase 9 tx families only "
                f"({_TX_FAMILIES}); got {family!r}."
            )
        super().__init__(
            family=family,
            deploy_mode=_deploy_mode_for_family(family),
            gamma=gamma,
        )
        self.in_dim = int(feature_dim(
            family,
            drafter_hidden_dim=drafter_hidden_dim,
            verifier_hidden_dim=verifier_hidden_dim,
        ))
        self.d_model = int(d_model)
        self.encoder = PerPositionTransformerEncoder(
            in_dim=self.in_dim, gamma=gamma,
            d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.head = nn.Linear(self.d_model, 1)

    def forward(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Return Q_hat in [0, 1]^(batch, gamma).

        Accepts (gamma, F_in) or (B, gamma, F_in); shape-matches the
        AcceptanceMLPPerPos convention so training / online paths work
        without modification.
        """
        squeezed = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeezed = True
        h = self.encoder(features)                              # (B, gamma, d)
        logits = self.head(h).squeeze(-1)                       # (B, gamma)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        training_flag = self.training
        self.eval()
        try:
            return self.forward(features)
        finally:
            if training_flag:
                self.train()


class LengthTx(PredictorBase):
    """Per-position transformer committed-length predictor, tau-conditioned.

    Encoder output (B, gamma, d_model) -> mean-pool over gamma ->
    LayerNorm -> concat tau scalar -> 2-layer classifier -> logits over
    {0, ..., gamma}.

    tau is concatenated AFTER the LayerNorm so it is not whitened out
    (same lesson as LengthMLP._pack_input). Training uses a random tau
    per item from FrozenLengthDataset's grid; at inference, the caller
    passes the target tau directly to predict_L.
    """
    kind: ClassVar[str] = "committed_length"

    def __init__(
        self,
        family: str,
        gamma: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        drafter_hidden_dim: int = 768,
        verifier_hidden_dim: int = 1600,
    ) -> None:
        if family not in _TX_FAMILIES:
            raise ValueError(
                f"LengthTx supports Phase 9 tx families only "
                f"({_TX_FAMILIES}); got {family!r}."
            )
        super().__init__(
            family=family,
            deploy_mode=_deploy_mode_for_family(family),
            gamma=gamma,
        )
        self.in_dim = int(feature_dim(
            family,
            drafter_hidden_dim=drafter_hidden_dim,
            verifier_hidden_dim=verifier_hidden_dim,
        ))
        self.d_model = int(d_model)
        self.n_classes = int(gamma) + 1

        self.encoder = PerPositionTransformerEncoder(
            in_dim=self.in_dim, gamma=gamma,
            d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.pool_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.n_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        tau: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        """Return logits of shape (B, gamma+1).

        Accepts (gamma, F_in)+scalar-tau and (B, gamma, F_in)+(B,)-tau.
        """
        squeezed_feat = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeezed_feat = True
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        B = features.shape[0]
        if tau.shape[0] != B:
            raise ValueError(
                f"features batch {B} != tau batch {tau.shape[0]}"
            )
        h = self.encoder(features)                              # (B, gamma, d)
        pooled = h.mean(dim=1)                                  # (B, d)
        pooled = self.pool_norm(pooled)                         # (B, d)
        tau_col = tau.to(pooled.dtype).reshape(-1, 1)           # (B, 1)
        x = torch.cat([pooled, tau_col], dim=-1)                # (B, d+1)
        logits = self.classifier(x)                             # (B, gamma+1)
        return logits.squeeze(0) if squeezed_feat else logits

    def predict_L(
        self,
        features: torch.Tensor,
        tau: float,
        **_kwargs,
    ) -> int:
        """Inference: argmax over categorical logits for the given tau."""
        training_flag = self.training
        self.eval()
        try:
            tau_t = torch.tensor(float(tau), dtype=features.dtype)
            logits = self.forward(features, tau_t)
            if logits.dim() == 2:
                logits = logits.squeeze(0)
            return int(logits.argmax().item())
        finally:
            if training_flag:
                self.train()
