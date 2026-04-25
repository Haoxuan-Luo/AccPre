"""Frozen acceptance predictor — feature-MLP variant.

A small MLP that takes a flat feature vector (from
`accpre.collect.features.extract_features`) and outputs
`Q̂ ∈ [0,1]^γ`, one scalar per draft position.

Architecture (fixed for v1; see DESIGN_PHASE2.md §F):

  feature_dim
    → LayerNorm
    → Linear(feature_dim, hidden_dim), ReLU, Dropout
    → Linear(hidden_dim, hidden_dim), ReLU, Dropout
    → Linear(hidden_dim, γ)
    → Sigmoid

`kind = "acceptance"`, `deploy_mode` depends on `family`.

The same class serves all three main-lane families. Feature
dimensionality is computed from `accpre.collect.features.feature_dim`.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from accpre.collect.features import feature_dim
from accpre.predictors.base import PredictorBase


class AcceptanceMLP(PredictorBase):
    """Pooled-feature acceptance predictor.

    Feeds a flat feature vector to a shared MLP that emits γ scalars.
    Supports families: numeric, hidden, hidden_numeric, upper_bound.
    Only `hidden` is V-free; the rest require a verifier prefix forward.

    `family` and `deploy_mode` are per-instance attributes set by the
    base class from the __init__ kwargs (no class-attribute mutation).
    """
    kind: ClassVar[str] = "acceptance"

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

        in_dim = feature_dim(
            family,
            drafter_hidden_dim=drafter_hidden_dim,
            verifier_hidden_dim=verifier_hidden_dim,
        )
        self.in_dim = in_dim
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.gamma),
        )

    def forward(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Return `Q̂ ∈ [0,1]^{batch, γ}` via sigmoid.

        Args:
            features: (batch, in_dim) float tensor.
        """
        logits = self.net(features)
        return torch.sigmoid(logits)

    def predict_q2(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Inference Q̂ for one or a batch of feature vectors."""
        training_flag = self.training
        self.eval()
        try:
            if features.dim() == 1:
                out = self.forward(features.unsqueeze(0))
                return out.squeeze(0)
            return self.forward(features)
        finally:
            if training_flag:
                self.train()


class AcceptanceMLPPerPos(PredictorBase):
    """Per-position acceptance predictor (audit issue 7 fix).

    Input feature shape: `(batch, γ, H)` or `(γ, H)`. A small MLP is
    applied POSITION-WISE (same shared weights across all γ positions);
    output is one scalar `Q̂_j` per position.

    Architecture (shared across positions):
      LayerNorm(H) → Linear(H, hidden) → ReLU → Dropout
                   → Linear(hidden, hidden) → ReLU → Dropout
                   → Linear(hidden, 1) → sigmoid

    Unlike `AcceptanceMLP`, each position sees its own H-dim input, so
    the head can emit genuinely position-specific Q̂ values. Intended
    for the `hidden_per_pos` feature family; that family is V-free.
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        family: str,
        gamma: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        drafter_hidden_dim: int = 768,
    ) -> None:
        if family not in (
            "hidden_per_pos", "hidden_per_pos_hs", "hidden_per_pos_s",
            "hidden_per_pos_ml", "hidden_per_pos_numeric",
        ):
            raise ValueError(
                f"AcceptanceMLPPerPos supports per-position families only, "
                f"got {family!r}."
            )
        # Deploy mode depends on family. All drafter-only per-position
        # families are V-free; the Phase 7 `hidden_per_pos_numeric`
        # family concatenates verifier-prefix numeric scalars and
        # therefore requires a V-prefix online decode path.
        deploy_mode = (
            "V-prefix" if family == "hidden_per_pos_numeric" else "V-free"
        )
        super().__init__(family=family, deploy_mode=deploy_mode, gamma=gamma)

        # in_dim is the per-position feature dim for the chosen family.
        #   "hidden_per_pos"         → H            (e.g. 768)
        #   "hidden_per_pos_hs"      → H + 4        (Phase 5 richer)
        #   "hidden_per_pos_s"       → 4            (Phase 5 scalars-only)
        #   "hidden_per_pos_ml"      → 2H           (Phase 6 multi-layer)
        #   "hidden_per_pos_numeric" → H + 3        (Phase 7 V-prefix numeric)
        from accpre.collect.features import feature_dim
        self.in_dim = int(feature_dim(family, drafter_hidden_dim=drafter_hidden_dim))
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Shared MLP applied position-wise.

        Accepts `(batch, γ, H)` or `(γ, H)`; returns `(batch, γ)` or
        `(γ,)` respectively (sigmoid-activated).
        """
        squeezed = False
        if features.dim() == 2:            # (γ, H) → (1, γ, H)
            features = features.unsqueeze(0)
            squeezed = True
        if features.dim() != 3:
            raise ValueError(
                f"features must be (batch, γ, H) or (γ, H), got shape "
                f"{tuple(features.shape)}"
            )
        batch, gamma, H = features.shape
        if H != self.in_dim:
            raise ValueError(
                f"features hidden dim {H} != predictor in_dim {self.in_dim}"
            )
        # Apply MLP position-wise via flatten (batch × γ, H).
        flat = features.reshape(batch * gamma, H)
        logits = self.net(flat).reshape(batch, gamma)
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


class AcceptanceMLPPerPosWide(PredictorBase):
    """Phase 4 Stage-A `pp_wide` variant.

    Same per-position input as `AcceptanceMLPPerPos`, but with:
      - a learned positional embedding concatenated to each position's
        feature vector (explicit "which position is this"),
      - a wider shared MLP (256-wide with an extra layer).

    Architecture (shared across γ positions):
      cat(x_j, pe[j])                         ∈ R^(H + pe_dim)
      → LayerNorm → Linear(., 256) → ReLU → Dropout
      → Linear(256, 256) → ReLU → Dropout
      → Linear(256, 128) → ReLU → Dropout
      → Linear(128, 1) → σ

    Still applied position-wise with shared weights; the position identity
    enters ONLY via `pe[j]`.
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        family: str,
        gamma: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        drafter_hidden_dim: int = 768,
        pe_dim: int = 64,
    ) -> None:
        if family != "hidden_per_pos":
            raise ValueError(
                f"AcceptanceMLPPerPosWide only supports family='hidden_per_pos', "
                f"got {family!r}."
            )
        super().__init__(family=family, deploy_mode="V-free", gamma=gamma)

        self.drafter_hidden_dim = int(drafter_hidden_dim)
        self.pe_dim = int(pe_dim)
        self.hidden_dim = int(hidden_dim)
        in_dim = self.drafter_hidden_dim + self.pe_dim

        self.pos_emb = nn.Embedding(self.gamma, self.pe_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        squeezed = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeezed = True
        if features.dim() != 3:
            raise ValueError(
                f"features must be (batch, γ, H) or (γ, H); got {tuple(features.shape)}"
            )
        batch, gamma, H = features.shape
        if H != self.drafter_hidden_dim:
            raise ValueError(
                f"features hidden dim {H} != expected {self.drafter_hidden_dim}"
            )
        # Build (batch, γ, H+pe_dim) by broadcasting pos_emb along batch.
        positions = torch.arange(gamma, device=features.device)
        pe = self.pos_emb(positions)                      # (γ, pe_dim)
        pe_b = pe.unsqueeze(0).expand(batch, -1, -1)      # (batch, γ, pe_dim)
        x = torch.cat([features, pe_b], dim=-1)           # (batch, γ, H+pe_dim)
        flat = x.reshape(batch * gamma, -1)
        logits = self.net(flat).reshape(batch, gamma)
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


class AcceptanceMLPPerPosAttn(PredictorBase):
    """Phase 4 Stage-A `pp_attn` variant — LIGHT cross-position attention.

    Intentionally kept small:
      - one MultiheadAttention layer, SINGLE head, d_model=128.
      - a small FFN residual (128 → 256 → 128).
      - learned per-position embedding.

    Architecture (shape shown is (batch, γ, ·)):
      x ∈ (B, γ, H=768)
      x = proj(x) + pe(positions)              ∈ (B, γ, 128)
      x = x + attn(norm1(x))                    # 1-head self-attn, bidirectional
      x = x + ffn(norm2(x))                     # Linear(128,256)→ReLU→Dropout→Linear(256,128)
      Q̂_j = σ(head(x_j))                         ∈ (B, γ)

    Total params ≈ 230K — still very light. γ=8 is too short to benefit
    from masking; bidirectional attention lets each position see full
    context.
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        family: str,
        gamma: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        drafter_hidden_dim: int = 768,
        num_heads: int = 1,
        ffn_mult: int = 2,
    ) -> None:
        if family != "hidden_per_pos":
            raise ValueError(
                f"AcceptanceMLPPerPosAttn only supports family='hidden_per_pos', "
                f"got {family!r}."
            )
        super().__init__(family=family, deploy_mode="V-free", gamma=gamma)

        self.drafter_hidden_dim = int(drafter_hidden_dim)
        self.d_model = int(hidden_dim)
        self.num_heads = int(num_heads)

        # Input LayerNorm on the raw per-position drafter hidden. Raw
        # MDLM hidden states have std ~55; without this LN the residual
        # stream dominates the attention+FFN contributions. Same lesson
        # applied in AcceptanceMLPPerPos and PerPositionTransformerEncoder.
        self.input_norm = nn.LayerNorm(self.drafter_hidden_dim)
        self.proj = nn.Linear(self.drafter_hidden_dim, self.d_model)
        self.pos_emb = nn.Embedding(self.gamma, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * int(ffn_mult)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model * int(ffn_mult), self.d_model),
            nn.Dropout(float(dropout)),
        )

        self.head = nn.Linear(self.d_model, 1)

    def forward(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        squeezed = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeezed = True
        if features.dim() != 3:
            raise ValueError(
                f"features must be (batch, γ, H) or (γ, H); got {tuple(features.shape)}"
            )
        batch, gamma, H = features.shape
        if H != self.drafter_hidden_dim:
            raise ValueError(
                f"features hidden dim {H} != expected {self.drafter_hidden_dim}"
            )

        features = self.input_norm(features)              # (B, γ, H) unit-scale
        x = self.proj(features)                           # (B, γ, d_model)
        positions = torch.arange(gamma, device=features.device)
        pe = self.pos_emb(positions).unsqueeze(0)          # (1, γ, d_model)
        x = x + pe

        # Pre-norm transformer block: attn + FFN, each with residual.
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n, need_weights=False)
        x = x + attn_out

        x_n = self.norm2(x)
        x = x + self.ffn(x_n)

        logits = self.head(x).squeeze(-1)                  # (B, γ)
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
