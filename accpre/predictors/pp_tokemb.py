"""Phase 12 — token-embedding acceptance predictors (V-free only).

Two heads that both take the existing `hidden_per_pos_v2` float block
(drafter hidden + 5 per-position scalars) PLUS per-position drafted
token IDs, embed the token IDs through a learned `nn.Embedding`, and
concatenate the embedding onto the float features before their body:

  AcceptanceMLPPerPosTokEmb (1A)
      LN(in_dim+emb_dim) → shared-weight position-wise MLP → σ
      No cross-position interaction — each position is scored from its
      own (hidden, scalars, token_emb) independently.

  AcceptanceTxTokEmb        (1B)
      PerPositionTransformerEncoder(in_dim+emb_dim) → Linear(d, 1) → σ
      Light cross-position attention on top of the same inputs.

The token embedding table is owned by the predictor (not the drafter
model), so online decode does not need to expose the drafter's weight
matrix. Vocab size defaults to 50258 (GPT-2 + MASK, MDLM's vocab).

Both heads are V-free: token IDs are already available on the drafter
side at inference time (they are the sampled `draft_tokens`), so no
verifier forward is required.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import torch
from torch import nn

from accpre.collect.features import feature_dim
from accpre.predictors.base import PredictorBase, _V_FREE_FAMILIES
from accpre.predictors.pp_transformer import PerPositionTransformerEncoder


# Default vocab size: MDLM = GPT-2 (50257) + MASK = 50258.
# See accpre/models/drafter_mdlm.py:72-86.
DEFAULT_TOK_VOCAB_SIZE: int = 50258


def _assemble_input(
    features: torch.Tensor,           # (B, γ, F_in) or (γ, F_in)
    token_ids: torch.Tensor,          # (B, γ) or (γ,)
    embedding: nn.Embedding,
) -> tuple[torch.Tensor, bool]:
    """Concat token embeddings onto per-position float features.

    Returns (assembled, squeezed) where `squeezed` says whether the
    caller handed in an unbatched (γ, F) shape and needs the batch axis
    dropped on the way back out.
    """
    if features.dim() == 2:
        features = features.unsqueeze(0)
        token_ids = token_ids.unsqueeze(0) if token_ids.dim() == 1 else token_ids
        squeezed = True
    else:
        squeezed = False
    if token_ids.dim() != 2:
        raise ValueError(
            f"token_ids must be (B, γ) or (γ,); got shape {tuple(token_ids.shape)}"
        )
    if token_ids.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"features batch/γ {tuple(features.shape[:2])} != token_ids "
            f"batch/γ {tuple(token_ids.shape)}"
        )
    emb = embedding(token_ids.long())                    # (B, γ, emb_dim)
    return torch.cat([features, emb], dim=-1), squeezed  # (B, γ, F_in+emb_dim)


_TOKEMB_MLP_SUPPORTED: tuple = (
    "hidden_per_pos_v2",   # 1A original — V-free
    "hidden_per_pos_v3",   # Phase 19 Vlite-N — V-prefix (drafter + 3 prefix numerics)
)


class AcceptanceMLPPerPosTokEmb(PredictorBase):
    """1A — per-position MLP + drafted-token embedding.

    Shared-weight position-wise MLP (no cross-position interaction),
    matching AcceptanceMLPPerPos's architecture except for the extra
    embedded-token channel at the input.

    Supported families (feature_dim auto-sizes):
        hidden_per_pos_v2  (768 + 5)       — V-free, original 1A.
        hidden_per_pos_v3  (768 + 3)       — V-prefix, Phase 19 Vlite-N.
    Deploy mode is derived from the family via `_V_FREE_FAMILIES`, so
    the online loop knows whether to fire a per-round verifier prefix
    forward for this predictor.
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        family: str,
        gamma: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        token_vocab_size: int = DEFAULT_TOK_VOCAB_SIZE,
        token_emb_dim: int = 64,
    ) -> None:
        if family not in _TOKEMB_MLP_SUPPORTED:
            raise ValueError(
                f"AcceptanceMLPPerPosTokEmb supports family in "
                f"{_TOKEMB_MLP_SUPPORTED}; got {family!r}."
            )
        deploy_mode = "V-free" if family in _V_FREE_FAMILIES else "V-prefix"
        super().__init__(family=family, deploy_mode=deploy_mode, gamma=gamma)

        self.feat_in_dim = int(feature_dim(family))
        self.token_vocab_size = int(token_vocab_size)
        self.token_emb_dim = int(token_emb_dim)
        self.in_dim = self.feat_in_dim + self.token_emb_dim
        self.hidden_dim = int(hidden_dim)

        self.token_emb = nn.Embedding(self.token_vocab_size, self.token_emb_dim)
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

    def forward(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError(
                "AcceptanceMLPPerPosTokEmb.forward requires `token_ids` "
                "(per-position drafted-token IDs, long tensor)."
            )
        x, squeezed = _assemble_input(features, token_ids, self.token_emb)
        B, gamma, F_in = x.shape
        flat = x.reshape(B * gamma, F_in)
        logits = self.net(flat).reshape(B, gamma)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        training_flag = self.training
        self.eval()
        try:
            return self.forward(features, token_ids=token_ids)
        finally:
            if training_flag:
                self.train()


class AcceptanceMLPPerPosTokEmbSepLN(PredictorBase):
    """1A fusion-C ablation: separate LN on float features and token embedding.

    Same body as `AcceptanceMLPPerPosTokEmb` (shared-weight position-wise
    MLP, no cross-position interaction), same target (Q2_j), same loss
    (plain MSE). The only change is *how* float features and the
    drafted-token embedding are fused at the input:

      default 1A  :  concat([feats, emb], -1)  →  LayerNorm(in_dim)  →  MLP
      sep-LN (C)  :  concat([ LN_f(feats), LN_e(emb) ], -1)          →  MLP

    Rationale: the default applies a single LayerNorm over (768 + 5 + 64)
    = 837 dims, so the 64-dim token embedding contributes ~7.6 % of the
    per-sample statistics. Separate LN normalises each modality to unit
    scale before they are mixed, so the linear's first layer sees
    token-emb coordinates and float-feature coordinates at comparable
    magnitudes regardless of the training dynamics of each branch.
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        family: str,
        gamma: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        token_vocab_size: int = DEFAULT_TOK_VOCAB_SIZE,
        token_emb_dim: int = 64,
    ) -> None:
        if family != "hidden_per_pos_v2":
            raise ValueError(
                f"AcceptanceMLPPerPosTokEmbSepLN supports family="
                f"'hidden_per_pos_v2' only; got {family!r}."
            )
        super().__init__(family=family, deploy_mode="V-free", gamma=gamma)

        self.feat_in_dim = int(feature_dim(family))
        self.token_vocab_size = int(token_vocab_size)
        self.token_emb_dim = int(token_emb_dim)
        self.in_dim = self.feat_in_dim + self.token_emb_dim
        self.hidden_dim = int(hidden_dim)

        self.token_emb = nn.Embedding(self.token_vocab_size, self.token_emb_dim)
        self.feat_norm = nn.LayerNorm(self.feat_in_dim)
        self.emb_norm = nn.LayerNorm(self.token_emb_dim)

        # Same MLP body as 1A but WITHOUT the leading LayerNorm (the two
        # branches normalise their own inputs already).
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError(
                "AcceptanceMLPPerPosTokEmbSepLN.forward requires `token_ids`."
            )
        if features.dim() == 2:
            features = features.unsqueeze(0)
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        B, gamma, _ = features.shape
        feat_n = self.feat_norm(features)                       # (B, γ, F_in)
        emb = self.emb_norm(self.token_emb(token_ids.long()))   # (B, γ, D_emb)
        x = torch.cat([feat_n, emb], dim=-1)                    # (B, γ, F_in+D_emb)
        flat = x.reshape(B * gamma, self.in_dim)
        logits = self.net(flat).reshape(B, gamma)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        training_flag = self.training
        self.eval()
        try:
            return self.forward(features, token_ids=token_ids)
        finally:
            if training_flag:
                self.train()


class AcceptanceTxTokEmb(PredictorBase):
    """1B — small per-position transformer on (hidden_per_pos_v2) + token emb.

    Same shape of body as `AcceptanceTx` (PerPositionTransformerEncoder +
    Linear(d, 1) + sigmoid), same hyperparameters (d_model=128,
    num_layers=2, num_heads=4, dropout=0.1 by default), but with the
    extra embedded-token channel prepended to the input. Leading input
    LayerNorm is inside the shared encoder (post-LN-fix).
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
        token_vocab_size: int = DEFAULT_TOK_VOCAB_SIZE,
        token_emb_dim: int = 64,
    ) -> None:
        if family != "hidden_per_pos_v2":
            raise ValueError(
                f"AcceptanceTxTokEmb supports family='hidden_per_pos_v2' only; "
                f"got {family!r}."
            )
        super().__init__(family=family, deploy_mode="V-free", gamma=gamma)

        self.feat_in_dim = int(feature_dim(family))
        self.token_vocab_size = int(token_vocab_size)
        self.token_emb_dim = int(token_emb_dim)
        self.in_dim = self.feat_in_dim + self.token_emb_dim
        self.d_model = int(d_model)

        self.token_emb = nn.Embedding(self.token_vocab_size, self.token_emb_dim)
        self.encoder = PerPositionTransformerEncoder(
            in_dim=self.in_dim, gamma=gamma,
            d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.head = nn.Linear(self.d_model, 1)

    def forward(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError(
                "AcceptanceTxTokEmb.forward requires `token_ids` "
                "(per-position drafted-token IDs, long tensor)."
            )
        x, squeezed = _assemble_input(features, token_ids, self.token_emb)
        h = self.encoder(x)                                 # (B, γ, d)
        logits = self.head(h).squeeze(-1)                   # (B, γ)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        training_flag = self.training
        self.eval()
        try:
            return self.forward(features, token_ids=token_ids)
        finally:
            if training_flag:
                self.train()


class AcceptanceMLPPerPosTokEmbVHProj(PredictorBase):
    """Phase 19 Vlite-H — 1A + projected verifier-hidden prefix feature.

    Input contract: `hidden_per_pos_v4` feature block, shape (γ, F_v4)
    with F_v4 = drafter_hidden_dim (768) + 5 drafter scalars + 1600-d
    verifier-hidden-at-last-prefix-token broadcast to every draft
    position. The last-prefix verifier hidden is IDENTICAL across γ
    rows by construction (see `extract_features` / the online v4 path),
    so we slice one copy, learn a compact projection Linear(1600, vh_proj_dim),
    then broadcast the projection back to γ positions and concatenate
    with the drafter-side features + token embedding.

    Head body is the same shared-weight position-wise MLP as 1A, with
    a leading LayerNorm over (drafter_hidden + 5 + vh_proj_dim + token_emb_dim).
    No cross-position interaction.

    Inference cost: exactly ONE verifier prefix forward per round (same
    as hidden_per_pos_v3 / V-prefix mode). The projection is tiny
    (~102K params for 1600→64).
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        family: str,
        gamma: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        token_vocab_size: int = DEFAULT_TOK_VOCAB_SIZE,
        token_emb_dim: int = 64,
        verifier_hidden_dim: int = 1600,
        drafter_hidden_dim: int = 768,
        vh_proj_dim: int = 64,
    ) -> None:
        if family != "hidden_per_pos_v4":
            raise ValueError(
                f"AcceptanceMLPPerPosTokEmbVHProj supports family="
                f"'hidden_per_pos_v4' only; got {family!r}."
            )
        super().__init__(family=family, deploy_mode="V-prefix", gamma=gamma)

        self.drafter_hidden_dim = int(drafter_hidden_dim)
        self.verifier_hidden_dim = int(verifier_hidden_dim)
        self.n_drafter_scalars = 5
        self.token_vocab_size = int(token_vocab_size)
        self.token_emb_dim = int(token_emb_dim)
        self.vh_proj_dim = int(vh_proj_dim)

        # Input layout of the v4 feature block per position:
        #   [0, drafter_hidden_dim)                           drafter hidden
        #   [drafter_hidden_dim, drafter_hidden_dim+5)        5 scalars
        #   [drafter_hidden_dim+5, drafter_hidden_dim+5+H_v)  verifier hidden bc
        self._drafter_slice = self.drafter_hidden_dim + self.n_drafter_scalars
        self._vh_slice_lo = self._drafter_slice
        self._vh_slice_hi = self._drafter_slice + self.verifier_hidden_dim

        # Compact prefix-hidden summary + token embedding.
        self.v_proj = nn.Linear(self.verifier_hidden_dim, self.vh_proj_dim)
        self.token_emb = nn.Embedding(self.token_vocab_size, self.token_emb_dim)

        # Per-position input dim: drafter(H+5) + vh_proj + token_emb.
        self.per_pos_feat_dim = self._drafter_slice + self.vh_proj_dim
        self.in_dim = self.per_pos_feat_dim + self.token_emb_dim
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

    def _assemble(
        self,
        features: torch.Tensor,        # (B, γ, F_v4) or (γ, F_v4)
        token_ids: Optional[torch.Tensor],
    ) -> tuple:
        squeezed = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            if token_ids is not None and token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            squeezed = True
        if features.dim() != 3:
            raise ValueError(
                f"features must be (B, γ, F_v4) or (γ, F_v4); "
                f"got {tuple(features.shape)}"
            )
        B, gamma, F_v4 = features.shape
        expected_F = self._vh_slice_hi
        if F_v4 != expected_F:
            raise ValueError(
                f"feature dim {F_v4} != expected v4 dim {expected_F} "
                f"(drafter_hidden={self.drafter_hidden_dim} + 5 + "
                f"verifier_hidden={self.verifier_hidden_dim})"
            )
        drafter_part = features[..., :self._drafter_slice]         # (B, γ, 773)
        vh_bc = features[..., self._vh_slice_lo:self._vh_slice_hi]  # (B, γ, 1600)
        # All γ rows are the same vh broadcast; slice one and project once.
        vh_one = vh_bc[:, 0, :]                                     # (B, 1600)
        vh_proj = self.v_proj(vh_one)                               # (B, vh_proj_dim)
        vh_proj_bc = vh_proj.unsqueeze(1).expand(B, gamma, -1)      # (B, γ, D_proj)
        emb = self.token_emb(token_ids.long())                      # (B, γ, D_emb)
        x = torch.cat([drafter_part, vh_proj_bc, emb], dim=-1)      # (B, γ, in_dim)
        return x, squeezed, B, gamma

    def forward(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError(
                "AcceptanceMLPPerPosTokEmbVHProj.forward requires `token_ids`."
            )
        x, squeezed, B, gamma = self._assemble(features, token_ids)
        flat = x.reshape(B * gamma, self.in_dim)
        logits = self.net(flat).reshape(B, gamma)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        training_flag = self.training
        self.eval()
        try:
            return self.forward(features, token_ids=token_ids)
        finally:
            if training_flag:
                self.train()
