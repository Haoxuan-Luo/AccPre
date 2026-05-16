"""0505_OWT_compare — predictor heads.

Three V-free heads. All consume the same per-position input bundle:

    features  (B, gamma, 773)        from extract_features('hidden_per_pos_v2', ...)
                                      = drafter hidden (768) + 5 scalars
    token_ids (B, gamma)             drafter-sampled token IDs at each position
    + j/gamma (B, gamma, 1)          per-position position scalar (built inside head)
    + token embedding (B, gamma, 64) learned, owned by the head
    --------------------------------
    = (B, gamma, 838) per-position bundle

`mlp_pos`                    : shared-weight position-wise MLP, no cross-position interaction.
`causal_transformer_pos`     : 2-layer causal Transformer over gamma positions
                               (j attends only to <= j), learned position embedding.
`bidirectional_transformer_pos`: same as causal but no causal mask.

All three subclass `accpre.predictors.base.PredictorBase` so the existing
online decode loop accepts them without changes (it dispatches on
`predictor.family` / `predictor.deploy_mode`).

V-free invariant: no head reads any verifier-derived feature. Forbidden inputs:
p_v, min_pq_j, accepted_j, verifier hidden, verifier prefix numerics, target
files. The shared `_assemble_input` is the single place per-position inputs
get built; an AST-grep sanity check (sanity_vfree_inference.py) verifies no
`verifier.` reference appears inside any head.

This module is in the experiment workspace; it does NOT modify accpre/.
"""
from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch
from torch import nn

# Import only the base class and the feature-dim helper from accpre.
# We do NOT subclass any production head — the position-scalar handling and
# the d_model=256 / causal-mask choices are layered in this file.
from accpre.predictors.base import PredictorBase, _V_FREE_FAMILIES
from accpre.collect.features import feature_dim


# Constants pinned for this experiment.
SUPPORTED_FAMILY: str = "hidden_per_pos_v2"   # H_d=768 + 5 scalars per position.
DRAFTER_HIDDEN_DIM: int = 768                  # MDLM-OWT.
TOKEN_VOCAB_SIZE: int = 50258                  # MDLM = GPT-2 (50257) + MASK.
DEFAULT_TOKEN_EMB_DIM: int = 64
DEFAULT_GAMMA: int = 15

# Each head's input width after we concat features + j/gamma + token_emb.
# = feature_dim('hidden_per_pos_v2', 768) + 1 + token_emb_dim
#   = 773 + 1 + 64 = 838 by default.
def _head_in_dim(token_emb_dim: int) -> int:
    return int(feature_dim(SUPPORTED_FAMILY, drafter_hidden_dim=DRAFTER_HIDDEN_DIM)) + 1 + int(token_emb_dim)


# ----------------------------------------------------------------------
# Shared per-position input assembly.
# ----------------------------------------------------------------------


def _assemble_input(
    features: torch.Tensor,         # (B, gamma, F_v2=773) or (gamma, F_v2)
    token_ids: torch.Tensor,        # (B, gamma) or (gamma,) long
    pos_scalar: torch.Tensor,       # (gamma,) buffer, value j/gamma in [0, 1)
    token_emb: nn.Embedding,
) -> Tuple[torch.Tensor, bool]:
    """Build the (B, gamma, F_in) per-position bundle used by every head.

    Returns (assembled, squeezed_back). `squeezed_back` is True iff the caller
    handed in unbatched (gamma, F_v2) input and expects (gamma,) back.
    """
    if features.dim() == 2:
        features = features.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    if features.dim() != 3:
        raise ValueError(
            f"features must be (B, gamma, F) or (gamma, F); got {tuple(features.shape)}"
        )
    if token_ids.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"token_ids batch/gamma {tuple(token_ids.shape)} != "
            f"features batch/gamma {tuple(features.shape[:2])}"
        )
    B, gamma, _ = features.shape
    if pos_scalar.shape[0] < gamma:
        raise ValueError(
            f"pos_scalar buffer length {pos_scalar.shape[0]} < gamma={gamma}; "
            f"the head's gamma must be >= the runtime gamma."
        )
    pos = pos_scalar[:gamma].to(features.dtype).to(features.device)        # (gamma,)
    pos_b = pos.view(1, gamma, 1).expand(B, gamma, 1)                       # (B, gamma, 1)
    emb = token_emb(token_ids.long())                                       # (B, gamma, E)
    x = torch.cat([features, pos_b, emb], dim=-1)                           # (B, gamma, F_in)
    return x, squeezed


# ----------------------------------------------------------------------
# 1. mlp_pos — pointwise MLP, no cross-position interaction.
# ----------------------------------------------------------------------


class MLPPosHead(PredictorBase):
    """Shared-weight per-position MLP head with explicit j/gamma scalar.

    Architecture (per position, weights shared across positions):
        LayerNorm(F_in)
        Linear(F_in, hidden_dim) + ReLU + Dropout
        Linear(hidden_dim, hidden_dim) + ReLU + Dropout
        Linear(hidden_dim, 1) + sigmoid

    `F_in = 773 + 1 (j/gamma) + token_emb_dim`. No cross-position attention;
    position information enters only via the j/gamma scalar.

    Frozen-only architecture in this experiment (see architecture_registry.md
    for rationale). Joint training of this head is not run here, but no code
    blocks it — the trainer dispatches by arch name only.
    """

    kind: ClassVar[str] = "acceptance"

    def __init__(
        self,
        gamma: int = DEFAULT_GAMMA,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        token_emb_dim: int = DEFAULT_TOKEN_EMB_DIM,
        token_vocab_size: int = TOKEN_VOCAB_SIZE,
    ) -> None:
        if SUPPORTED_FAMILY not in _V_FREE_FAMILIES:
            raise AssertionError(
                f"{SUPPORTED_FAMILY!r} expected to be V-free; package invariant violated"
            )
        super().__init__(family=SUPPORTED_FAMILY, deploy_mode="V-free", gamma=int(gamma))

        self.token_emb_dim = int(token_emb_dim)
        self.token_vocab_size = int(token_vocab_size)
        self.in_dim = _head_in_dim(self.token_emb_dim)
        self.hidden_dim = int(hidden_dim)

        self.token_emb = nn.Embedding(self.token_vocab_size, self.token_emb_dim)
        self.register_buffer(
            "pos_scalar",
            torch.arange(int(gamma), dtype=torch.float32) / float(gamma),
            persistent=False,
        )
        self.body = nn.Sequential(
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
            raise ValueError("MLPPosHead.forward requires `token_ids`.")
        x, squeezed = _assemble_input(features, token_ids, self.pos_scalar, self.token_emb)
        B, gamma, F_in = x.shape
        flat = x.reshape(B * gamma, F_in)
        logits = self.body(flat).reshape(B, gamma)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            return self.forward(features, **kwargs)
        finally:
            if was_training:
                self.train()


# ----------------------------------------------------------------------
# 2. causal_transformer_pos / bidirectional_transformer_pos.
# ----------------------------------------------------------------------


class _PerPositionTransformerEncoder(nn.Module):
    """2-layer Transformer encoder over gamma positions with learned position embedding.

    This class is a thin local copy of accpre.predictors.pp_transformer.
    PerPositionTransformerEncoder, modified to (a) default to d_model=256
    (the existing module is fixed at 128), and (b) take a `causal: bool`
    flag that controls whether a causal (upper-triangular −inf) mask is
    passed to the underlying TransformerEncoder forward.

    Kept inline in this file rather than imported because the causal-mask
    path is the central correctness invariant for `causal_transformer_pos`
    and we want it visible right next to the head class that uses it.
    """

    def __init__(
        self,
        in_dim: int,
        gamma: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.gamma = int(gamma)
        self.d_model = int(d_model)
        self.causal = bool(causal)

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

        if self.causal:
            # Upper-triangular -inf mask of shape (gamma, gamma).
            # mask[i, j] = -inf if j > i (forbid j attending forward); 0 otherwise.
            mask = torch.full(
                (self.gamma, self.gamma), float("-inf"), dtype=torch.float32
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.causal_mask = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"encoder expects (B, gamma, in_dim); got {tuple(x.shape)}"
            )
        B, gamma, F = x.shape
        if F != self.in_dim:
            raise ValueError(f"input feature dim {F} != encoder in_dim {self.in_dim}")
        if gamma > self.gamma:
            raise ValueError(
                f"input gamma={gamma} exceeds encoder's gamma={self.gamma}"
            )
        h = self.input_norm(x)
        h = self.input_proj(h)
        positions = torch.arange(gamma, device=x.device)
        h = h + self.pos_emb(positions).unsqueeze(0)
        if self.causal:
            mask = self.causal_mask
            if gamma != self.gamma:
                mask = mask[:gamma, :gamma]
            return self.encoder(h, mask=mask)
        return self.encoder(h)


class _TransformerPosHead(PredictorBase):
    """Common implementation for causal_transformer_pos and bidirectional_transformer_pos.

    Subclasses set `_causal: bool` and pick a name. Architecture:

        _assemble_input(features, token_ids, pos_scalar, token_emb)        (B, gamma, 838)
        _PerPositionTransformerEncoder(in_dim=838, d_model=256,
                                       num_layers=2, num_heads=4,
                                       dropout=0.1, causal=...)            (B, gamma, 256)
        Linear(256, 1) + sigmoid                                           (B, gamma)
    """

    kind: ClassVar[str] = "acceptance"
    _causal: ClassVar[bool] = False  # subclass overrides

    def __init__(
        self,
        gamma: int = DEFAULT_GAMMA,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        token_emb_dim: int = DEFAULT_TOKEN_EMB_DIM,
        token_vocab_size: int = TOKEN_VOCAB_SIZE,
    ) -> None:
        super().__init__(family=SUPPORTED_FAMILY, deploy_mode="V-free", gamma=int(gamma))

        self.token_emb_dim = int(token_emb_dim)
        self.token_vocab_size = int(token_vocab_size)
        self.in_dim = _head_in_dim(self.token_emb_dim)
        self.d_model = int(d_model)

        self.token_emb = nn.Embedding(self.token_vocab_size, self.token_emb_dim)
        self.register_buffer(
            "pos_scalar",
            torch.arange(int(gamma), dtype=torch.float32) / float(gamma),
            persistent=False,
        )
        self.encoder = _PerPositionTransformerEncoder(
            in_dim=self.in_dim,
            gamma=int(gamma),
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            causal=bool(self._causal),
        )
        self.head = nn.Linear(self.d_model, 1)

    def forward(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError(f"{type(self).__name__}.forward requires `token_ids`.")
        x, squeezed = _assemble_input(features, token_ids, self.pos_scalar, self.token_emb)
        h = self.encoder(x)                              # (B, gamma, d_model)
        logits = self.head(h).squeeze(-1)                # (B, gamma)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            return self.forward(features, **kwargs)
        finally:
            if was_training:
                self.train()


class CausalTransformerPosHead(_TransformerPosHead):
    """2-layer causal Transformer head; j attends only to <= j."""
    _causal: ClassVar[bool] = True


class BidirTransformerPosHead(_TransformerPosHead):
    """2-layer bidirectional Transformer head; j attends to all gamma positions."""
    _causal: ClassVar[bool] = False


# ----------------------------------------------------------------------
# Factory.
# ----------------------------------------------------------------------


_ARCH_TABLE = {
    "mlp_pos": MLPPosHead,
    "causal_transformer_pos": CausalTransformerPosHead,
    "bidirectional_transformer_pos": BidirTransformerPosHead,
}


def build_head(
    arch: str,
    gamma: int = DEFAULT_GAMMA,
    *,
    hidden_dim: int = 128,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    token_emb_dim: int = DEFAULT_TOKEN_EMB_DIM,
    token_vocab_size: int = TOKEN_VOCAB_SIZE,
) -> nn.Module:
    """Build one of the three head architectures.

    `arch` selects mlp_pos / causal_transformer_pos / bidirectional_transformer_pos.
    `hidden_dim` applies to mlp_pos only; transformer hyperparameters apply to the
    other two only. Unused kwargs are ignored for the chosen arch.
    """
    if arch not in _ARCH_TABLE:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCH_TABLE)}")
    if arch == "mlp_pos":
        return MLPPosHead(
            gamma=gamma,
            hidden_dim=hidden_dim,
            dropout=dropout,
            token_emb_dim=token_emb_dim,
            token_vocab_size=token_vocab_size,
        )
    cls = _ARCH_TABLE[arch]
    return cls(
        gamma=gamma,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        token_emb_dim=token_emb_dim,
        token_vocab_size=token_vocab_size,
    )


def head_in_dim(token_emb_dim: int = DEFAULT_TOKEN_EMB_DIM) -> int:
    """Public helper for sanity scripts and tests."""
    return _head_in_dim(token_emb_dim)
