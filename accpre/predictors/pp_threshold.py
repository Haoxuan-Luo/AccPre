"""Phase 14 — direct threshold classifier (V-free, multi-head).

Tests whether *classifying* `1[Q2_j >= τ]` directly beats *regressing* Q2.
Same feature block as 1A (`hidden_per_pos_v2` + drafted-token embedding),
same per-position MLP body. The only change is the output: a single
shared trunk feeds K=len(TAU_GRID) sigmoid heads, one per threshold, and
deployment routes the caller's τ to the matching head.

This is explicitly NOT:
  - a transformer variant (body is the 1A MLP body),
  - a temperature-calibrated head (no calibration),
  - a weighted-MSE auxiliary (no regression target at all),
  - a length model (still feeds `commit_threshold` at deploy).

Labels at position j:
    y_j(τ)  =  1[record.min_pq_j[j]  >=  τ]        τ ∈ TAU_GRID

Loss: survived-masked BCE, averaged per head then averaged over heads.

Deploy rule (PredictorBase.predict_L override):
    for a caller-provided τ ∈ TAU_GRID:
        head_idx = TAU_GRID.index(τ)
        scores   = σ(logits[:, :, head_idx])       # (γ,) per example
        L_hat    = commit_threshold(scores, cutoff=0.5)

The `cutoff=0.5` threshold-on-the-classifier keeps the "contiguous
prefix of positive predictions" rule identical to what callers use
everywhere else in the pipeline — it is not a free hyperparameter.

TAU_GRID is a ClassVar tuple; changing it requires a retrain because
the output head count depends on it.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch
from torch import nn

from accpre.collect.features import feature_dim
from accpre.core.commit import commit_threshold
from accpre.predictors.base import PredictorBase
from accpre.predictors.pp_tokemb import DEFAULT_TOK_VOCAB_SIZE, _assemble_input


class AcceptanceThresholdMultiHead(PredictorBase):
    """V-free per-position multi-head threshold classifier.

    Shares the 1A body exactly; the only architectural delta is the
    final Linear's out-features (1 → K=len(TAU_GRID)).

    At inference, `predict_L` picks the head matching the caller's τ
    and applies `commit_threshold(scores, cutoff=0.5)`.
    """
    kind: ClassVar[str] = "acceptance"
    TAU_GRID: ClassVar[Tuple[float, ...]] = (0.5, 0.7, 0.9)

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
                f"AcceptanceThresholdMultiHead supports family='hidden_per_pos_v2' "
                f"only; got {family!r}."
            )
        super().__init__(family=family, deploy_mode="V-free", gamma=gamma)

        self.feat_in_dim = int(feature_dim(family))
        self.token_vocab_size = int(token_vocab_size)
        self.token_emb_dim = int(token_emb_dim)
        self.in_dim = self.feat_in_dim + self.token_emb_dim
        self.hidden_dim = int(hidden_dim)
        self.n_heads = len(self.TAU_GRID)

        self.token_emb = nn.Embedding(self.token_vocab_size, self.token_emb_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.n_heads),
        )

    def forward(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Return sigmoid scores of shape (B, γ, K) or (γ, K)."""
        if token_ids is None:
            raise ValueError(
                "AcceptanceThresholdMultiHead.forward requires `token_ids`."
            )
        x, squeezed = _assemble_input(features, token_ids, self.token_emb)
        B, gamma, F_in = x.shape
        flat = x.reshape(B * gamma, F_in)
        logits = self.net(flat).reshape(B, gamma, self.n_heads)
        out = torch.sigmoid(logits)
        return out.squeeze(0) if squeezed else out

    def predict_q2(
        self,
        features: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Threshold classifier does not estimate Q2.

        Raising here makes sure any downstream caller that expected a
        Q2 regressor fails loudly instead of silently feeding classifier
        scores into a `commit_threshold(Qhat, τ)` path that assumes
        regression semantics.
        """
        raise NotImplementedError(
            "AcceptanceThresholdMultiHead is a classifier over "
            f"1[Q2 >= τ] for τ ∈ {self.TAU_GRID}; call "
            "`predict_L(features, tau)` or `forward(...)` directly."
        )

    def _tau_to_idx(self, tau: float) -> int:
        for i, t in enumerate(self.TAU_GRID):
            if abs(float(tau) - float(t)) < 1e-6:
                return i
        raise ValueError(
            f"AcceptanceThresholdMultiHead only supports τ ∈ {self.TAU_GRID}; "
            f"got τ={tau}. Retrain with a larger TAU_GRID to extend."
        )

    def predict_L(
        self,
        features: torch.Tensor,
        tau: float,
        token_ids: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> int:
        """Route τ to the matching head and apply the prefix rule.

        Caller's τ MUST be in `TAU_GRID`; we assert this rather than
        interpolate so the evaluation convention stays unambiguous.
        """
        idx = self._tau_to_idx(float(tau))
        training_flag = self.training
        self.eval()
        try:
            scores = self.forward(features, token_ids=token_ids)
            # scores shape: (γ, K) when features was (γ, H); (B, γ, K) otherwise.
            if scores.dim() == 3:
                scores = scores.squeeze(0)
            head = scores[:, idx]                                    # (γ,)
            head_list = [float(x) for x in head.detach().cpu().tolist()]
            return commit_threshold(head_list, 0.5)
        finally:
            if training_flag:
                self.train()

    def describe(self):
        d = dict(super().describe())
        d["head_arch"] = "thresh_multihead"
        d["tau_grid"] = str(tuple(self.TAU_GRID))
        return d
