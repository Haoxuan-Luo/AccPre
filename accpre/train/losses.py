"""Predictor training losses.

Three primary losses in Phase 2:
  - masked_mse_q2    (acceptance, primary)
  - masked_bce_accepted (acceptance, auxiliary — off by default)
  - ce_length        (committed-length, primary)

Plus two multitask combiners (Stage 2B only):
  - weighted_sum       (Stage 2B.2 fixed weights)
  - homoscedastic_sum  (Stage 2B.3 learnable log σ_k, Kendall & Gal)

All masked losses use `survived_j == 1` as the mask — positions that
strict never reached have no defined conditional-acceptance label and
should not contribute gradient.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def masked_mse_q2(
    q2_hat: torch.Tensor,
    q2_target: torch.Tensor,
    survived: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error on Q2, masked by `survived`.

    Args:
        q2_hat:    (batch, γ) predicted Q̂ ∈ [0, 1]
        q2_target: (batch, γ) canonical Q2 = min(1, p/q)
        survived:  (batch, γ) 0/1 mask, float

    Returns:
        scalar mean squared error over survived positions.
    """
    se = (q2_hat - q2_target) ** 2
    masked = se * survived
    denom = survived.sum().clamp(min=1e-9)
    return masked.sum() / denom


def masked_threshold_bce_q2(
    q2_hat: torch.Tensor,
    q2_target: torch.Tensor,
    survived: torch.Tensor,
    taus=(0.5, 0.7, 0.9),
    sharpness: float = 10.0,
) -> torch.Tensor:
    """Threshold-aware BCE auxiliary on `1[Q2 ≥ τ]` for τ in the target set.

    Per position j and per τ, define
        z_{j,τ} = 1[Q2_target_j ≥ τ]
        p_{j,τ} = σ( sharpness · (Q̂_j − τ) )        (soft indicator "Q̂ ≥ τ")
        bce_{j,τ} = -[ z · log p + (1 − z) · log(1 − p) ]

    Returns the survived-masked mean of bce, averaged over the τ set. Keep
    `sharpness` moderate (~10) so the indicator is soft enough for stable
    gradients but sharp enough to actually penalise wrong-side predictions.
    Intended as a **weak auxiliary** added to the main Q2 regression loss:
    scale the return with a small λ at the call site.
    """
    if q2_hat.dim() != q2_target.dim():
        raise ValueError(
            f"q2_hat {tuple(q2_hat.shape)} and q2_target "
            f"{tuple(q2_target.shape)} must match"
        )
    eps = 1e-7
    mask = survived
    denom = mask.sum().clamp(min=1e-9)

    total = q2_hat.new_zeros(())
    k = float(sharpness)
    for tau in taus:
        tau_t = float(tau)
        z = (q2_target >= tau_t).float()
        p = torch.sigmoid(k * (q2_hat - tau_t)).clamp(eps, 1.0 - eps)
        bce = -(z * torch.log(p) + (1.0 - z) * torch.log(1.0 - p))
        total = total + (bce * mask).sum() / denom
    return total / float(len(taus))


def masked_weighted_mse_q2(
    q2_hat: torch.Tensor,
    q2_target: torch.Tensor,
    survived: torch.Tensor,
    q_weight_lambda: float = 0.0,
) -> torch.Tensor:
    """MSE on Q2 with a mild per-position weight `1 + λ · Q2_target`.

    `q_weight_lambda=0.0` recovers uniform `masked_mse_q2`. Positive λ
    upweights positions with high Q2 (close to 1 — "easier to accept"),
    biasing the regressor toward fitting the high-Q tail more
    accurately. This is the target regime for commit_threshold at
    τ ≥ 0.5 — we care most about whether Q̂ ≥ τ is correctly ranked in
    the high-Q regime where commits actually happen.

    Still masked by `survived`. Weight normalisation: the denominator
    uses `sum(survived · w)` so the overall loss scale is comparable to
    vanilla masked MSE.
    """
    w = 1.0 + float(q_weight_lambda) * q2_target.clamp(0.0, 1.0)
    se = (q2_hat - q2_target) ** 2
    mask = survived * w
    denom = mask.sum().clamp(min=1e-9)
    return (se * mask).sum() / denom


def masked_multi_thresh_bce(
    y_hat: torch.Tensor,          # (B, γ, K) sigmoid scores from the K heads
    q2_target: torch.Tensor,      # (B, γ) canonical Q2 = min(1, p/q)
    survived: torch.Tensor,       # (B, γ) 0/1 mask
    taus=(0.5, 0.7, 0.9),
) -> torch.Tensor:
    """Multi-head BCE on `1[Q2 >= τ]` for τ in `taus`.

    Per-head survived-masked BCE (each head normalised by its own
    masked count so imbalance across heads doesn't skew the overall
    gradient), then averaged across heads.

    Shape contract: `y_hat[..., k]` corresponds to threshold `taus[k]`.
    The classifier that owns `y_hat` is responsible for keeping the
    head-to-τ mapping stable at deploy time (see
    `AcceptanceThresholdMultiHead.TAU_GRID`).
    """
    if y_hat.dim() != q2_target.dim() + 1:
        raise ValueError(
            f"y_hat {tuple(y_hat.shape)} expected to be q2_target "
            f"{tuple(q2_target.shape)} with a trailing K-axis"
        )
    if y_hat.shape[-1] != len(taus):
        raise ValueError(
            f"y_hat last dim {y_hat.shape[-1]} != len(taus)={len(taus)}"
        )
    eps = 1e-7
    denom = survived.sum().clamp(min=1e-9)
    total = y_hat.new_zeros(())
    for k, tau in enumerate(taus):
        z = (q2_target >= float(tau)).float()
        p = y_hat[..., k].clamp(eps, 1.0 - eps)
        bce = -(z * torch.log(p) + (1.0 - z) * torch.log(1.0 - p))
        total = total + (bce * survived).sum() / denom
    return total / float(len(taus))


def masked_bce_accepted(
    q2_hat: torch.Tensor,
    accepted: torch.Tensor,
    survived: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy against `accepted_j`, masked by `survived`.

    Auxiliary target — links the soft prediction to the Bernoulli
    sample. Off by default in v1.
    """
    eps = 1e-7
    q = q2_hat.clamp(eps, 1.0 - eps)
    bce = -(accepted * torch.log(q) + (1.0 - accepted) * torch.log(1.0 - q))
    masked = bce * survived
    denom = survived.sum().clamp(min=1e-9)
    return masked.sum() / denom


def masked_soft_bce_q2(
    q2_hat: torch.Tensor,
    q2_target: torch.Tensor,
    survived: torch.Tensor,
) -> torch.Tensor:
    """Soft binary cross-entropy with the continuous `Q2` target.

    `L_j = -[ Q2_j · log Q̂_j + (1 − Q2_j) · log(1 − Q̂_j) ]`

    This is BCE treating Q2 ∈ [0, 1] as a *soft target probability*,
    NOT as a binary label. Unlike MSE, it has large gradient magnitude
    near the boundaries (`Q̂ → 0` or `Q̂ → 1`), which pushes predictions
    toward 0/1 rather than collapsing to the mean — the specific
    failure mode MSE exhibited on the bimodal Q2 distribution in Phase
    2A (see debug report, §1).

    Masked by `survived`, same convention as `masked_mse_q2`.
    """
    eps = 1e-7
    q = q2_hat.clamp(eps, 1.0 - eps)
    loss = -(q2_target * torch.log(q) + (1.0 - q2_target) * torch.log(1.0 - q))
    masked = loss * survived
    denom = survived.sum().clamp(min=1e-9)
    return masked.sum() / denom


def ce_length(
    logits: torch.Tensor,
    L_target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy for categorical L prediction.

    Args:
        logits:        (batch, γ+1)
        L_target:      (batch,) long in [0, γ]
        class_weights: optional (γ+1,) float tensor of per-class weights
                       (torch's `cross_entropy(weight=...)` convention).

    If `class_weights` is provided, this is the weighted-CE variant used
    for the Stage-2A length fix (inverse-frequency weighting; see
    `compute_length_class_weights`). Otherwise plain mean-reduction CE.
    """
    return nn.functional.cross_entropy(logits, L_target, weight=class_weights)


def compute_length_class_weights(
    records,
    tau_grid,
    gamma: int,
    cap: float = 10.0,
) -> torch.Tensor:
    """Inverse-frequency class weights for committed-length CE.

    Iterates over `records × tau_grid`, counts `L_τ^oracle` values, and
    returns a `(γ+1,)` float tensor of weights normalised so the mean
    weight across classes is 1. Rare classes receive larger weight;
    classes with zero count receive weight 0 (so CE ignores them, since
    they never appear as targets anyway). The final weights are capped
    at `cap` to avoid extreme imbalance amplifying rare-class noise.
    """
    from accpre.core.commit import commit_threshold

    n_classes = int(gamma) + 1
    counts = [0.0] * n_classes
    total = 0.0
    for r in records:
        for t in tau_grid:
            L = commit_threshold(r.min_pq_j, float(t))
            counts[int(L)] += 1.0
            total += 1.0

    # Inverse-frequency weighting: w_c = N / (K_nonzero * count_c).
    k_nonzero = sum(1 for c in counts if c > 0)
    weights = [0.0] * n_classes
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = total / (k_nonzero * counts[c])
    # Cap.
    weights = [min(w, float(cap)) for w in weights]
    # Renormalise so mean of nonzero weights is 1.
    nonzero_sum = sum(w for w in weights if w > 0)
    if k_nonzero > 0 and nonzero_sum > 0:
        scale = k_nonzero / nonzero_sum
        weights = [w * scale for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def weighted_sum(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """Fixed-weight multitask combiner (Stage 2B.2)."""
    total = None
    for k, v in losses.items():
        w = float(weights.get(k, 1.0))
        contrib = w * v
        total = contrib if total is None else total + contrib
    assert total is not None
    return total


class HomoscedasticWeights(nn.Module):
    """Learnable log-sigma^2 weights per task (Kendall & Gal 2018).

    L = Σ_k [ (1 / (2 σ_k²)) · L_k + log σ_k ]

    Stored as `log_sigma` so `σ = exp(log_sigma) > 0` at all times.
    """

    def __init__(self, task_names: tuple) -> None:
        super().__init__()
        self.task_names = tuple(task_names)
        self.log_sigma = nn.Parameter(torch.zeros(len(self.task_names)))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for i, name in enumerate(self.task_names):
            if name not in losses:
                raise KeyError(
                    f"HomoscedasticWeights: task {name!r} missing from losses "
                    f"(got {list(losses.keys())})."
                )
            sigma_sq = (2.0 * self.log_sigma[i]).exp()
            term = losses[name] / sigma_sq + self.log_sigma[i]
            total = term if total is None else total + term
        assert total is not None
        return total
