"""0505_OWT_compare — expected-survival weighted MSE loss.

Single primary loss for all three target families (relmax, alpha_q2, dep) and
both training regimes (frozen, joint). The choice of *target* selects what
goes into `y_target`; the choice of *regime* selects whether `alpha` is read
from a precomputed file (frozen) or computed live each batch (joint). The
loss formula is identical in either case.

Definition (per batch item):

    alpha_j = min(1, p_v(draft_j) / q_theta(draft_j))    (in [0, 1])
    w_j     = prod_{i < j} alpha_i                         (w_0 = 1.0)
    L_pred  = sum_j w_j * (y_hat_j - y_target_j)^2 / sum_j w_j

`alpha` is detached before the prefix product; `y_target` is detached by the
target builder. Gradients flow only through `y_hat`. In particular for
joint training, no gradient flows through the drafter or verifier (both run
under no_grad).

Why expected (smooth) survival weight rather than the sampled `survived_j`
(0/1) used by `accpre.train.losses.masked_mse_q2`:
- Sampled survival is high-variance: a single accept-test draw zeroes whole
  positions for a record.
- Expected survival is the mean of that 0/1 process, smooth and bounded.
- For frozen training, `alpha = record.min_pq_j` is already the bit-identical
  Leviathan ratio; this is "expected" by construction.
- For joint training, `alpha` is the live ratio computed from the current
  drafter and verifier — same source as the live `alpha_q2` target.

This module is self-contained: it imports only torch.
"""
from __future__ import annotations

import torch


def expected_survival_weights(alpha: torch.Tensor) -> torch.Tensor:
    """Return w_j = prod_{i<j} alpha_i (with w_0 = 1.0).

    Args:
        alpha: shape (B, gamma) or (gamma,), values clamped to [0, 1].

    Returns:
        Tensor of the same shape as `alpha`, detached. w[..., 0] = 1.
    """
    with torch.no_grad():
        a = alpha.clamp(min=0.0, max=1.0)
        if a.dim() == 0:
            raise ValueError("alpha must be at least 1-D (gamma,); got scalar")
        if a.dim() == 1:
            cum = torch.cumprod(a, dim=0)
            ones = torch.ones(1, dtype=a.dtype, device=a.device)
            w = torch.cat([ones, cum[:-1]], dim=0)
        else:
            cum = torch.cumprod(a, dim=-1)
            ones = torch.ones(*a.shape[:-1], 1, dtype=a.dtype, device=a.device)
            w = torch.cat([ones, cum[..., :-1]], dim=-1)
    return w.detach()


def expected_survival_weighted_mse(
    y_hat: torch.Tensor,         # (B, gamma) or (gamma,) in [0, 1]
    y_target: torch.Tensor,      # same shape as y_hat in [0, 1]
    alpha: torch.Tensor,         # same shape as y_hat in [0, 1]
    eps: float = 1e-9,
) -> torch.Tensor:
    """Expected-survival weighted MSE.

    Returns a scalar: mean over batch of (sum_j w_j (y_hat_j - y_target_j)^2)
    / (sum_j w_j + eps), where w_j = prod_{i<j} alpha_i.

    Shapes:
        y_hat / y_target / alpha must all be (B, gamma) or all be (gamma,).
        A 1-D input is treated as a single-item batch.

    Numerical safety:
        - alpha is clamped to [0, 1] before the cumulative product (cheap).
        - eps prevents divide-by-zero if every weight is 0 (extreme case
          where alpha[..., 0] = 0).
    """
    if y_hat.shape != y_target.shape or y_hat.shape != alpha.shape:
        raise ValueError(
            f"y_hat / y_target / alpha shape mismatch: "
            f"{tuple(y_hat.shape)} vs {tuple(y_target.shape)} vs {tuple(alpha.shape)}"
        )
    w = expected_survival_weights(alpha)
    se = (y_hat - y_target) ** 2 * w
    if se.dim() == 1:
        return se.sum() / (w.sum() + eps)
    per_item = se.sum(dim=-1) / (w.sum(dim=-1) + eps)
    return per_item.mean()


# Self-test reference for sanity scripts.
def _reference_weights_for_alpha(alpha_list: list[float]) -> list[float]:
    """Hand-computed reference: w_0 = 1, w_j = prod_{i<j} alpha_i."""
    out = [1.0]
    cum = 1.0
    for i, a in enumerate(alpha_list):
        cum *= float(a)
        if i + 1 < len(alpha_list):
            out.append(cum)
    return out


if __name__ == "__main__":
    # Tiny self-test; run as `python -m experiments.0505_OWT_compare.scripts.losses`
    # or `python scripts/losses.py` from the experiment folder.
    a = torch.tensor([0.9, 0.8, 0.5, 1.0])
    w = expected_survival_weights(a).tolist()
    expected = _reference_weights_for_alpha(a.tolist())
    print(f"alpha           = {a.tolist()}")
    print(f"weights (got)   = {w}")
    print(f"weights (expect)= {expected}")
    assert all(abs(x - y) < 1e-7 for x, y in zip(w, expected)), "weight mismatch"

    # Closed form: y_hat = 0.5 * ones, y_target = ones, alpha as above.
    # se per pos = 0.25, weighted sum = 0.25 * sum(w), denom = sum(w),
    # so ratio = 0.25.
    y_hat = torch.full_like(a, 0.5)
    y_target = torch.ones_like(a)
    loss = expected_survival_weighted_mse(y_hat, y_target, a)
    assert abs(loss.item() - 0.25) < 1e-7, f"loss {loss.item()} != 0.25"
    print(f"loss            = {loss.item():.6f} (expected 0.25)  OK")
