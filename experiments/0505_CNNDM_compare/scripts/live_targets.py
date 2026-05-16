"""0505_OWT_compare — live target and survival-weight computation.

Joint training requires per-batch live computation of (a) the regression
target y_j for the configured target family and (b) the per-position
acceptance probability alpha_j = min(1, p_v / q_theta) used as the
expected-survival weight in the loss.

For target='alpha_q2' and target='relmax', both y and alpha use the verifier
softmax row at position j; we therefore share one verifier forward between
them. For target='dep', y is drafter-only (1 - TV) and we still need alpha
for the survival weight, which costs one verifier forward per record.

V-free invariant:
    These functions are called ONLY at training time (joint regime) and at
    offline evaluation time (post-hoc verifier NLL pass). They are NEVER
    called inside the V-free online decode loop. The eval driver's round
    loop must contain zero verifier references; the post-hoc NLL pass is
    structurally outside that loop.

Imports from accpre are deliberate and minimal:
- `compute_dependence_target` for the 1-TV dep target (drafter-only).
- `_compute_live_q2` is NOT imported — we re-implement the live alpha_q2
  inline here so this module is the single source of all three live
  computations and the verifier-forward sharing is explicit.

This module is self-contained except for the `compute_dependence_target`
import; it does NOT modify accpre/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from accpre.core.dependence import compute_dependence_target


EPS: float = 1e-10


@dataclass
class LiveTargetOutputs:
    """Per-record outputs of one live-target call.

    Shapes are (gamma,) and tensors live on the same device as the inputs.
    All fields are detached (no grad).
    """
    y_target: torch.Tensor    # the regression target the head is supervised against
    alpha: torch.Tensor       # min(1, p_v / q_theta), used for the survival weight
    target_log_probs: Optional[torch.Tensor] = None
    """If a verifier forward was performed, the resulting (L+gamma, V_v) log_probs
    on the verifier device, fp32. None for target='dep' if no verifier forward
    was made (the survival weight still requires one, so this is always populated
    in practice — kept Optional for readability)."""


@torch.no_grad()
def _verifier_log_probs(
    verifier,
    prefix_ids: torch.Tensor,    # (L_pref,) long, on verifier device
    draft_tokens: torch.Tensor,  # (gamma,) long, on verifier device
) -> torch.Tensor:
    """One verifier forward over (prefix, draft_tokens). Returns (L+gamma, V_v)
    fp32 log-probabilities on the verifier device.

    The verifier is loaded on its own device (typically GPU); inputs must already
    be on that device. The returned tensor is fp32 even if the verifier is bf16:
    the downstream ratio + min are evaluated in fp32 to match the offline target
    builders (which are also fp32 after the exp + clamp).
    """
    candidate = torch.cat([prefix_ids.long(), draft_tokens.long()], dim=0)
    log_probs = verifier.score(candidate)              # (L+gamma, V_v)
    return log_probs.to(torch.float32)


@torch.no_grad()
def _alpha_from_verifier_log_probs(
    target_log_probs: torch.Tensor,    # (L+gamma, V_v) fp32 on verifier device
    draft_tokens: torch.Tensor,        # (gamma,) long on same device
    draft_log_probs: torch.Tensor,     # (gamma, V_d) — drafter log-probs row per position
    prefix_len: int,
    gamma: int,
) -> torch.Tensor:
    """alpha_j = min(1, p_v(draft_j) / q_theta(draft_j)) on the verifier device.

    `draft_log_probs` is detached and cast to fp32 before the q lookup; the
    returned tensor is fp32, shape (gamma,), detached.
    """
    device = target_log_probs.device
    idx = torch.arange(int(gamma), device=device)
    target_pos = int(prefix_len) - 1 + idx                  # (gamma,)
    p_draft = (
        target_log_probs[target_pos, draft_tokens.long()]
        .exp()
        .clamp(min=EPS)
    )                                                       # (gamma,)
    q_theta = (
        draft_log_probs.detach().to(torch.float32).to(device)[
            idx, draft_tokens.long()
        ]
        .exp()
        .clamp(min=EPS)
    )                                                       # (gamma,)
    alpha = torch.minimum(torch.ones_like(p_draft), p_draft / q_theta)
    return alpha.detach()


@torch.no_grad()
def compute_live_alpha_q2(
    verifier,
    prefix_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_log_probs: torch.Tensor,
) -> LiveTargetOutputs:
    """Live target = alpha_q2 = min(1, p_v / q_theta). Same as alpha.

    Equivalent to `accpre.train.cli._compute_live_q2`, but reformulated so
    the (target, alpha) pair is returned as a single object alongside the
    cached verifier forward.
    """
    device = verifier.device
    prefix_ids_d = prefix_ids.to(device).long()
    draft_tokens_d = draft_tokens.to(device).long()
    target_log_probs = _verifier_log_probs(verifier, prefix_ids_d, draft_tokens_d)
    gamma = int(draft_tokens.shape[0])
    prefix_len = int(prefix_ids.shape[0])
    alpha = _alpha_from_verifier_log_probs(
        target_log_probs, draft_tokens_d, draft_log_probs, prefix_len, gamma,
    )
    # Target IS alpha for alpha_q2.
    return LiveTargetOutputs(
        y_target=alpha.clone(),
        alpha=alpha,
        target_log_probs=target_log_probs,
    )


@torch.no_grad()
def compute_live_relmax(
    verifier,
    prefix_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_log_probs: torch.Tensor,
) -> LiveTargetOutputs:
    """Live target = p_v(draft_j) / p_v(argmax_v_j) under the verifier softmax,
    plus alpha computed from the same verifier forward.

    Both outputs are on the verifier's device, fp32, detached, shape (gamma,).
    """
    device = verifier.device
    prefix_ids_d = prefix_ids.to(device).long()
    draft_tokens_d = draft_tokens.to(device).long()
    target_log_probs = _verifier_log_probs(verifier, prefix_ids_d, draft_tokens_d)

    gamma = int(draft_tokens.shape[0])
    prefix_len = int(prefix_ids.shape[0])
    idx = torch.arange(gamma, device=device)
    target_pos = prefix_len - 1 + idx                        # (gamma,)

    p_draft = (
        target_log_probs[target_pos, draft_tokens_d].exp().clamp(min=EPS)
    )                                                        # (gamma,)
    # max_v p_v(v at j) = exp(max(log p_v(v at j))) since exp is monotone.
    p_max = (
        target_log_probs[target_pos].max(dim=-1).values.exp().clamp(min=EPS)
    )                                                        # (gamma,)
    # By construction p_draft <= p_max, so the ratio is in (0, 1]; clamp for safety.
    y = (p_draft / p_max).clamp(min=0.0, max=1.0).detach()

    alpha = _alpha_from_verifier_log_probs(
        target_log_probs, draft_tokens_d, draft_log_probs, prefix_len, gamma,
    )
    return LiveTargetOutputs(
        y_target=y,
        alpha=alpha,
        target_log_probs=target_log_probs,
    )


@torch.no_grad()
def compute_live_dep(
    drafter,
    verifier,
    prefix_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_log_probs: torch.Tensor,
    gamma: int,
    sigma: float = 1.0,
) -> LiveTargetOutputs:
    """Live target = 1 - TV(Q_prefix_j, Q_revealed_j) (drafter-only); alpha
    requires a verifier forward.

    Two model forwards happen here:
      1. drafter batched gamma-row forward (for the dep target itself, via
         `accpre.core.dependence.compute_dependence_target`).
      2. verifier forward over (prefix, draft_tokens) (for alpha).

    For dep-joint training the verifier is genuinely required at training
    time only because we need alpha for the survival weight; the dep target
    itself never references the verifier. (The alternative — replacing alpha
    with a self-consistency-based weight — is documented in target_registry.md
    §4 as a possible follow-on. Not done here.)

    Returns LiveTargetOutputs with y_target on the drafter's device after a
    detach + .to(device) handled by the caller (we keep tensors on the
    devices their model produced them on; the trainer aligns devices).
    """
    # Dep target via the existing drafter helper.
    y = compute_dependence_target(
        drafter, prefix_ids.to(drafter.device), draft_tokens.to(drafter.device).long(), int(gamma), sigma=float(sigma),
    )                                                          # (gamma,) fp32 on CPU per accpre signature
    y = y.detach()

    # Alpha via verifier.
    device = verifier.device
    prefix_ids_d = prefix_ids.to(device).long()
    draft_tokens_d = draft_tokens.to(device).long()
    target_log_probs = _verifier_log_probs(verifier, prefix_ids_d, draft_tokens_d)
    prefix_len = int(prefix_ids.shape[0])
    alpha = _alpha_from_verifier_log_probs(
        target_log_probs, draft_tokens_d, draft_log_probs, prefix_len, int(gamma),
    )
    return LiveTargetOutputs(
        y_target=y,
        alpha=alpha,
        target_log_probs=target_log_probs,
    )


def compute_live_target(
    target_name: str,
    *,
    drafter,
    verifier,
    prefix_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_log_probs: torch.Tensor,
    gamma: int,
) -> LiveTargetOutputs:
    """Single dispatch point used by the joint trainer.

    Always returns y_target AND alpha; both are detached. The trainer then
    computes the loss with `expected_survival_weighted_mse(y_hat, y_target, alpha)`.

    The target_name MUST be one of {'relmax', 'alpha_q2', 'dep'}. The trainer
    enforces this via cfg-level assertion before this function is called.
    """
    if target_name == "alpha_q2":
        return compute_live_alpha_q2(verifier, prefix_ids, draft_tokens, draft_log_probs)
    if target_name == "relmax":
        return compute_live_relmax(verifier, prefix_ids, draft_tokens, draft_log_probs)
    if target_name == "dep":
        return compute_live_dep(drafter, verifier, prefix_ids, draft_tokens, draft_log_probs, gamma)
    raise ValueError(
        f"unknown target_name {target_name!r}; "
        f"expected one of {{'relmax', 'alpha_q2', 'dep'}}"
    )
