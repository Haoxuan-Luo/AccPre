"""Full-row 1-TV dependence target (Phase 23 branch (a)).

For each draft position j ∈ [0, γ), define a stability / dependence
score

    s_j = 1 - TV(Q^prefix_j,  Q^revealed_j)    ∈  [0, 1]

where

    Q^prefix_j(v)   := P_drafter(x_j = v  |  prefix,
                                             positions 0..γ-1 all MASK)
    Q^revealed_j(v) := P_drafter(x_j = v  |  prefix,
                                             positions 0..j-1 ← draft_tokens,
                                             positions j..γ-1 MASK)
    TV(P, Q)        := 0.5 · Σ_v |P(v) - Q(v)|    ∈  [0, 1]

By construction s_0 = 1.0 (the two inputs are identical when nothing
is revealed). Generally,

    s_j ≈ 1  ⇒ position j's distribution barely depends on the
               intermediate drafted tokens — position is "parallel-safe".
    s_j ≈ 0  ⇒ position j's distribution shifts a lot once earlier
               positions are revealed — position is "sequential".

Implementation notes
  * One batched drafter forward of γ items per record. Row 0 is the
    all-MASK input (prefix + γ MASKs); its slice at
    `prefix_len + j` for every j gives Q^prefix_j in a single row.
  * Row j (j ≥ 1) reveals positions 0..j-1 to `draft_tokens[0..j-1]`
    and leaves the rest MASK; its slice at `prefix_len + j` gives
    Q^revealed_j.
  * Both forwards use sigma = 1.0 (the drafter's step-0 timestep),
    so the prefix-only query matches the DDPM step-0 forward exactly.
    The SUBS parameterization pins unmasked positions to their
    values regardless of sigma, so the sigma choice mostly affects
    the revealed-case distribution — using 1.0 for both keeps the
    two queries on the same "MLM-style" footing.
  * No gradients flow through this target: it is a supervised label
    for the predictor head.
"""

from __future__ import annotations

import torch


def _apply_subs_batched(
    drafter,
    logits: torch.Tensor,   # (B, L, V)
    xt: torch.Tensor,       # (B, L) long
) -> torch.Tensor:
    """Batched SUBS parameterization — same math as
    `MDLMDrafter._subs_parameterization` applied row-wise.
    """
    NEG_INF = -1e9
    mask_index = drafter.mask_index
    logits = logits.clone()
    logits[..., mask_index] = NEG_INF
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked = xt != mask_index
    if unmasked.any():
        mask_rows = unmasked.unsqueeze(-1).expand_as(log_p)
        log_p = torch.where(
            mask_rows, torch.full_like(log_p, NEG_INF), log_p,
        )
        b_idx, l_idx = unmasked.nonzero(as_tuple=True)
        v_idx = xt[b_idx, l_idx].long()
        log_p[b_idx, l_idx, v_idx] = 0.0
    return log_p


def _build_dep_inputs(
    prefix_ids: torch.Tensor,    # (L_pref,) long on device
    draft_tokens: torch.Tensor,  # (γ,) long on device
    gamma: int,
    mask_idx: int,
) -> torch.Tensor:
    """Construct the (γ, L_pref + γ) batched input for the dependence forward.

    Row 0: [prefix, MASK × γ]                         (all-mask, Q^prefix source)
    Row j (1 ≤ j < γ):
           [prefix, dt[0..j-1], MASK × (γ - j)]       (Q^revealed_j source)
    """
    device = prefix_ids.device
    L_pref = int(prefix_ids.shape[0])
    total_len = L_pref + int(gamma)
    x = torch.full(
        (int(gamma), total_len), int(mask_idx),
        dtype=torch.long, device=device,
    )
    x[:, :L_pref] = prefix_ids.unsqueeze(0)
    for j in range(1, int(gamma)):
        x[j, L_pref:L_pref + j] = draft_tokens[:j]
    return x


def _trim_for_ctx(
    prefix_ids: torch.Tensor, gamma: int, max_seq_len: int,
) -> torch.Tensor:
    """If prefix + γ exceeds the drafter's context, left-trim the prefix.

    Mirrors `MDLMDrafter._run_ddpm_loop`'s trim logic so the dependence
    target is comparable to what the drafter sees when it decodes past
    the context window.
    """
    total = int(prefix_ids.shape[0]) + int(gamma)
    if total <= int(max_seq_len):
        return prefix_ids
    trim = total - int(max_seq_len)
    return prefix_ids[trim:]


@torch.no_grad()
def compute_dependence_target(
    drafter,
    prefix_ids: torch.Tensor,    # (L_pref,) long
    draft_tokens: torch.Tensor,  # (γ,) long
    gamma: int,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Return (γ,) float32 s_j = 1 - TV(Q^prefix_j, Q^revealed_j) on CPU.

    Expected:
      - prefix_ids and draft_tokens on `drafter.device`.
      - draft_tokens.shape[0] == gamma.
      - s[0] == 1.0 (up to numeric noise); s in [0, 1].
    """
    if int(draft_tokens.shape[0]) != int(gamma):
        raise ValueError(
            f"draft_tokens length {int(draft_tokens.shape[0])} != gamma={gamma}"
        )
    device = drafter.device
    prefix_ids = _trim_for_ctx(
        prefix_ids.to(device), int(gamma), drafter.max_seq_len,
    )
    L_pref = int(prefix_ids.shape[0])
    mask_idx = drafter.mask_index

    # Build the γ × L batched input.
    x = _build_dep_inputs(
        prefix_ids, draft_tokens.to(device).long(), int(gamma), mask_idx,
    )

    # Single batched forward; sigma = constant across the batch.
    sigma_t = torch.full(
        (int(gamma),), float(sigma), dtype=torch.float32, device=device,
    )
    out = drafter.model(input_ids=x, timesteps=sigma_t)
    logits = out.logits if hasattr(out, "logits") else out        # (γ, L, V)

    # SUBS-parameterize each row with its own xt.
    log_p = _apply_subs_batched(drafter, logits, x)               # (γ, L, V)

    # Row 0 at positions L_pref + j = Q^prefix_j (same row for all j).
    # Row j at position  L_pref + j = Q^revealed_j.
    idx_j = torch.arange(int(gamma), device=device)
    pos_j = L_pref + idx_j

    prefix_rows   = log_p[0, pos_j, :]          # (γ, V)
    revealed_rows = log_p[idx_j, pos_j, :]      # (γ, V)

    p_prefix   = prefix_rows.exp().to(torch.float32)
    p_revealed = revealed_rows.exp().to(torch.float32)
    tv = 0.5 * (p_prefix - p_revealed).abs().sum(dim=-1)   # (γ,)
    s = (1.0 - tv).clamp(min=0.0, max=1.0)                 # (γ,)
    return s.detach().cpu()
