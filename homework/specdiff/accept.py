"""Canonical per-position accept logic.

Vendored from `accpre/core/accept.py`. Single implementation of the
Leviathan speculative-decoding accept test.

Matched-randomness discipline:
  - `accept_rng` is a torch.Generator seeded from
        round_rng_seed XOR protocol.accept_salt
    inside `draft_verify._make_generator`.
  - Under temperature > 0, `U_j[j]` is drawn from `accept_rng` for every j
    in [0, gamma), regardless of the calling lane's commit rule.
  - Under temperature == 0, acceptance is deterministic
    `argmax(target) == draft_tok`; U_j is still drawn (and logged) to keep
    the stream position fixed across lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch

from .protocol import ProtocolConfig


@dataclass
class AcceptOutcome:
    """Per-position outputs of the accept test (length = gamma)."""
    accepted_j: List[int]
    survived_j: List[int]
    q_j: List[float]
    p_j: List[float]
    min_pq_j: List[float]
    U_j: List[float]


def per_position_accept(
    draft_tokens: torch.Tensor,        # (gamma,) long
    draft_log_probs: torch.Tensor,     # (gamma, vocab_drafter)
    target_log_probs: torch.Tensor,    # (seq_len_total, vocab_verifier)
    prefix_len: int,
    protocol: ProtocolConfig,
    accept_rng: torch.Generator,
) -> AcceptOutcome:
    """Leviathan per-position accept test."""
    gamma = int(draft_tokens.shape[0])
    EPS = 1e-10

    accepted_j: List[int] = []
    survived_j: List[int] = []
    q_j: List[float] = []
    p_j: List[float] = []
    min_pq_j: List[float] = []
    U_j: List[float] = []

    alive = True
    for j in range(gamma):
        target_pos = prefix_len - 1 + j
        tok = int(draft_tokens[j].item())

        q_val = float(draft_log_probs[j, tok].exp().clamp(min=EPS).item())
        p_val = float(target_log_probs[target_pos, tok].exp().clamp(min=EPS).item())
        ratio = min(1.0, p_val / q_val)

        u = float(
            torch.rand(1, generator=accept_rng, device=accept_rng.device).item()
        )

        if protocol.temperature == 0.0:
            top = int(target_log_probs[target_pos].argmax().item())
            accepted = (top == tok)
        else:
            accepted = (u < ratio)

        survived_j.append(1 if alive else 0)
        accepted_j.append(1 if accepted else 0)
        q_j.append(q_val)
        p_j.append(p_val)
        min_pq_j.append(ratio)
        U_j.append(u)

        if not accepted:
            alive = False

    return AcceptOutcome(
        accepted_j=accepted_j,
        survived_j=survived_j,
        q_j=q_j,
        p_j=p_j,
        min_pq_j=min_pq_j,
        U_j=U_j,
    )
