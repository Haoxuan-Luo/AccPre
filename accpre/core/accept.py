"""Canonical per-position accept logic.

This file is THE single implementation of the Leviathan speculative-
decoding accept test in the repository. A meta-test (in
`tests/test_schema.py`) asserts the distinctive pattern does not appear
anywhere else under `accpre/`.

Matched-randomness discipline (DESIGN.md §D.5.2):
  - `accept_rng` is a `torch.Generator` seeded from
        round_rng_seed XOR protocol.accept_salt
    inside `accpre.core.draft_verify._make_generator`. It is separate
    from `draft_rng` and `fallback_rng`.
  - Under `temperature > 0` (v1 main protocol), `U_j[j]` is drawn from
    `accept_rng` for every j in [0, gamma), regardless of whether the
    calling lane uses it for its commit decision. This keeps the
    accept_rng stream position identical across lanes that share the
    same `round_rng_seed`, which matters for offline auditability even
    though v1 fallback RNG is already isolated on its own generator.
  - Under `temperature == 0` (tests only), acceptance is deterministic
    argmax(target) == draft_tok; `U_j` is still drawn (and logged) for
    schema completeness and to keep the stream position fixed.

The function returns per-position arrays suitable for direct use by
`commit_strict`, `commit_threshold`, and for logging into a
`RoundRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch

from accpre.core.protocol import ProtocolConfig


@dataclass
class AcceptOutcome:
    """Per-position outputs of the accept test.

    All lists have length `gamma` and align position-wise.
    """
    accepted_j: List[int]   # 0/1: Bernoulli outcome (temp>0) or argmax match (temp==0)
    survived_j: List[int]   # 0/1: monotone non-increasing from survived_0=1
    q_j: List[float]        # drafter's prob of the drafted token
    p_j: List[float]        # verifier's prob of the drafted token
    min_pq_j: List[float]   # Q2_j := min(1, p_j / q_j)
    U_j: List[float]        # uniform draws from accept_rng (always populated)


def per_position_accept(
    draft_tokens: torch.Tensor,        # (gamma,) long
    draft_log_probs: torch.Tensor,     # (gamma, vocab_drafter)
    target_log_probs: torch.Tensor,    # (seq_len_total, vocab_verifier)
    prefix_len: int,
    protocol: ProtocolConfig,
    accept_rng: torch.Generator,
) -> AcceptOutcome:
    """Leviathan per-position accept test — the only implementation.

    Args:
        draft_tokens: drafter-sampled token IDs, length gamma.
        draft_log_probs: per-position log-prob rows from the drafter; the
            value q_j = exp(draft_log_probs[j, draft_tokens[j]]).
        target_log_probs: full-sequence verifier log-probs on the
            `concat(prefix_ids, draft_tokens)` input. Shape
            (prefix_len + gamma, vocab_verifier). The value
            p_j = exp(target_log_probs[prefix_len - 1 + j, draft_tokens[j]]).
        prefix_len: length of the prefix at round start.
        protocol: the pinned `ProtocolConfig` for this run.
        accept_rng: the per-round accept generator. Consumed for U_j
            regardless of temperature.

    Returns:
        AcceptOutcome with all per-position arrays populated.
    """
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

        # Always draw U_j. This keeps the accept_rng stream position
        # deterministic across lanes that share the same round_rng_seed,
        # regardless of the lane's commit rule or temperature.
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
