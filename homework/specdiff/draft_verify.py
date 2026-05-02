"""Draft -> verify -> accept composer + bonus/fallback sampler.

Vendored (subset) from `accpre/core/draft_verify.py`. The original module
also contained a `RoundRecord`-emitting strict-rule wrapper used by the
predictor-training paths; that machinery is not needed for the coursework
experiment and is omitted here.

Matched-randomness discipline:
  - `round_rng_seed` is the only randomness input.
  - Three independent sub-streams are derived via protocol salts:
        draft_rng    = Generator(seed XOR protocol.draft_salt)
        accept_rng   = Generator(seed XOR protocol.accept_salt)
        fallback_rng = Generator(seed XOR protocol.fallback_salt)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Union

import torch

from .accept import AcceptOutcome, per_position_accept
from .commit import commit_strict
from .protocol import ProtocolConfig


def _make_generator(
    device: Union[str, torch.device], seed: int
) -> torch.Generator:
    """Build a torch.Generator on the right device with a 63-bit-masked seed."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


@dataclass
class RoundArtifacts:
    """Raw outputs of `draft_verify_accept`, shared across lanes."""
    draft_tokens: torch.Tensor          # (gamma,)
    draft_log_probs: torch.Tensor       # (gamma, vocab_drafter)
    target_log_probs: torch.Tensor      # (prefix_len+gamma, vocab_verifier)
    outcome: AcceptOutcome
    prefix_len: int
    gamma: int
    T: int
    round_rng_seed: int
    draft_time: float
    verify_time: float
    fallback_rng: torch.Generator


@torch.no_grad()
def draft_verify_accept(
    prefix_ids: torch.Tensor,
    drafter,
    verifier,
    gamma: int,
    T: int,
    protocol: ProtocolConfig,
    round_rng_seed: int,
) -> RoundArtifacts:
    """Shared draft + verify + accept scaffold (no commit applied)."""
    device = prefix_ids.device
    prefix_len = int(prefix_ids.shape[0])

    draft_rng = _make_generator(device, round_rng_seed ^ protocol.draft_salt)
    accept_rng = _make_generator(device, round_rng_seed ^ protocol.accept_salt)
    fallback_rng = _make_generator(device, round_rng_seed ^ protocol.fallback_salt)

    t0 = time.time()
    draft_tokens, draft_log_probs = drafter.draft(
        prefix_ids=prefix_ids,
        gamma=gamma,
        T=T,
        temperature=protocol.temperature,
        q_mode=protocol.q_mode,
        generator=draft_rng,
    )
    draft_time = time.time() - t0

    t0 = time.time()
    candidate = torch.cat([prefix_ids, draft_tokens])
    target_log_probs = verifier.score(candidate)
    verify_time = time.time() - t0

    outcome = per_position_accept(
        draft_tokens=draft_tokens,
        draft_log_probs=draft_log_probs,
        target_log_probs=target_log_probs,
        prefix_len=prefix_len,
        protocol=protocol,
        accept_rng=accept_rng,
    )

    return RoundArtifacts(
        draft_tokens=draft_tokens,
        draft_log_probs=draft_log_probs,
        target_log_probs=target_log_probs,
        outcome=outcome,
        prefix_len=prefix_len,
        gamma=int(gamma),
        T=int(T),
        round_rng_seed=int(round_rng_seed),
        draft_time=float(draft_time),
        verify_time=float(verify_time),
        fallback_rng=fallback_rng,
    )


def sample_fallback_or_bonus(
    L: int,
    gamma: int,
    draft_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    prefix_len: int,
    protocol: ProtocolConfig,
    fallback_rng: torch.Generator,
) -> int:
    """Sample the single extra token appended after the accepted prefix.

    L == gamma : BONUS from `target_log_probs` at position prefix_len-1+gamma.
    L <  gamma : FALLBACK from norm(max(0, p - q)) at position L (or argmax
                 at temp=0).
    """
    if L == gamma:
        pos = prefix_len - 1 + gamma
        logits_row = target_log_probs[pos]
        if protocol.temperature == 0.0:
            return int(logits_row.argmax().item())
        probs = (logits_row / protocol.temperature).softmax(dim=-1)
        return int(
            torch.multinomial(probs, num_samples=1, generator=fallback_rng).item()
        )

    pos = prefix_len - 1 + L
    target_row = target_log_probs[pos]
    draft_row = draft_log_probs[L]

    if protocol.temperature == 0.0:
        return int(target_row.argmax().item())

    # MDLM 50258 vs GPT-2 50257 vocab-size mismatch guard.
    p = target_row.exp()
    q = draft_row.exp()
    min_size = min(p.shape[0], q.shape[0])
    p_trim = p[:min_size]
    q_trim = q[:min_size]

    adjusted = torch.clamp(p_trim - q_trim, min=0.0)
    total = adjusted.sum()
    if float(total.item()) < 1e-10:
        return int(p_trim.argmax().item())
    adjusted = adjusted / total
    return int(
        torch.multinomial(adjusted, num_samples=1, generator=fallback_rng).item()
    )


@torch.no_grad()
def draft_verify_round_strict(
    prefix_ids: torch.Tensor,
    drafter,
    verifier,
    gamma: int,
    T: int,
    protocol: ProtocolConfig,
    round_rng_seed: int,
):
    """Canonical strict-rule round: returns (L, draft_tokens, bonus_token, gamma).

    Equivalent of the original `draft_verify_round` minus the RoundRecord
    schema bookkeeping the homework lanes don't need.
    """
    artifacts = draft_verify_accept(
        prefix_ids=prefix_ids,
        drafter=drafter,
        verifier=verifier,
        gamma=gamma,
        T=T,
        protocol=protocol,
        round_rng_seed=round_rng_seed,
    )
    L = commit_strict(artifacts.outcome.accepted_j)
    bonus = sample_fallback_or_bonus(
        L=L,
        gamma=artifacts.gamma,
        draft_log_probs=artifacts.draft_log_probs,
        target_log_probs=artifacts.target_log_probs,
        prefix_len=artifacts.prefix_len,
        protocol=protocol,
        fallback_rng=artifacts.fallback_rng,
    )
    return L, artifacts.draft_tokens, bonus, artifacts.gamma
