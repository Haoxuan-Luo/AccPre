"""Tests for accpre.core.accept — the single canonical accept test."""

from __future__ import annotations

import pytest
import torch

from accpre.core.accept import per_position_accept
from accpre.core.protocol import ProtocolConfig


def _make_log_dists(
    gamma: int,
    vocab: int,
    prefix_len: int,
    draft_tok: int = 7,
    q_logit: float = -1.0,
    p_logit: float = 0.0,
):
    """Build synthetic draft and target log-prob tensors.

    q_logit < p_logit ⇒ q < p ⇒ min(1, p/q) = 1 at the drafted token
    ⇒ strict Bernoulli always accepts.
    """
    draft_tokens = torch.full((gamma,), draft_tok, dtype=torch.long)
    draft_lp = torch.full((gamma, vocab), -5.0)
    draft_lp[:, draft_tok] = q_logit
    draft_lp = draft_lp - torch.logsumexp(draft_lp, dim=-1, keepdim=True)

    target_lp = torch.full((prefix_len + gamma, vocab), -5.0)
    for j in range(gamma):
        target_lp[prefix_len - 1 + j, draft_tok] = p_logit
    target_lp = target_lp - torch.logsumexp(target_lp, dim=-1, keepdim=True)
    return draft_tokens, draft_lp, target_lp


def test_leviathan_bernoulli_always_accepts_when_p_ge_q():
    gamma, vocab, prefix_len = 5, 32, 8
    dt, dlp, tlp = _make_log_dists(gamma, vocab, prefix_len, q_logit=-2, p_logit=0)
    proto = ProtocolConfig(temperature=1.0, q_mode="A")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    out = per_position_accept(dt, dlp, tlp, prefix_len, proto, rng)
    assert all(a == 1 for a in out.accepted_j)
    assert all(s == 1 for s in out.survived_j)
    assert len(out.U_j) == gamma


def test_greedy_argmax_acceptance_at_temperature_zero():
    """temperature=0: accept iff argmax(target) == draft_tok. Tests only."""
    gamma, vocab, prefix_len = 3, 16, 4
    dt, dlp, _ = _make_log_dists(gamma, vocab, prefix_len, draft_tok=2)

    # Override target so position 0 disagrees, positions 1,2 agree.
    tlp = torch.full((prefix_len + gamma, vocab), -10.0)
    tlp[prefix_len - 1 + 0, 5] = 0.0   # argmax=5, draft_tok=2 -> reject
    tlp[prefix_len - 1 + 1, 2] = 0.0
    tlp[prefix_len - 1 + 2, 2] = 0.0
    tlp = tlp - torch.logsumexp(tlp, dim=-1, keepdim=True)

    proto = ProtocolConfig(temperature=0.0, q_mode="A")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    out = per_position_accept(dt, dlp, tlp, prefix_len, proto, rng)
    assert out.accepted_j[0] == 0
    assert out.survived_j == [1, 0, 0]


def test_survival_monotone_after_rejection():
    """Once survived flips to 0, it stays 0."""
    gamma, vocab, prefix_len = 5, 16, 3
    draft_tokens = torch.full((gamma,), 4, dtype=torch.long)
    draft_lp = torch.full((gamma, vocab), -5.0)
    draft_lp[:, 4] = 0.0
    draft_lp = draft_lp - torch.logsumexp(draft_lp, dim=-1, keepdim=True)

    # Force the verifier to assign tiny p to drafted token at position 2,
    # so the ratio ≈ 0 and Bernoulli almost certainly rejects.
    target_lp = torch.full((prefix_len + gamma, vocab), 0.0)
    target_lp[prefix_len - 1 + 2, 4] = -30.0
    target_lp = target_lp - torch.logsumexp(target_lp, dim=-1, keepdim=True)

    proto = ProtocolConfig(temperature=1.0, q_mode="A")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(123)
    out = per_position_accept(
        draft_tokens, draft_lp, target_lp, prefix_len, proto, rng
    )
    prev = 1
    for s in out.survived_j:
        assert s <= prev
        prev = s


def test_min_pq_is_clamped_and_bounded():
    """min_pq is always in [0, 1] even with pathological q."""
    gamma, vocab, prefix_len = 2, 8, 2
    draft_tokens = torch.zeros(gamma, dtype=torch.long)
    # Almost-uniform draft distribution → small q
    draft_lp = torch.full((gamma, vocab), 0.0)
    draft_lp = draft_lp - torch.logsumexp(draft_lp, dim=-1, keepdim=True)
    target_lp = torch.full((prefix_len + gamma, vocab), 0.0)
    target_lp = target_lp - torch.logsumexp(target_lp, dim=-1, keepdim=True)

    proto = ProtocolConfig(temperature=1.0, q_mode="A")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(7)
    out = per_position_accept(
        draft_tokens, draft_lp, target_lp, prefix_len, proto, rng
    )
    for v in out.min_pq_j:
        assert 0.0 <= v <= 1.0


def test_u_j_stream_independent_of_temperature():
    """U_j is drawn regardless of protocol.temperature."""
    gamma, vocab, prefix_len = 4, 16, 3
    dt, dlp, tlp = _make_log_dists(gamma, vocab, prefix_len)

    u_at_zero = None
    u_at_one = None
    for T, bucket in [(0.0, "z"), (1.0, "o")]:
        proto = ProtocolConfig(temperature=T, q_mode="A")
        rng = torch.Generator(device="cpu")
        rng.manual_seed(999)
        out = per_position_accept(dt, dlp, tlp, prefix_len, proto, rng)
        if bucket == "z":
            u_at_zero = out.U_j
        else:
            u_at_one = out.U_j
    # Same seed ⇒ same stream, regardless of temperature branching.
    assert u_at_zero == u_at_one
