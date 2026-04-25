"""Fast tests for the cheap online predictor decode loop.

Uses mock drafter / verifier / predictor so no real models or GPU are
required. Verifies the three semantic commitments in DESIGN_PHASE2.md
§C.5:

  1. NO verifier-on-draft pass (never). The verifier's `.score()` method
     is never called by the online loop — even the full strict-style
     pass is absent.
  2. Commit convention: `max(1, L̂)` draft tokens per round.
     - `L̂ == 0` commits 1 token (progress guarantee).
     - `L̂ == k > 0` commits exactly k tokens, no bonus/fallback.
  3. Deploy modes: V-free → zero `verifier.prefix_features` calls;
     V-prefix → exactly one such call per round.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest
import torch
from torch import nn

from accpre.core.protocol import ProtocolConfig
from accpre.eval.online_decode import run_predictor_lane
from accpre.predictors.base import PredictorBase


# ------------------------------------------------------------------
# Mock models
# ------------------------------------------------------------------


class _MockDrafter:
    """Counts calls; returns deterministic draft tokens and a zero hidden."""

    def __init__(self, hidden_dim: int = 768, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.call_count = 0

    def draft_with_features(
        self, prefix_ids, gamma, T, temperature, q_mode, generator=None,
        pool="mean", layers=(-1,), **_kwargs,
    ):
        # `layers` and any future kwargs are accepted but ignored; the
        # mock returns a zero hidden tensor regardless. Real MDLMDrafter
        # honors these; online_decode.py unconditionally passes them
        # (see accpre/eval/online_decode.py::_run_single_prompt).
        self.call_count += 1
        prefix_len = int(prefix_ids.shape[0])
        tokens = torch.arange(
            prefix_len, prefix_len + gamma, dtype=torch.long,
            device=prefix_ids.device,
        )
        vocab = 50258
        log_probs = torch.full((gamma, vocab), math.log(1.0 / vocab))
        hidden = torch.zeros(self.hidden_dim)
        return tokens, log_probs, hidden


class _MockVerifier:
    """Counts calls to prefix_features AND score — both must stay at zero
    in V-free online decode; prefix_features should equal n_rounds in
    V-prefix mode; score should never be called by online_decode at all."""

    def __init__(self, hidden_dim: int = 1600) -> None:
        self.hidden_dim = hidden_dim
        self.prefix_features_count = 0
        self.score_count = 0

    def prefix_features(self, prefix_ids):
        self.prefix_features_count += 1
        return (
            torch.zeros(self.hidden_dim),
            {"entropy": 1.0, "margin": 0.5, "top1_prob": 0.5},
        )

    def score(self, token_ids):
        # The online loop MUST NOT invoke this. If it does, the test fails.
        self.score_count += 1
        vocab = 50257
        return torch.full((int(token_ids.shape[0]), vocab), -10.0)


# ------------------------------------------------------------------
# Stub predictor
# ------------------------------------------------------------------


class _StubPredictor(PredictorBase):
    """Returns a fixed L_hat regardless of inputs.

    `family` and `deploy_mode` flow through PredictorBase.__init__ as
    kwargs; the base class validates that V-free is only used with a
    drafter-only family.
    """
    kind: ClassVar[str] = "acceptance"

    def __init__(self, family: str, gamma: int, returns_L: int) -> None:
        deploy_mode = "V-free" if family == "hidden" else "V-prefix"
        super().__init__(family=family, deploy_mode=deploy_mode, gamma=gamma)
        self._returns_L = int(returns_L)
        # Dummy parameter so .parameters() is non-empty.
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        # Return a (batch, gamma) Q̂ where first self._returns_L entries
        # are 1 and the rest are 0. Only used if some caller decides to
        # invoke forward directly; the inference path uses predict_L.
        b = features.shape[0] if features.dim() == 2 else 1
        q = torch.zeros(b, self.gamma)
        q[:, : max(0, min(self._returns_L, self.gamma))] = 1.0
        return q

    def predict_L(self, features, tau, **_kwargs):
        return self._returns_L


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def _single_prompt_pair(device: str = "cpu", prefix_len: int = 16):
    prefix = torch.zeros(prefix_len, dtype=torch.long, device=device)
    return [(prefix, "stub")]


def test_progress_guarantee_when_L_hat_zero():
    """L_hat == 0 must still commit 1 token per round."""
    drafter = _MockDrafter()
    verifier = _MockVerifier()
    predictor = _StubPredictor(family="hidden", gamma=8, returns_L=0)
    prompts = _single_prompt_pair()
    proto = ProtocolConfig()

    lane = run_predictor_lane(
        predictor=predictor, drafter=drafter, verifier=verifier,
        prompts=prompts, protocol=proto,
        gamma=8, T=2, max_new_tokens=20, tau=0.5,
    )
    p = lane.per_prompt[0]
    assert p.n_new_tokens == 20          # 20 rounds × 1 commit
    assert len(p.rounds) == 20
    for rd in p.rounds:
        assert rd.L_hat == 0
        assert rd.n_committed == 1
        assert len(rd.committed_tokens) == 1


def test_exact_L_hat_commits_no_bonus():
    """L_hat == 5 must commit exactly 5 tokens per round — no extras."""
    drafter = _MockDrafter()
    verifier = _MockVerifier()
    predictor = _StubPredictor(family="hidden", gamma=8, returns_L=5)
    prompts = _single_prompt_pair()
    proto = ProtocolConfig()

    lane = run_predictor_lane(
        predictor=predictor, drafter=drafter, verifier=verifier,
        prompts=prompts, protocol=proto,
        gamma=8, T=2, max_new_tokens=20, tau=0.5,
    )
    for rd in lane.per_prompt[0].rounds:
        assert rd.L_hat == 5
        assert rd.n_committed == 5
        assert len(rd.committed_tokens) == 5
        assert rd.committed_tokens == rd.draft_tokens[:5]


def test_v_free_zero_verifier_calls():
    """V-free predictor must NEVER call verifier."""
    drafter = _MockDrafter()
    verifier = _MockVerifier()
    predictor = _StubPredictor(family="hidden", gamma=8, returns_L=4)
    prompts = _single_prompt_pair()
    proto = ProtocolConfig()

    run_predictor_lane(
        predictor=predictor, drafter=drafter, verifier=verifier,
        prompts=prompts, protocol=proto,
        gamma=8, T=2, max_new_tokens=16, tau=0.5,
    )
    assert verifier.prefix_features_count == 0
    assert verifier.score_count == 0


def test_v_prefix_exactly_one_prefix_call_per_round():
    """V-prefix predictor must call verifier.prefix_features once per round,
    and must NEVER call verifier.score (no verifier-on-draft)."""
    drafter = _MockDrafter()
    verifier = _MockVerifier()
    predictor = _StubPredictor(family="hidden_numeric", gamma=8, returns_L=4)
    prompts = _single_prompt_pair()
    proto = ProtocolConfig()

    lane = run_predictor_lane(
        predictor=predictor, drafter=drafter, verifier=verifier,
        prompts=prompts, protocol=proto,
        gamma=8, T=2, max_new_tokens=16, tau=0.5,
    )
    n_rounds = len(lane.per_prompt[0].rounds)
    # 4 tokens per round × 4 rounds = 16 new tokens
    assert n_rounds == 4
    assert verifier.prefix_features_count == n_rounds
    assert verifier.score_count == 0   # strict verifier-on-draft never invoked


def test_verifier_score_never_invoked_even_at_full_accept():
    """L_hat = gamma (full accept) must still not trigger verifier.score."""
    drafter = _MockDrafter()
    verifier = _MockVerifier()
    predictor = _StubPredictor(family="hidden", gamma=8, returns_L=8)
    prompts = _single_prompt_pair()
    proto = ProtocolConfig()

    run_predictor_lane(
        predictor=predictor, drafter=drafter, verifier=verifier,
        prompts=prompts, protocol=proto,
        gamma=8, T=2, max_new_tokens=16, tau=0.5,
    )
    assert verifier.score_count == 0


def test_base_rejects_v_free_with_non_hidden_family():
    """Configuration guard: `V-free` is only valid for drafter-only families.

    After the P1 fix, `family` and `deploy_mode` are keyword-only
    __init__ parameters on `PredictorBase`. The V-free guard fires when
    `deploy_mode='V-free'` is paired with a family not in
    `_V_FREE_FAMILIES`. The test pairs family='numeric' (verifier-prefix
    numeric, NOT drafter-only) with deploy_mode='V-free' and expects
    the guard to raise.
    """
    class Bad(PredictorBase):
        kind: ClassVar[str] = "acceptance"

        def forward(self, features, **kwargs):
            return features

    with pytest.raises(TypeError, match="V-free"):
        Bad(family="numeric", deploy_mode="V-free", gamma=8)
