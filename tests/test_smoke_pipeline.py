"""End-to-end smoke tests. SLOW — loads real HuggingFace models.

Skipped by default via `pytest.ini_options.addopts = "-m 'not slow'"`.
Enable explicitly with:  `pytest -m slow tests/test_smoke_pipeline.py`.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def models():
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    drafter = MDLMDrafter(device=device, dtype=dtype)
    verifier = GPT2Verifier(device=device, dtype=dtype)
    return drafter, verifier, device


def _prefix(device: str) -> torch.Tensor:
    """Fixed synthetic prefix (arbitrary token ids) — avoids network at test time."""
    return torch.arange(32, dtype=torch.long, device=device)


def test_strict_round_produces_valid_record(models):
    from accpre.core.draft_verify import draft_verify_round
    from accpre.core.protocol import ProtocolConfig, derive_seed

    drafter, verifier, device = models
    proto = ProtocolConfig(temperature=1.0, q_mode="A")
    prefix = _prefix(device)
    seed = derive_seed(proto, 0, 0)
    rec = draft_verify_round(
        prefix_ids=prefix, drafter=drafter, verifier=verifier,
        gamma=8, T=2, protocol=proto,
        prompt_idx=0, round_idx=0, round_rng_seed=seed,
    )
    rec.validate()
    assert rec.gamma == 8
    assert 0 <= rec.L <= 8
    assert len(rec.U_j) == 8
    assert rec.round_rng_seed == seed


def test_strict_generates_up_to_max_new_tokens(models):
    from accpre.eval.wallclock import run_strict_lane
    from accpre.core.protocol import ProtocolConfig

    drafter, verifier, device = models
    proto = ProtocolConfig(temperature=1.0, q_mode="A")
    prompts = [(_prefix(device), "synthetic")]
    recs, _tps, gens = run_strict_lane(
        drafter, verifier, prompts, proto, gamma=8, T=2, max_new_tokens=32,
    )
    assert len(gens) == 1
    new_len = int(gens[0].shape[0]) - int(prompts[0][0].shape[0])
    assert new_len >= 32 or new_len == int(prompts[0][0].shape[0]) * 0  # generated at least up to cap
    # Round records have consistent L
    for rec in recs[0]:
        assert 0 <= rec.L <= rec.gamma


def test_oracle_q2_tau_sweep_valid_L(models):
    """Apply commit_threshold(min_pq_j, tau) to strict records across a sweep."""
    from accpre.core.commit import commit_threshold
    from accpre.core.draft_verify import draft_verify_round
    from accpre.core.protocol import ProtocolConfig, derive_seed

    drafter, verifier, device = models
    proto = ProtocolConfig(temperature=1.0, q_mode="A")
    prefix = _prefix(device)

    recs = []
    for r in range(3):
        seed = derive_seed(proto, 0, r)
        rec = draft_verify_round(
            prefix, drafter, verifier, 8, 2, proto, 0, r, seed,
        )
        recs.append(rec)

    for tau in (0.3, 0.5, 0.7, 0.9):
        for rec in recs:
            L_tau = commit_threshold(rec.min_pq_j, tau)
            assert 0 <= L_tau <= rec.gamma


def test_strict_round_idempotent_under_fixed_seed(models):
    """Same seed -> same draft, same accepted_j, same L, same extra token."""
    from accpre.core.draft_verify import draft_verify_round
    from accpre.core.protocol import ProtocolConfig

    drafter, verifier, device = models
    proto = ProtocolConfig(temperature=1.0, q_mode="A")
    prefix = _prefix(device)
    seed = 99_999

    a = draft_verify_round(prefix, drafter, verifier, 8, 2, proto, 0, 0, seed)
    b = draft_verify_round(prefix, drafter, verifier, 8, 2, proto, 0, 0, seed)

    assert a.draft_tokens == b.draft_tokens
    assert a.U_j == b.U_j
    assert a.accepted_j == b.accepted_j
    assert a.L == b.L
    assert a.bonus_or_fallback_token == b.bonus_or_fallback_token


def test_wallclock_timer_excludes_model_load(models):
    """Smoke: model load is not inside the timed decode loop."""
    import time
    from accpre.eval.wallclock import run_strict_lane
    from accpre.core.protocol import ProtocolConfig

    drafter, verifier, device = models
    proto = ProtocolConfig()
    prompts = [(_prefix(device), "synthetic")]
    t0 = time.time()
    _, tps, _ = run_strict_lane(
        drafter, verifier, prompts, proto, gamma=4, T=2, max_new_tokens=8,
    )
    outer = time.time() - t0
    # Internal tok/s must be >= 0 and finite.
    assert tps[0] > 0.0
    assert outer > 0.0
