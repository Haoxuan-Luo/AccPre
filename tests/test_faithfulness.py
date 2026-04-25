"""Tests for accpre.eval.faithfulness (Controlled Faithfulness@1)."""

from __future__ import annotations

from typing import Sequence

import pytest

from accpre.core.commit import commit_threshold
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import RoundRecord
from accpre.eval.faithfulness import controlled_faithfulness_at_1


def _dummy_record(
    prompt_idx: int,
    round_idx: int,
    gamma: int,
    L: int,
    min_pq_j: Sequence[float],
) -> RoundRecord:
    proto = ProtocolConfig()
    return RoundRecord(
        schema_version=proto.schema_version,
        protocol=proto,
        prompt_idx=prompt_idx,
        round_idx=round_idx,
        round_rng_seed=0,
        gamma=gamma,
        T=2,
        prefix_len=5,
        draft_tokens=[0] * gamma,
        q_j=[0.5] * gamma,
        p_j=[0.5] * gamma,
        min_pq_j=list(min_pq_j),
        U_j=[0.0] * gamma,
        accepted_j=[1 if j < L else 0 for j in range(gamma)],
        survived_j=[1 if j <= L else 0 for j in range(gamma)],
        L=L,
        bonus_or_fallback_token=0,
        draft_time=0.0,
        verify_time=0.0,
    )


def test_cf1_is_1_when_method_equals_strict():
    """If method returns the record's L, CF@1 aggregate = 1.0."""
    records = [
        _dummy_record(0, i, gamma=4, L=2, min_pq_j=[0.9, 0.8, 0.3, 0.95])
        for i in range(5)
    ]
    out = controlled_faithfulness_at_1(records, lambda r: r.L)
    assert out["aggregate"] == 1.0
    assert all(v == 1 for v in out["per_round"])


def test_cf1_is_0_when_method_always_disagrees():
    records = [
        _dummy_record(0, i, gamma=4, L=2, min_pq_j=[0.9] * 4)
        for i in range(5)
    ]
    # method returns L+1 for every record -> always differs from L=2
    out = controlled_faithfulness_at_1(
        records, lambda r: min(r.L + 1, r.gamma)
    )
    assert out["aggregate"] == 0.0
    assert all(v == 0 for v in out["per_round"])


def test_cf1_equals_L_agreement_rate():
    """CF@1 reduces to mean(1[method_L == strict_L]) under v1's shared-fallback coupling."""
    rs = [
        _dummy_record(0, 0, 4, 2, [0.9, 0.8, 0.3, 0.9]),
        _dummy_record(0, 1, 4, 4, [0.9] * 4),
        _dummy_record(0, 2, 4, 0, [0.1] + [0.9] * 3),
    ]

    def method_fn(r: RoundRecord) -> int:
        return {0: 2, 1: 4, 2: 2}[r.round_idx]

    out = controlled_faithfulness_at_1(rs, method_fn)
    # Agrees on 0 and 1; disagrees on 2 -> 2/3.
    assert abs(out["aggregate"] - 2.0 / 3.0) < 1e-12
    assert out["per_round"] == [1, 1, 0]


def test_cf1_extreme_taus_for_oracle_q2():
    """tau=0: L_tau=gamma for all; tau>1: L_tau=0 for all."""
    records = [
        _dummy_record(0, i, 4, L=2, min_pq_j=[0.9, 0.8, 0.3, 0.95])
        for i in range(10)
    ]
    out_zero = controlled_faithfulness_at_1(
        records, lambda r: commit_threshold(r.min_pq_j, 0.0)
    )
    assert out_zero["aggregate"] == 0.0   # L_tau = 4 != 2

    out_high = controlled_faithfulness_at_1(
        records, lambda r: commit_threshold(r.min_pq_j, 2.0)
    )
    assert out_high["aggregate"] == 0.0   # L_tau = 0 != 2

    # At tau=0.5, the first position with alpha<0.5 is j=2; L_tau=2 matches L=2.
    out_mid = controlled_faithfulness_at_1(
        records, lambda r: commit_threshold(r.min_pq_j, 0.5)
    )
    assert out_mid["aggregate"] == 1.0


def test_cf1_per_prompt_and_bootstrap_ci_fields():
    records = (
        [_dummy_record(p_idx, r_idx, 4, 2, [0.9, 0.8, 0.3, 0.9])
         for p_idx in range(3) for r_idx in range(4)]
    )
    out = controlled_faithfulness_at_1(records, lambda r: r.L,
                                       bootstrap_samples=100)
    assert out["n_rounds"] == 12
    assert out["n_prompts"] == 3
    assert set(out["per_prompt"].keys()) == {0, 1, 2}
    lo, hi = out["bootstrap_ci"]
    assert 0.0 <= lo <= hi <= 1.0


def test_cf1_empty_records():
    out = controlled_faithfulness_at_1([], lambda r: 0)
    assert out["aggregate"] == 0.0
    assert out["n_rounds"] == 0
    assert out["per_round"] == []
    assert out["per_prompt"] == {}
