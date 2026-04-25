"""Tests for accpre/collect/features.py.

Uses synthetic RoundRecords (no models) to verify shapes and field
requirements for each family.
"""

from __future__ import annotations

import pytest
import torch

from accpre.collect.features import (
    NUMERIC_DIM,
    extract_features,
    feature_dim,
)
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import RoundRecord


def _make_record(
    *,
    with_drafter_hidden: bool = True,
    with_verifier_hidden: bool = True,
    with_numeric: bool = True,
    drafter_dim: int = 768,
    verifier_dim: int = 1600,
    gamma: int = 4,
) -> RoundRecord:
    proto = ProtocolConfig()
    rec = RoundRecord(
        schema_version=proto.schema_version,
        protocol=proto,
        prompt_idx=0, round_idx=0, round_rng_seed=1,
        gamma=gamma, T=2, prefix_len=5,
        draft_tokens=[0] * gamma,
        q_j=[0.5] * gamma, p_j=[0.5] * gamma,
        min_pq_j=[0.5] * gamma, U_j=[0.0] * gamma,
        accepted_j=[1] * gamma, survived_j=[1] * gamma,
        L=gamma, bonus_or_fallback_token=0,
        draft_time=0.0, verify_time=0.0,
    )
    if with_drafter_hidden:
        rec.drafter_hidden = torch.randn(drafter_dim)
    if with_verifier_hidden:
        rec.verifier_hidden = torch.randn(verifier_dim)
    if with_numeric:
        rec.verifier_entropy = 2.3
        rec.verifier_margin = 1.1
        rec.verifier_top1_prob = 0.4
    return rec


def test_numeric_family():
    rec = _make_record()
    f = extract_features(rec, "numeric")
    assert f.shape == (NUMERIC_DIM,)
    assert f.dtype == torch.float32
    assert float(f[0]) == pytest.approx(2.3)
    assert float(f[1]) == pytest.approx(1.1)
    assert float(f[2]) == pytest.approx(0.4)


def test_hidden_family():
    rec = _make_record()
    f = extract_features(rec, "hidden")
    assert f.shape == (768,)
    assert f.dtype == torch.float32


def test_hidden_numeric_family():
    rec = _make_record()
    f = extract_features(rec, "hidden_numeric")
    assert f.shape == (768 + NUMERIC_DIM,)
    # First 768 should be the drafter_hidden; last 3 the numeric.
    assert torch.equal(f[:768], rec.drafter_hidden.to(torch.float32))
    assert float(f[768]) == pytest.approx(2.3)


def test_upper_bound_family():
    rec = _make_record()
    f = extract_features(rec, "upper_bound")
    assert f.shape == (1600 + 768 + NUMERIC_DIM,)


def test_numeric_missing_field_raises():
    rec = _make_record(with_numeric=False)
    with pytest.raises(ValueError, match="verifier_entropy"):
        extract_features(rec, "numeric")


def test_hidden_missing_raises():
    rec = _make_record(with_drafter_hidden=False)
    with pytest.raises(ValueError, match="drafter_hidden"):
        extract_features(rec, "hidden")


def test_upper_bound_missing_verifier_hidden_raises():
    rec = _make_record(with_verifier_hidden=False)
    with pytest.raises(ValueError, match="verifier_hidden"):
        extract_features(rec, "upper_bound")


def test_unknown_family_raises():
    rec = _make_record()
    with pytest.raises(ValueError, match="Unknown family"):
        extract_features(rec, "not_a_family")


def test_feature_dim_matches_extract():
    rec = _make_record()
    for family in ("numeric", "hidden", "hidden_numeric", "upper_bound"):
        assert extract_features(rec, family).shape[0] == feature_dim(family)


def test_feature_dim_custom_hidden_sizes():
    assert feature_dim("hidden", drafter_hidden_dim=512) == 512
    assert feature_dim(
        "upper_bound", drafter_hidden_dim=512, verifier_hidden_dim=1024
    ) == 1024 + 512 + NUMERIC_DIM


def test_hidden_per_pos_ml_feature_dim():
    # Phase 6 main candidate. Per-position dim = 2H.
    assert feature_dim("hidden_per_pos_ml") == 2 * 768
    assert feature_dim("hidden_per_pos_ml", drafter_hidden_dim=512) == 2 * 512


def test_hidden_per_pos_ml_shape_and_missing():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma)
    # missing field raises
    with pytest.raises(ValueError, match="drafter_hidden_per_pos_ml"):
        extract_features(rec, "hidden_per_pos_ml")
    # populated → returns (γ, 2H)
    rec.drafter_hidden_per_pos_ml = torch.randn(gamma, 2 * H)
    f = extract_features(rec, "hidden_per_pos_ml")
    assert f.shape == (gamma, 2 * H)
    assert f.dtype == torch.float32


def test_hidden_per_pos_numeric_feature_dim():
    # Phase 7 V1. Per-position dim = H + 3.
    assert feature_dim("hidden_per_pos_numeric") == 768 + NUMERIC_DIM
    assert feature_dim("hidden_per_pos_numeric", drafter_hidden_dim=512) == 512 + NUMERIC_DIM


def test_hidden_per_pos_numeric_shape_missing_and_broadcast():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma, with_drafter_hidden=False, with_numeric=True)
    # missing drafter per-pos hidden raises
    with pytest.raises(ValueError, match="drafter_hidden_per_pos"):
        extract_features(rec, "hidden_per_pos_numeric")
    rec.drafter_hidden_per_pos = torch.randn(gamma, H)
    f = extract_features(rec, "hidden_per_pos_numeric")
    assert f.shape == (gamma, H + NUMERIC_DIM)
    assert f.dtype == torch.float32
    # first H columns = drafter hidden
    assert torch.allclose(f[:, :H], rec.drafter_hidden_per_pos.to(torch.float32))
    # last 3 columns = the 3 numeric scalars, identical across all γ positions
    for j in range(gamma):
        assert float(f[j, H]) == pytest.approx(2.3)
        assert float(f[j, H + 1]) == pytest.approx(1.1)
        assert float(f[j, H + 2]) == pytest.approx(0.4)


def test_hidden_per_pos_numeric_missing_verifier_numeric_raises():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma, with_numeric=False)
    rec.drafter_hidden_per_pos = torch.randn(gamma, H)
    with pytest.raises(ValueError, match="verifier_"):
        extract_features(rec, "hidden_per_pos_numeric")


# -----------------------------------------------------------------------
# Phase 9 per-position transformer feature families (v1..v4).
# -----------------------------------------------------------------------


def _populate_phase9_fields(rec, gamma: int, H: int):
    """Attach the drafter-side per-position fields required by v2/v4."""
    rec.drafter_hidden_per_pos = torch.randn(gamma, H)
    rec.drafter_entropy_j = [1.0] * gamma
    rec.drafter_margin_j = [0.5] * gamma
    rec.drafter_top1_prob_j = [0.3] * gamma
    return rec


def test_hidden_per_pos_v1_shape_and_dim():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma)
    rec.drafter_hidden_per_pos = torch.randn(gamma, H)
    f = extract_features(rec, "hidden_per_pos_v1")
    assert f.shape == (gamma, H)
    assert f.dtype == torch.float32
    assert feature_dim("hidden_per_pos_v1") == H


def test_hidden_per_pos_v2_shape_and_dim():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma)
    _populate_phase9_fields(rec, gamma, H)
    f = extract_features(rec, "hidden_per_pos_v2")
    assert f.shape == (gamma, H + 5)
    assert f.dtype == torch.float32
    assert feature_dim("hidden_per_pos_v2") == H + 5
    # Scalar column order is [q, log q, entropy, margin, top1].
    # q_j defaults to 0.5 in _make_record, so log q_j = log(0.5).
    for j in range(gamma):
        assert float(f[j, H]) == pytest.approx(0.5, abs=1e-5)
        assert float(f[j, H + 1]) == pytest.approx(
            torch.log(torch.tensor(0.5)).item(), abs=1e-5,
        )
        assert float(f[j, H + 2]) == pytest.approx(1.0)    # entropy
        assert float(f[j, H + 3]) == pytest.approx(0.5)    # margin
        assert float(f[j, H + 4]) == pytest.approx(0.3)    # top1


def test_hidden_per_pos_v3_shape_and_dim():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma)
    rec.drafter_hidden_per_pos = torch.randn(gamma, H)
    f = extract_features(rec, "hidden_per_pos_v3")
    assert f.shape == (gamma, H + NUMERIC_DIM)
    assert feature_dim("hidden_per_pos_v3") == H + NUMERIC_DIM
    # Verifier numeric broadcast is identical across γ.
    for j in range(gamma):
        assert float(f[j, H]) == pytest.approx(2.3)        # entropy
        assert float(f[j, H + 1]) == pytest.approx(1.1)    # margin
        assert float(f[j, H + 2]) == pytest.approx(0.4)    # top1_prob


def test_hidden_per_pos_v4_shape_and_dim():
    gamma, H, Hv = 4, 768, 1600
    rec = _make_record(gamma=gamma, verifier_dim=Hv)
    _populate_phase9_fields(rec, gamma, H)
    f = extract_features(rec, "hidden_per_pos_v4")
    assert f.shape == (gamma, H + 5 + Hv)
    assert feature_dim("hidden_per_pos_v4") == H + 5 + Hv
    # Layout: [H drafter hidden | 5 local scalars | 1600 verifier hidden].
    # Verifier block is identical across γ.
    vh_row0 = f[0, H + 5 :]
    for j in range(1, gamma):
        assert torch.equal(f[j, H + 5 :], vh_row0)
    # Local scalars: q_j default = 0.5, so log q = log 0.5.
    assert float(f[0, H + 1]) == pytest.approx(
        torch.log(torch.tensor(0.5)).item(), abs=1e-5,
    )


def test_hidden_per_pos_v4_missing_verifier_hidden_raises():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma, with_verifier_hidden=False)
    _populate_phase9_fields(rec, gamma, H)
    with pytest.raises(ValueError, match="verifier_hidden"):
        extract_features(rec, "hidden_per_pos_v4")


def test_hidden_per_pos_v2_missing_scalars_raises():
    gamma, H = 4, 768
    rec = _make_record(gamma=gamma)
    rec.drafter_hidden_per_pos = torch.randn(gamma, H)
    # drafter_entropy_j etc. intentionally not set.
    with pytest.raises(ValueError, match="drafter_"):
        extract_features(rec, "hidden_per_pos_v2")
