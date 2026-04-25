"""Fast tests for the Phase 9 per-position transformer predictor family.

Covers:
  - PerPositionTransformerEncoder: output shape, in_dim / gamma guards.
  - AcceptanceTx:   sigmoid range, (γ, F) and (B, γ, F) input handling,
                    deploy_mode derivation, family rejection.
  - LengthTx:       classification logit shape, τ influences output,
                    predict_L returns a valid int, family rejection.

No models loaded, no GPU. Should run in under a second.
"""

from __future__ import annotations

import pytest
import torch

from accpre.predictors.pp_transformer import (
    AcceptanceTx,
    LengthTx,
    PerPositionTransformerEncoder,
)


# ------------------------------------------------------------------
# Encoder
# ------------------------------------------------------------------


def test_encoder_output_shape():
    enc = PerPositionTransformerEncoder(in_dim=64, gamma=8, d_model=128)
    x = torch.randn(2, 8, 64)
    out = enc(x)
    assert out.shape == (2, 8, 128)
    assert out.dtype == torch.float32


def test_encoder_accepts_shorter_gamma():
    """Tail rounds with cur_gamma < trained gamma must still work."""
    enc = PerPositionTransformerEncoder(in_dim=64, gamma=8, d_model=128)
    x = torch.randn(2, 5, 64)
    out = enc(x)
    assert out.shape == (2, 5, 128)


def test_encoder_rejects_gamma_too_large():
    enc = PerPositionTransformerEncoder(in_dim=64, gamma=8, d_model=128)
    x = torch.randn(2, 10, 64)  # γ=10 > trained γ=8
    with pytest.raises(ValueError, match="gamma"):
        enc(x)


def test_encoder_rejects_wrong_in_dim():
    enc = PerPositionTransformerEncoder(in_dim=64, gamma=8, d_model=128)
    x = torch.randn(2, 8, 65)
    with pytest.raises(ValueError, match="feature dim"):
        enc(x)


def test_encoder_rejects_2d_input():
    enc = PerPositionTransformerEncoder(in_dim=64, gamma=8)
    x = torch.randn(8, 64)  # missing batch dim
    with pytest.raises(ValueError, match=r"\(B, gamma, in_dim\)"):
        enc(x)


# ------------------------------------------------------------------
# AcceptanceTx
# ------------------------------------------------------------------


def test_acceptance_tx_v1_shape_and_sigmoid():
    p = AcceptanceTx(family="hidden_per_pos_v1", gamma=8)
    assert p.kind == "acceptance"
    assert p.deploy_mode == "V-free"
    assert p.in_dim == 768
    x = torch.randn(4, 8, 768)
    out = p(x)
    assert out.shape == (4, 8)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_acceptance_tx_v2_v3_v4_in_dim_and_deploy():
    cases = [
        ("hidden_per_pos_v2", 768 + 5,         "V-free"),
        ("hidden_per_pos_v3", 768 + 3,         "V-prefix"),
        ("hidden_per_pos_v4", 768 + 5 + 1600,  "V-prefix"),
    ]
    for fam, in_dim, deploy in cases:
        p = AcceptanceTx(family=fam, gamma=8)
        assert p.deploy_mode == deploy, (
            f"{fam}: expected deploy_mode={deploy}, got {p.deploy_mode}"
        )
        assert p.in_dim == in_dim, (
            f"{fam}: expected in_dim={in_dim}, got {p.in_dim}"
        )
        x = torch.randn(2, 8, in_dim)
        out = p(x)
        assert out.shape == (2, 8)


def test_acceptance_tx_accepts_unbatched_input():
    p = AcceptanceTx(family="hidden_per_pos_v1", gamma=8)
    x = torch.randn(8, 768)
    out = p(x)
    assert out.shape == (8,)


def test_acceptance_tx_predict_q2_eval_mode():
    p = AcceptanceTx(family="hidden_per_pos_v1", gamma=8, dropout=0.5)
    p.train()
    x = torch.randn(8, 768)
    q = p.predict_q2(x)
    assert q.shape == (8,)
    # predict_q2 toggles to eval; verify the module is restored to train.
    assert p.training is True


def test_acceptance_tx_predict_L_uses_commit_threshold():
    p = AcceptanceTx(family="hidden_per_pos_v1", gamma=8)
    p.eval()
    x = torch.randn(8, 768)
    L = p.predict_L(x, tau=0.5)
    assert isinstance(L, int)
    assert 0 <= L <= 8


def test_acceptance_tx_rejects_non_phase9_family():
    with pytest.raises(ValueError, match="Phase 9"):
        AcceptanceTx(family="hidden", gamma=8)
    with pytest.raises(ValueError, match="Phase 9"):
        AcceptanceTx(family="hidden_per_pos", gamma=8)
    with pytest.raises(ValueError, match="Phase 9"):
        AcceptanceTx(family="hidden_per_pos_ml", gamma=8)


# ------------------------------------------------------------------
# LengthTx
# ------------------------------------------------------------------


def test_length_tx_v1_shape_and_classes():
    p = LengthTx(family="hidden_per_pos_v1", gamma=8)
    assert p.kind == "committed_length"
    assert p.deploy_mode == "V-free"
    assert p.n_classes == 9
    x = torch.randn(4, 8, 768)
    tau = torch.tensor([0.3, 0.5, 0.7, 0.9])
    logits = p(x, tau)
    assert logits.shape == (4, 9)


def test_length_tx_v3_v4_deploy_and_shape():
    cases = [
        ("hidden_per_pos_v3", 768 + 3,         "V-prefix"),
        ("hidden_per_pos_v4", 768 + 5 + 1600,  "V-prefix"),
    ]
    for fam, in_dim, deploy in cases:
        p = LengthTx(family=fam, gamma=8)
        assert p.deploy_mode == deploy
        x = torch.randn(2, 8, in_dim)
        tau = torch.tensor([0.5, 0.5])
        logits = p(x, tau)
        assert logits.shape == (2, 9)


def test_length_tx_predict_L_returns_int():
    p = LengthTx(family="hidden_per_pos_v1", gamma=8)
    p.eval()
    x = torch.randn(8, 768)
    L = p.predict_L(x, tau=0.5)
    assert isinstance(L, int)
    assert 0 <= L <= 8


def test_length_tx_logits_depend_on_tau():
    """τ-conditioning must actually change the logits.

    If post-encoder concat was whitened out (as happened with the
    original LengthMLP before the fix), this test would catch it.
    """
    torch.manual_seed(42)
    p = LengthTx(family="hidden_per_pos_v1", gamma=8)
    p.eval()
    x = torch.randn(1, 8, 768)
    logits_lo = p(x, torch.tensor([0.1]))
    logits_hi = p(x, torch.tensor([0.9]))
    assert not torch.allclose(logits_lo, logits_hi), (
        "LengthTx logits are identical across τ — τ-conditioning not "
        "functional (post-encoder concat may have been whitened out)."
    )


def test_length_tx_accepts_unbatched_input():
    p = LengthTx(family="hidden_per_pos_v1", gamma=8)
    x = torch.randn(8, 768)
    tau = torch.tensor(0.5)
    logits = p(x, tau)
    assert logits.shape == (9,)


def test_length_tx_rejects_non_phase9_family():
    with pytest.raises(ValueError, match="Phase 9"):
        LengthTx(family="hidden_numeric", gamma=8)
    with pytest.raises(ValueError, match="Phase 9"):
        LengthTx(family="hidden_per_pos", gamma=8)


def test_length_tx_predict_L_restores_train_mode():
    p = LengthTx(family="hidden_per_pos_v1", gamma=8, dropout=0.5)
    p.train()
    x = torch.randn(8, 768)
    _ = p.predict_L(x, tau=0.5)
    assert p.training is True


# ------------------------------------------------------------------
# End-to-end shape round-trip with the actual extract_features helper
# ------------------------------------------------------------------


def test_tx_predictors_accept_extracted_features():
    """Features from accpre.collect.features.extract_features must flow
    end-to-end into AcceptanceTx / LengthTx without shape surgery."""
    import torch as _torch
    from accpre.collect.features import extract_features
    from accpre.core.protocol import ProtocolConfig
    from accpre.core.schema import RoundRecord

    gamma, H, Hv = 4, 768, 1600
    proto = ProtocolConfig()
    rec = RoundRecord(
        schema_version=proto.schema_version, protocol=proto,
        prompt_idx=0, round_idx=0, round_rng_seed=1,
        gamma=gamma, T=2, prefix_len=5,
        draft_tokens=[0] * gamma,
        q_j=[0.5] * gamma, p_j=[0.5] * gamma,
        min_pq_j=[0.5] * gamma, U_j=[0.0] * gamma,
        accepted_j=[1] * gamma, survived_j=[1] * gamma,
        L=gamma, bonus_or_fallback_token=0,
        draft_time=0.0, verify_time=0.0,
    )
    rec.drafter_hidden_per_pos = _torch.randn(gamma, H)
    rec.verifier_hidden = _torch.randn(Hv)
    rec.verifier_entropy = 2.3
    rec.verifier_margin = 1.1
    rec.verifier_top1_prob = 0.4
    rec.drafter_entropy_j = [1.0] * gamma
    rec.drafter_margin_j = [0.5] * gamma
    rec.drafter_top1_prob_j = [0.3] * gamma

    for fam in (
        "hidden_per_pos_v1", "hidden_per_pos_v2",
        "hidden_per_pos_v3", "hidden_per_pos_v4",
    ):
        feats = extract_features(rec, fam)
        acc = AcceptanceTx(family=fam, gamma=gamma)
        assert acc(feats).shape == (gamma,)
        lenp = LengthTx(family=fam, gamma=gamma)
        logits = lenp(feats, torch.tensor(0.5))
        assert logits.shape == (gamma + 1,)
