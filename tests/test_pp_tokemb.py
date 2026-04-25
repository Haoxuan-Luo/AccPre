"""Shape / correctness tests for the Phase 12 token-embedding heads."""

from __future__ import annotations

import pytest
import torch

from accpre.predictors.pp_tokemb import (
    AcceptanceMLPPerPosTokEmb,
    AcceptanceTxTokEmb,
    DEFAULT_TOK_VOCAB_SIZE,
)


GAMMA = 8
H = 768
F_V2 = H + 5       # hidden_per_pos_v2 = drafter hidden + 5 scalars


@pytest.mark.parametrize(
    "cls,extra_kw",
    [
        (AcceptanceMLPPerPosTokEmb, {"hidden_dim": 64, "dropout": 0.0}),
        (AcceptanceTxTokEmb, {"d_model": 32, "num_layers": 1, "num_heads": 2, "dropout": 0.0}),
    ],
)
def test_forward_shapes_batched(cls, extra_kw):
    p = cls(family="hidden_per_pos_v2", gamma=GAMMA, token_emb_dim=16, **extra_kw)
    p.eval()
    feats = torch.randn(3, GAMMA, F_V2)
    tok = torch.randint(0, DEFAULT_TOK_VOCAB_SIZE, (3, GAMMA))
    out = p(feats, token_ids=tok)
    assert out.shape == (3, GAMMA)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize(
    "cls,extra_kw",
    [
        (AcceptanceMLPPerPosTokEmb, {"hidden_dim": 64, "dropout": 0.0}),
        (AcceptanceTxTokEmb, {"d_model": 32, "num_layers": 1, "num_heads": 2, "dropout": 0.0}),
    ],
)
def test_forward_shapes_unbatched(cls, extra_kw):
    p = cls(family="hidden_per_pos_v2", gamma=GAMMA, token_emb_dim=16, **extra_kw)
    p.eval()
    feats = torch.randn(GAMMA, F_V2)
    tok = torch.randint(0, DEFAULT_TOK_VOCAB_SIZE, (GAMMA,))
    out = p(feats, token_ids=tok)
    assert out.shape == (GAMMA,)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_requires_token_ids():
    for cls in (AcceptanceMLPPerPosTokEmb, AcceptanceTxTokEmb):
        p = cls(family="hidden_per_pos_v2", gamma=GAMMA, token_emb_dim=8)
        feats = torch.randn(GAMMA, F_V2)
        with pytest.raises(ValueError):
            p(feats)


def test_rejects_wrong_family():
    for cls in (AcceptanceMLPPerPosTokEmb, AcceptanceTxTokEmb):
        with pytest.raises(ValueError):
            cls(family="hidden_per_pos_v1", gamma=GAMMA, token_emb_dim=8)


def test_predict_L_uses_token_ids():
    """predict_L passes token_ids through kwargs → predict_q2 → forward."""
    p = AcceptanceMLPPerPosTokEmb(
        family="hidden_per_pos_v2", gamma=GAMMA, hidden_dim=32,
        dropout=0.0, token_emb_dim=8,
    )
    p.eval()
    feats = torch.randn(GAMMA, F_V2)
    tok = torch.randint(0, DEFAULT_TOK_VOCAB_SIZE, (GAMMA,))
    L = p.predict_L(feats, tau=0.5, token_ids=tok)
    assert isinstance(L, int)
    assert 0 <= L <= GAMMA


def test_deploy_mode_is_vfree():
    for cls in (AcceptanceMLPPerPosTokEmb, AcceptanceTxTokEmb):
        p = cls(family="hidden_per_pos_v2", gamma=GAMMA, token_emb_dim=8)
        assert p.deploy_mode == "V-free"
        assert p.kind == "acceptance"
