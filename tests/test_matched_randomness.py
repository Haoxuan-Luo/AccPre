"""Tests for the matched-randomness discipline.

These tests ensure:
  - round_rng_seed derivation is deterministic and unique across prompts
    and rounds.
  - Same seed ⇒ same torch.Generator output stream.
  - The three sub-streams (draft_rng, accept_rng, fallback_rng) derived
    via protocol salts are independent of each other.
"""

from __future__ import annotations

import torch

from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig, derive_seed


def test_derive_seed_is_deterministic():
    """Same inputs always produce the same seed (cross-process stable)."""
    proto = ProtocolConfig()
    s1 = derive_seed(proto, 0, 0)
    s2 = derive_seed(proto, 0, 0)
    assert s1 == s2


def test_derive_seed_unique_over_small_grid():
    proto = ProtocolConfig()
    seen: set[int] = set()
    for p in range(20):
        for r in range(10):
            seen.add(derive_seed(proto, p, r))
    assert len(seen) == 200


def test_derive_seed_differs_across_protocols():
    """Changing the protocol changes the fingerprint, hence the seed."""
    a = ProtocolConfig(q_mode="A")
    b = ProtocolConfig(q_mode="B")
    assert derive_seed(a, 0, 0) != derive_seed(b, 0, 0)


def test_derive_seed_differs_across_schema_versions():
    a = ProtocolConfig(schema_version=1)
    b = ProtocolConfig(schema_version=2)
    assert derive_seed(a, 0, 0) != derive_seed(b, 0, 0)


def test_uniform_stream_replays_from_seed():
    """torch.Generator with the same seed yields identical uniforms."""
    seed = 12345
    g1 = torch.Generator(device="cpu")
    g1.manual_seed(seed)
    u1 = torch.rand(10, generator=g1)

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(seed)
    u2 = torch.rand(10, generator=g2)

    assert torch.equal(u1, u2)


def test_sub_streams_differ_across_salts():
    """Different salts yield different uniform streams."""
    proto = ProtocolConfig()
    seed = 0xABCDEF
    g_draft = _make_generator("cpu", seed ^ proto.draft_salt)
    g_acc = _make_generator("cpu", seed ^ proto.accept_salt)
    g_fb = _make_generator("cpu", seed ^ proto.fallback_salt)

    u_draft = torch.rand(16, generator=g_draft)
    u_acc = torch.rand(16, generator=g_acc)
    u_fb = torch.rand(16, generator=g_fb)

    assert not torch.allclose(u_draft, u_acc)
    assert not torch.allclose(u_draft, u_fb)
    assert not torch.allclose(u_acc, u_fb)


def test_make_generator_masks_large_seeds():
    """Large seeds are masked to int63 so manual_seed doesn't choke."""
    seed_way_too_big = (1 << 70) + 17
    g = _make_generator("cpu", seed_way_too_big)
    # Just drawing should not raise.
    _ = torch.rand(1, generator=g)
