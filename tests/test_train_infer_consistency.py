"""Phase-4 anti-drift test, promoted to Stage 2B.1 hard gate.

Invariant: when the drafter is in `.eval()` mode with frozen weights,
`drafter.draft(...)` (inference path, fully no_grad) must produce
bit-identical `draft_tokens` and `draft_log_probs` as
`drafter.draft_with_features_grad(...)` (training path, grad-enabled
at the final DDPM step) under the SAME prefix and generator state.

If this test fails, training-time drafting has drifted from the
inference-time drafting, which is exactly the audit risk the design
was written to prevent.

Marked `slow` because it loads the real MDLM.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def drafter():
    from accpre.models.drafter_mdlm import MDLMDrafter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dr = MDLMDrafter(device=device, dtype=torch.float32)
    dr.model.eval()  # deterministic forward under fixed weights
    return dr


def test_draft_and_draft_with_features_grad_match(drafter):
    """draft() vs draft_with_features_grad() produce identical tokens + q rows."""
    device = drafter.device
    prefix = torch.arange(32, dtype=torch.long, device=device)

    # --- no-grad path ---
    g1 = torch.Generator(device=device)
    g1.manual_seed(12345)
    tokens_ng, log_probs_ng = drafter.draft(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g1,
    )

    # --- grad path (same seed, same model) ---
    g2 = torch.Generator(device=device)
    g2.manual_seed(12345)
    tokens_g, log_probs_g, drafter_hidden = drafter.draft_with_features_grad(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g2,
    )

    # Token selection must be bit-identical.
    assert torch.equal(tokens_ng.cpu(), tokens_g.detach().cpu()), (
        "draft_tokens differ between no_grad and grad paths"
    )
    # q rows must be numerically identical (same logits under .eval()).
    assert torch.allclose(
        log_probs_ng.cpu(), log_probs_g.detach().cpu(), atol=1e-6
    ), "draft_log_probs differ between no_grad and grad paths"
    # The returned drafter_hidden is on the same device type (cuda vs cpu).
    # Note: torch.device("cuda") has no index, drafter_hidden.device has
    # index 0 under real GPU — compare by type to avoid that mismatch.
    assert drafter_hidden.device.type == torch.device(device).type
    # Gradient tracking: must be a non-leaf tensor with grad_fn.
    assert drafter_hidden.requires_grad or drafter_hidden.grad_fn is not None, (
        "draft_with_features_grad must return a hidden with grad tracking"
    )


def test_draft_with_features_grad_is_reproducible(drafter):
    """Calling grad path twice with the same seed produces identical outputs."""
    device = drafter.device
    prefix = torch.arange(32, dtype=torch.long, device=device)

    g1 = torch.Generator(device=device); g1.manual_seed(42)
    t1, lp1, h1 = drafter.draft_with_features_grad(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g1,
    )
    g2 = torch.Generator(device=device); g2.manual_seed(42)
    t2, lp2, h2 = drafter.draft_with_features_grad(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g2,
    )

    assert torch.equal(t1.cpu(), t2.cpu())
    assert torch.allclose(lp1.detach().cpu(), lp2.detach().cpu(), atol=1e-6)
    assert torch.allclose(h1.detach().cpu(), h2.detach().cpu(), atol=1e-6)


def test_multilayer_layers_does_not_change_tokens(drafter):
    """Phase 6 invariant: passing `layers=(-2, -1)` to draft_with_features
    must not change `draft_tokens` / `draft_log_probs` vs the inference-only
    `draft(...)` path with the same generator state. The multi-layer
    extraction side-path only affects the returned hidden tensor."""
    device = drafter.device
    prefix = torch.arange(32, dtype=torch.long, device=device)

    g1 = torch.Generator(device=device); g1.manual_seed(777)
    tokens_ng, log_probs_ng = drafter.draft(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g1,
    )

    g2 = torch.Generator(device=device); g2.manual_seed(777)
    tokens_ml, log_probs_ml, hidden_ml = drafter.draft_with_features(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g2,
        pool="per_position", layers=(-2, -1),
    )

    assert torch.equal(tokens_ng.cpu(), tokens_ml.cpu()), (
        "multi-layer extraction changed drafted tokens — anti-drift violated"
    )
    assert torch.allclose(
        log_probs_ng.cpu(), log_probs_ml.cpu(), atol=1e-6
    ), "multi-layer extraction changed log-probs — anti-drift violated"

    H = drafter.hidden_size
    assert tuple(hidden_ml.shape) == (8, 2 * H), (
        f"expected (γ=8, 2H={2*H}), got {tuple(hidden_ml.shape)}"
    )


def test_multilayer_final_slice_equals_single_layer(drafter):
    """Phase 6 invariant: the LAST H columns of the multi-layer hidden
    equal the single-layer (default `layers=(-1,)`) hidden, position-wise.
    This is load-bearing for storing one (γ, 2H) field and recovering
    (γ, H) for pp_base by slicing [:, H:]."""
    device = drafter.device
    prefix = torch.arange(32, dtype=torch.long, device=device)

    g1 = torch.Generator(device=device); g1.manual_seed(4242)
    _, _, h_final = drafter.draft_with_features(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g1,
        pool="per_position", layers=(-1,),
    )
    g2 = torch.Generator(device=device); g2.manual_seed(4242)
    _, _, h_ml = drafter.draft_with_features(
        prefix, gamma=8, T=2, temperature=1.0, q_mode="A", generator=g2,
        pool="per_position", layers=(-2, -1),
    )

    H = drafter.hidden_size
    assert torch.allclose(h_final.cpu(), h_ml[:, H:].cpu(), atol=1e-6), (
        "final-layer slice of multi-layer hidden != single-layer hidden"
    )
