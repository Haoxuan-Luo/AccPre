"""Runtime sanity check: causal_transformer_pos enforces a real causal mask.

Strategy (information-flow probe — preferred over NaN propagation):

  Build two inputs (A and B) that AGREE on positions 0..j-1 and DIFFER on
  positions >= j (we use random vs. zero fill). Forward both through the
  head and compare outputs.

  - Causal head      : output at positions 0..j-1 must be IDENTICAL between
                       A and B (those positions cannot attend forward).
                       Output at position >= j MAY differ.
  - Bidirectional head: output at every position MAY differ; in practice a
                       random vs zero fill at later positions perturbs
                       earlier outputs through full attention. We assert
                       that AT LEAST one position 0..j-1 differs (positive
                       leakage signal).

Why not NaN-propagation: in IEEE-754 NaN + (-inf) = NaN, and the mask is
APPLIED ADDITIVELY to attention scores. So a NaN at a "blocked" key still
contaminates the row through softmax. Testing causal masking with NaN is
unreliable; testing with finite inputs is sound.

CPU-only; runs in <1 second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXP_ROOT))

from scripts.heads import build_head, head_in_dim


def _two_inputs(gamma: int, F_v2: int, j_split: int, seed: int = 1234):
    """Two inputs equal on positions 0..j_split-1; differ on >= j_split."""
    g = torch.Generator().manual_seed(seed)
    common = torch.randn(1, j_split, F_v2, generator=g)
    fillA_late = torch.randn(1, gamma - j_split, F_v2, generator=g)
    fillB_late = torch.zeros(1, gamma - j_split, F_v2)
    A = torch.cat([common, fillA_late], dim=1)
    B = torch.cat([common, fillB_late], dim=1)
    # Token IDs: same across A and B (we only stress positional information flow).
    token_ids = torch.zeros(1, gamma, dtype=torch.long)
    return A, B, token_ids


def main() -> int:
    print("=== sanity_causal_mask ===")
    gamma = 4
    j_split = 3   # positions 0..2 must be unaffected by 3 under causal mask.

    # Causal head: positions < j_split must be identical between A and B.
    causal = build_head("causal_transformer_pos", gamma=gamma).eval()
    F_v2 = head_in_dim(causal.token_emb_dim) - 1 - causal.token_emb_dim
    A, B, tok = _two_inputs(gamma, F_v2, j_split)
    with torch.no_grad():
        out_a = causal(A, token_ids=tok).squeeze(0)
        out_b = causal(B, token_ids=tok).squeeze(0)
    diff_safe = (out_a[:j_split] - out_b[:j_split]).abs().max().item()
    diff_late = (out_a[j_split:] - out_b[j_split:]).abs().max().item()
    print(f"causal head: max|A-B| at pos<{j_split} = {diff_safe:.2e}; "
          f"max|A-B| at pos>={j_split} = {diff_late:.2e}")
    if diff_safe > 1e-6:
        raise AssertionError(
            f"[FAIL] causal_transformer_pos: positions 0..{j_split-1} should be "
            f"identical between two inputs differing only at >= {j_split} "
            f"(causal mask blocks future). Got max diff {diff_safe:.4e}."
        )
    # Sanity: positions >= j_split SHOULD differ at least somewhat (those
    # positions DO see the differing inputs); otherwise our test is degenerate.
    if diff_late <= 1e-6:
        raise AssertionError(
            f"[FAIL] causal_transformer_pos: positions {j_split}..gamma-1 are "
            f"identical between A and B (max diff {diff_late:.4e}); test is "
            f"degenerate (perhaps the heads are returning a constant?)."
        )
    print(f"[ok ] causal_transformer_pos: causal mask blocks forward leakage")

    # Bidirectional head: outputs at positions 0..j_split-1 SHOULD differ
    # because they attend forward to the perturbed positions.
    bidir = build_head("bidirectional_transformer_pos", gamma=gamma).eval()
    F_v2 = head_in_dim(bidir.token_emb_dim) - 1 - bidir.token_emb_dim
    A, B, tok = _two_inputs(gamma, F_v2, j_split)
    with torch.no_grad():
        out_a = bidir(A, token_ids=tok).squeeze(0)
        out_b = bidir(B, token_ids=tok).squeeze(0)
    diff_safe = (out_a[:j_split] - out_b[:j_split]).abs().max().item()
    diff_late = (out_a[j_split:] - out_b[j_split:]).abs().max().item()
    print(f"bidir  head: max|A-B| at pos<{j_split} = {diff_safe:.2e}; "
          f"max|A-B| at pos>={j_split} = {diff_late:.2e}")
    if diff_safe <= 1e-6:
        raise AssertionError(
            f"[FAIL] bidirectional_transformer_pos: positions 0..{j_split-1} are "
            f"identical between A and B (max diff {diff_safe:.4e}); the "
            f"bidirectional head is NOT actually bidirectional (no leakage from "
            f"positions >= {j_split} into earlier positions)."
        )
    print(f"[ok ] bidirectional_transformer_pos: forward attention leaks as expected")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
