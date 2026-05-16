"""Static / runtime sanity check: every head consumes the j/gamma position scalar.

For each head class:
  1. Build the head with gamma=4 in eval mode.
  2. Run forward A with the head's standard pos_scalar buffer ([0, 0.25, 0.5, 0.75]).
  3. Replace the pos_scalar buffer with a different vector ([0.99, 0.66, 0.33, 0.0])
     and run forward B on the same features/token_ids.
  4. Assert that A != B componentwise.

If a head accidentally drops the position scalar (wrong concat axis, missing
buffer registration, etc.), the outputs will be identical and this check fires.

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


_TARGET_ARCHS = ("mlp_pos", "causal_transformer_pos", "bidirectional_transformer_pos")


def _check_one(arch: str, gamma: int = 4, seed: int = 1234) -> None:
    torch.manual_seed(seed)
    head = build_head(arch, gamma=gamma).eval()

    # Reference inputs (random but fixed via seed).
    F_v2 = head_in_dim(head.token_emb_dim) - 1 - head.token_emb_dim
    features = torch.randn(2, gamma, F_v2)
    token_ids = torch.randint(0, 50256, (2, gamma), dtype=torch.long)

    # Forward A — default pos_scalar.
    with torch.no_grad():
        out_a = head(features, token_ids=token_ids).clone()

    # Replace pos_scalar with a non-trivial transform.
    new_pos = torch.tensor([0.99, 0.66, 0.33, 0.0], dtype=head.pos_scalar.dtype)[:gamma]
    head.pos_scalar.copy_(new_pos)

    with torch.no_grad():
        out_b = head(features, token_ids=token_ids).clone()

    diff = (out_a - out_b).abs().max().item()
    if diff < 1e-6:
        raise AssertionError(
            f"[FAIL] {arch}: outputs identical for two different pos_scalar values "
            f"(max abs diff = {diff:.2e}); position feature is NOT consumed."
        )
    print(f"[ok ] {arch}: max abs diff between two pos_scalar settings = {diff:.4e}")


def main() -> int:
    print("=== sanity_position_feature ===")
    failures = 0
    for arch in _TARGET_ARCHS:
        try:
            _check_one(arch)
        except AssertionError as e:
            print(str(e))
            failures += 1
    if failures:
        print(f"FAILED ({failures}/{len(_TARGET_ARCHS)})")
        return 1
    print(f"PASS ({len(_TARGET_ARCHS)}/{len(_TARGET_ARCHS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
