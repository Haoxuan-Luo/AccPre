"""Canonical prompt splits for v1.

Deterministic train / val / test split over a fixed prompt pool. For
Phase 1 we only need a small test split (for baselines). Phase 2+ will
use train + val from the same pool with the same seed.

Split discipline:
  - `seed=42`, `prefix_len=32` (matches the pool used throughout the old
    project).
  - Indices 0..TRAIN_N-1          -> train
  - Indices TRAIN_N..TRAIN_N+VAL_N-1 -> val
  - Indices TRAIN_N+VAL_N..         -> test
  - Split sizes are exposed as module-level constants so changes are
    visible in one place and force a schema-version bump.
"""

from __future__ import annotations

from typing import List, Tuple
import torch

from accpre.data.prompts import load_owt_prompts


TRAIN_N: int = 40
VAL_N: int = 20
TEST_N: int = 20
POOL_SIZE: int = TRAIN_N + VAL_N + TEST_N   # 80

PROMPT_SEED: int = 42
PREFIX_LEN: int = 32


def _load_pool(n: int = POOL_SIZE) -> List[Tuple[torch.Tensor, str]]:
    return load_owt_prompts(
        n_prompts=n, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )


def test_split() -> List[Tuple[torch.Tensor, str]]:
    pool = _load_pool(POOL_SIZE)
    return pool[TRAIN_N + VAL_N:TRAIN_N + VAL_N + TEST_N]


def val_split() -> List[Tuple[torch.Tensor, str]]:
    pool = _load_pool(POOL_SIZE)
    return pool[TRAIN_N:TRAIN_N + VAL_N]


def train_split() -> List[Tuple[torch.Tensor, str]]:
    pool = _load_pool(POOL_SIZE)
    return pool[:TRAIN_N]
