"""Canonical prompt splits.

Deterministic train / val / test split over a fixed prompt pool, per
dataset. The OWT defaults are preserved as module-level constants
(`TRAIN_N=40, VAL_N=20, TEST_N=20, POOL_SIZE=80, PREFIX_LEN=32,
PROMPT_SEED=42`) so legacy phase scripts continue to work.

For new code (sweep framework, multi-dataset collection / dep-targets)
use the dataset-aware API instead:
    cfg = get_split_config(dataset)
    cfg.train_n / cfg.val_n / cfg.test_n / cfg.pool_size
    cfg.prefix_len / cfg.seed
    train_split(dataset)  # also val_split / test_split
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

from accpre.data.prompts import load_prompts, load_owt_prompts


# --- Legacy OWT constants (used directly by Phase 22/23/25 scripts). ---
TRAIN_N: int = 40
VAL_N: int = 20
TEST_N: int = 20
POOL_SIZE: int = TRAIN_N + VAL_N + TEST_N   # 80

PROMPT_SEED: int = 42
PREFIX_LEN: int = 32


@dataclass(frozen=True)
class DatasetSplitConfig:
    name: str
    pool_size: int
    train_n: int
    val_n: int
    test_n: int
    prefix_len: int
    seed: int

    def __post_init__(self) -> None:
        if self.train_n + self.val_n + self.test_n != self.pool_size:
            raise ValueError(
                f"split sizes {self.train_n}+{self.val_n}+{self.test_n}="
                f"{self.train_n + self.val_n + self.test_n} != "
                f"pool_size={self.pool_size}"
            )


# Per-dataset split discipline. For OWT and CNN/DM the 80-prompt pool
# matches the historical OWT layout. MT-Bench is gated by usable
# question count: only 48 of the 80 official questions clear the
# prefix_len=32 filter, so we cap the pool at 40 (20/10/10) which
# leaves a small headroom margin and matches the OWT 0.5/0.25/0.25
# train/val/test ratio.
DATASETS: Dict[str, DatasetSplitConfig] = {
    "owt": DatasetSplitConfig(
        name="owt",
        pool_size=80, train_n=40, val_n=20, test_n=20,
        prefix_len=32, seed=42,
    ),
    "cnn_dm": DatasetSplitConfig(
        name="cnn_dm",
        pool_size=80, train_n=40, val_n=20, test_n=20,
        prefix_len=32, seed=42,
    ),
    "mt_bench": DatasetSplitConfig(
        name="mt_bench",
        pool_size=40, train_n=20, val_n=10, test_n=10,
        prefix_len=32, seed=42,
    ),
    # OWT_Frozen_0429 experiment splits.
    "owt_300": DatasetSplitConfig(
        name="owt_300",
        pool_size=300, train_n=160, val_n=40, test_n=100,
        prefix_len=32, seed=42,
    ),
    "owt_smoke": DatasetSplitConfig(
        name="owt_smoke",
        pool_size=8, train_n=4, val_n=2, test_n=2,
        prefix_len=32, seed=42,
    ),
    # 0505_CNNDM_compare experiment split — mirrors owt_300 sizes.
    "cnn_dm_300": DatasetSplitConfig(
        name="cnn_dm_300",
        pool_size=300, train_n=160, val_n=40, test_n=100,
        prefix_len=32, seed=42,
    ),
}


def get_split_config(dataset: str) -> DatasetSplitConfig:
    if dataset not in DATASETS:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}"
        )
    return DATASETS[dataset]


def _load_pool_for(dataset: str) -> List[Tuple[torch.Tensor, str]]:
    cfg = get_split_config(dataset)
    return load_prompts(
        dataset=cfg.name, n_prompts=cfg.pool_size,
        prefix_len=cfg.prefix_len, seed=cfg.seed,
    )


def train_split(dataset: str = "owt") -> List[Tuple[torch.Tensor, str]]:
    cfg = get_split_config(dataset)
    pool = _load_pool_for(dataset)
    return pool[:cfg.train_n]


def val_split(dataset: str = "owt") -> List[Tuple[torch.Tensor, str]]:
    cfg = get_split_config(dataset)
    pool = _load_pool_for(dataset)
    return pool[cfg.train_n:cfg.train_n + cfg.val_n]


def test_split(dataset: str = "owt") -> List[Tuple[torch.Tensor, str]]:
    cfg = get_split_config(dataset)
    pool = _load_pool_for(dataset)
    return pool[cfg.train_n + cfg.val_n:cfg.pool_size]
