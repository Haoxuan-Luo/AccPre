"""Dataset registry for predictor sweeps.

A uniform API that lets sweep scripts stay dataset-agnostic. Each
`DatasetBundle` tells the framework:

  - where the pre-collected RoundRecords live (training data),
  - where the precomputed dependence-target file lives (dep families),
  - how to load online-eval prompts (by count, drawn from the test split).

Wired datasets (loaders exist in code; whether the records / dep files
are populated on disk is a runtime check that callers handle):

  owt       — OpenWebText, 80-prompt pool (40/20/20), prefix_len=32.
  cnn_dm    — CNN/DailyMail v3.0.0, 80-prompt pool (40/20/20).
  mt_bench  — MT-Bench (lmsys),     40-prompt pool (20/10/10) — capped
              by usable-question count (only 48/80 official MT-Bench
              prompts clear prefix_len=32 after tokenization).

Per-dataset split sizes come from `accpre.data.splits.DATASETS`. Path
convention for new datasets is:

    data_collected/stage1_pp_<dataset>.pt
    data_collected/stage1_pp_<dataset>_dep.pt

OWT keeps its legacy un-suffixed path (`stage1_pp.pt` / `stage1_pp_dep.pt`)
for backward compatibility with files already on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


PromptLoader = Callable[[int], Tuple[List[Tuple[torch.Tensor, str]], List[int]]]


@dataclass
class DatasetBundle:
    name: str
    records_path: Path
    dep_targets_path: Optional[Path]
    prompt_loader: PromptLoader
    description: str
    wired: bool


def _make_test_prompt_loader(dataset_name: str) -> PromptLoader:
    """Build a `(n_prompts) -> (prompts, indices)` loader for `dataset_name`.

    Returns prompts from the dataset's TEST split (last `test_n` entries
    of the pool). Indices are the global pool indices, used by the eval
    harness as `prompt_idx` when seeding round RNGs and feature
    extraction — they must match the indices used during stage-1
    collection on the same dataset.
    """
    def loader(
        n_prompts: int,
    ) -> Tuple[List[Tuple[torch.Tensor, str]], List[int]]:
        from accpre.data.prompts import load_prompts
        from accpre.data.splits import get_split_config

        cfg = get_split_config(dataset_name)
        pool = load_prompts(
            dataset=cfg.name, n_prompts=cfg.pool_size,
            prefix_len=cfg.prefix_len, seed=cfg.seed,
        )
        test_offset = cfg.train_n + cfg.val_n
        test_pool = pool[test_offset:cfg.pool_size]
        if n_prompts > len(test_pool):
            raise ValueError(
                f"{dataset_name!r} test split has {len(test_pool)} "
                f"prompts; requested {n_prompts}."
            )
        prompts = test_pool[:n_prompts]
        indices = list(range(test_offset, test_offset + n_prompts))
        return prompts, indices

    return loader


_REGISTRY = {
    "owt": DatasetBundle(
        name="owt",
        records_path=_REPO_ROOT / "data_collected/stage1_pp.pt",
        dep_targets_path=_REPO_ROOT / "data_collected/stage1_pp_dep.pt",
        prompt_loader=_make_test_prompt_loader("owt"),
        description=(
            "OpenWebText, 80-prompt pool (40/20/20), prefix_len=32, seed=42."
        ),
        wired=True,
    ),
    "cnn_dm": DatasetBundle(
        name="cnn_dm",
        records_path=_REPO_ROOT / "data_collected/stage1_pp_cnn_dm.pt",
        dep_targets_path=_REPO_ROOT / "data_collected/stage1_pp_cnn_dm_dep.pt",
        prompt_loader=_make_test_prompt_loader("cnn_dm"),
        description=(
            "CNN/DailyMail v3.0.0, 80-prompt pool (40/20/20), "
            "prefix_len=32, seed=42."
        ),
        wired=True,
    ),
    "mt_bench": DatasetBundle(
        name="mt_bench",
        records_path=_REPO_ROOT / "data_collected/stage1_pp_mt_bench.pt",
        dep_targets_path=_REPO_ROOT / "data_collected/stage1_pp_mt_bench_dep.pt",
        prompt_loader=_make_test_prompt_loader("mt_bench"),
        description=(
            "MT-Bench (lmsys), 40-prompt pool (20/10/10), "
            "prefix_len=32, seed=42."
        ),
        wired=True,
    ),
    "cnn_dm_300": DatasetBundle(
        name="cnn_dm_300",
        records_path=_REPO_ROOT / "data_collected/stage1_pp_cnn_dm_300_g15_T1.pt",
        dep_targets_path=_REPO_ROOT / "data_collected/stage1_pp_cnn_dm_300_g15_T1_dep.pt",
        prompt_loader=_make_test_prompt_loader("cnn_dm_300"),
        description=(
            "CNN/DailyMail v3.0.0, 300-prompt pool (160/40/100), "
            "prefix_len=32, seed=42."
        ),
        wired=True,
    ),
}


def get_dataset(name: str) -> DatasetBundle:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown dataset {name!r}; registered: {sorted(_REGISTRY)}. "
            f"Add an entry to accpre/sweep/datasets.py to extend the registry."
        )
    return _REGISTRY[name]


def list_datasets() -> List[Tuple[str, bool, str]]:
    """Return [(name, wired, description), ...] for all registered datasets."""
    return [(b.name, b.wired, b.description) for b in _REGISTRY.values()]
