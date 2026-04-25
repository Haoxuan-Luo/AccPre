"""Dataset registry for predictor sweeps.

A uniform API that lets sweep scripts stay dataset-agnostic. Each
`DatasetBundle` tells the framework:

  - where the pre-collected RoundRecords live (training data for frozen),
  - where the precomputed dependence-target file lives (if any),
  - how to load online-eval prompts (by count).

Currently wired:
  owt — canonical 80-prompt OpenWebText pool (split 40/20/20).

Stubbed (raises a clear NotImplementedError when selected):
  cnn_dm — CNN/DailyMail. Needs (a) a CNN/DM prompt loader added to
           `accpre/data/prompts.py`, (b) a stage-1 RoundRecord collection
           on CNN/DM (via `scripts/recollect_all.py`-style pipeline),
           and (c) a precomputed dep-targets file on that collection.

If/when you wire a third dataset, add a DatasetBundle entry below.
Keep the `name` key lowercase and match it to the --dataset flag.
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


def _owt_test_prompts(
    n_prompts: int,
) -> Tuple[List[Tuple[torch.Tensor, str]], List[int]]:
    from accpre.data.splits import test_split, TRAIN_N, VAL_N
    pool = test_split()
    if n_prompts > len(pool):
        raise ValueError(
            f"OWT test split has {len(pool)} prompts; requested {n_prompts}."
        )
    prompts = pool[:n_prompts]
    offset = TRAIN_N + VAL_N
    indices = list(range(offset, offset + n_prompts))
    return prompts, indices


def _stub_loader(name: str, reason: str) -> PromptLoader:
    def raiser(n_prompts: int):
        raise NotImplementedError(
            f"dataset {name!r} is not wired: {reason}. "
            f"Edit accpre/sweep/datasets.py to register it."
        )
    return raiser


_REGISTRY = {
    "owt": DatasetBundle(
        name="owt",
        records_path=_REPO_ROOT / "data_collected/stage1_pp.pt",
        dep_targets_path=_REPO_ROOT / "data_collected/stage1_pp_dep.pt",
        prompt_loader=_owt_test_prompts,
        description="OpenWebText, 80-prompt pool, prefix_len=32, seed=42.",
        wired=True,
    ),
    "cnn_dm": DatasetBundle(
        name="cnn_dm",
        records_path=_REPO_ROOT / "data_collected/cnn_dm_pp.pt",
        dep_targets_path=_REPO_ROOT / "data_collected/cnn_dm_pp_dep.pt",
        prompt_loader=_stub_loader(
            "cnn_dm",
            "needs a load_cnn_dm_prompts helper in accpre/data/prompts.py, "
            "a stage-1 collection on that pool, and (for dep training) a "
            "precomputed dep-targets file",
        ),
        description=(
            "[NOT WIRED] CNN/DailyMail. Missing: prompt loader, collected "
            "records, dep-targets file."
        ),
        wired=False,
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
