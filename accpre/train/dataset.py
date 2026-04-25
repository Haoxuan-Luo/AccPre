"""Datasets that wrap Phase 2 `RoundRecord`s for predictor training.

Two dataset classes:
  - FrozenAcceptanceDataset: (features, q2_target, survived) per record.
  - FrozenLengthDataset:     (features, tau, L_tau) per (record, τ).

Both consume lists of `RoundRecord` produced by Stage 2A.0 collection;
features are computed on demand via
`accpre.collect.features.extract_features`.

No dataset class calls the drafter or verifier — Stage 2A.0 already
logged every feature they need.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from accpre.collect.features import extract_features
from accpre.core.commit import commit_threshold
from accpre.core.schema import RoundRecord


def _load_dep_targets(path: str) -> Dict:
    """Load the Phase-23 dependence-target file produced by
    `scripts/collect_dep_targets.py`.

    Layout on disk (torch.save):
        {"protocol_fp": str,
         "drafter_fp":  str | None,
         "sigma":        float,
         "gamma":        int,
         "targets":      dict[(int prompt_idx, int round_idx) -> list[float γ]]}

    We accept either a flat "targets" mapping or a dict-of-dicts keyed by
    prompt_idx then round_idx; return a normalized
    `dict[(prompt_idx, round_idx)] -> list[float]`.
    """
    blob = torch.load(path, weights_only=False)
    if isinstance(blob, dict) and "targets" in blob:
        raw = blob["targets"]
    else:
        raw = blob
    out: Dict = {}
    for k, v in raw.items():
        if isinstance(k, tuple):
            key = (int(k[0]), int(k[1]))
        else:
            # Accept a string serialization fallback "p|r".
            parts = str(k).split("|")
            key = (int(parts[0]), int(parts[1]))
        out[key] = [float(x) for x in v]
    return out


class FrozenAcceptanceDataset(Dataset):
    """Yields `(features, q2_target, survived)` for each RoundRecord.

    Default target (v1): `q2_target[j] = record.min_pq_j[j]` (Q2_j).
    `survived[j]` masks loss contributions so only positions actually
    reached by strict contribute to training.

    Phase-23 option (`dep_targets_path` != None): the `q2_target` slot is
    *overridden* with the precomputed dependence target s_j loaded from
    the given file (see `accpre/core/dependence.py`). The key stays
    named `q2_target` for codepath reuse — downstream trainer / loss
    treat it identically as a [0, 1] regression target. The actual
    semantic is logged via the trainer's config.
    """

    def __init__(
        self,
        records: List[RoundRecord],
        family: str,
        dep_targets_path: str | None = None,
    ) -> None:
        self.records = list(records)
        self.family = family
        self.dep_targets_path = dep_targets_path
        self._deps: Dict | None = None
        if dep_targets_path is not None:
            self._deps = _load_dep_targets(dep_targets_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        features = extract_features(r, self.family)                    # (D,)
        if self._deps is not None:
            key = (int(r.prompt_idx), int(r.round_idx))
            if key not in self._deps:
                raise KeyError(
                    f"dep target missing for prompt={r.prompt_idx} "
                    f"round={r.round_idx} (path={self.dep_targets_path!r})"
                )
            vals = self._deps[key]
            if len(vals) != int(r.gamma):
                raise ValueError(
                    f"dep target length {len(vals)} != gamma={r.gamma} "
                    f"at prompt={r.prompt_idx} round={r.round_idx}"
                )
            q2 = torch.tensor(vals, dtype=torch.float32)               # (γ,)
        else:
            q2 = torch.tensor(r.min_pq_j, dtype=torch.float32)         # (γ,)
        survived = torch.tensor(r.survived_j, dtype=torch.float32)      # (γ,)
        accepted = torch.tensor(r.accepted_j, dtype=torch.float32)      # (γ,)
        # Per-position drafted token IDs. Always yielded; predictors that
        # don't consume them ignore the key via **_kwargs in forward.
        # Long tensor for nn.Embedding lookup.
        token_ids = torch.tensor(r.draft_tokens, dtype=torch.long)      # (γ,)
        return {
            "features": features,
            "q2_target": q2,
            "survived": survived,
            "accepted": accepted,
            "token_ids": token_ids,
            "record_idx": torch.tensor(idx, dtype=torch.long),
        }


# Default τ sampling grid for length predictor training (DESIGN_PHASE2 §B.2).
DEFAULT_TAU_GRID: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9)


def _tau_sampler_from_grid(
    grid: Sequence[float], rng: random.Random
) -> Callable[[], float]:
    grid = tuple(float(t) for t in grid)

    def _draw() -> float:
        return rng.choice(grid)

    return _draw


class FrozenLengthDataset(Dataset):
    """Yields `(features, tau, L_tau)` for each RoundRecord.

    At each `__getitem__`, a fresh τ is sampled (deterministically per
    `(idx, epoch-epoch_seed)`) from the configured grid. `L_tau =
    commit_threshold(record.min_pq_j, τ)`, an int in `{0, ..., γ}`.
    """

    def __init__(
        self,
        records: List[RoundRecord],
        family: str,
        tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
        seed: int = 0,
    ) -> None:
        self.records = list(records)
        self.family = family
        self.tau_grid = tuple(float(t) for t in tau_grid)
        self._rng = random.Random(int(seed))
        self._sample_tau = _tau_sampler_from_grid(self.tau_grid, self._rng)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        tau = self._sample_tau()
        features = extract_features(r, self.family)                    # (D,)
        L_tau = commit_threshold(r.min_pq_j, tau)                       # int
        return {
            "features": features,
            "tau": torch.tensor(tau, dtype=torch.float32),
            "L_target": torch.tensor(L_tau, dtype=torch.long),
            "record_idx": torch.tensor(idx, dtype=torch.long),
        }


_JOINT_FAMILIES = (
    "hidden", "hidden_numeric", "hidden_per_pos", "hidden_per_pos_v2",
)


def joint_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Custom collate for `JointAcceptanceDataset`.

    Post-Phase-15: `prefix_ids` is the FULL reconstructed prefix per
    record, whose length varies across rounds (32 at round 0, growing
    through the prompt's trajectory). `torch.stack` on the default
    collate path fails with a size-mismatch error.

    This collate leaves `prefix_ids` as a plain Python list of 1-D
    LongTensors (downstream `_forward_item` iterates per-item anyway)
    and stacks everything else via the default collate.

    All other keys have fixed shape (γ, …) or are scalars, so the
    default collate stacks them correctly.
    """
    from torch.utils.data._utils.collate import default_collate
    keys = batch[0].keys()
    out: Dict[str, object] = {}
    for k in keys:
        if k == "prefix_ids":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = default_collate([b[k] for b in batch])
    return out


def _build_full_prefix_cache(
    records: Sequence[RoundRecord],
) -> List[torch.Tensor]:
    """Reconstruct the FULL prefix each record saw during collection.

    Replaces the legacy last-32-token `prefix_tail` truncation. The
    reconstruction mirrors `accpre.collect.cli.run_collection` exactly:

        running_{k+1} = running_k  ⊕  rec_k.draft_tokens[: rec_k.L]
                                  ⊕ [rec_k.bonus_or_fallback_token]

    starting from the initial 32-token prompt prefix pulled from the
    canonical pool (seeded by `accpre.data.splits.PROMPT_SEED`).

    Returns a list of 1-D long tensors, `len(records)` entries, in the
    SAME order as `records`, so `__getitem__(idx)` can index into it
    with the same `idx` the dataset exposes.
    """
    from accpre.data.prompts import load_owt_prompts
    from accpre.data.splits import POOL_SIZE, PREFIX_LEN, PROMPT_SEED

    # Initial pool-wide prefixes, keyed by global prompt index 0..POOL_SIZE-1.
    pool = load_owt_prompts(
        n_prompts=POOL_SIZE, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )
    initial_prefix: Dict[int, List[int]] = {
        i: [int(x) for x in pool[i][0].tolist()] for i in range(POOL_SIZE)
    }

    # Group records by prompt, preserving original position so we can
    # write into the output list in-place.
    by_prompt: Dict[int, List[tuple]] = defaultdict(list)
    for pos, r in enumerate(records):
        by_prompt[int(r.prompt_idx)].append((pos, r))

    out: List[torch.Tensor] = [None] * len(records)  # type: ignore[assignment]
    for p_idx, entries in by_prompt.items():
        entries.sort(key=lambda pr: int(pr[1].round_idx))
        # Guard against missing round 0 (would break the chain).
        if entries[0][1].round_idx != 0:
            raise ValueError(
                f"JointAcceptanceDataset full-prefix reconstruction needs "
                f"prompt {p_idx} to start at round_idx=0; got "
                f"{entries[0][1].round_idx}."
            )
        if p_idx not in initial_prefix:
            raise ValueError(
                f"Prompt index {p_idx} out of canonical pool range "
                f"[0, {POOL_SIZE}); cannot reconstruct initial prefix."
            )
        running: List[int] = list(initial_prefix[p_idx])
        prev_round = -1
        for pos, r in entries:
            if int(r.round_idx) != prev_round + 1:
                raise ValueError(
                    f"Prompt {p_idx}: non-contiguous rounds "
                    f"(saw {prev_round} then {r.round_idx})."
                )
            # Snapshot the running prefix AT round start (before this
            # round's commits are appended), matching draft_verify semantics.
            out[pos] = torch.tensor(list(running), dtype=torch.long)
            # Advance by this round's committed tokens + bonus/fallback.
            running.extend(int(t) for t in r.draft_tokens[: int(r.L)])
            running.append(int(r.bonus_or_fallback_token))
            prev_round = int(r.round_idx)

    # Any None slot means a record wasn't visited — shouldn't happen.
    for pos, p in enumerate(out):
        if p is None:
            raise RuntimeError(
                f"full-prefix reconstruction missed record position {pos}"
            )
    return out


class JointAcceptanceDataset(Dataset):
    """Stage 2B.1 joint-training dataset.

    Yields one item per `RoundRecord`:
      - prefix_ids    (LongTensor, actual prefix_len at round start —
                       reconstructed from the full trajectory, NOT the
                       last-32-token `record.prefix_tail` truncation)
      - round_rng_seed (int)
      - q2_target   (γ,) — stored Stage-1 Q2 (lazy target; see
                    DESIGN_PHASE2 §2B design notes)
      - survived    (γ,) 0/1
      - accepted    (γ,) 0/1
      - token_ids   (γ,) long — record.draft_tokens, needed by token-emb
                    heads; ignored by heads that don't consume it
      - prompt_idx, round_idx (for bookkeeping)
      - numeric     (3,) — populated when family needs it

    The drafter's hidden state is NOT pre-computed here; the joint
    trainer re-runs the drafter with grad at each step using
    `prefix_ids` and `round_rng_seed`. Full-prefix reconstruction is
    cached at __init__ so every __getitem__ is O(1).
    """

    def __init__(
        self,
        records: List[RoundRecord],
        family: str,
        dep_targets_path: str | None = None,
    ) -> None:
        if family not in _JOINT_FAMILIES:
            raise ValueError(
                f"JointAcceptanceDataset supports family in "
                f"{_JOINT_FAMILIES}; got {family!r}. "
                f"(Numeric-only joint makes no sense — no drafter signal.)"
            )
        self.records = list(records)
        self.family = family
        self.dep_targets_path = dep_targets_path
        self._deps: Dict | None = (
            _load_dep_targets(dep_targets_path)
            if dep_targets_path is not None else None
        )
        # Precompute full prefixes once (cheap for a ~6k-record file).
        self._full_prefixes = _build_full_prefix_cache(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        prefix_ids = self._full_prefixes[idx].clone()
        # Regression target: Q2 by default, precomputed s_j if dep_targets_path
        # was provided (lazy dep target — Phase 23 frozen-only path).
        if self._deps is not None:
            key = (int(r.prompt_idx), int(r.round_idx))
            if key not in self._deps:
                raise KeyError(
                    f"dep target missing for prompt={r.prompt_idx} "
                    f"round={r.round_idx} (path={self.dep_targets_path!r})"
                )
            vals = self._deps[key]
            if len(vals) != int(r.gamma):
                raise ValueError(
                    f"dep target length {len(vals)} != gamma={r.gamma} "
                    f"at prompt={r.prompt_idx} round={r.round_idx}"
                )
            target = torch.tensor(vals, dtype=torch.float32)
        else:
            target = torch.tensor(r.min_pq_j, dtype=torch.float32)
        item: Dict[str, torch.Tensor] = {
            "prefix_ids": prefix_ids,
            "round_rng_seed": torch.tensor(int(r.round_rng_seed), dtype=torch.long),
            "q2_target": target,
            "min_pq_j": torch.tensor(r.min_pq_j, dtype=torch.float32),
            "survived": torch.tensor(r.survived_j, dtype=torch.float32),
            "accepted": torch.tensor(r.accepted_j, dtype=torch.float32),
            "token_ids": torch.tensor(r.draft_tokens, dtype=torch.long),
            "prompt_idx": torch.tensor(int(r.prompt_idx), dtype=torch.long),
            "round_idx": torch.tensor(int(r.round_idx), dtype=torch.long),
            "record_idx": torch.tensor(idx, dtype=torch.long),
            "gamma": torch.tensor(int(r.gamma), dtype=torch.long),
            "T": torch.tensor(int(r.T), dtype=torch.long),
        }
        if self.family == "hidden_numeric":
            if any(getattr(r, n) is None for n in
                   ("verifier_entropy", "verifier_margin", "verifier_top1_prob")):
                raise ValueError(
                    f"record missing numeric features (prompt {r.prompt_idx}, round {r.round_idx})"
                )
            item["numeric"] = torch.tensor([
                float(r.verifier_entropy),
                float(r.verifier_margin),
                float(r.verifier_top1_prob),
            ], dtype=torch.float32)
        return item


def split_records_by_prompt(
    records: List[RoundRecord],
    train_prompt_indices: Sequence[int],
    val_prompt_indices: Sequence[int],
    test_prompt_indices: Sequence[int],
) -> Dict[str, List[RoundRecord]]:
    """Partition a record list by prompt_idx membership.

    Indices refer to the canonical `accpre.data.splits` global prompt
    indices (0..79). Records whose prompt_idx is not in any split are
    dropped.
    """
    train_set = set(int(i) for i in train_prompt_indices)
    val_set = set(int(i) for i in val_prompt_indices)
    test_set = set(int(i) for i in test_prompt_indices)

    out: Dict[str, List[RoundRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        if r.prompt_idx in train_set:
            out["train"].append(r)
        elif r.prompt_idx in val_set:
            out["val"].append(r)
        elif r.prompt_idx in test_set:
            out["test"].append(r)
    return out
