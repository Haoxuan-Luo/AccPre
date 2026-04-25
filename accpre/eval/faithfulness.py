"""Controlled Faithfulness@1 (CF@1) — the primary v1 Pareto x-axis.

Definition (DESIGN.md §D.5.2): under matched randomness and a shared
draft proposal, the fraction of rounds where the method under test
produces the same committed output sequence as strict SpecDiff.

Matched-randomness shortcut (valid for every v1 method):
Under v1's shared-fallback coupling — every method uses the same
drafter with the same `draft_rng`, the same verifier (deterministic),
the same `accept_rng` stream (U_j is always drawn), and the same
fallback/bonus sampling procedure with the same `fallback_rng` — two
lanes that agree on `L` produce identical committed sequences, and
two lanes that disagree produce different sequences. So CF@1(round)
reduces to `1[method_L == record.L]`. See DESIGN.md §D.5.2 for the
full derivation.

The API accepts an arbitrary `method_commit_fn: RoundRecord -> int` so
that future methods with divergent fallback behavior remain evaluable
with the same function signature; at that point a second callable hook
for fallback-token comparison can be added without changing callers.

What the record must carry for this function to work:
  - `record.L`: strict's committed length (ground truth for comparison).
  - The protocol and round_rng_seed fields are not consulted here (the
    seed is metadata for replayability), but callers who want to audit
    sampled artifacts can reconstruct `U_j` and fallback states from
    the seed via `accpre.core.draft_verify._make_generator`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Any

from accpre.core.schema import RoundRecord


def controlled_faithfulness_at_1(
    records: List[RoundRecord],
    method_commit_fn: Callable[[RoundRecord], int],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    """Compute CF@1 over a list of strict records.

    Args:
        records: strict RoundRecords (ground-truth L is `record.L`).
        method_commit_fn: given a record, returns the method's committed
            length `L_hat` in [0, record.gamma]. Must be deterministic
            given the record and the method's own τ / hyperparameters.
        bootstrap_samples: number of per-prompt bootstrap resamples.
        bootstrap_seed: seed for the bootstrap RNG (NOT related to the
            records' own round_rng_seed).

    Returns:
        dict with keys:
          - "aggregate":   float, overall CF@1 mean over rounds
          - "per_prompt":  dict[prompt_idx -> float], per-prompt mean
          - "per_round":   list[int], 1/0 per input record
          - "bootstrap_ci": (lo, hi) 95% CI over per-prompt means
          - "n_rounds":    int, number of rounds evaluated
          - "n_prompts":   int, number of distinct prompts
    """
    per_round: List[int] = []
    per_prompt_vals: Dict[int, List[int]] = defaultdict(list)

    for r in records:
        L_hat = int(method_commit_fn(r))
        # v1 shared-fallback reduction: CF@1(round) = 1[L_hat == L_strict].
        is_faithful = 1 if L_hat == int(r.L) else 0
        per_round.append(is_faithful)
        per_prompt_vals[int(r.prompt_idx)].append(is_faithful)

    per_prompt: Dict[int, float] = {
        p: (sum(vs) / len(vs)) for p, vs in per_prompt_vals.items()
    }

    if len(per_round) == 0:
        return {
            "aggregate": 0.0,
            "per_prompt": {},
            "per_round": [],
            "bootstrap_ci": (0.0, 0.0),
            "n_rounds": 0,
            "n_prompts": 0,
        }

    aggregate = sum(per_round) / float(len(per_round))

    # Per-prompt cluster bootstrap: resample prompts with replacement,
    # take the mean of per-prompt means, repeat.
    rng = random.Random(bootstrap_seed)
    prompt_means = list(per_prompt.values())
    n = len(prompt_means)
    if n == 0:
        lo = hi = 0.0
    else:
        boots: List[float] = []
        for _ in range(bootstrap_samples):
            sample = [prompt_means[rng.randint(0, n - 1)] for _ in range(n)]
            boots.append(sum(sample) / n)
        boots.sort()
        lo_idx = max(int(0.025 * bootstrap_samples), 0)
        hi_idx = min(int(0.975 * bootstrap_samples), bootstrap_samples - 1)
        lo = boots[lo_idx]
        hi = boots[hi_idx]

    return {
        "aggregate": aggregate,
        "per_prompt": per_prompt,
        "per_round": per_round,
        "bootstrap_ci": (lo, hi),
        "n_rounds": len(per_round),
        "n_prompts": n,
    }
