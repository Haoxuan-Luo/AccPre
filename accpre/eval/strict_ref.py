"""Resolve the strict SpecDiff tok/s reference for comparison scripts.

Priority:
  1. Environment variable `STRICT_TOK_S` if set (manual override).
  2. Canonical output of the global-idx strict refresh, at
     `outputs/strict_global_idx/summary.json`, produced by
     `jobs/phase8_strict_refresh.slurm` /
     `python -m accpre.eval.strict_refresh ...`.
  3. Phase 1 baseline (LOCAL p_idx) of 19.75 tok/s with a warning —
     this is not randomness-matched to the Phase 3+ predictor lanes
     but is preserved as a fallback so compare scripts still run
     before the refresh job has been submitted.

Returns `(tok_s, source_label)` so compare scripts can print where
the number came from.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Tuple


STRICT_FALLBACK: float = 19.75
STRICT_GLOBAL_JSON: str = "outputs/strict_global_idx/summary.json"


def load_strict_tok_s() -> Tuple[float, str]:
    """Resolve strict tok/s + a label describing where it came from."""
    env = os.environ.get("STRICT_TOK_S")
    if env:
        try:
            return float(env), f"env STRICT_TOK_S={env}"
        except ValueError:
            print(
                f"[strict_ref] WARN: env STRICT_TOK_S={env!r} not a float; "
                f"ignoring.",
                file=sys.stderr,
            )
    if os.path.isfile(STRICT_GLOBAL_JSON):
        try:
            with open(STRICT_GLOBAL_JSON, "r") as f:
                d = json.load(f)
            val = float(d["strict"]["tok_s_mean"])
            return val, STRICT_GLOBAL_JSON
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(
                f"[strict_ref] WARN: could not parse {STRICT_GLOBAL_JSON}: "
                f"{e}; falling back to Phase 1 baseline.",
                file=sys.stderr,
            )
    print(
        f"[strict_ref] WARN: {STRICT_GLOBAL_JSON} not found; using Phase 1 "
        f"baseline {STRICT_FALLBACK} tok/s (measured with LOCAL p_idx, not "
        f"randomness-matched to Phase 3+ predictor lanes). Submit "
        f"jobs/phase8_strict_refresh.slurm to refresh.",
        file=sys.stderr,
    )
    return STRICT_FALLBACK, f"Phase 1 baseline ({STRICT_FALLBACK}, LOCAL p_idx)"
