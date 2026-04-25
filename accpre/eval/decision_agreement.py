"""L2 — decision-level agreement metrics (offline).

Given predictor outputs on a list of test records, compare the
predictor-induced committed length `L̂(τ)` against:
  - `L_strict`     (record.L)
  - `L_τ^oracle`   = commit_threshold(record.min_pq_j, τ)

For acceptance predictors, `L̂(τ) = commit_threshold(Q̂, τ)`.
For length predictors, `L̂(τ)` is the argmax of the categorical output.

See DESIGN_PHASE2.md §H.1 (L2 block). Note that `L_agreement_strict`
under the v1 shared-fallback coupling numerically equals the offline
CF@1 value; it is reported here under the L2 name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from accpre.core.commit import commit_threshold
from accpre.core.schema import RoundRecord


def _by_record_idx(records: List[RoundRecord]) -> Dict[int, RoundRecord]:
    # Fallback index-by-position since RoundRecord has no uuid.
    return {i: r for i, r in enumerate(records)}


def acceptance_agreement(
    pred_rows: List[Dict[str, Any]],
    records: List[RoundRecord],
    taus: Sequence[float],
) -> Dict[str, Any]:
    """L2 for acceptance predictors. Reports per-τ agreement tables.

    Args:
        pred_rows: rows with keys `record_idx`, `q2_hat` (list of γ floats).
        records:   the list of `RoundRecord`s that the `record_idx` values
                   in pred_rows index into (as positional indices).
        taus:      τ values to evaluate over.

    Returns:
        dict with key `per_tau` -> dict keyed by str(tau) -> agreement dict.
    """
    idx2rec = _by_record_idx(records)
    out: Dict[str, Dict[str, Any]] = {}
    for tau in taus:
        tau_f = float(tau)
        n = 0
        agree_strict = 0
        agree_q2 = 0
        over_strict = 0
        under_strict = 0
        overcommit_tokens = 0
        for pr in pred_rows:
            r = idx2rec[int(pr["record_idx"])]
            q_hat = list(pr["q2_hat"])
            L_hat = commit_threshold(q_hat, tau_f)
            L_strict = int(r.L)
            L_q2 = commit_threshold(r.min_pq_j, tau_f)
            n += 1
            if L_hat == L_strict:
                agree_strict += 1
            if L_hat == L_q2:
                agree_q2 += 1
            if L_hat > L_strict:
                over_strict += 1
                overcommit_tokens += (L_hat - L_strict)
            elif L_hat < L_strict:
                under_strict += 1
        out[str(tau_f)] = {
            "n": n,
            "L_agreement_strict": agree_strict / max(n, 1),
            "L_agreement_oracleQ2": agree_q2 / max(n, 1),
            "L_overcommit_rate_strict": over_strict / max(n, 1),
            "L_undercommit_rate_strict": under_strict / max(n, 1),
            "L_overcommit_token_count": overcommit_tokens,
        }
    return {"per_tau": out}


def length_agreement(
    pred_rows: List[Dict[str, Any]],
    records: List[RoundRecord],
) -> Dict[str, Any]:
    """L2 for length predictors. Each row already has a τ; compute per-τ.

    Args:
        pred_rows: rows with keys `record_idx`, `L_hat`, `L_target`, `tau`.
        records:   the list of `RoundRecord`s indexed by position.
    """
    idx2rec = _by_record_idx(records)
    by_tau: Dict[str, Dict[str, Any]] = {}
    for pr in pred_rows:
        r = idx2rec[int(pr["record_idx"])]
        L_hat = int(pr["L_hat"])
        tau = float(pr["tau"])
        L_strict = int(r.L)
        L_q2 = commit_threshold(r.min_pq_j, tau)
        key = str(tau)
        bucket = by_tau.setdefault(key, {
            "n": 0, "agree_strict": 0, "agree_q2": 0,
            "over_strict": 0, "under_strict": 0, "overcommit_tokens": 0,
        })
        bucket["n"] += 1
        if L_hat == L_strict:
            bucket["agree_strict"] += 1
        if L_hat == L_q2:
            bucket["agree_q2"] += 1
        if L_hat > L_strict:
            bucket["over_strict"] += 1
            bucket["overcommit_tokens"] += (L_hat - L_strict)
        elif L_hat < L_strict:
            bucket["under_strict"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for tau_key, b in by_tau.items():
        n = max(int(b["n"]), 1)
        out[tau_key] = {
            "n": int(b["n"]),
            "L_agreement_strict": b["agree_strict"] / n,
            "L_agreement_oracleQ2": b["agree_q2"] / n,
            "L_overcommit_rate_strict": b["over_strict"] / n,
            "L_undercommit_rate_strict": b["under_strict"] / n,
            "L_overcommit_token_count": b["overcommit_tokens"],
        }
    return {"per_tau": out}
