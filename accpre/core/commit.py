"""Canonical commit rules.

Three deterministic functions that reduce a per-position signal to a
commit length `L` in [0, gamma]:

  commit_strict(accepted_j)          -> first-zero reducer (Leviathan rule)
  commit_threshold(alpha, tau)       -> threshold-prefix reducer
  commit_confidence(alpha, tau_conf) -> prefix-product reducer

The Bernoulli commit rule is reserved for v2 and will raise
`NotImplementedError` if called.
"""

from __future__ import annotations

from typing import Sequence


def commit_strict(accepted_j: Sequence[int]) -> int:
    """First-zero reducer.

    Returns the index of the first 0 in `accepted_j`; if every value is 1,
    returns `len(accepted_j)`. Under the Leviathan speculative-decoding
    rule, this is the length of the contiguous accepted prefix.
    """
    for j, a in enumerate(accepted_j):
        if a == 0:
            return j
    return len(accepted_j)


def commit_threshold(alpha: Sequence[float], tau: float) -> int:
    """Threshold-prefix reducer.

    Returns the length of the longest prefix with `alpha[j] >= tau` for
    every j. The oracle-Q2 threshold baseline calls this with
    `alpha = min_pq_j` (the Leviathan ratio; see `accpre.core.accept`).
    """
    for j, a in enumerate(alpha):
        if a < tau:
            return j
    return len(alpha)


def commit_confidence(alpha: Sequence[float], tau_conf: float) -> int:
    """Prefix-product (confidence) reducer.

    Returns the largest `k ∈ [0, len(alpha)]` such that the cumulative
    prefix product `C_k := prod_{j<k} alpha[j]` stays ≥ `tau_conf`.
    `C_0 = 1` by convention, so `k=0` always satisfies the predicate for
    `tau_conf ≤ 1`. Because every `alpha[j] ∈ [0, 1]`, `C_k` is monotone
    non-increasing, so the first `k` at which `C_k < tau_conf` is the
    cutoff and every subsequent `k` also fails.
    """
    c = 1.0
    for j, a in enumerate(alpha):
        c *= float(a)
        if c < tau_conf:
            return j
    return len(alpha)


def commit_bernoulli(*_args: object, **_kwargs: object) -> int:
    """Reserved for v2. Do not implement in v1.

    Kept as a named stub so any Phase 1 code that accidentally tries to
    use a Bernoulli commit rule fails at call time with a clear message,
    rather than silently falling through.
    """
    raise NotImplementedError(
        "Bernoulli commit rule is reserved for v2; not implemented in v1."
    )
