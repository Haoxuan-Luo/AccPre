"""Canonical commit rules.

Vendored from `accpre/core/commit.py`. Three deterministic functions that
reduce a per-position signal to a commit length L in [0, gamma]:

  commit_strict(accepted_j)          -> first-zero reducer (Leviathan rule)
  commit_threshold(alpha, tau)       -> threshold-prefix reducer
  commit_confidence(alpha, tau_conf) -> prefix-product reducer
"""

from __future__ import annotations

from typing import Sequence


def commit_strict(accepted_j: Sequence[int]) -> int:
    """First-zero reducer (Leviathan strict speculative decoding rule)."""
    for j, a in enumerate(accepted_j):
        if a == 0:
            return j
    return len(accepted_j)


def commit_threshold(alpha: Sequence[float], tau: float) -> int:
    """Threshold-prefix reducer: longest prefix with alpha[j] >= tau."""
    for j, a in enumerate(alpha):
        if a < tau:
            return j
    return len(alpha)


def commit_confidence(alpha: Sequence[float], tau_conf: float) -> int:
    """Prefix-product (confidence) reducer: largest k with prod_{j<k} alpha[j] >= tau_conf."""
    c = 1.0
    for j, a in enumerate(alpha):
        c *= float(a)
        if c < tau_conf:
            return j
    return len(alpha)
