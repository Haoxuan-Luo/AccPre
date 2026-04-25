"""Tests for accpre.core.commit."""

from __future__ import annotations

import pytest

from accpre.core.commit import commit_bernoulli, commit_strict, commit_threshold


def test_commit_strict_all_accept():
    assert commit_strict([1, 1, 1, 1]) == 4


def test_commit_strict_first_reject():
    assert commit_strict([0, 1, 1]) == 0


def test_commit_strict_mid_reject():
    assert commit_strict([1, 1, 0, 1]) == 2


def test_commit_strict_empty():
    assert commit_strict([]) == 0


def test_commit_threshold_all_above():
    assert commit_threshold([0.9, 0.9, 0.9], tau=0.5) == 3


def test_commit_threshold_first_below():
    assert commit_threshold([0.1, 0.9, 0.9], tau=0.5) == 0


def test_commit_threshold_mid_below():
    assert commit_threshold([0.9, 0.8, 0.3, 0.9], tau=0.5) == 2


def test_commit_threshold_monotone_in_tau():
    alpha = [0.9, 0.7, 0.5, 0.3]
    prev = len(alpha) + 1
    for tau in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]:
        L = commit_threshold(alpha, tau)
        assert L <= prev, f"L={L} at tau={tau} not <= prev={prev}"
        prev = L


def test_commit_threshold_binary_matches_strict():
    alpha = [1.0, 1.0, 0.0, 1.0]
    L_t = commit_threshold(alpha, 0.5)
    L_s = commit_strict([int(a >= 0.5) for a in alpha])
    assert L_t == L_s


def test_commit_threshold_tau_above_one_returns_zero_if_any_pos():
    assert commit_threshold([0.99, 0.99], tau=1.5) == 0
    assert commit_threshold([], tau=1.5) == 0


def test_commit_bernoulli_is_reserved():
    with pytest.raises(NotImplementedError):
        commit_bernoulli()
