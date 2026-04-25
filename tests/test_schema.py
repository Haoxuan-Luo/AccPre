"""Tests for accpre.core.schema and the canonical-accept meta-test."""

from __future__ import annotations

import pathlib
import re

import pytest
import torch

from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import RoundRecord, load_records, save_records


def _make_record(
    protocol: ProtocolConfig | None = None,
    gamma: int = 3,
    prompt_idx: int = 0,
    round_idx: int = 0,
    L: int = 2,
) -> RoundRecord:
    protocol = protocol or ProtocolConfig()
    return RoundRecord(
        schema_version=protocol.schema_version,
        protocol=protocol,
        prompt_idx=prompt_idx,
        round_idx=round_idx,
        round_rng_seed=1234,
        gamma=gamma,
        T=2,
        prefix_len=5,
        draft_tokens=[1, 2, 3][:gamma],
        q_j=[0.5, 0.6, 0.7][:gamma],
        p_j=[0.4, 0.5, 0.6][:gamma],
        min_pq_j=[0.8, 0.833, 0.857][:gamma],
        U_j=[0.1, 0.2, 0.3][:gamma],
        accepted_j=[1, 1, 0][:gamma],
        survived_j=[1, 1, 1][:gamma],
        L=L,
        bonus_or_fallback_token=42,
        draft_time=0.01,
        verify_time=0.02,
    )


def test_roundtrip_save_load(tmp_path):
    records = [_make_record(prompt_idx=i) for i in range(3)]
    path = str(tmp_path / "records.pt")
    save_records(records, path)
    loaded = load_records(path, expected_protocol=ProtocolConfig())
    assert len(loaded) == 3
    for a, b in zip(records, loaded):
        assert a.round_rng_seed == b.round_rng_seed
        assert a.draft_tokens == b.draft_tokens
        assert a.L == b.L
        assert a.protocol.fingerprint() == b.protocol.fingerprint()


def test_schema_version_mismatch_raises(tmp_path):
    proto_v1 = ProtocolConfig(schema_version=1)
    r = _make_record(protocol=proto_v1)
    path = str(tmp_path / "records.pt")
    save_records([r], path)

    proto_v2 = ProtocolConfig(schema_version=2)
    with pytest.raises(AssertionError, match="schema_version"):
        load_records(path, expected_protocol=proto_v2)


def test_protocol_fingerprint_mismatch_raises(tmp_path):
    proto_A = ProtocolConfig(q_mode="A")
    r = _make_record(protocol=proto_A)
    path = str(tmp_path / "records.pt")
    save_records([r], path)

    proto_B = ProtocolConfig(q_mode="B")
    with pytest.raises(AssertionError, match="fingerprint"):
        load_records(path, expected_protocol=proto_B)


def test_per_position_array_length_mismatch_raises():
    r = _make_record(gamma=3)
    r.q_j = [0.5, 0.6]  # wrong length
    with pytest.raises(ValueError, match="length"):
        r.validate()


def test_L_out_of_range_raises():
    r = _make_record(gamma=3)
    r.L = 5
    with pytest.raises(ValueError, match="in \\[0"):
        r.validate()


def test_survived_must_be_monotone():
    r = _make_record(gamma=3)
    r.survived_j = [1, 0, 1]  # rises after 0 — illegal
    with pytest.raises(ValueError, match="monotone"):
        r.validate()


# --- Meta-test: canonical accept pattern only in core/accept.py -------------

_LEVIATHAN_PATTERNS = [
    # Compact forms most likely to appear if someone re-implemented the
    # Bernoulli accept test outside the canonical module.
    re.compile(r"min\(\s*1\.?0?\s*,\s*p_val\s*/\s*q_val\s*\)"),
    re.compile(r"min\(\s*1\.?0?\s*,\s*p_j\s*/\s*q_j\s*\)"),
]


def test_no_duplicate_accept_implementation():
    """The Leviathan accept pattern must appear only in core/accept.py.

    Scans every .py under `accpre/` (excluding `__pycache__`) and reports
    any file other than `core/accept.py` containing the distinctive
    pattern. Callers that route through `per_position_accept` are fine;
    inline re-implementations are not.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    accept_path = (repo_root / "accpre" / "core" / "accept.py").resolve()

    offenders: list[str] = []
    for path in (repo_root / "accpre").rglob("*.py"):
        if path.resolve() == accept_path:
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for pat in _LEVIATHAN_PATTERNS:
            if pat.search(text):
                offenders.append(f"{path} :: {pat.pattern}")
                break

    assert not offenders, (
        "Leviathan accept pattern found outside core/accept.py:\n  "
        + "\n  ".join(offenders)
    )
