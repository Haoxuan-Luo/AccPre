"""Canonical `RoundRecord` schema and strict save/load.

One record per draft-verify-accept-commit round. All Phase 1 fields are
required. Phase 2+ feature fields (verifier_hidden, drafter_hidden,
prefix_tail, verifier_{entropy,margin,top1_prob}) are declared as
`Optional[...] = None` so adding them does NOT bump `schema_version`.

Matched-randomness discipline (DESIGN.md §D.5.2) makes the following
fields load-bearing for CF@1 replay:
  - `round_rng_seed`: the sole randomness input for the round. Every
    sub-stream (draft_rng, accept_rng, fallback_rng) is derived from it
    via protocol salts, so replaying the round under matched randomness
    only requires this seed plus the protocol.
  - `U_j`: the per-position uniform draws consumed by strict's Bernoulli
    accept test. Logged here for two reasons: (1) so offline audits can
    verify the strict accept-decision against the logged q/p without
    re-running; (2) so lossy/oracle-Q2 evaluation can confirm it is
    operating on the same accept_rng stream position (strict consumes
    gamma uniforms; any replaying lane must consume gamma uniforms too,
    otherwise fallback_rng state would drift — but fallback_rng is its
    OWN generator, so this is actually only for auditability).
  - `bonus_or_fallback_token`: the single extra token strict committed
    after the accepted draft prefix (bonus when L==gamma, fallback when
    L<gamma). Under the shared-fallback coupling, any v1 method whose
    L agrees with strict's L will produce the same token here, by
    construction; logging it makes offline replay verifiable.

`save_records` / `load_records` are the only supported I/O paths.
`load_records` asserts `schema_version` and (if provided) protocol
fingerprint consistency on every record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import torch

from accpre.core.protocol import ProtocolConfig


@dataclass
class RoundRecord:
    # --- Provenance ---
    schema_version: int
    protocol: ProtocolConfig
    prompt_idx: int
    round_idx: int
    round_rng_seed: int

    # --- Action ---
    gamma: int
    T: int
    prefix_len: int

    # --- Per-position arrays (length gamma, order 0..gamma-1) ---
    draft_tokens: List[int]
    q_j: List[float]
    p_j: List[float]
    min_pq_j: List[float]      # == Q2_j in v1 terminology
    U_j: List[float]            # uniform draws used by strict Bernoulli test
    accepted_j: List[int]       # 0/1
    survived_j: List[int]       # 0/1 (monotone non-increasing)

    # --- Commit outputs ---
    L: int
    bonus_or_fallback_token: int

    # --- Timing ---
    draft_time: float
    verify_time: float

    # --- Optional Phase 2+ feature fields ---
    verifier_hidden: Optional[torch.Tensor] = None
    drafter_hidden: Optional[torch.Tensor] = None              # mean-pooled, (H,)
    drafter_hidden_per_pos: Optional[torch.Tensor] = None      # per-position, (γ, H)
    prefix_tail: Optional[torch.Tensor] = None
    verifier_entropy: Optional[float] = None
    verifier_margin: Optional[float] = None
    verifier_top1_prob: Optional[float] = None

    # --- Phase 5: per-position drafter-side scalars (computed from the
    # SUBS-processed `log p_θ(x_0 | x_t)` row at the commit step for j) ---
    drafter_entropy_j: Optional[List[float]] = None    # (γ,)
    drafter_margin_j: Optional[List[float]] = None     # (γ,) — log p(top1) − log p(top2)
    drafter_top1_prob_j: Optional[List[float]] = None  # (γ,) — max_v p_θ[v]

    # --- Phase 6: multi-layer per-position drafter hidden ---
    # Concat of the final and penultimate transformer block outputs at
    # the γ draft positions, in (penultimate, final) order along the
    # hidden axis. Shape `(γ, 2H)`. The last H columns are exactly what
    # `drafter_hidden_per_pos` stores (final layer only) so slicing
    # `[:, H:]` recovers the legacy pp_base feature.
    drafter_hidden_per_pos_ml: Optional[torch.Tensor] = None   # (γ, 2H)

    def validate(self) -> None:
        """Assert per-position arrays have length gamma and L is in range.

        Also validates OPTIONAL rich per-position fields when present:
          - drafter_entropy_j / drafter_margin_j / drafter_top1_prob_j
            (Phase 5 scalars, length γ lists)
          - drafter_hidden_per_pos   (shape[0] == γ)
          - drafter_hidden_per_pos_ml (shape[0] == γ)

        Called automatically by `load_records` on every record. Also
        callable directly as a self-check.
        """
        g = int(self.gamma)
        for name in ("draft_tokens", "q_j", "p_j", "min_pq_j", "U_j",
                     "accepted_j", "survived_j"):
            v = getattr(self, name)
            if len(v) != g:
                raise ValueError(
                    f"RoundRecord.{name} length={len(v)} but gamma={g} "
                    f"(prompt_idx={self.prompt_idx}, round_idx={self.round_idx})"
                )
        if not (0 <= int(self.L) <= g):
            raise ValueError(
                f"RoundRecord.L={self.L} not in [0, {g}] "
                f"(prompt_idx={self.prompt_idx}, round_idx={self.round_idx})"
            )
        # Cheap sanity: survived_j is monotone non-increasing.
        prev = 1
        for j, s in enumerate(self.survived_j):
            if s > prev:
                raise ValueError(
                    f"survived_j must be monotone non-increasing; "
                    f"position {j} jumped from {prev} to {s}"
                )
            prev = s

        # Optional Phase 5 per-position scalar lists: length == γ when present.
        for name in ("drafter_entropy_j", "drafter_margin_j", "drafter_top1_prob_j"):
            v = getattr(self, name, None)
            if v is None:
                continue
            if len(v) != g:
                raise ValueError(
                    f"RoundRecord.{name} length={len(v)} but gamma={g} "
                    f"(prompt_idx={self.prompt_idx}, round_idx={self.round_idx})"
                )

        # Optional per-position hidden tensors: shape[0] == γ when present.
        for name in ("drafter_hidden_per_pos", "drafter_hidden_per_pos_ml"):
            v = getattr(self, name, None)
            if v is None:
                continue
            if v.dim() < 2:
                raise ValueError(
                    f"RoundRecord.{name} must be ≥ 2-D, got shape "
                    f"{tuple(v.shape)} "
                    f"(prompt_idx={self.prompt_idx}, round_idx={self.round_idx})"
                )
            if int(v.shape[0]) != g:
                raise ValueError(
                    f"RoundRecord.{name} shape[0]={int(v.shape[0])} but "
                    f"gamma={g} "
                    f"(prompt_idx={self.prompt_idx}, round_idx={self.round_idx})"
                )


def save_records(records: List[RoundRecord], path: str) -> None:
    """Save a list of RoundRecords to a `.pt` file.

    Each record carries its own protocol + schema_version, so the file
    format is simply a list. `torch.save` handles tensor fields in the
    Phase 2+ optional slots when they appear.
    """
    torch.save(records, path)


def load_records(
    path: str,
    expected_protocol: Optional[ProtocolConfig] = None,
) -> List[RoundRecord]:
    """Load RoundRecords and assert schema + protocol consistency.

    Two kinds of protocol check:

      1. Within-file consistency (ALWAYS enforced, even when
         `expected_protocol` is None): every record in the file must
         share the same `protocol.fingerprint()` as record[0]. Catches
         any mixed-protocol file silently.

      2. Against-config consistency (only when `expected_protocol` is
         provided): every record's fingerprint and schema_version must
         match `expected_protocol` exactly. Protects callers who know
         which protocol they expect.

    Errors from either check are `AssertionError` with the exact
    `(prompt_idx, round_idx)` and the two conflicting fingerprints,
    so the offending record is easy to trace.
    """
    records = torch.load(path, weights_only=False)
    if not isinstance(records, list):
        raise ValueError(
            f"Expected list of RoundRecords in {path!r}, got {type(records)}"
        )
    if len(records) == 0:
        return records

    # Schema version — use expected if provided, else infer from record[0].
    expected_sv = (
        expected_protocol.schema_version if expected_protocol is not None
        else records[0].schema_version
    )
    # Fingerprint — use expected if provided, else infer from record[0]
    # (enforces within-file consistency).
    expected_fp = (
        expected_protocol.fingerprint() if expected_protocol is not None
        else records[0].protocol.fingerprint()
    )
    for i, r in enumerate(records):
        if not isinstance(r, RoundRecord):
            raise ValueError(
                f"record[{i}] is not a RoundRecord (got {type(r).__name__})"
            )
        if r.schema_version != expected_sv:
            raise AssertionError(
                f"record[{i}] schema_version={r.schema_version} "
                f"!= expected {expected_sv} "
                f"(prompt_idx={r.prompt_idx}, round_idx={r.round_idx}, path={path!r})"
            )
        if r.protocol.fingerprint() != expected_fp:
            raise AssertionError(
                f"record[{i}] protocol fingerprint mismatch "
                f"(prompt_idx={r.prompt_idx}, round_idx={r.round_idx}, "
                f"path={path!r}).\n"
                f"  expected: {expected_fp}\n"
                f"  got:      {r.protocol.fingerprint()}"
            )
        r.validate()
    return records
