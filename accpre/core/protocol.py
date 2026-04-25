"""Frozen v1 protocol configuration and per-round seed derivation.

The `ProtocolConfig` pins every runtime parameter that could affect the
definition of q_j, accepted_j, L, CF@1, or any logged artifact. Every
`RoundRecord` carries a copy of the protocol it was produced under; a
loader that receives a record whose protocol fingerprint does not match
the expected protocol fails loudly.

Matched-randomness discipline (DESIGN.md §D.5.2):
  - `round_rng_seed` is the sole randomness input to a round.
  - It is derived deterministically from
        SHA-256( protocol.fingerprint() | prompt_idx | round_idx )
    so the same (protocol, prompt, round) always yields the same seed,
    across Python processes and machine restarts.
  - From `round_rng_seed` we derive three *independent* sub-streams by
    XORing with the three salts on this dataclass:
        draft_rng     (drives drafter.draft's torch.multinomial calls)
        accept_rng    (drives per-position U_j draws in per_position_accept)
        fallback_rng  (drives bonus/fallback token sampling)
    These are used in `accpre.core.draft_verify._make_generator` and
    fed directly into the relevant tensor ops.
  - The salts are **constants**. They are dataclass fields purely for
    transparency so the RNG derivation is self-documenting; changing them
    invalidates all existing records.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import hashlib
import json


@dataclass(frozen=True)
class ProtocolConfig:
    # --- Versioning ---
    schema_version: int = 1

    # --- Decoding discipline (pinned for v1) ---
    # temperature == 1.0 is the main v1 protocol; accepted_j is the
    # Bernoulli test with the Leviathan ratio (see accpre.core.accept).
    # temperature == 0 is reserved for unit tests and flips accepted_j
    # to argmax(target) == draft_tok.
    temperature: float = 1.0
    # q_mode "A" is the canonical v1 q definition: the SUBS-parameterized
    # log p_theta(x0 | xt) row at the DDPM step where each draft position
    # was first unmasked, evaluated on the drafted token (no factor shift).
    # q_mode "B" applies a per-step log-factor shift; kept for future
    # ablation only, not exercised by any v1 main experiment.
    q_mode: str = "A"

    # --- Models ---
    drafter_model: str = "kuleshov-group/mdlm-owt"
    verifier_model: str = "gpt2-xl"
    dtype: str = "float32"

    # --- Bounds ---
    max_verifier_ctx: int = 1024

    # --- RNG stream salts (constants; see module docstring) ---
    draft_salt: int = 0xD1EC0DE
    accept_salt: int = 0xACCE71
    fallback_salt: int = 0xFA11BAC

    def __post_init__(self) -> None:
        if self.q_mode not in ("A", "B"):
            raise ValueError(f"q_mode must be 'A' or 'B', got {self.q_mode!r}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"unsupported dtype: {self.dtype!r}")

    def fingerprint(self) -> str:
        """Stable, canonical string key for this protocol.

        JSON with `sort_keys=True` → insensitive to field-order. Used by
        `derive_seed` and by schema mismatch assertions on load.
        """
        d: dict[str, Any] = asdict(self)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


def derive_seed(protocol: ProtocolConfig, prompt_idx: int, round_idx: int) -> int:
    """Deterministic per-round seed.

    Returns a non-negative int fitting in int63 so it can be passed to
    `torch.Generator.manual_seed` without overflow surprises.

    This function is pure: same inputs → same output, across Python
    processes, shells, and machines.
    """
    key = f"{protocol.fingerprint()}|p={int(prompt_idx)}|r={int(round_idx)}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    # Take the top 8 bytes, mask to 63 bits for torch compatibility.
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)
