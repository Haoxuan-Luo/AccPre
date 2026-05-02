"""Frozen protocol configuration and per-round seed derivation.

Vendored from `accpre/core/protocol.py`. Pins every runtime parameter that
could affect the definition of q_j, accepted_j, L, or any logged artifact.

Matched-randomness discipline:
  - `round_rng_seed` is the sole randomness input to a round.
  - It is derived deterministically from
        SHA-256( protocol.fingerprint() | prompt_idx | round_idx ).
  - From `round_rng_seed` we derive three independent sub-streams by XORing
    with the three salts on this dataclass (draft_rng / accept_rng /
    fallback_rng). They are used in `draft_verify._make_generator`.
  - The salts are constants. Changing them invalidates all existing records.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import hashlib
import json


@dataclass(frozen=True)
class ProtocolConfig:
    schema_version: int = 1

    # temperature == 1.0 is the v1 main protocol; accepted_j is the
    # Bernoulli test with the Leviathan ratio (see accept.py).
    # temperature == 0.0 flips accepted_j to argmax(target) == draft_tok.
    temperature: float = 1.0
    # q_mode "A" is the canonical v1 q definition: the SUBS-parameterized
    # log p_theta(x0 | xt) row at the DDPM step where each draft position
    # was first unmasked, evaluated on the drafted token.
    q_mode: str = "A"

    drafter_model: str = "kuleshov-group/mdlm-owt"
    verifier_model: str = "gpt2-xl"
    dtype: str = "float32"

    max_verifier_ctx: int = 1024

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
        """Stable, canonical string key for this protocol."""
        d: dict[str, Any] = asdict(self)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


def derive_seed(protocol: ProtocolConfig, prompt_idx: int, round_idx: int) -> int:
    """Deterministic per-round seed (non-negative, fits in int63)."""
    key = f"{protocol.fingerprint()}|p={int(prompt_idx)}|r={int(round_idx)}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)
