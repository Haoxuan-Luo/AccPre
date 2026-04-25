"""Canonical predictor-input feature extractor.

ONE function (`extract_features`) is the single source of predictor
feature vectors across training and online decode. No bespoke
extractors in any training script or eval lane.

Four families are supported in v1 (see DESIGN_PHASE2.md §C):

  - "numeric":        3 scalars from verifier prefix features
                      (entropy, margin, top1_prob).
                      Requires a verifier-prefix forward at inference.
  - "hidden":         drafter hidden (mean-pooled over draft positions).
                      Drafter-only; the ONLY truly verifier-free family.
  - "hidden_numeric": concatenation of the two above.
                      Requires a verifier-prefix forward at inference.
  - "upper_bound":    verifier-prefix hidden + drafter hidden + numeric.
                      Reference ceiling, NOT on the main cheap lane.
                      Requires a verifier-prefix forward at inference.

`feature_dim(family, ...)` reports the expected flat vector length so
predictor heads can declare input sizes statically.
"""

from __future__ import annotations

from typing import Optional

import torch

from accpre.core.schema import RoundRecord


# Default model hidden-dim assumptions (can be overridden per call).
MDLM_HIDDEN_DIM_DEFAULT: int = 768
GPT2XL_HIDDEN_DIM_DEFAULT: int = 1600

NUMERIC_DIM: int = 3  # (entropy, margin, top1_prob)


_SUPPORTED = (
    "numeric", "hidden", "hidden_numeric", "upper_bound",
    "hidden_per_pos",
    "hidden_per_pos_hs",       # Phase 5: drafter hidden + 4 scalars per position
    "hidden_per_pos_s",        # Phase 5: 4 drafter scalars per position (ablation)
    "hidden_per_pos_ml",       # Phase 6: concat(penultimate, final) per position
    "hidden_per_pos_numeric",  # Phase 7 V1 (pp_vpn): per-position drafter hidden
                                #   + prefix-numeric (3 scalars) broadcast to all γ.
                                #   V-prefix deploy mode.
    # Phase 9 — per-position transformer family. Four feature bundles
    # fed into a shared transformer encoder (see accpre.predictors.
    # pp_transformer). Position embedding lives inside the model, not
    # the offline feature. v1 is content-equivalent to hidden_per_pos;
    # v3 is content-equivalent to hidden_per_pos_numeric. The Phase 9
    # classes require these family names specifically (not the legacy
    # aliases) so the acceptance / length heads can be paired one-to-one.
    "hidden_per_pos_v1",   # A1/L1: drafter hidden only         (γ, H)         V-free
    "hidden_per_pos_v2",   # A2/L2: + 5 local scalars           (γ, H+5)       V-free
    "hidden_per_pos_v3",   # A3/L3: + 3 prefix-numeric bcast    (γ, H+3)       V-prefix
    "hidden_per_pos_v4",   # A4/L4: + 5 local + verifier_hidden (γ, H+5+H_v)   V-prefix
)


# Phase 5: each per-position scalar contributes 1 dim.
PP_SCALARS_DIM: int = 4  # q_j, entropy, margin, top1_prob

# Phase 6: multi-layer per-position drafter hidden uses the last N
# transformer block outputs. v1 pp_ml uses N=2 (penultimate + final).
PP_ML_N_LAYERS: int = 2

# Phase 9: per-position local scalars for v2 / v4 families.
#   [q_j, log q_j, drafter_entropy_j, drafter_margin_j, drafter_top1_prob_j]
# log q_j is derived at extraction time as log(clamp(q_j, min=EPS)).
PP_TX_LOCAL_SCALARS_DIM: int = 5


def _require(record: RoundRecord, field: str) -> None:
    if getattr(record, field) is None:
        raise ValueError(
            f"RoundRecord is missing field `{field}` "
            f"(prompt_idx={record.prompt_idx}, round_idx={record.round_idx}). "
            f"Stage 2A.0 collection must have been run."
        )


def extract_features(record: RoundRecord, family: str) -> torch.Tensor:
    """Return a 1-D float32 feature vector for the given family.

    Raises ValueError if required fields are missing from the record.
    """
    if family not in _SUPPORTED:
        raise ValueError(f"Unknown family {family!r}; supported: {_SUPPORTED}")

    if family == "numeric":
        for f in ("verifier_entropy", "verifier_margin", "verifier_top1_prob"):
            _require(record, f)
        return torch.tensor(
            [
                float(record.verifier_entropy),
                float(record.verifier_margin),
                float(record.verifier_top1_prob),
            ],
            dtype=torch.float32,
        )

    if family == "hidden":
        _require(record, "drafter_hidden")
        return record.drafter_hidden.to(torch.float32).flatten()

    if family == "hidden_numeric":
        h = extract_features(record, "hidden")
        n = extract_features(record, "numeric")
        return torch.cat([h, n])

    if family == "upper_bound":
        _require(record, "verifier_hidden")
        vh = record.verifier_hidden.to(torch.float32).flatten()
        hn = extract_features(record, "hidden_numeric")
        return torch.cat([vh, hn])

    if family == "hidden_per_pos":
        _require(record, "drafter_hidden_per_pos")
        # Shape: (γ, H). DO NOT flatten — the per-position head consumes
        # it as a 2-D tensor and applies a shared MLP position-wise.
        return record.drafter_hidden_per_pos.to(torch.float32)

    if family in ("hidden_per_pos_hs", "hidden_per_pos_s"):
        # Phase 5: per-position drafter-side scalars.
        for f in ("drafter_entropy_j", "drafter_margin_j", "drafter_top1_prob_j"):
            _require(record, f)
        gamma = int(record.gamma)
        # (γ, 4):  [q_j, entropy, margin, top1_prob]
        scalars = torch.stack([
            torch.tensor(record.q_j, dtype=torch.float32),
            torch.tensor(record.drafter_entropy_j, dtype=torch.float32),
            torch.tensor(record.drafter_margin_j, dtype=torch.float32),
            torch.tensor(record.drafter_top1_prob_j, dtype=torch.float32),
        ], dim=1)  # (γ, 4)
        if family == "hidden_per_pos_s":
            return scalars
        # hidden_per_pos_hs: concat drafter_hidden_per_pos + scalars along dim 1.
        _require(record, "drafter_hidden_per_pos")
        h = record.drafter_hidden_per_pos.to(torch.float32)  # (γ, H)
        return torch.cat([h, scalars], dim=1)  # (γ, H+4)

    if family == "hidden_per_pos_ml":
        # Phase 6: concat of last 2 transformer block outputs per draft
        # position. Shape `(γ, 2H)`; last H columns = final layer.
        _require(record, "drafter_hidden_per_pos_ml")
        return record.drafter_hidden_per_pos_ml.to(torch.float32)

    if family == "hidden_per_pos_numeric":
        # Phase 7 V1: per-position drafter hidden `(γ, H)` concatenated
        # with the 3 prefix-numeric scalars broadcast to every position.
        # Result shape: `(γ, H + 3)`.
        _require(record, "drafter_hidden_per_pos")
        for f in ("verifier_entropy", "verifier_margin", "verifier_top1_prob"):
            _require(record, f)
        h = record.drafter_hidden_per_pos.to(torch.float32)  # (γ, H)
        gamma = int(h.shape[0])
        numeric = torch.tensor([
            float(record.verifier_entropy),
            float(record.verifier_margin),
            float(record.verifier_top1_prob),
        ], dtype=torch.float32)                              # (3,)
        numeric_b = numeric.unsqueeze(0).expand(gamma, -1)   # (γ, 3)
        return torch.cat([h, numeric_b], dim=1)              # (γ, H+3)

    if family == "hidden_per_pos_v1":
        # Phase 9 A1/L1: drafter hidden only (content-equivalent to
        # hidden_per_pos; renamed so the A_n / L_n family-lookup is 1:1).
        _require(record, "drafter_hidden_per_pos")
        return record.drafter_hidden_per_pos.to(torch.float32)  # (γ, H)

    if family == "hidden_per_pos_v2":
        # Phase 9 A2/L2: drafter hidden + 5 local scalars per position.
        # Scalars: [q_j, log q_j, entropy_j, margin_j, top1_j].
        # log q_j uses the same EPS=1e-10 clamp as accpre.core.accept.
        _require(record, "drafter_hidden_per_pos")
        for f in ("drafter_entropy_j", "drafter_margin_j",
                  "drafter_top1_prob_j"):
            _require(record, f)
        h = record.drafter_hidden_per_pos.to(torch.float32)       # (γ, H)
        q = torch.tensor(record.q_j, dtype=torch.float32)         # (γ,)
        log_q = torch.log(q.clamp(min=1e-10))                     # (γ,)
        scalars = torch.stack([
            q,
            log_q,
            torch.tensor(record.drafter_entropy_j, dtype=torch.float32),
            torch.tensor(record.drafter_margin_j, dtype=torch.float32),
            torch.tensor(record.drafter_top1_prob_j, dtype=torch.float32),
        ], dim=1)                                                 # (γ, 5)
        return torch.cat([h, scalars], dim=1)                     # (γ, H+5)

    if family == "hidden_per_pos_v3":
        # Phase 9 A3/L3: drafter hidden + 3 prefix-numeric scalars
        # broadcast. Content-equivalent to hidden_per_pos_numeric.
        _require(record, "drafter_hidden_per_pos")
        for f in ("verifier_entropy", "verifier_margin",
                  "verifier_top1_prob"):
            _require(record, f)
        h = record.drafter_hidden_per_pos.to(torch.float32)       # (γ, H)
        gamma = int(h.shape[0])
        numeric = torch.tensor([
            float(record.verifier_entropy),
            float(record.verifier_margin),
            float(record.verifier_top1_prob),
        ], dtype=torch.float32)                                   # (3,)
        numeric_b = numeric.unsqueeze(0).expand(gamma, -1)        # (γ, 3)
        return torch.cat([h, numeric_b], dim=1)                   # (γ, H+3)

    if family == "hidden_per_pos_v4":
        # Phase 9 A4/L4: drafter hidden + 5 local scalars (as v2) +
        # raw verifier_hidden (last-prefix-token hidden from
        # verifier.prefix_features) broadcast to every draft position.
        # The learned input projection inside the transformer encoder
        # compresses the 1600-dim verifier slice; we do not hand-design
        # a pooled prefix summary here.
        _require(record, "drafter_hidden_per_pos")
        _require(record, "verifier_hidden")
        for f in ("drafter_entropy_j", "drafter_margin_j",
                  "drafter_top1_prob_j"):
            _require(record, f)
        h = record.drafter_hidden_per_pos.to(torch.float32)       # (γ, H)
        q = torch.tensor(record.q_j, dtype=torch.float32)         # (γ,)
        log_q = torch.log(q.clamp(min=1e-10))                     # (γ,)
        scalars = torch.stack([
            q,
            log_q,
            torch.tensor(record.drafter_entropy_j, dtype=torch.float32),
            torch.tensor(record.drafter_margin_j, dtype=torch.float32),
            torch.tensor(record.drafter_top1_prob_j, dtype=torch.float32),
        ], dim=1)                                                 # (γ, 5)
        vh = record.verifier_hidden.to(torch.float32)             # (H_v,)
        gamma = int(h.shape[0])
        vh_b = vh.unsqueeze(0).expand(gamma, -1)                  # (γ, H_v)
        return torch.cat([h, scalars, vh_b], dim=1)               # (γ, H+5+H_v)

    # unreachable
    raise AssertionError(f"family {family!r} not handled")


def feature_dim(
    family: str,
    drafter_hidden_dim: int = MDLM_HIDDEN_DIM_DEFAULT,
    verifier_hidden_dim: int = GPT2XL_HIDDEN_DIM_DEFAULT,
) -> int:
    """Expected flat feature vector length for the given family."""
    if family == "numeric":
        return NUMERIC_DIM
    if family == "hidden":
        return drafter_hidden_dim
    if family == "hidden_numeric":
        return drafter_hidden_dim + NUMERIC_DIM
    if family == "upper_bound":
        return verifier_hidden_dim + drafter_hidden_dim + NUMERIC_DIM
    if family == "hidden_per_pos":
        # Per-position: the predictor head sees `drafter_hidden_dim` features
        # at each of γ positions. `feature_dim` here reports the per-position
        # dimension, not γ * H — callers that need γ pass it separately.
        return drafter_hidden_dim
    if family == "hidden_per_pos_hs":
        return drafter_hidden_dim + PP_SCALARS_DIM
    if family == "hidden_per_pos_s":
        return PP_SCALARS_DIM
    if family == "hidden_per_pos_ml":
        return drafter_hidden_dim * PP_ML_N_LAYERS
    if family == "hidden_per_pos_numeric":
        return drafter_hidden_dim + NUMERIC_DIM
    if family == "hidden_per_pos_v1":
        return drafter_hidden_dim
    if family == "hidden_per_pos_v2":
        return drafter_hidden_dim + PP_TX_LOCAL_SCALARS_DIM
    if family == "hidden_per_pos_v3":
        return drafter_hidden_dim + NUMERIC_DIM
    if family == "hidden_per_pos_v4":
        return drafter_hidden_dim + PP_TX_LOCAL_SCALARS_DIM + verifier_hidden_dim
    raise ValueError(f"Unknown family {family!r}; supported: {_SUPPORTED}")
