"""Build the Leviathan acceptance-probability (q2) target file.

Per record, per drafted position j:

    y_j = min(1, p_v(draft_j) / q(draft_j))   ∈ [0, 1]

This is the Leviathan acceptance probability for speculative decoding
under temperature=1.

Implementation note
-------------------
The collection pipeline already stores `min_pq_j` per record (see
`accpre/core/schema.py:RoundRecord` — the schema comment explicitly
labels it `# == Q2_j in v1 terminology`). The values are bit-identical
to `min(1, p_j / q_j)` regardless of the protocol's temperature, so
this builder requires NO verifier replay and NO drafter replay.

It is purely a load + repack + save operation.

Output dict shape (matches the relmax / dep target files so
`accpre/train/dataset.py:_load_dep_targets` can ingest it via the
existing `dep_targets_path` mechanism):

    {
      "protocol_fp":      str,
      "drafter_fp":       str,
      "sigma":            0.0,            # placeholder; not used here
      "gamma":            int,
      "targets":          dict[(p, r) -> list[float γ]],
      "_label_semantics": "min(1, p_v(draft) / q(draft)) under T=1, sourced from RoundRecord.min_pq_j",
      "_summary":         {n_records, n_total_positions, value_min, value_max,
                           value_mean, n_eq_1, n_lt_0p1},
    }

CLI
---
    python build_target_q2.py \
        --records data_collected/stage1_pp_owt_300_g15_T1.pt \
        --protocol configs/protocol.yaml \
        --gamma 15 \
        --out data_collected/stage1_pp_owt_300_g15_T1_q2.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--gamma", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    protocol = load_protocol(Path(args.protocol))
    print(f"[q2] protocol fp: {protocol.fingerprint()[:60]}...")
    print(f"[q2] temperature: {protocol.temperature}")

    records = load_records(args.records, expected_protocol=protocol)
    gamma_records = [r for r in records if int(r.gamma) == int(args.gamma)]
    print(f"[q2] loaded {len(records)} records; γ={args.gamma} → {len(gamma_records)} kept")

    targets: Dict = {}
    n_total_positions = 0
    n_eq_1 = 0
    n_lt_0p1 = 0
    val_min = 1e9
    val_max = -1e9
    val_sum = 0.0

    for r in gamma_records:
        if len(r.min_pq_j) != int(args.gamma):
            raise ValueError(
                f"record min_pq_j length={len(r.min_pq_j)} != gamma={args.gamma} "
                f"at prompt_idx={r.prompt_idx} round_idx={r.round_idx}"
            )
        vals = [float(x) for x in r.min_pq_j]
        # Range check: schema guarantees [0, 1] but verify here as a safety net.
        for v in vals:
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"min_pq_j out of range [0,1] at prompt_idx={r.prompt_idx} "
                    f"round_idx={r.round_idx}: {v}"
                )
        targets[(int(r.prompt_idx), int(r.round_idx))] = vals
        n_total_positions += len(vals)
        for v in vals:
            val_sum += v
            if v < val_min: val_min = v
            if v > val_max: val_max = v
            if v >= 1.0 - 1e-9: n_eq_1 += 1
            if v < 0.1: n_lt_0p1 += 1

    if not targets:
        raise RuntimeError(f"no records at gamma={args.gamma} in {args.records}")

    summary = {
        "n_records": int(len(targets)),
        "n_total_positions": int(n_total_positions),
        "value_min": float(val_min),
        "value_max": float(val_max),
        "value_mean": float(val_sum / max(n_total_positions, 1)),
        "n_eq_1": int(n_eq_1),
        "n_lt_0p1": int(n_lt_0p1),
    }
    print(f"[q2] sanity:\n  {json.dumps(summary, indent=2)}")

    blob = {
        "protocol_fp":      protocol.fingerprint(),
        "drafter_fp":       protocol.drafter_model,
        "sigma":            0.0,
        "gamma":            int(args.gamma),
        "targets":          targets,
        "_label_semantics": "min(1, p_v(draft) / q(draft)) under T=1, sourced from RoundRecord.min_pq_j",
        "_summary":         summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, str(out_path))
    print(f"[q2] saved {len(targets)} targets to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
