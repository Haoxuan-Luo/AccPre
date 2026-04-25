"""Phase 25 aggregator — combine per-lane finalize JSONs into one table.

Reads every `lane_*.json` produced by
`scripts/phase25_finalize_lane.py` and emits:
  - <out_dir>/unified_table_final.md
  - <out_dir>/unified_table_final.json

ΔNLL is filled in if and only if the `lane_strict.json` file is present.
Lanes that were not evaluated simply do not appear in the output —
nothing is fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CANONICAL_ORDER: List[Tuple[str, Optional[float]]] = [
    ("strict", None),
    ("oracle_exec",   0.5), ("oracle_exec",   0.7), ("oracle_exec",   0.9),
    ("frozen_1A",     0.5), ("frozen_1A",     0.7), ("frozen_1A",     0.9),
    ("live_jnt_10ep", 0.5), ("live_jnt_10ep", 0.7), ("live_jnt_10ep", 0.9),
    ("frozen_dep",    0.5), ("frozen_dep",    0.7), ("frozen_dep",    0.9),
    ("live_jnt_dep",  0.5), ("live_jnt_dep",  0.7), ("live_jnt_dep",  0.9),
]


def _fmt_opt(v, width: int = 10, precision: int = 4) -> str:
    if v is None:
        return f"{'—':>{width}}"
    return f"{float(v):>{width}.{precision}f}"


def _fmt_signed(v, width: int = 10, precision: int = 4) -> str:
    if v is None:
        return f"{'—':>{width}}"
    return f"{float(v):>+{width}.{precision}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane_dir", required=True,
                    help="Directory containing lane_*.json")
    ap.add_argument("--out_dir", default=None,
                    help="Output dir for the table (defaults to lane_dir)")
    ap.add_argument("--decode_dir", default=None,
                    help="Decode dir (recorded in unified_table_final.json "
                         "for provenance)")
    args = ap.parse_args()

    lane_dir = Path(args.lane_dir).resolve()
    out_dir = Path(args.out_dir or lane_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lane_files = sorted(lane_dir.glob("lane_*.json"))
    if not lane_files:
        print(f"[agg] FATAL: no lane_*.json in {lane_dir}")
        return 1

    by_key: Dict[Tuple[str, Optional[float]], Dict[str, Any]] = {}
    for lf in lane_files:
        with open(lf) as f:
            row = json.load(f)
        key = (row["method"], row.get("tau"))
        by_key[key] = row

    strict_nll = None
    if ("strict", None) in by_key:
        strict_nll = float(by_key[("strict", None)]["NLL"])

    rows: List[Dict[str, Any]] = []
    for key in CANONICAL_ORDER:
        if key not in by_key:
            continue
        r = dict(by_key[key])
        if strict_nll is not None:
            r["delta_NLL"] = r["NLL"] - strict_nll
        else:
            r["delta_NLL"] = None
        rows.append(r)

    # ----- markdown table -----
    md: List[str] = []
    md.append(
        "# Phase 25 — long-horizon greedy, per-lane finalize "
        "(20 prompts × ≤992 new tok, γ=8, T=2, temperature=0)\n\n"
    )
    md.append(
        "tok/s measured inside the decode loop (decode-job stats — "
        "untouched by this finalize). NLL, ΔNLL, and verifier-success "
        "metrics are from the offline per-lane finalize passes on the "
        "saved decode JSONs. Under temp=0, `tok_succ` is the fraction of "
        "committed tokens whose drafter-sampled token matches the "
        "verifier's argmax (Leviathan greedy accept rule). Oracle "
        "verifier-success is read from the decode-time cache; predictor "
        "verifier-success is reconstructed by replaying drafter + "
        "verifier forwards against the stored round seeds.\n\n"
    )
    if strict_nll is None:
        md.append(
            "> ⚠ lane_strict.json is missing — ΔNLL is not computable.\n\n"
        )
    md.append("```\n")
    md.append(
        f"  {'method':<14}{'τ':>6}{'tok/s':>8}{'NLL':>9}{'ΔNLL':>10}"
        f"{'tok_succ':>10}{'rnd_mean':>10}{'all_pass':>10}\n"
    )
    for r in rows:
        tau_s = "—" if r["tau"] is None else f"{float(r['tau']):.1f}"
        md.append(
            f"  {r['method']:<14}{tau_s:>6}"
            f"{float(r['tok_s_mean']):>8.2f}"
            f"{_fmt_opt(r.get('NLL'), 9, 4)}"
            f"{_fmt_signed(r.get('delta_NLL'), 10, 4)}"
            f"{_fmt_opt(r.get('tok_succ'), 10, 4)}"
            f"{_fmt_opt(r.get('rnd_mean'), 10, 4)}"
            f"{_fmt_opt(r.get('all_pass'), 10, 4)}\n"
        )
    md.append("```\n")
    text = "".join(md)
    print(text)

    with open(out_dir / "unified_table_final.md", "w") as f:
        f.write(text)
    with open(out_dir / "unified_table_final.json", "w") as f:
        json.dump({
            "rows": rows,
            "strict_nll": strict_nll,
            "lane_dir":   str(lane_dir),
            "decode_dir": str(args.decode_dir) if args.decode_dir else None,
            "lanes_seen": [f.name for f in lane_files],
        }, f, indent=2, default=str)
    print(f"[agg] wrote {out_dir}/unified_table_final.md")
    print(f"[agg] wrote {out_dir}/unified_table_final.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
