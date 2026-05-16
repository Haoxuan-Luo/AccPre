"""Baseline reuse gate.

Reuses the 5 T=1 baseline JSONs from `experiments/OWT_Frozen_0429/results/baselines/`,
filtering out the stale `online_strict_temp0.json` (only 2 prompts; from a
greedy-protocol run that does not match this experiment's frozen v1 / T=1
discipline).

Sanity gates (PASS-required):
  1. Source dir exists.
  2. Each accepted baseline file has `n_prompts >= 100` (the protocol pool
     test split is 100).
  3. Each accepted baseline filename ends in `_T1.json` (T=1 protocol).
  4. The expected 5 names are all present:
        online_strict_soft_T1.json
        online_lossy_soft_T1_l0p3.json
        online_lossy_soft_T1_l0p5.json
        online_lossy_soft_T1_l0p7.json
        online_lossy_soft_T1_l1.json
  5. `online_strict_temp0.json` is in the source dir but is filtered out
     (its presence is OK; reusing it would be wrong).

Output:
  results/baselines/<basename>          (copies, NOT symlinks — protects
                                         this experiment from upstream edits)
  results/baselines_metrics.csv         (one row per accepted baseline:
                                         method_name, n_prompts, tok_s_mean,
                                         nll_mean, tok_succ_mean, tokens_per_round)

Exit code: 0 if all gates pass; 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]


# Authoritative names. Anything outside this set is filtered.
_ACCEPT = {
    "online_strict_soft_T1.json",
    "online_lossy_soft_T1_l0p3.json",
    "online_lossy_soft_T1_l0p5.json",
    "online_lossy_soft_T1_l0p7.json",
    "online_lossy_soft_T1_l1.json",
}
_FORBIDDEN = {
    "online_strict_temp0.json",   # n_prompts=2; greedy-temp0 protocol
}


def _mean_tokens_per_round(d: Dict[str, Any]) -> float:
    n_total = 0
    n_rounds = 0
    for pp in d.get("per_prompt", []):
        for rd in pp.get("rounds", []):
            n_total += int(rd.get("n_committed", rd.get("n_commit", 0)))
            n_rounds += 1
    return n_total / n_rounds if n_rounds else 0.0


def _check_one(p: Path) -> List[str]:
    msgs: List[str] = []
    name = p.name
    if name in _FORBIDDEN:
        msgs.append(f"{name}: explicitly forbidden (filter is the gate, not a fail)")
        return msgs
    if name not in _ACCEPT:
        msgs.append(f"{name}: unknown baseline (not in accept list)")
        return msgs
    # The protocol tag must be `_T1` (Leviathan T=1). Both `..._T1.json`
    # (strict) and `..._T1_l<lambda>.json` (lossy) are accepted.
    if "_T1" not in name:
        msgs.append(f"{name}: filename must contain `_T1` (Leviathan T=1 protocol)")
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception as e:
        msgs.append(f"{name}: load failed: {e}")
        return msgs
    n = len(d.get("per_prompt", []))
    if n < 100:
        msgs.append(f"{name}: n_prompts={n} < 100")
    return msgs


def _row(p: Path) -> Dict[str, Any]:
    with open(p) as f:
        d = json.load(f)
    return dict(
        method_name=p.stem.replace("online_", ""),
        n_prompts=len(d.get("per_prompt", [])),
        tok_s_mean=float(d.get("tok_s_mean", 0.0)),
        nll_mean=float(d.get("nll_mean", float("nan"))),
        tok_succ_mean=float(d.get("tok_succ_mean", float("nan"))),
        tokens_per_round=_mean_tokens_per_round(d),
        source_path=str(p),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source_dir",
        default=str(_REPO_ROOT / "experiments/OWT_Frozen_0429/results/baselines"),
        help="upstream baselines directory (the predecessor experiment)",
    )
    ap.add_argument(
        "--out_dir",
        default=str(_EXP_ROOT / "results" / "baselines"),
        help="local copy destination",
    )
    ap.add_argument(
        "--out_csv",
        default=str(_EXP_ROOT / "results" / "baselines_metrics.csv"),
    )
    ap.add_argument("--no_copy", action="store_true",
                    help="run gates only; do not copy files")
    args = ap.parse_args()

    src = Path(args.source_dir)
    if not src.exists():
        print(f"[baselines] ERROR source dir missing: {src}")
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. List source files; gate forbidden / unknown / missing.
    found_names = {p.name for p in src.iterdir() if p.is_file() and p.suffix == ".json"}
    missing = sorted(_ACCEPT - found_names)
    if missing:
        print(f"[baselines] ERROR missing required baselines: {missing}")
        return 1

    # 2. Per-file gates.
    failures: List[str] = []
    accepted: List[Path] = []
    for name in sorted(_ACCEPT):
        p = src / name
        msgs = _check_one(p)
        if msgs:
            failures.extend(f"{name}: {m}" for m in msgs)
            continue
        accepted.append(p)

    if failures:
        for m in failures:
            print(f"[baselines] FAIL {m}")
        return 1

    # 3. Verify forbidden file is filtered.
    for name in _FORBIDDEN:
        if name in found_names:
            print(f"[baselines] OK filtering {name} (present upstream but not reused)")

    # 4. Copy.
    if not args.no_copy:
        for p in accepted:
            dst = out_dir / p.name
            if not dst.exists():
                shutil.copyfile(str(p), str(dst))
                print(f"[baselines] copied {p.name} -> {dst}")
            else:
                print(f"[baselines] SKIP {p.name} (already present at {dst})")

    # 5. Write metrics CSV.
    rows = [_row(p) for p in accepted]
    cols = ["method_name", "n_prompts", "tok_s_mean", "nll_mean",
            "tok_succ_mean", "tokens_per_round", "source_path"]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[baselines] wrote {out_csv} ({len(rows)} rows)")

    print(f"[baselines] PASS ({len(accepted)} accepted; "
          f"{len(_FORBIDDEN & found_names)} forbidden filtered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
