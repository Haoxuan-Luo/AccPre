"""Phase 9 Step-2 sanity report for len_tx_v1.

Checks three things:

  (a) Training trajectory is healthy:
        - train_loss decreases over at least the first 2 epochs.
        - val_loss is finite.
  (b) Predicted L distribution over {0..gamma} is non-degenerate:
        - at least 3 distinct L_hat values across the test split;
        - no single class covers >= 95% of predictions.
  (c) Per-tau |mean(L_hat) - mean(L_strict)| is within 1.5 on the
      online-decode records at the four standard taus.

This is a sanity check, not a performance gate.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Let this script run as `python scripts/sanity_l1.py ...` from the repo
# root (the Phase 9 SLURM job invokes it that way) by adding the repo
# root to sys.path BEFORE we import accpre.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

CKPT = "checkpoints/len_tx_v1"
TAUS = (0.3, 0.5, 0.7, 0.9)


def _load_history() -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(CKPT, "train_history.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f).get("history")


def _load_preds_test() -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(CKPT, "preds_test.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, weights_only=False)


def _load_strict_records():
    """Filter stage1_pp.pt to the test split with gamma=8."""
    from accpre.core.schema import load_records
    from accpre.data.splits import TRAIN_N, VAL_N, POOL_SIZE

    recs = load_records("data_collected/stage1_pp.pt")
    lo = TRAIN_N + VAL_N
    hi = POOL_SIZE
    return [r for r in recs if lo <= r.prompt_idx < hi and r.gamma == 8]


def _online_L_stats(online_dir: str, tau: float,
                    strict_records) -> Optional[Dict[str, float]]:
    tag = f"{tau:.1f}".replace(".", "p")
    path = os.path.join(online_dir, f"online_committed_length_tau_{tag}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        j = json.load(f)
    lane = j["online_lane"]

    # L_hat: collect from every round of every prompt in the lane.
    l_hats: List[int] = []
    for p_res in lane["per_prompt"]:
        for r in p_res["rounds"]:
            l_hats.append(int(r["L_hat"]))

    # Strict reference: L from the strict records covering the same 10
    # global prompt indices. Online has ~80 rounds; strict records have
    # up to 160 (max_new_tokens=128 vs 64). We compare mean of a
    # per-round signal so the scale is fine.
    allowed = set(int(p_res["prompt_idx"]) for p_res in lane["per_prompt"])
    l_strict = [int(r.L) for r in strict_records if int(r.prompt_idx) in allowed]

    if not l_hats or not l_strict:
        return None
    mean_hat = sum(l_hats) / len(l_hats)
    mean_strict = sum(l_strict) / len(l_strict)
    return {
        "n_hat": len(l_hats),
        "n_strict": len(l_strict),
        "mean_L_hat": mean_hat,
        "mean_L_strict": mean_strict,
        "abs_delta": abs(mean_hat - mean_strict),
        "cf_at_1": float(j["l3"]["cf_at_1_aggregate"]),
        "tok_s_mean": float(j["l3"]["tok_s_mean"]),
    }


def main(online_dir: str) -> int:
    print("=" * 80)
    print("Phase 9 Step-2 sanity: len_tx_v1")
    print("=" * 80)

    # (a) Training trajectory.
    history = _load_history()
    print()
    if history is None:
        print("(a) training trajectory : MISSING train_history.json")
        stage_a_ok = False
    else:
        print("(a) training trajectory")
        print(f"   {'epoch':>5} {'train':>10} {'val':>10}")
        for h in history:
            print(f"   {h['epoch']:>5} {h['train_loss']:>10.5f} {h['val_loss']:>10.5f}")
        train_decreasing = all(
            history[i + 1]["train_loss"] <= history[i]["train_loss"] + 1e-4
            for i in range(min(2, len(history) - 1))
        )
        val_finite = all(
            h["val_loss"] == h["val_loss"] and h["val_loss"] < float("inf")
            for h in history
        )
        stage_a_ok = train_decreasing and val_finite
        print(f"   train decreasing (first 2 epochs): {train_decreasing}")
        print(f"   val finite: {val_finite}")
        print(f"   (a) {'PASS' if stage_a_ok else 'FAIL'}")

    # (b) Non-degenerate distribution on preds_test.pt.
    preds = _load_preds_test()
    print()
    if preds is None:
        print("(b) prediction distribution : MISSING preds_test.pt")
        stage_b_ok = False
    else:
        hats = collections.Counter(int(r["L_hat"]) for r in preds)
        tgts = collections.Counter(int(r["L_target"]) for r in preds)
        n = len(preds)
        print("(b) prediction distribution on preds_test.pt")
        print(f"   n = {n}")
        print(f"   L_hat    : {dict(sorted(hats.items()))}")
        print(f"   L_target : {dict(sorted(tgts.items()))}")
        distinct = len(hats)
        max_frac = (max(hats.values()) / n) if n else 1.0
        stage_b_ok = (distinct >= 3) and (max_frac < 0.95)
        print(f"   distinct L_hat classes: {distinct}  (>=3?)")
        print(f"   top class frac       : {max_frac:.3f}  (<0.95?)")
        print(f"   (b) {'PASS' if stage_b_ok else 'FAIL'}")

    # (c) Per-tau mean L_hat vs mean L_strict on online records.
    try:
        strict = _load_strict_records()
    except Exception as e:  # pragma: no cover
        strict = None
        print(f"(c) could not load strict records: {e}")
    print()
    stage_c_rows = []
    if strict is None:
        print("(c) per-tau mean-L comparison : STRICT RECORDS MISSING")
        stage_c_ok = False
    else:
        print("(c) per-tau mean predicted L vs mean strict L")
        print(f"   {'tau':<4}  {'n_hat':>6} {'n_strict':>8}  "
              f"{'mean_Lhat':>10} {'mean_Lstr':>10} {'|d|':>6}  "
              f"{'CF@1':>6} {'tok/s':>7}")
        stage_c_ok = True
        for t in TAUS:
            r = _online_L_stats(online_dir, t, strict)
            if r is None:
                print(f"   {t:<4}  MISSING")
                stage_c_ok = False
                continue
            line = (
                f"   {t:<4}  {r['n_hat']:>6} {r['n_strict']:>8}  "
                f"{r['mean_L_hat']:>10.3f} {r['mean_L_strict']:>10.3f} "
                f"{r['abs_delta']:>6.2f}  {r['cf_at_1']:>6.3f} "
                f"{r['tok_s_mean']:>7.2f}"
            )
            print(line)
            stage_c_rows.append((t, r))
            if r["abs_delta"] > 1.5:
                stage_c_ok = False
        print(
            f"   (c) {'PASS' if stage_c_ok else 'FAIL (>1.5 at some tau)'}"
        )

    print()
    print("Summary")
    print(f"  (a) training trajectory : {'PASS' if stage_a_ok else 'FAIL'}")
    print(f"  (b) non-degenerate L_hat : {'PASS' if stage_b_ok else 'FAIL'}")
    print(f"  (c) mean-L within 1.5    : {'PASS' if stage_c_ok else 'FAIL'}")
    print()
    print("This is a plumbing sanity check; absolute performance is not yet")
    print("the point. The rest of the L2..L4 family runs regardless.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python sanity_l1.py <online_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
