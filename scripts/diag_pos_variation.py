"""Zero-training diagnostic for Issue 7 (mean-pool position-blindness).

For each acceptance predictor, reads the saved `preds_test.pt` and
measures, WITHOUT running any model:

  - per-record std of Q̂_j across γ positions
  - per-record std of Q2_j (ground truth) across γ positions
  - ratio std(Q̂)/std(Q2) (position-discrimination retention)
  - per-position mean of Q̂_j and Q2_j

If std(Q̂) is systematically much smaller than std(Q2), the predictor
cannot discriminate by position — the mean-pooled input is the dominant
information bottleneck.

No training, no model loads. Uses artifacts already on disk.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_ROOT = "checkpoints"
PREDICTORS = [
    "acc_frz_num",
    "acc_frz_hid",
    "acc_frz_hidnum",
    "acc_frz_num_sbce",
    "acc_frz_hid_sbce",
    "acc_frz_hidnum_sbce",
    "acc_jnt_hid",
    "acc_jnt_hidnum",
    "acc_jnt_mt_hid",
    "acc_jnt_mt_hidnum",
]


def _sec(s: str) -> None:
    print()
    print("=" * 80)
    print(s)
    print("=" * 80)


def _quantiles(x: np.ndarray) -> str:
    if len(x) == 0:
        return "no data"
    return (
        f"mean={x.mean():.4f}  median={np.median(x):.4f}  "
        f"min={x.min():.4f}  max={x.max():.4f}  "
        f"p10={np.quantile(x, 0.1):.4f}  p90={np.quantile(x, 0.9):.4f}"
    )


def main() -> None:
    _sec("§1+§2+§3  per-record across-position std of Q̂ vs Q2")
    print(
        f"{'predictor':<24} {'n':>5}  "
        f"{'std(Q̂) mean':>12}  {'std(Q2) mean':>12}  "
        f"{'ratio μ':>8}  {'ratio med':>10}"
    )
    all_stats = []
    for name in PREDICTORS:
        path = os.path.join(CKPT_ROOT, name, "preds_test.pt")
        if not os.path.isfile(path):
            print(f"  SKIP {name}: no preds_test.pt at {path}")
            continue
        rows = torch.load(path, weights_only=False)
        n = len(rows)
        stds_qhat = []
        stds_q2 = []
        qhat_per_pos = []  # list of γ-arrays
        q2_per_pos = []
        for r in rows:
            qh = np.array(r["q2_hat"], dtype=np.float64)
            qt = np.array(r["q2_target"], dtype=np.float64)
            stds_qhat.append(qh.std())
            stds_q2.append(qt.std())
            qhat_per_pos.append(qh)
            q2_per_pos.append(qt)
        stds_qhat = np.array(stds_qhat)
        stds_q2 = np.array(stds_q2)
        # Safe ratio (per record)
        eps = 1e-8
        ratios = stds_qhat / (stds_q2 + eps)
        all_stats.append({
            "name": name, "n": n,
            "stds_qhat": stds_qhat, "stds_q2": stds_q2, "ratios": ratios,
            "qhat_stack": np.stack(qhat_per_pos, axis=0),
            "q2_stack": np.stack(q2_per_pos, axis=0),
            "path": path,
        })
        print(
            f"{name:<24} {n:>5}  "
            f"{stds_qhat.mean():>12.4f}  {stds_q2.mean():>12.4f}  "
            f"{ratios.mean():>8.3f}  {np.median(ratios):>10.3f}"
        )

    _sec("§1  Detailed distribution of std(Q̂) and std(Q2)")
    for s in all_stats:
        print(f"\n  [{s['name']}]  n={s['n']}  preds_test: {s['path']}")
        print(f"    std(Q̂)  : {_quantiles(s['stds_qhat'])}")
        print(f"    std(Q2) : {_quantiles(s['stds_q2'])}")
        print(f"    ratio   : {_quantiles(s['ratios'])}")
        # Threshold evidence: fraction of records with std(Q̂) < 0.05 (Q̂ essentially flat)
        flat_qhat_rate = float(np.mean(s["stds_qhat"] < 0.05))
        flat_q2_rate = float(np.mean(s["stds_q2"] < 0.05))
        print(
            f"    records with std < 0.05:  Q̂: {flat_qhat_rate:6.1%}   "
            f"Q2: {flat_q2_rate:6.1%}"
        )

    _sec("§4  Per-position mean of Q̂_j and Q2_j (γ=8)")
    print(
        f"{'predictor':<24}  "
        + " ".join(f"{'j='+str(j):>8}" for j in range(8))
    )
    for s in all_stats:
        qhat_mean_per_j = s["qhat_stack"].mean(axis=0)
        q2_mean_per_j = s["q2_stack"].mean(axis=0)
        print(f"{s['name']:<24}  [Q̂]  " + " ".join(f"{v:>7.4f}" for v in qhat_mean_per_j))
        print(f"{'':<24}  [Q2]  " + " ".join(f"{v:>7.4f}" for v in q2_mean_per_j))

    _sec("§3  Direct comparison verdict")
    print("  A ratio of std(Q̂) / std(Q2) well below 1.0 indicates the predictor")
    print("  is not discriminating by position — it outputs near-constant Q̂ across")
    print("  positions while Q2 actually varies meaningfully by position.")
    print()
    worst_ratio_name, worst_ratio_val = None, float("inf")
    best_ratio_name, best_ratio_val = None, 0.0
    for s in all_stats:
        mean_ratio = float(s["ratios"].mean())
        if mean_ratio < worst_ratio_val:
            worst_ratio_val, worst_ratio_name = mean_ratio, s["name"]
        if mean_ratio > best_ratio_val:
            best_ratio_val, best_ratio_name = mean_ratio, s["name"]
    print(
        f"  Most position-blind predictor:   {worst_ratio_name}  "
        f"(mean std ratio = {worst_ratio_val:.3f})"
    )
    print(
        f"  Least position-blind predictor:  {best_ratio_name}  "
        f"(mean std ratio = {best_ratio_val:.3f})"
    )


if __name__ == "__main__":
    main()
