"""Compare pooled-hidden vs per-position-hidden acceptance predictors.

3 V-free pairs:
  acc_frz_hid            vs  acc_frz_hid_pp          (frozen, MSE)
  acc_frz_hid_sbce       vs  acc_frz_hid_sbce_pp     (frozen, soft-BCE)
  acc_jnt_hid            vs  acc_jnt_hid_pp          (joint single-task)

Uses preds_test.pt for L1 (+ Q̂ stats + std ratio). For joint variants,
preds_test.pt for the NEW per-pos one is against records that went
through the fine-tuned drafter (matching DESIGN), and the OLD baseline
uses its own preds_test.pt as written during training.

Online tables pulled from each run's online_decode JSONs.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accpre.core.commit import commit_threshold
from accpre.core.schema import load_records
from accpre.eval.predictor_metrics import acceptance_metrics


STRICT_TOK_S = 19.75
EVAL_TAUS = (0.3, 0.5, 0.7, 0.9)


PAIRS = [
    # (label, old_ckpt, old_online_dir, new_ckpt, new_online_dir)
    ("frozen MSE",
     "acc_frz_hid",        "outputs/phase2_online_11791962",
     "acc_frz_hid_pp",     "outputs/phase3_online_pp_11799951"),
    ("frozen sBCE",
     "acc_frz_hid_sbce",   "outputs/phase2_online_fix_11794789",
     "acc_frz_hid_sbce_pp", "outputs/phase3_online_pp_11799951"),
    ("joint single",
     "acc_jnt_hid",        "outputs/phase2b_online_refix_11799136",
     "acc_jnt_hid_pp",     "outputs/phase3_online_pp_11799951"),
]


def _l1(preds_rows) -> Dict[str, float]:
    m = acceptance_metrics(preds_rows)
    q_hat = np.array([
        float(x)
        for row in preds_rows
        for j, x in enumerate(row["q2_hat"])
        if row["survived"][j] > 0.5
    ])
    q_tgt = np.array([
        float(x)
        for row in preds_rows
        for j, x in enumerate(row["q2_target"])
        if row["survived"][j] > 0.5
    ])
    # Per-record std across positions (no survival masking — we care about
    # whether the predictor emits position-dependent values for all γ).
    stds_qhat = np.array([np.array(r["q2_hat"]).std() for r in preds_rows])
    stds_q2 = np.array([np.array(r["q2_target"]).std() for r in preds_rows])
    ratios = stds_qhat / (stds_q2 + 1e-9)
    return {
        "q2_mae": float(m["q2_mae"]),
        "q2_mse": float(m["q2_brier"]),
        "q2_bias": float(m["q2_bias"]),
        "q2_ece": float(m["q2_ece"]),
        "acc_auc": float(m["acc_auc"]) if m["acc_auc"] == m["acc_auc"] else float("nan"),
        "q_hat_min": float(q_hat.min()) if len(q_hat) else 0.0,
        "q_hat_max": float(q_hat.max()) if len(q_hat) else 0.0,
        "q_hat_med": float(np.median(q_hat)) if len(q_hat) else 0.0,
        "std_qhat_mean": float(stds_qhat.mean()),
        "std_q2_mean": float(stds_q2.mean()),
        "ratio_mean": float(ratios.mean()),
        "ratio_median": float(np.median(ratios)),
        "n": len(preds_rows),
    }


def _l2(records, predictor_ckpt_dir: str, is_per_pos: bool, taus) -> Dict[float, Dict[str, float]]:
    """Compute decision-level metrics using saved preds_test.pt Q̂.

    For joint-per-pos, the "records" should be the RE-COLLECTED records
    (passed in). For frozen, use stage1.pt test records.
    """
    # Load preds_test.pt's q2_hat by record_idx (row order)
    rows = torch.load(os.path.join(predictor_ckpt_dir, "preds_test.pt"),
                       weights_only=False)
    # rows aligned with the dataset's order for the test split; its length
    # should equal len(records) (after γ=8 filter).
    # Defensive: min length.
    n = min(len(rows), len(records))
    out = {}
    for tau in taus:
        diffs = []
        L_hats = []
        for i in range(n):
            q_hat = rows[i]["q2_hat"]
            L_hat = commit_threshold([float(x) for x in q_hat], float(tau))
            L_strict = int(records[i].L)
            diffs.append(L_hat - L_strict)
            L_hats.append(L_hat)
        diffs = np.array(diffs); L_hats = np.array(L_hats)
        out[tau] = {
            "L_mae":      float(np.abs(diffs).mean()) if n else 0.0,
            "exact":      float(np.mean(diffs == 0)) if n else 0.0,
            "over":       float(np.mean(diffs > 0)) if n else 0.0,
            "under":      float(np.mean(diffs < 0)) if n else 0.0,
            "mean_L_hat": float(L_hats.mean()) if n else 0.0,
            "zero_rate":  float(np.mean(L_hats == 0)) if n else 0.0,
        }
    return out


def _l3(online_dir: str, name: str, taus):
    out = {}
    for tau in taus:
        tag = f"{tau:.1f}".replace(".", "p")
        path = os.path.join(online_dir, name, f"online_acceptance_tau_{tag}.json")
        if not os.path.isfile(path):
            out[tau] = None
            continue
        with open(path, "r") as f:
            s = json.load(f)
        l3 = s["l3"]
        out[tau] = {
            "tok_s":   float(l3["tok_s_mean"]),
            "cf_at_1": float(l3["cf_at_1_aggregate"]),
            "xl_aux":  float(l3["xl_audit_aux_mean"]),
        }
    return out


def _records_for_ckpt(ckpt_name: str, online_dir: str):
    if ckpt_name.endswith("_pp") and ckpt_name.startswith("acc_jnt"):
        # Joint per-pos uses re-collected records.
        path = os.path.join(online_dir, ckpt_name, "strict_records_recollected.pt")
        recs = load_records(path)
        return [r for r in recs if r.gamma == 8]
    if ckpt_name.startswith("acc_jnt"):
        # Old joint (refix run).
        path = os.path.join(online_dir, ckpt_name, "strict_records_recollected.pt")
        recs = load_records(path)
        return [r for r in recs if r.gamma == 8]
    # Frozen: stage1.pt / stage1_pp.pt test split γ=8.
    data_path = "data_collected/stage1_pp.pt" if ckpt_name.endswith("_pp") else "data_collected/stage1.pt"
    if not os.path.isfile(data_path):
        data_path = "data_collected/stage1.pt"
    recs = load_records(data_path)
    return [r for r in recs if 60 <= r.prompt_idx < 80 and r.gamma == 8]


def main() -> None:
    STRICT = "strict SpecDiff: tok/s=19.75  CF@1=1.000"

    for label, old_ck, old_dir, new_ck, new_dir in PAIRS:
        print("\n" + "=" * 80)
        print(f"{label.upper()}:  {old_ck}  (pooled)   →   {new_ck}  (per-pos)")
        print("=" * 80)

        # L1: load preds_test.pt directly
        for tag, ck in (("OLD (pooled)", old_ck), ("NEW (per-pos)", new_ck)):
            rows = torch.load(os.path.join("checkpoints", ck, "preds_test.pt"),
                              weights_only=False)
            l1 = _l1(rows)
            print(
                f"\n  [{tag}]  n={l1['n']}"
                f"\n    MAE={l1['q2_mae']:.4f}  MSE={l1['q2_mse']:.4f}  "
                f"bias={l1['q2_bias']:+.4f}  ECE={l1['q2_ece']:.4f}  "
                f"AUC={l1['acc_auc']:.3f}"
                f"\n    Q̂ range: min={l1['q_hat_min']:.3f}  max={l1['q_hat_max']:.3f}  "
                f"median={l1['q_hat_med']:.3f}"
                f"\n    per-record std: std(Q̂)={l1['std_qhat_mean']:.4f}  "
                f"std(Q2)={l1['std_q2_mean']:.4f}  "
                f"ratio mean={l1['ratio_mean']:.3f}  median={l1['ratio_median']:.3f}"
            )

        # L2 and L3
        old_records = _records_for_ckpt(old_ck, old_dir)
        new_records = _records_for_ckpt(new_ck, new_dir)
        l2_old = _l2(old_records, os.path.join("checkpoints", old_ck),
                     is_per_pos=False, taus=EVAL_TAUS)
        l2_new = _l2(new_records, os.path.join("checkpoints", new_ck),
                     is_per_pos=True, taus=EVAL_TAUS)
        l3_old = _l3(old_dir, old_ck, EVAL_TAUS)
        l3_new = _l3(new_dir, new_ck, EVAL_TAUS)

        print("\n  Decision + System  (OLD / NEW):")
        header = (
            f"    {'τ':<4} "
            f"{'L_MAE':>11}  {'exact':>11}  {'over':>11}  {'under':>11}  "
            f"{'meanL':>11}  {'zero%':>11}  "
            f"{'tok/s':>13}  {'CF@1':>13}"
        )
        print(header)
        for tau in EVAL_TAUS:
            do = l2_old[tau]; dn = l2_new[tau]
            so = l3_old.get(tau) or {}; sn = l3_new.get(tau) or {}
            ok_tok = so.get("tok_s", float("nan"))
            nw_tok = sn.get("tok_s", float("nan"))
            ok_cf = so.get("cf_at_1", float("nan"))
            nw_cf = sn.get("cf_at_1", float("nan"))
            print(
                f"    {tau:<4} "
                f"{do['L_mae']:>5.2f}/{dn['L_mae']:>5.2f}  "
                f"{do['exact']:>4.1%}/{dn['exact']:>4.1%}  "
                f"{do['over']:>4.1%}/{dn['over']:>4.1%}  "
                f"{do['under']:>4.1%}/{dn['under']:>4.1%}  "
                f"{do['mean_L_hat']:>5.2f}/{dn['mean_L_hat']:>5.2f}  "
                f"{do['zero_rate']:>4.1%}/{dn['zero_rate']:>4.1%}  "
                f"{ok_tok:>5.1f}/{nw_tok:>5.1f}  "
                f"{ok_cf:>5.3f}/{nw_cf:>5.3f}"
            )

    print("\n" + "=" * 80)
    print(f"Reference:  {STRICT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
