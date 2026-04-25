"""Unified acceptance-side summary.

Pulls L1 (predictor-level), L2 (decision-level), and L3 (system-level)
metrics for every acceptance predictor variant trained so far, onto a
single set of tables. For each predictor the "record set" used for
L1/L2 matches the drafter state the online CF@1 was measured on:

  - frozen predictors  → stage1.pt test split (original drafter)
  - joint predictors   → re-collected records (fine-tuned drafter)

Strict reference comes from the most recent Phase-1 run. No new
training; this is a pure post-hoc analysis.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accpre.collect.features import extract_features
from accpre.core.commit import commit_threshold
from accpre.core.schema import RoundRecord, load_records
from accpre.eval.predictor_metrics import acceptance_metrics
from accpre.predictors.acceptance_mlp import AcceptanceMLP


STRICT_TOK_S = 19.75   # Phase 1 strict reference
CKPT_ROOT = "checkpoints"
DATA = "data_collected/stage1.pt"


PREDICTORS = [
    # (ckpt_name, stage_label, deploy, online_dir, recollect_records?)
    ("acc_frz_num",          "frozen MSE",      "V-prefix",
     "outputs/phase2_online_11791962",        False),
    ("acc_frz_hid",          "frozen MSE",      "V-free",
     "outputs/phase2_online_11791962",        False),
    ("acc_frz_hidnum",       "frozen MSE",      "V-prefix",
     "outputs/phase2_online_11791962",        False),

    ("acc_frz_num_sbce",     "frozen sBCE",     "V-prefix",
     "outputs/phase2_online_fix_11794789",    False),
    ("acc_frz_hid_sbce",     "frozen sBCE",     "V-free",
     "outputs/phase2_online_fix_11794789",    False),
    ("acc_frz_hidnum_sbce",  "frozen sBCE",     "V-prefix",
     "outputs/phase2_online_fix_11794789",    False),

    ("acc_jnt_hid",          "joint single",    "V-free",
     "outputs/phase2b_online_refix_11799136", True),
    ("acc_jnt_hidnum",       "joint single",    "V-prefix",
     "outputs/phase2b_online_refix_11799136", True),

    ("acc_jnt_mt_hid",       "joint multi",     "V-free",
     "outputs/phase2b2_online_11799180",      True),
    ("acc_jnt_mt_hidnum",    "joint multi",     "V-prefix",
     "outputs/phase2b2_online_11799180",      True),
]

EVAL_TAUS = (0.3, 0.5, 0.7, 0.9)


def _load_predictor(name: str):
    d = os.path.join(CKPT_ROOT, name)
    with open(os.path.join(d, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    p = AcceptanceMLP(
        family=cfg["family"], gamma=int(cfg["gamma"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    p.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
    p.eval()
    return p, cfg


def _records_for(predictor_entry) -> List[RoundRecord]:
    name, stage, deploy, online_dir, is_joint = predictor_entry
    if is_joint:
        # Use online_decode's in-memory-then-saved re-collected records
        # (these use the fine-tuned drafter → matching CF@1 measurement).
        path = os.path.join(online_dir, name, "strict_records_recollected.pt")
        recs = load_records(path)
        # Strip tail rounds where cur_gamma < γ (same rule applied during
        # training; these can't be batch-stacked with the γ=8 majority).
        return [r for r in recs if r.gamma == 8]
    # Frozen: stage1.pt test split, γ=8 only.
    all_records = load_records(DATA)
    return [r for r in all_records if 60 <= r.prompt_idx < 80 and r.gamma == 8]


def _compute_q_hats(predictor, records):
    """Return list of (γ,) numpy Q̂ vectors, one per record."""
    out = []
    with torch.no_grad():
        for r in records:
            feats = extract_features(r, predictor.family)
            q = predictor.predict_q2(feats).detach().cpu().numpy()
            out.append(q)
    return out


def _level1(predictor, records, q_hats):
    """Pack records + Q̂ into acceptance_metrics's row format and run it."""
    pred_rows = []
    for r, q_hat in zip(records, q_hats):
        pred_rows.append({
            "record_idx": 0,  # unused by metrics
            "q2_hat": [float(x) for x in q_hat.tolist()],
            "q2_target": [float(x) for x in r.min_pq_j],
            "survived": [float(x) for x in r.survived_j],
            "accepted": [float(x) for x in r.accepted_j],
        })
    m = acceptance_metrics(pred_rows)
    q_hat_flat = np.array([
        float(x)
        for row in pred_rows
        for j, x in enumerate(row["q2_hat"])
        if row["survived"][j] > 0.5
    ])
    return {
        "q2_mae":    float(m["q2_mae"]),
        "q2_mse":    float(m["q2_brier"]),
        "q2_bias":   float(m["q2_bias"]),
        "q2_ece":    float(m["q2_ece"]),
        "acc_auc":   float(m["acc_auc"]) if m["acc_auc"] == m["acc_auc"] else float("nan"),
        "q_hat_min": float(q_hat_flat.min()) if len(q_hat_flat) else 0.0,
        "q_hat_max": float(q_hat_flat.max()) if len(q_hat_flat) else 0.0,
        "q_hat_med": float(np.median(q_hat_flat)) if len(q_hat_flat) else 0.0,
    }


def _level2(records, q_hats, taus):
    """Decision-level metrics per τ."""
    out = {}
    for tau in taus:
        L_hats = np.array([
            commit_threshold([float(x) for x in q.tolist()], float(tau))
            for q in q_hats
        ])
        L_strict = np.array([int(r.L) for r in records])
        diffs = L_hats - L_strict
        n = len(diffs)
        out[tau] = {
            "n":           n,
            "L_mae":       float(np.abs(diffs).mean()) if n else 0.0,
            "exact":       float(np.mean(diffs == 0)) if n else 0.0,
            "over":        float(np.mean(diffs > 0)) if n else 0.0,
            "under":       float(np.mean(diffs < 0)) if n else 0.0,
            "mean_L_hat":  float(L_hats.mean()) if n else 0.0,
            "zero_rate":   float(np.mean(L_hats == 0)) if n else 0.0,
            # n_commit = max(1, L̂); progress guard activated ⇔ L̂ == 0.
            "progress_guard_rate": float(np.mean(L_hats == 0)) if n else 0.0,
            "mean_n_commit": float(np.maximum(L_hats, 1).mean()) if n else 0.0,
        }
    return out


def _level3(online_dir, name, taus):
    out = {}
    for tau in taus:
        tag = f"{tau:.1f}".replace(".", "p")
        path = os.path.join(online_dir, name,
                            f"online_acceptance_tau_{tag}.json")
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


# --------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------


def _section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def main() -> None:
    results = []
    for entry in PREDICTORS:
        name, stage, deploy, online_dir, is_joint = entry
        print(f"[report] {name} ({stage}, {deploy}) ...")
        p, cfg = _load_predictor(name)
        records = _records_for(entry)
        q_hats = _compute_q_hats(p, records)
        l1 = _level1(p, records, q_hats)
        l2 = _level2(records, q_hats, EVAL_TAUS)
        l3 = _level3(online_dir, name, EVAL_TAUS)
        results.append({
            "name": name, "stage": stage, "deploy": deploy,
            "n_records": len(records),
            "l1": l1, "l2": l2, "l3": l3,
        })

    # --- §1 L1 table ---
    _section("§1  Predictor-level metrics (L1) — on the per-predictor record set")
    print(
        f"{'predictor':<24} {'stage':<13} {'deploy':<9} {'n':>5} "
        f"{'MAE':>7} {'MSE':>7} {'bias':>8} {'ECE':>7} {'AUC':>6} "
        f"{'Q_min':>7} {'Q_max':>7} {'Q_med':>7}"
    )
    for r in results:
        l1 = r["l1"]
        auc_str = f"{l1['acc_auc']:.3f}" if np.isfinite(l1["acc_auc"]) else "  nan"
        print(
            f"{r['name']:<24} {r['stage']:<13} {r['deploy']:<9} {r['n_records']:>5} "
            f"{l1['q2_mae']:>7.4f} {l1['q2_mse']:>7.4f} {l1['q2_bias']:+8.4f} "
            f"{l1['q2_ece']:>7.4f} {auc_str:>6} "
            f"{l1['q_hat_min']:>7.3f} {l1['q_hat_max']:>7.3f} {l1['q_hat_med']:>7.3f}"
        )

    # --- §2+§3 per-tau combined table ---
    for tau in EVAL_TAUS:
        _section(f"§2+§3  Decision + system metrics at τ={tau:.1f}")
        hdr = (
            f"{'predictor':<24} {'stage':<13} "
            f"{'L_MAE':>6} {'exact':>6} {'over':>6} {'under':>6} "
            f"{'meanL':>6} {'zero%':>6} "
            f"{'tok/s':>7} {'×strict':>8} {'CF@1':>7} {'XL-aux':>7}"
        )
        print(hdr)
        for r in results:
            d = r["l2"][tau]
            s = r["l3"].get(tau) or {}
            tok_s = s.get("tok_s", float("nan"))
            speedup = tok_s / STRICT_TOK_S if tok_s == tok_s else float("nan")
            cf = s.get("cf_at_1", float("nan"))
            xl = s.get("xl_aux", float("nan"))
            print(
                f"{r['name']:<24} {r['stage']:<13} "
                f"{d['L_mae']:>6.3f} {d['exact']:>6.1%} {d['over']:>6.1%} {d['under']:>6.1%} "
                f"{d['mean_L_hat']:>6.2f} {d['zero_rate']:>6.1%} "
                f"{tok_s:>7.2f} {speedup:>7.2f}× {cf:>7.3f} {xl:>7.3f}"
            )

    _section("Reference")
    print(f"  strict SpecDiff: tok/s={STRICT_TOK_S:.2f}  CF@1=1.000  (baseline)")
    print("  oracle-Q2 best  : tok/s=22.25  CF@1=0.823  (Pareto ceiling)")


if __name__ == "__main__":
    main()
