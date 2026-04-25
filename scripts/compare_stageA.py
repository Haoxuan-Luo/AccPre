"""Phase 4 Stage A comparison — pp_base vs pp_wide vs pp_attn."""
from __future__ import annotations

import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accpre.core.commit import commit_threshold
from accpre.core.schema import load_records
from accpre.eval.predictor_metrics import acceptance_metrics

STRICT = 19.75
TAUS = (0.3, 0.5, 0.7, 0.9)

VARIANTS = [
    ("pp_base", "acc_frz_hid_pp",   "outputs/phase3_online_pp_11799951"),
    ("pp_wide", "acc_frz_pp_wide",  "outputs/phase4_A_11800305"),
    ("pp_attn", "acc_frz_pp_attn",  "outputs/phase4_A_11800305"),
]


def l1(rows):
    m = acceptance_metrics(rows)
    qh = np.array([float(x) for r in rows for j, x in enumerate(r["q2_hat"]) if r["survived"][j] > 0.5])
    stds_qh = np.array([np.array(r["q2_hat"]).std() for r in rows])
    stds_q2 = np.array([np.array(r["q2_target"]).std() for r in rows])
    return dict(
        n=len(rows),
        mae=float(m["q2_mae"]), mse=float(m["q2_brier"]),
        bias=float(m["q2_bias"]), ece=float(m["q2_ece"]),
        auc=float(m["acc_auc"]) if m["acc_auc"] == m["acc_auc"] else float("nan"),
        q_min=float(qh.min()) if len(qh) else 0.0,
        q_max=float(qh.max()) if len(qh) else 0.0,
        q_med=float(np.median(qh)) if len(qh) else 0.0,
        q_mean=float(qh.mean()) if len(qh) else 0.0,
        std_qh=float(stds_qh.mean()),
        std_q2=float(stds_q2.mean()),
        ratio=float((stds_qh / (stds_q2 + 1e-9)).mean()),
    )


def l2(rows, records, taus):
    n = min(len(rows), len(records))
    out = {}
    for t in taus:
        diffs, Lh = [], []
        for i in range(n):
            Lhat = commit_threshold([float(x) for x in rows[i]["q2_hat"]], float(t))
            diffs.append(Lhat - int(records[i].L))
            Lh.append(Lhat)
        diffs = np.array(diffs); Lh = np.array(Lh)
        out[t] = dict(
            mae=float(np.abs(diffs).mean()),
            exact=float(np.mean(diffs == 0)),
            over=float(np.mean(diffs > 0)),
            under=float(np.mean(diffs < 0)),
            mean_L=float(Lh.mean()),
            zero=float(np.mean(Lh == 0)),
        )
    return out


def l3(dir_, name, taus):
    out = {}
    for t in taus:
        tag = f"{t:.1f}".replace(".", "p")
        p = os.path.join(dir_, name, f"online_acceptance_tau_{tag}.json")
        if os.path.isfile(p):
            s = json.load(open(p))["l3"]
            out[t] = dict(tok_s=float(s["tok_s_mean"]),
                          cf=float(s["cf_at_1_aggregate"]),
                          xl=float(s["xl_audit_aux_mean"]))
        else:
            out[t] = None
    return out


def main():
    # frozen V-free on stage1_pp.pt test split, γ=8
    data_path = "data_collected/stage1_pp.pt" if os.path.isfile("data_collected/stage1_pp.pt") else "data_collected/stage1.pt"
    all_recs = load_records(data_path)
    test_records = [r for r in all_recs if 60 <= r.prompt_idx < 80 and r.gamma == 8]
    print(f"[cmp] loaded {len(test_records)} test records from {data_path}")

    results = []
    for label, ck, online_dir in VARIANTS:
        preds = torch.load(f"checkpoints/{ck}/preds_test.pt", weights_only=False)
        m1 = l1(preds)
        m2 = l2(preds, test_records, TAUS)
        m3 = l3(online_dir, ck, TAUS)
        results.append((label, ck, m1, m2, m3))

    print()
    print("=" * 96)
    print("§1  Predictor-level  (test split, survived positions unless noted)")
    print("=" * 96)
    print(f"  {'variant':<10} {'ckpt':<20} {'n':>5} {'MAE':>7} {'MSE':>7} {'bias':>8} {'ECE':>7} {'AUC':>6}  {'Q̂ min':>6} {'Q̂ max':>6} {'Q̂ med':>6}  {'std(Q̂)':>8} {'ratio':>6}")
    for label, ck, m1, _, _ in results:
        print(f"  {label:<10} {ck:<20} {m1['n']:>5} "
              f"{m1['mae']:>7.4f} {m1['mse']:>7.4f} {m1['bias']:+8.4f} "
              f"{m1['ece']:>7.4f} {m1['auc']:>6.3f}  "
              f"{m1['q_min']:>6.3f} {m1['q_max']:>6.3f} {m1['q_med']:>6.3f}  "
              f"{m1['std_qh']:>8.4f} {m1['ratio']:>6.3f}")

    for t in TAUS:
        print()
        print("=" * 96)
        print(f"§2+§3  Decision + system at τ={t:.1f}")
        print("=" * 96)
        print(f"  {'variant':<10} {'L_MAE':>6} {'exact':>7} {'over':>7} {'under':>7} {'meanL':>6} {'zero%':>7}  {'tok/s':>7} {'×strict':>8} {'CF@1':>7} {'XL-aux':>7}")
        for label, _, _, m2, m3 in results:
            d = m2[t]; s = m3.get(t) or {}
            tok = s.get("tok_s", float("nan"))
            cf = s.get("cf", float("nan"))
            xl = s.get("xl", float("nan"))
            speedup = tok / STRICT if tok == tok else float("nan")
            print(f"  {label:<10} "
                  f"{d['mae']:>6.2f} {d['exact']:>7.1%} {d['over']:>7.1%} {d['under']:>7.1%} "
                  f"{d['mean_L']:>6.2f} {d['zero']:>7.1%}  "
                  f"{tok:>7.2f} {speedup:>7.2f}× {cf:>7.3f} {xl:>7.3f}")

    print()
    print(f"Reference: strict SpecDiff = {STRICT} tok/s, CF@1=1.000")


if __name__ == "__main__":
    main()
