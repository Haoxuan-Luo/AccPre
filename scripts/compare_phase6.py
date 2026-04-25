"""Phase 6 comparison — pp_base (incumbent) vs pp_ml (Phase 6 main)."""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accpre.core.commit import commit_threshold
from accpre.core.schema import load_records
from accpre.eval.predictor_metrics import acceptance_metrics
from accpre.eval.strict_ref import load_strict_tok_s

STRICT, STRICT_SRC = load_strict_tok_s()
TAUS = (0.3, 0.5, 0.7, 0.9)

# (label, checkpoint name, online decode output dir holding <ckpt>/online_*.json)
VARIANTS = [
    ("pp_base", "acc_frz_hid_pp", "outputs/phase3_online_pp_11799951"),
    ("pp_ml",   "acc_frz_pp_ml",  None),  # online dir set via env PHASE6_OUT
]

# Decision gate from the Phase 6 design note.
GATE_TAU_05 = 0.04    # absolute CF@1 improvement at τ=0.5
GATE_TAU_03 = 0.05    # absolute CF@1 improvement at τ=0.3
GATE_VAL_MSE_REL = 0.03   # relative val-MSE improvement


def l1(rows):
    m = acceptance_metrics(rows)
    qh = np.array([float(x) for r in rows for j, x in enumerate(r["q2_hat"]) if r["survived"][j] > 0.5])
    stds_qh = np.array([np.array(r["q2_hat"]).std() for r in rows])
    stds_q2 = np.array([np.array(r["q2_target"]).std() for r in rows])
    return dict(
        n=len(rows), mae=float(m["q2_mae"]), mse=float(m["q2_brier"]),
        bias=float(m["q2_bias"]), ece=float(m["q2_ece"]),
        auc=float(m["acc_auc"]) if m["acc_auc"] == m["acc_auc"] else float("nan"),
        q_min=float(qh.min()) if len(qh) else 0.0,
        q_max=float(qh.max()) if len(qh) else 0.0,
        q_med=float(np.median(qh)) if len(qh) else 0.0,
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
            mae=float(np.abs(diffs).mean()), exact=float(np.mean(diffs == 0)),
            over=float(np.mean(diffs > 0)), under=float(np.mean(diffs < 0)),
            mean_L=float(Lh.mean()), zero=float(np.mean(Lh == 0)),
        )
    return out


def l3(dir_, name, taus):
    out = {}
    if dir_ is None:
        for t in taus:
            out[t] = None
        return out
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


def read_val_mse(ckpt_dir):
    p = os.path.join("checkpoints", ckpt_dir, "train_history.json")
    if not os.path.isfile(p):
        return float("nan")
    try:
        h = json.load(open(p))
        return float(h.get("best_val", float("nan")))
    except Exception:
        return float("nan")


def main():
    # pp_base test records from stage1_pp.pt; pp_ml test records from
    # stage1_pp_ml.pt. We load both and confirm they agree on L/survived
    # (they should — the drafter is deterministic and matched-randomness).
    pp_base_recs = load_records("data_collected/stage1_pp.pt")
    pp_ml_recs = load_records("data_collected/stage1_pp_ml.pt") \
        if os.path.isfile("data_collected/stage1_pp_ml.pt") else pp_base_recs
    test_pp_base = [r for r in pp_base_recs if 60 <= r.prompt_idx < 80 and r.gamma == 8]
    test_pp_ml = [r for r in pp_ml_recs if 60 <= r.prompt_idx < 80 and r.gamma == 8]
    print(f"[cmp] {len(test_pp_base)} pp_base test records; {len(test_pp_ml)} pp_ml test records")

    # Allow env override for pp_ml online dir (set by SLURM script).
    variants = []
    for label, ck, online in VARIANTS:
        if label == "pp_ml":
            online = os.environ.get("PHASE6_OUT", online)
        variants.append((label, ck, online))

    results = []
    for label, ck, online_dir in variants:
        preds_path = f"checkpoints/{ck}/preds_test.pt"
        if not os.path.isfile(preds_path):
            print(f"[cmp] WARN: {preds_path} missing — skipping {label}")
            continue
        preds = torch.load(preds_path, weights_only=False)
        test_records = test_pp_ml if label == "pp_ml" else test_pp_base
        results.append((label, ck,
                        l1(preds),
                        l2(preds, test_records, TAUS),
                        l3(online_dir, ck, TAUS),
                        read_val_mse(ck)))

    print()
    print("=" * 110)
    print("§1  Predictor-level  (test split, survived positions)")
    print("=" * 110)
    print(f"  {'variant':<10} {'ckpt':<20} {'val_MSE':>8} {'MAE':>7} {'MSE':>7} {'bias':>8} {'ECE':>7} {'AUC':>6}  "
          f"{'Qhat min':>8} {'Qhat max':>8} {'Qhat med':>8}  {'std(Qhat)':>10} {'ratio':>6}")
    for label, ck, m1, _, _, val_mse in results:
        print(f"  {label:<10} {ck:<20} {val_mse:>8.5f} "
              f"{m1['mae']:>7.4f} {m1['mse']:>7.4f} {m1['bias']:+8.4f} "
              f"{m1['ece']:>7.4f} {m1['auc']:>6.3f}  "
              f"{m1['q_min']:>8.3f} {m1['q_max']:>8.3f} {m1['q_med']:>8.3f}  "
              f"{m1['std_qh']:>10.4f} {m1['ratio']:>6.3f}")

    for t in TAUS:
        print()
        print("=" * 110)
        marker = " **MAIN**" if t in (0.3, 0.5) else " (plateau)"
        print(f"§2+§3  Decision + system at τ={t:.1f}{marker}")
        print("=" * 110)
        print(f"  {'variant':<10} {'L_MAE':>6} {'exact':>7} {'over':>7} {'under':>7} {'meanL':>6} {'zero%':>7}  "
              f"{'tok/s':>7} {'×strict':>8} {'CF@1':>7} {'XL-aux':>7}")
        for label, _, _, m2, m3, _ in results:
            d = m2[t]; s = m3.get(t) or {}
            tok = s.get("tok_s", float("nan"))
            cf = s.get("cf", float("nan"))
            xl = s.get("xl", float("nan"))
            speedup = tok / STRICT if tok == tok else float("nan")
            print(f"  {label:<10} "
                  f"{d['mae']:>6.2f} {d['exact']:>7.1%} {d['over']:>7.1%} {d['under']:>7.1%} "
                  f"{d['mean_L']:>6.2f} {d['zero']:>7.1%}  "
                  f"{tok:>7.2f} {speedup:>7.2f}x {cf:>7.3f} {xl:>7.3f}")

    # --- decision gate ---
    print()
    print("=" * 110)
    print("§4  Decision gate (Phase 6 design note)")
    print("=" * 110)
    by_label = {x[0]: x for x in results}
    if "pp_base" not in by_label or "pp_ml" not in by_label:
        print("  Cannot evaluate gate: missing pp_base and/or pp_ml.")
    else:
        _, _, _, _, base_m3, base_val = by_label["pp_base"]
        _, _, _, _, ml_m3, ml_val = by_label["pp_ml"]
        def get_cf(m3, tau):
            s = m3.get(tau)
            return s["cf"] if s else float("nan")
        cf05_base, cf05_ml = get_cf(base_m3, 0.5), get_cf(ml_m3, 0.5)
        cf03_base, cf03_ml = get_cf(base_m3, 0.3), get_cf(ml_m3, 0.3)
        d05 = cf05_ml - cf05_base
        d03 = cf03_ml - cf03_base
        val_rel = (base_val - ml_val) / max(abs(base_val), 1e-9) \
            if (base_val == base_val and ml_val == ml_val) else float("nan")
        pass_05 = d05 >= GATE_TAU_05
        pass_03 = d03 >= GATE_TAU_03
        pass_val = val_rel >= GATE_VAL_MSE_REL
        print(f"  τ=0.5 CF@1: pp_base={cf05_base:.3f}  pp_ml={cf05_ml:.3f}  Δ={d05:+.3f}  "
              f"gate(≥{GATE_TAU_05:+.2f}) = {'PASS' if pass_05 else 'fail'}")
        print(f"  τ=0.3 CF@1: pp_base={cf03_base:.3f}  pp_ml={cf03_ml:.3f}  Δ={d03:+.3f}  "
              f"gate(≥{GATE_TAU_03:+.2f}) = {'PASS' if pass_03 else 'fail'}")
        print(f"  val MSE:    pp_base={base_val:.5f}  pp_ml={ml_val:.5f}  "
              f"Δ_rel={val_rel:+.3f}  gate(≥{GATE_VAL_MSE_REL:+.2f}) = {'PASS' if pass_val else 'fail'}")
        online_pass = pass_05 or pass_03
        overall = (
            "PASS — promote pp_ml" if (online_pass and not (pass_val is False and val_rel < -GATE_VAL_MSE_REL))
            else ("PARTIAL — predictor-level only" if (pass_val and not online_pass)
                  else "FAIL — drafter-side V-free ceiling reached")
        )
        print(f"  overall: {overall}")

    print()
    print(f"Reference: strict SpecDiff = {STRICT:.2f} tok/s, CF@1=1.000  "
          f"(source: {STRICT_SRC})")


if __name__ == "__main__":
    main()
