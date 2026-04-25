"""Phase 7 V1 comparison — pp_base (V-free ref) vs pp_vpn (V-prefix min)."""
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

# (label, checkpoint name, online output dir for that ckpt)
VARIANTS = [
    ("pp_base", "acc_frz_hid_pp", "outputs/phase3_online_pp_11799951"),
    ("pp_vpn",  "acc_frz_pp_vpn", None),   # online dir set via env PHASE7_OUT
]

# Phase 7 V1 gate (design note).
GATE_TAU_05_CF = 0.58      # absolute CF@1 at τ=0.5
GATE_TAU_05_TOKS = 25.0    # tok/s at τ=0.5
GATE_TAU_03_CF = 0.40      # absolute CF@1 at τ=0.3
GATE_TAU_03_TOKS = 35.0    # tok/s at τ=0.3
GATE_VAL_MSE_REL = 0.05    # ≥5% relative improvement vs pp_base
GATE_MIN_SPEEDUP = 1.25    # tok/s > 1.25× strict at both main τ


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
    """Read tok/s, CF@1, XL-aux per τ. Also derive wall-clock breakdown."""
    out = {}
    if dir_ is None:
        for t in taus:
            out[t] = None
        return out
    for t in taus:
        tag = f"{t:.1f}".replace(".", "p")
        p = os.path.join(dir_, name, f"online_acceptance_tau_{tag}.json")
        if not os.path.isfile(p):
            out[t] = None
            continue
        j = json.load(open(p))
        s = j["l3"]
        lane = j["online_lane"]
        # Aggregate per-round wall-clock breakdown across all prompts.
        draft_sum = verifier_sum = predictor_sum = 0.0
        n_rounds = 0
        elapsed_sum = 0.0
        tokens_sum = 0
        for p_res in lane["per_prompt"]:
            elapsed_sum += p_res["elapsed_s"]
            tokens_sum += p_res["n_new_tokens"]
            for r in p_res["rounds"]:
                draft_sum += r["draft_time_s"]
                verifier_sum += r["verifier_prefix_time_s"]
                predictor_sum += r["predictor_time_s"]
                n_rounds += 1
        out[t] = dict(
            tok_s=float(s["tok_s_mean"]),
            cf=float(s["cf_at_1_aggregate"]),
            xl=float(s["xl_audit_aux_mean"]),
            n_rounds=int(n_rounds),
            draft_ms=float(draft_sum / max(n_rounds, 1) * 1000.0),
            verifier_ms=float(verifier_sum / max(n_rounds, 1) * 1000.0),
            predictor_ms=float(predictor_sum / max(n_rounds, 1) * 1000.0),
            draft_frac=float(draft_sum / max(elapsed_sum, 1e-9)),
            verifier_frac=float(verifier_sum / max(elapsed_sum, 1e-9)),
            predictor_frac=float(predictor_sum / max(elapsed_sum, 1e-9)),
        )
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
    all_recs = load_records("data_collected/stage1_pp.pt")
    test_records = [r for r in all_recs if 60 <= r.prompt_idx < 80 and r.gamma == 8]
    print(f"[cmp] {len(test_records)} test records")

    # Allow env override for pp_vpn online dir.
    variants = []
    for label, ck, online in VARIANTS:
        if label == "pp_vpn":
            online = os.environ.get("PHASE7_OUT", online)
        variants.append((label, ck, online))

    results = []
    for label, ck, online_dir in variants:
        preds_path = f"checkpoints/{ck}/preds_test.pt"
        if not os.path.isfile(preds_path):
            print(f"[cmp] WARN: {preds_path} missing — skipping {label}")
            continue
        preds = torch.load(preds_path, weights_only=False)
        results.append((label, ck,
                        l1(preds),
                        l2(preds, test_records, TAUS),
                        l3(online_dir, ck, TAUS),
                        read_val_mse(ck)))

    # --- §1 predictor-level ---
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

    # --- §2 + §3 decision + system ---
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

    # --- §3b wall-clock breakdown ---
    print()
    print("=" * 110)
    print("§3b  Wall-clock breakdown (mean per-round ms; share-of-total per τ)")
    print("=" * 110)
    print(f"  {'variant':<10} {'tau':<5} {'rounds':>6}  "
          f"{'draft_ms':>9} {'verif_ms':>9} {'pred_ms':>9}  "
          f"{'draft_%':>8} {'verif_%':>8} {'pred_%':>8}  {'other_%':>8}")
    for label, _, _, _, m3, _ in results:
        for t in TAUS:
            s = m3.get(t)
            if s is None:
                continue
            other_frac = 1.0 - s["draft_frac"] - s["verifier_frac"] - s["predictor_frac"]
            print(f"  {label:<10} {t:<5} {s['n_rounds']:>6}  "
                  f"{s['draft_ms']:>9.2f} {s['verifier_ms']:>9.2f} {s['predictor_ms']:>9.2f}  "
                  f"{s['draft_frac']:>7.1%} {s['verifier_frac']:>7.1%} {s['predictor_frac']:>7.1%}  {other_frac:>7.1%}")

    # --- §4 decision gate ---
    print()
    print("=" * 110)
    print("§4  Decision gate (Phase 7 V1 design note)")
    print("=" * 110)
    by_label = {x[0]: x for x in results}
    if "pp_base" not in by_label or "pp_vpn" not in by_label:
        print("  Cannot evaluate gate: missing pp_base and/or pp_vpn.")
        return
    _, _, _, _, base_m3, base_val = by_label["pp_base"]
    _, _, vpn_l1, _, vpn_m3, vpn_val = by_label["pp_vpn"]

    def get(m3, tau, k):
        s = m3.get(tau)
        return (s[k] if s else float("nan"))

    cf05_base, cf05_vpn = get(base_m3, 0.5, "cf"), get(vpn_m3, 0.5, "cf")
    cf03_base, cf03_vpn = get(base_m3, 0.3, "cf"), get(vpn_m3, 0.3, "cf")
    tok05_vpn = get(vpn_m3, 0.5, "tok_s")
    tok03_vpn = get(vpn_m3, 0.3, "tok_s")
    val_rel = (
        (base_val - vpn_val) / max(abs(base_val), 1e-9)
        if (base_val == base_val and vpn_val == vpn_val) else float("nan")
    )
    d05 = cf05_vpn - cf05_base
    d03 = cf03_vpn - cf03_base

    pass_primary_cf = cf05_vpn >= GATE_TAU_05_CF
    pass_primary_tok = tok05_vpn >= GATE_TAU_05_TOKS
    pass_secondary_cf = cf03_vpn >= GATE_TAU_03_CF
    pass_secondary_tok = tok03_vpn >= GATE_TAU_03_TOKS
    pass_val = val_rel >= GATE_VAL_MSE_REL
    speedup05 = tok05_vpn / STRICT if tok05_vpn == tok05_vpn else float("nan")
    speedup03 = tok03_vpn / STRICT if tok03_vpn == tok03_vpn else float("nan")
    pass_speedup = (speedup05 >= GATE_MIN_SPEEDUP) and (speedup03 >= GATE_MIN_SPEEDUP)

    print(f"  τ=0.5  CF@1: pp_base={cf05_base:.3f}  pp_vpn={cf05_vpn:.3f}  Δ={d05:+.3f}  "
          f"gate(≥{GATE_TAU_05_CF:.2f}) = {'PASS' if pass_primary_cf else 'fail'}")
    print(f"         tok/s: pp_vpn={tok05_vpn:.2f}  (≥{GATE_TAU_05_TOKS:.1f}?) "
          f"= {'PASS' if pass_primary_tok else 'fail'}   "
          f"speedup={speedup05:.2f}× strict")
    print(f"  τ=0.3  CF@1: pp_base={cf03_base:.3f}  pp_vpn={cf03_vpn:.3f}  Δ={d03:+.3f}  "
          f"gate(≥{GATE_TAU_03_CF:.2f}) = {'PASS' if pass_secondary_cf else 'fail'}")
    print(f"         tok/s: pp_vpn={tok03_vpn:.2f}  (≥{GATE_TAU_03_TOKS:.1f}?) "
          f"= {'PASS' if pass_secondary_tok else 'fail'}   "
          f"speedup={speedup03:.2f}× strict")
    print(f"  val MSE: pp_base={base_val:.5f}  pp_vpn={vpn_val:.5f}  Δ_rel={val_rel:+.3f}  "
          f"gate(≥{GATE_VAL_MSE_REL:.2f}) = {'PASS' if pass_val else 'fail'}")
    print(f"  speedup floor (≥{GATE_MIN_SPEEDUP:.2f}× strict at both main τ): "
          f"{'PASS' if pass_speedup else 'fail'}")

    online_pass = (pass_primary_cf and pass_primary_tok) or (pass_secondary_cf and pass_secondary_tok)
    overall = (
        "PASS — open V-prefix lane" if (online_pass and pass_val and pass_speedup)
        else ("PARTIAL — predictor-level only"
              if (pass_val and not online_pass)
              else "FAIL")
    )
    print(f"  overall: {overall}")

    # --- failure-mode interpretation (only when online gates fail) ---
    if not online_pass:
        print()
        print("  Failure-mode diagnosis:")
        # Signal-insufficient vs head-doesn't-exploit.
        if val_rel < 0.02:
            print(f"    val MSE moved only {val_rel:+.3f} (rel) — the prefix-numeric broadcast")
            print(f"    adds little new information the predictor can use. SIGNAL INSUFFICIENT.")
        elif pass_val:
            print(f"    val MSE improved {val_rel:+.3f} (rel, ≥ gate) but CF@1 did not move. The")
            print(f"    predictor learns something but the head's decisions do not benefit at")
            print(f"    the commit thresholds we test. HEAD-FAILS-TO-EXPLOIT.")
        else:
            print(f"    val MSE moved {val_rel:+.3f} (rel) — measurable but below the 5% gate;")
            print(f"    signal is present but small, and the head does not translate it into")
            print(f"    online CF@1 gains. Borderline — leaning SIGNAL INSUFFICIENT.")

    print()
    print(f"Reference: strict SpecDiff = {STRICT:.2f} tok/s, CF@1=1.000  "
          f"(source: {STRICT_SRC})")


if __name__ == "__main__":
    main()
