"""Phase 9 unified comparison across A1..A4 and L1..L4.

Four sections:

  Section 1 — predictor-level
    Acceptance (A1..A4):
      val MSE  (from train_history.json::best_val)
      test MAE / Brier / ECE / AUROC on preds_test.pt's survived rows.
    Length (L1..L4):
      val CE   (from train_history.json::best_val)
      per-tau L_MAE / exact / over / under on preds_test.pt
      (tau is the one sampled during _save_predictions; a per-record
       point estimate of how well argmax tracks commit_threshold).

  Section 2 — decision-level
    Acceptance (A1..A4): CF@1 per tau from online_acceptance_tau_*.json
    Length     (L1..L4): CF@1 per tau from online_committed_length_tau_*.json
    (Same L = 1[ L_hat == record.L ] on strict records, under the v1
    shared-fallback coupling. For length, L_hat is argmax of the
    categorical head; for acceptance, L_hat = commit_threshold(Q_hat,
    tau).)

  Section 3 — system-level
    tok/s per predictor per tau from the online JSONs; `x strict`
    column pulled from accpre.eval.strict_ref.load_strict_tok_s.

  Section 4 — tau-sensitivity diagnostic (length family only)
    For each L predictor, evaluates logits at tau in TAU_PROBE on a
    sample of test records and reports:
      - mean number of distinct L_hat per record across tau
      - mean |L_hat(tau_hi) - L_hat(tau_lo)| per record
      - mean KL( softmax(tau=0.1) || softmax(tau=0.9) )
    Tells us whether tau is actually moving the classifier.

Inputs are resolved by environment variables so the script can be
pointed at any Phase 9 run:

  PHASE9_ACC_REST_OUT  = outputs/phase9_acc_rest_<jobid>   (for A2..A4)
  PHASE9_LEN_REST_OUT  = outputs/phase9_len_rest_<jobid>   (for L2..L4)

A1 and L1 paths come from their own sanity jobs and are looked up
below. If any online directory is missing, that predictor's decision
/ system row will show `-`.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Repo root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

# Cap CPU thread count. The login node advertises 40 cores to the
# container; at default thread count a single 2-layer transformer
# forward can take ~2.5s due to thread-pool thrashing. torch.set_num_threads(2)
# turns a 500-forward loop from 21 minutes into well under a second.
torch.set_num_threads(2)

from accpre.eval.predictor_metrics import acceptance_metrics
from accpre.eval.strict_ref import load_strict_tok_s
from accpre.core.commit import commit_threshold


TAUS = (0.3, 0.5, 0.7, 0.9)
TAU_PROBE = (0.1, 0.3, 0.5, 0.7, 0.9)   # finer grid for the sensitivity probe


# -----------------------------------------------------------------------------
# Resolve expected artefact paths. A1/L1 come from their own sanity jobs;
# A2..A4 / L2..L4 come from the acc_rest / len_rest jobs. Any missing
# predictor is reported but does not crash the script.
# -----------------------------------------------------------------------------


def _most_recent(prefix: str) -> Optional[str]:
    """Find the most recent outputs/<prefix>_<jobid> directory, if any."""
    base = "outputs"
    if not os.path.isdir(base):
        return None
    cands = [
        os.path.join(base, x) for x in os.listdir(base)
        if x.startswith(prefix + "_") and os.path.isdir(os.path.join(base, x))
    ]
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def _resolve_runs() -> Tuple[List[Dict], List[Dict]]:
    """Resolve (accept_runs, length_runs) — each entry names a ckpt
    and an online output directory."""
    a_rest = os.environ.get("PHASE9_ACC_REST_OUT") or _most_recent(
        "phase9_acc_rest"
    )
    l_rest = os.environ.get("PHASE9_LEN_REST_OUT") or _most_recent(
        "phase9_len_rest"
    )
    a1_out = os.environ.get("PHASE9_A1_OUT") or _most_recent("phase9_a1_online")
    l1_out = os.environ.get("PHASE9_L1_OUT") or _most_recent("phase9_l1_online")

    def _sub(parent: Optional[str], name: str) -> Optional[str]:
        if parent is None:
            return None
        path = os.path.join(parent, name)
        return path if os.path.isdir(path) else None

    acc = [
        {"tag": "A1", "ckpt": "checkpoints/acc_tx_v1",
         "online": _sub(a1_out, "acc_tx_v1"), "family": "hidden_per_pos_v1"},
        {"tag": "A2", "ckpt": "checkpoints/acc_tx_v2",
         "online": _sub(a_rest, "acc_tx_v2"), "family": "hidden_per_pos_v2"},
        {"tag": "A3", "ckpt": "checkpoints/acc_tx_v3",
         "online": _sub(a_rest, "acc_tx_v3"), "family": "hidden_per_pos_v3"},
        {"tag": "A4", "ckpt": "checkpoints/acc_tx_v4",
         "online": _sub(a_rest, "acc_tx_v4"), "family": "hidden_per_pos_v4"},
    ]
    length = [
        {"tag": "L1", "ckpt": "checkpoints/len_tx_v1",
         "online": _sub(l1_out, "len_tx_v1"), "family": "hidden_per_pos_v1"},
        {"tag": "L2", "ckpt": "checkpoints/len_tx_v2",
         "online": _sub(l_rest, "len_tx_v2"), "family": "hidden_per_pos_v2"},
        {"tag": "L3", "ckpt": "checkpoints/len_tx_v3",
         "online": _sub(l_rest, "len_tx_v3"), "family": "hidden_per_pos_v3"},
        {"tag": "L4", "ckpt": "checkpoints/len_tx_v4",
         "online": _sub(l_rest, "len_tx_v4"), "family": "hidden_per_pos_v4"},
    ]
    return acc, length


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _read_best_val(ckpt: str) -> Optional[float]:
    path = os.path.join(ckpt, "train_history.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return float(json.load(f).get("best_val", float("nan")))


def _read_online(online_dir: Optional[str], tau: float, kind: str):
    if online_dir is None:
        return None
    tag = f"{tau:.1f}".replace(".", "p")
    path = os.path.join(online_dir, f"online_{kind}_tau_{tag}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        j = json.load(f)
    s = j["l3"]
    return {
        "tok_s_mean": float(s["tok_s_mean"]),
        "cf_at_1": float(s["cf_at_1_aggregate"]),
        "xl_aux": float(s.get("xl_audit_aux_mean", float("nan"))),
    }


def _load_preds(ckpt: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(ckpt, "preds_test.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, weights_only=False)


# -----------------------------------------------------------------------------
# Section 1 — predictor-level
# -----------------------------------------------------------------------------


def _acceptance_l1(ckpt: str) -> Dict[str, Any]:
    val = _read_best_val(ckpt)
    preds = _load_preds(ckpt)
    if preds is None:
        return {"val_mse": val, "l1": None}
    m = acceptance_metrics(preds)
    # std(Q_hat) / std(Q2) ratio — matches the handoff-era reporting.
    qh = np.array([
        float(x)
        for r in preds for j, x in enumerate(r["q2_hat"])
        if r["survived"][j] > 0.5
    ])
    stds_qh = np.array([np.array(r["q2_hat"]).std() for r in preds])
    stds_q2 = np.array([np.array(r["q2_target"]).std() for r in preds])
    ratio = float((stds_qh / (stds_q2 + 1e-9)).mean())
    return {
        "val_mse": val,
        "mae": float(m["q2_mae"]),
        "brier": float(m["q2_brier"]),
        "ece": float(m["q2_ece"]),
        "auc": float(m["acc_auc"]) if m["acc_auc"] == m["acc_auc"] else float("nan"),
        "qh_mean": float(qh.mean()) if len(qh) else 0.0,
        "qh_std": float(qh.std()) if len(qh) else 0.0,
        "std_ratio": ratio,
    }


def _length_l1(ckpt: str) -> Dict[str, Any]:
    val = _read_best_val(ckpt)
    preds = _load_preds(ckpt)
    if preds is None:
        return {"val_ce": val, "l1": None}
    by_tau: Dict[float, List[Dict[str, int]]] = {}
    for r in preds:
        by_tau.setdefault(float(r["tau"]), []).append(
            {"L_hat": int(r["L_hat"]), "L_target": int(r["L_target"])}
        )
    per_tau = {}
    for tau in sorted(by_tau.keys()):
        rows = by_tau[tau]
        diffs = np.array([e["L_hat"] - e["L_target"] for e in rows])
        per_tau[tau] = {
            "n": len(rows),
            "mae": float(np.abs(diffs).mean()),
            "exact": float((diffs == 0).mean()),
            "over": float((diffs > 0).mean()),
            "under": float((diffs < 0).mean()),
        }
    return {"val_ce": val, "per_tau": per_tau}


# -----------------------------------------------------------------------------
# Section 4 — tau-sensitivity diagnostic (length family)
# -----------------------------------------------------------------------------


_TEST_RECS_CACHE = None


def _get_test_records(gamma: int = 8):
    """Load and cache the test-split records; avoid re-reading the 200MB
    stage1_pp.pt each time compare is called across 4 length predictors."""
    global _TEST_RECS_CACHE
    if _TEST_RECS_CACHE is not None:
        return _TEST_RECS_CACHE
    from accpre.core.schema import load_records
    from accpre.data.splits import TRAIN_N, VAL_N, POOL_SIZE
    all_recs = load_records("data_collected/stage1_pp.pt")
    lo = TRAIN_N + VAL_N
    hi = POOL_SIZE
    _TEST_RECS_CACHE = [
        r for r in all_recs
        if lo <= r.prompt_idx < hi and r.gamma == gamma
    ]
    return _TEST_RECS_CACHE


def _length_tau_sensitivity(
    ckpt: str, family: str, gamma: int = 8, max_samples: int = 400,
) -> Optional[Dict[str, float]]:
    """Load the L checkpoint, batch-evaluate at TAU_PROBE on the test
    split, report tau-sensitivity diagnostics.

    Batched: one forward per tau covering all records, not per-record
    per-tau. Turns a 4000-forward walk into 5 big forwards.
    """
    import yaml
    from accpre.collect.features import extract_features
    from accpre.predictors.pp_transformer import LengthTx

    cfg_path = os.path.join(ckpt, "config.yaml")
    model_path = os.path.join(ckpt, "model.pt")
    if not (os.path.isfile(cfg_path) and os.path.isfile(model_path)):
        return None

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    p = LengthTx(
        family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
        d_model=int(cfg.get("d_model", 128)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    p.load_state_dict(torch.load(model_path, map_location="cpu"))
    p.eval()

    test_recs = _get_test_records(p.gamma)
    if not test_recs:
        return None
    if len(test_recs) > max_samples:
        step = max(1, len(test_recs) // max_samples)
        test_recs = test_recs[::step][:max_samples]

    # Stack all features once: (N, gamma, F_in).
    feats_list = [extract_features(r, family) for r in test_recs]
    feats_batch = torch.stack(feats_list, dim=0)
    N = feats_batch.shape[0]

    # One batched forward per tau.
    per_tau_logits: List[torch.Tensor] = []
    with torch.no_grad():
        for tau in TAU_PROBE:
            tau_vec = torch.full((N,), float(tau), dtype=feats_batch.dtype)
            logits = p.forward(feats_batch, tau_vec)             # (N, C)
            per_tau_logits.append(logits)

    stack = torch.stack(per_tau_logits, dim=0)                    # (T, N, C)
    l_hats = stack.argmax(dim=-1)                                 # (T, N)

    # distinct L_hat per record across tau
    distinct_counts = [int(len(torch.unique(l_hats[:, i]))) for i in range(N)]
    span = [int(l_hats[:, i].max() - l_hats[:, i].min()) for i in range(N)]

    # KL( P(tau=0.1) || P(tau=0.9) ), mean across records.
    p_lo = torch.softmax(stack[0], dim=-1)                        # (N, C)
    p_hi = torch.softmax(stack[-1], dim=-1)                       # (N, C)
    eps = 1e-12
    kl_per_rec = (
        p_lo * (torch.log(p_lo + eps) - torch.log(p_hi + eps))
    ).sum(dim=-1)                                                 # (N,)
    mean_kl = float(kl_per_rec.mean())

    # Mean std of each logit coordinate across tau, averaged over (N, C).
    logit_std_mean = float(stack.std(dim=0).mean())

    return {
        "n_records": N,
        "mean_distinct_Lhat": float(np.mean(distinct_counts)),
        "mean_Lhat_span": float(np.mean(span)),
        "frac_records_tau_sensitive": float(
            np.mean([c > 1 for c in distinct_counts])
        ),
        "mean_kl_lo_hi": mean_kl,
        "logit_std_across_tau": logit_std_mean,
    }


# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------


def _fmt(x, fmt="{:.4f}", missing="   -   "):
    if x is None:
        return missing
    try:
        if x != x:   # NaN
            return missing
    except Exception:
        return missing
    return fmt.format(x)


def _print_section_header(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def main() -> int:
    acc_runs, len_runs = _resolve_runs()
    strict_tok_s, strict_src = load_strict_tok_s()

    _print_section_header("Section 1 — predictor-level")
    print()
    print("  ACCEPTANCE  (test split, survived positions)")
    print(f"  {'tag':<3} {'ckpt':<18} {'val_MSE':>9} {'MAE':>7} {'Brier':>7} "
          f"{'ECE':>7} {'AUC':>6}  {'Q_hat_std':>10} {'std_ratio':>10}")
    for run in acc_runs:
        info = _acceptance_l1(run["ckpt"])
        row = f"  {run['tag']:<3} {os.path.basename(run['ckpt']):<18} "
        row += f"{_fmt(info.get('val_mse'), '{:>9.5f}')} "
        row += f"{_fmt(info.get('mae'), '{:>7.4f}')} "
        row += f"{_fmt(info.get('brier'), '{:>7.4f}')} "
        row += f"{_fmt(info.get('ece'), '{:>7.4f}')} "
        row += f"{_fmt(info.get('auc'), '{:>6.3f}')}  "
        row += f"{_fmt(info.get('qh_std'), '{:>10.4f}')} "
        row += f"{_fmt(info.get('std_ratio'), '{:>10.3f}')}"
        print(row)

    print()
    print("  LENGTH  (test split; tau is the random draw at _save_predictions)")
    print(f"  {'tag':<3} {'ckpt':<18} {'val_CE':>9} "
          f"{'per-tau L_MAE (n)':>30}")
    for run in len_runs:
        info = _length_l1(run["ckpt"])
        per_tau = info.get("per_tau") or {}
        per_tau_str = " ".join(
            f"{tau}:{v['mae']:.2f}(n={v['n']})" for tau, v in sorted(per_tau.items())
        ) or "-"
        row = f"  {run['tag']:<3} {os.path.basename(run['ckpt']):<18} "
        row += f"{_fmt(info.get('val_ce'), '{:>9.5f}')}  "
        row += per_tau_str
        print(row)

    _print_section_header("Section 2 — decision-level (CF@1 per tau)")
    print()
    print(f"  {'tag':<3} {'ckpt':<18}  " + "  ".join(
        f"{'tau='+str(t):>8}" for t in TAUS
    ))
    for run in acc_runs + len_runs:
        kind = "acceptance" if run["tag"].startswith("A") else "committed_length"
        row = f"  {run['tag']:<3} {os.path.basename(run['ckpt']):<18}"
        for t in TAUS:
            got = _read_online(run["online"], t, kind)
            row += f"  {_fmt(got['cf_at_1'] if got else None, '{:>8.3f}')}"
        print(row)

    _print_section_header(
        f"Section 3 — system-level  (strict ref = {strict_tok_s:.2f} tok/s;"
        f" source: {strict_src})"
    )
    print()
    print(f"  {'tag':<3} {'ckpt':<18}  " + "  ".join(
        f"{'tau='+str(t)+' tok/s':>14}" for t in TAUS
    ) + "    " + "  ".join(f"{'x strict':>8}" for _ in TAUS))
    for run in acc_runs + len_runs:
        kind = "acceptance" if run["tag"].startswith("A") else "committed_length"
        tok_row = f"  {run['tag']:<3} {os.path.basename(run['ckpt']):<18}"
        mult_row = ""
        for t in TAUS:
            got = _read_online(run["online"], t, kind)
            tok_s = got["tok_s_mean"] if got else None
            mult = (tok_s / strict_tok_s) if tok_s is not None else None
            tok_row += f"  {_fmt(tok_s, '{:>14.2f}')}"
            mult_row += f"  {_fmt(mult, '{:>8.2f}')}"
        print(tok_row + "    " + mult_row)

    _print_section_header(
        "Section 4 — tau-sensitivity diagnostic (length family only)"
    )
    print()
    print("  For each L predictor, how does its output change as tau varies?")
    print("  tau probe grid: " + ", ".join(str(t) for t in TAU_PROBE))
    print()
    print(f"  {'tag':<3} {'ckpt':<18} {'n_rec':>6} "
          f"{'distinct_Lhat':>14} {'Lhat_span':>10} {'tau_sens_frac':>14} "
          f"{'KL(lo||hi)':>11} {'logit_std':>10}")
    print("    distinct_Lhat  = mean # distinct L_hat values across the 5 tau per record")
    print("    Lhat_span      = mean (max L_hat - min L_hat) across the 5 tau per record")
    print("    tau_sens_frac  = fraction of records with > 1 distinct L_hat across tau")
    print("    KL(lo||hi)     = mean KL( softmax(tau=0.1) || softmax(tau=0.9) ) per record")
    print("    logit_std      = mean std of each logit coordinate across tau")
    print()
    for run in len_runs:
        info = _length_tau_sensitivity(run["ckpt"], run["family"])
        if info is None:
            print(f"  {run['tag']:<3} {os.path.basename(run['ckpt']):<18} "
                  f"CHECKPOINT MISSING")
            continue
        print(
            f"  {run['tag']:<3} {os.path.basename(run['ckpt']):<18} "
            f"{info['n_records']:>6} "
            f"{info['mean_distinct_Lhat']:>14.3f} "
            f"{info['mean_Lhat_span']:>10.3f} "
            f"{info['frac_records_tau_sensitive']:>14.3f} "
            f"{info['mean_kl_lo_hi']:>11.5f} "
            f"{info['logit_std_across_tau']:>10.4f}"
        )

    print()
    print("Reference strict = " + f"{strict_tok_s:.2f} tok/s  (source: {strict_src})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
