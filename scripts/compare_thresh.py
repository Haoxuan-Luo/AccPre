"""Task B — acc_thresh vs acc_1a vs oracle-Q2, read-only comparison.

Assumes:
  checkpoints/acc_1a/preds_test.pt        (row.q2_hat is (γ,))
  checkpoints/acc_thresh/preds_test.pt    (row.q2_hat is (γ, K))
  outputs/oracle_q2_ceiling/summary.json  (Task A output)

For each τ in {0.5, 0.7, 0.9}:

  Predictor-level (per-head vs per-τ Q̂-slice):
    - BCE, AUC, Brier, ECE  (survived-masked)
    - positive rate of label
    - positive rate of prediction at the natural cutoff (0.5 for thresh,
      τ for acc_1a's implied classifier)

  Decision-level (offline CF@1 on full test split, n=1433):
    - acc_thresh: L̂ = commit_threshold(head_τ_scores, cutoff=0.5)
    - acc_1a   : L̂ = commit_threshold(Q̂, τ)
    - oracle   : from Task A
    - per-τ CF@1 / under / over / L̂ distribution

System-level (tok/s) requires a GPU online lane and is out of scope for
this script — it would need `online_decode.py` on a GPU node.

Outputs:
  outputs/compare_thresh/report.md
  outputs/compare_thresh/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.commit import commit_threshold
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N


TAUS = (0.5, 0.7, 0.9)


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def load_test_records(root: Path, protocol: ProtocolConfig) -> List:
    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"), expected_protocol=protocol,
    )
    test_prompts = set(range(TRAIN_N + VAL_N, POOL_SIZE))
    return [r for r in all_recs if r.prompt_idx in test_prompts and r.gamma == 8]


def bce(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def auc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if (y.sum() == 0) or (y.sum() == len(y)):
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-count bins."""
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    n = len(p_sorted)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    total = 0.0
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        bin_p = p_sorted[lo:hi].mean()
        bin_y = y_sorted[lo:hi].mean()
        total += (hi - lo) / n * abs(bin_p - bin_y)
    return float(total)


def flatten_preds(
    preds: List[Dict], test_records: List, score_key: str = "q2_hat",
) -> Dict[str, np.ndarray]:
    """Align preds and records, flatten survived positions.

    `score_key` is the predictor's score field. For acc_thresh that's a
    (γ, K) nested list; we return `scores_all_heads: (N, K)` alongside
    the 1-D `q2_target`, `survived`, etc.
    """
    q2 = []; surv = []; acc = []; scores = []
    for p in preds:
        rec = test_records[int(p["record_idx"])]
        gamma = len(rec.q_j)
        raw_scores = p[score_key]
        # Both (γ,) and (γ, K) list formats are tolerated here.
        for j in range(gamma):
            if int(rec.survived_j[j]) != 1:
                continue
            q2.append(float(rec.min_pq_j[j]))
            surv.append(1)
            acc.append(int(rec.accepted_j[j]))
            scores.append(raw_scores[j])
    q2 = np.asarray(q2, dtype=np.float64)
    surv = np.asarray(surv, dtype=np.int64)
    acc = np.asarray(acc, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    return {"q2": q2, "survived": surv, "accepted": acc, "scores": scores_arr}


def per_tau_predictor_metrics(
    flat_1a: Dict[str, np.ndarray],
    flat_thresh: Dict[str, np.ndarray],
) -> Dict[float, Dict]:
    """For each τ, metrics for the threshold head AND the implied classifier
    (1A's Q̂ treated as a score for 1[Q2 >= τ]).
    """
    out: Dict[float, Dict] = {}
    for k, tau in enumerate(TAUS):
        y = (flat_thresh["q2"] >= float(tau)).astype(np.float64)
        # acc_thresh head (K-axis is dim 1)
        s_th = flat_thresh["scores"][:, k].astype(np.float64)
        # acc_1a's Q̂ as the implied classifier score for τ
        s_1a = flat_1a["scores"].astype(np.float64)
        # Sanity: 1A flat must be same length as thresh flat.
        if s_1a.shape[0] != s_th.shape[0]:
            raise RuntimeError(
                f"1A vs thresh row count mismatch: "
                f"{s_1a.shape[0]} vs {s_th.shape[0]}"
            )
        out[float(tau)] = {
            "positive_rate_label":        float(y.mean()),
            "thresh_positive_rate_pred":  float((s_th >= 0.5).mean()),
            "acc1a_positive_rate_pred":   float((s_1a >= float(tau)).mean()),
            "thresh_bce":                 bce(y, s_th),
            "acc1a_bce_as_classifier":    bce(y, s_1a),
            "thresh_auc":                 auc(y, s_th),
            "acc1a_auc":                  auc(y, s_1a),
            "thresh_brier":               brier(y, s_th),
            "acc1a_brier":                brier(y, s_1a),
            "thresh_ece":                 ece(y, s_th),
            "acc1a_ece":                  ece(y, s_1a),
        }
    return out


def per_tau_decision_metrics(
    preds_1a: List[Dict],
    preds_thresh: List[Dict],
    test_records: List,
) -> Dict[float, Dict]:
    """Compute offline CF@1 / under / over / L̂ distribution for both
    predictors at each τ, on the full test split.

    acc_thresh: L̂ = commit_threshold(scores[:, k], 0.5)
    acc_1a   : L̂ = commit_threshold(Q̂, τ)
    """
    out: Dict[float, Dict] = {}
    gamma_max = 8
    for k, tau in enumerate(TAUS):
        rows = {
            "acc_thresh": {"exact": 0, "under": 0, "over": 0,
                           "dist": Counter(), "n": 0,
                           "mean_abs_diff": 0.0, "abs_diff_sum": 0.0,
                           "per_prompt": {}},
            "acc_1a":     {"exact": 0, "under": 0, "over": 0,
                           "dist": Counter(), "n": 0,
                           "mean_abs_diff": 0.0, "abs_diff_sum": 0.0,
                           "per_prompt": {}},
        }
        for p_th, p_1a in zip(preds_thresh, preds_1a):
            if int(p_th["record_idx"]) != int(p_1a["record_idx"]):
                raise RuntimeError(
                    "thresh and 1A preds are out of order; "
                    f"{p_th['record_idx']} vs {p_1a['record_idx']}"
                )
            rec = test_records[int(p_th["record_idx"])]
            L_strict = int(rec.L)
            # thresh L̂
            thresh_scores = [float(p_th["q2_hat"][j][k])
                             for j in range(rec.gamma)]
            L_th = commit_threshold(thresh_scores, 0.5)
            L_th = max(0, min(L_th, rec.gamma))
            # 1A L̂
            q_scores = [float(p_1a["q2_hat"][j]) for j in range(rec.gamma)]
            L_1a = commit_threshold(q_scores, float(tau))
            L_1a = max(0, min(L_1a, rec.gamma))
            for name, Lhat in (("acc_thresh", L_th), ("acc_1a", L_1a)):
                d = rows[name]
                d["n"] += 1
                d["dist"][Lhat] += 1
                d["abs_diff_sum"] += abs(Lhat - L_strict)
                if Lhat == L_strict:
                    d["exact"] += 1
                elif Lhat < L_strict:
                    d["under"] += 1
                else:
                    d["over"] += 1
                d["per_prompt"].setdefault(int(rec.prompt_idx), []).append(
                    1 if Lhat == L_strict else 0,
                )
        tau_out = {}
        for name, d in rows.items():
            n = max(d["n"], 1)
            tau_out[name] = {
                "cf_at_1":       d["exact"] / n,
                "under":         d["under"] / n,
                "over":          d["over"] / n,
                "mean_abs_diff": d["abs_diff_sum"] / n,
                "L_hat_distribution": {
                    str(kk): d["dist"][kk] / n
                    for kk in sorted(d["dist"])
                },
                "per_prompt_cf_mean": {
                    p: sum(v) / len(v) for p, v in d["per_prompt"].items()
                },
                "n": int(d["n"]),
            }
        out[float(tau)] = tau_out
    return out


def render_report(
    predictor_metrics: Dict[float, Dict],
    decision_metrics: Dict[float, Dict],
    oracle_full: Dict,
) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Task B — acc_thresh vs acc_1a vs oracle-Q2 (test split, n=1433)\n\n")

    # --- Predictor-level ---
    add("## Predictor-level (survived positions only, flattened)\n\n")
    add("                        τ=0.5        τ=0.7        τ=0.9\n")
    def col(metric: str, which: str, fmt: str = "{:>10.4f}") -> str:
        line = f"{metric:<24}"
        for t in TAUS:
            v = predictor_metrics[float(t)][which]
            if isinstance(v, float) and not np.isfinite(v):
                line += f"{'   —   ':>10}   "
            else:
                line += f"  {fmt.format(v)} "
        return line + "\n"
    add(col("positive rate (label) ",    "positive_rate_label"))
    add(col("pred rate (thresh cut 0.5)", "thresh_positive_rate_pred"))
    add(col("pred rate (1A  cut τ)    ", "acc1a_positive_rate_pred"))
    add("\n")
    add(col("BCE — acc_thresh head   ", "thresh_bce"))
    add(col("BCE — acc_1a Q̂ at τ     ", "acc1a_bce_as_classifier"))
    add("\n")
    add(col("AUC — acc_thresh head   ", "thresh_auc"))
    add(col("AUC — acc_1a Q̂ at τ     ", "acc1a_auc"))
    add("\n")
    add(col("Brier — acc_thresh      ", "thresh_brier"))
    add(col("Brier — acc_1a Q̂        ", "acc1a_brier"))
    add("\n")
    add(col("ECE  — acc_thresh       ", "thresh_ece"))
    add(col("ECE  — acc_1a Q̂        ", "acc1a_ece"))

    # --- Decision-level ---
    add("\n## Decision-level offline CF@1 (full test split)\n\n")
    add("                           τ=0.5        τ=0.7        τ=0.9\n")
    for tau in TAUS:
        t = float(tau)
        th = decision_metrics[t]["acc_thresh"]
        a1 = decision_metrics[t]["acc_1a"]
        orc = oracle_full[str(tau)] if str(tau) in oracle_full else oracle_full[tau]
    for label, get_fn in [
        ("CF@1 — acc_thresh        ",
         lambda t: decision_metrics[float(t)]["acc_thresh"]["cf_at_1"]),
        ("CF@1 — acc_1a            ",
         lambda t: decision_metrics[float(t)]["acc_1a"]["cf_at_1"]),
        ("CF@1 — oracle (Task A)   ",
         lambda t: oracle_full[str(t)]["cf_at_1"]),
    ]:
        row = f"{label:<26}"
        for t in TAUS:
            row += f"  {get_fn(t):>10.4f} "
        add(row + "\n")
    add("\n")
    for label, get_fn in [
        ("under rate — acc_thresh  ",
         lambda t: decision_metrics[float(t)]["acc_thresh"]["under"]),
        ("under rate — acc_1a      ",
         lambda t: decision_metrics[float(t)]["acc_1a"]["under"]),
        ("over  rate — acc_thresh  ",
         lambda t: decision_metrics[float(t)]["acc_thresh"]["over"]),
        ("over  rate — acc_1a      ",
         lambda t: decision_metrics[float(t)]["acc_1a"]["over"]),
        ("mean|L̂−L| — acc_thresh   ",
         lambda t: decision_metrics[float(t)]["acc_thresh"]["mean_abs_diff"]),
        ("mean|L̂−L| — acc_1a       ",
         lambda t: decision_metrics[float(t)]["acc_1a"]["mean_abs_diff"]),
    ]:
        row = f"{label:<26}"
        for t in TAUS:
            row += f"  {get_fn(t):>10.4f} "
        add(row + "\n")
    add("\n## Induced L̂ distributions\n\n")
    for tau in TAUS:
        t = float(tau)
        add(f"τ = {tau}:\n")
        add(f"  {'L̂':>3} {'acc_thresh':>12} {'acc_1a':>12}\n")
        d_th = decision_metrics[t]["acc_thresh"]["L_hat_distribution"]
        d_1a = decision_metrics[t]["acc_1a"]["L_hat_distribution"]
        for kk in range(0, 9):
            s = str(kk)
            v_th = d_th.get(s, 0.0)
            v_1a = d_1a.get(s, 0.0)
            add(f"  {kk:>3d} {v_th:>12.4f} {v_1a:>12.4f}\n")
        add("\n")

    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--out_dir", type=str,
                    default=str(_REPO_ROOT / "outputs/compare_thresh"))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(root / "configs/protocol.yaml")
    test_records = load_test_records(root, protocol)
    preds_1a = torch.load(
        str(root / "checkpoints/acc_1a/preds_test.pt"), weights_only=False,
    )
    preds_thresh = torch.load(
        str(root / "checkpoints/acc_thresh/preds_test.pt"), weights_only=False,
    )
    if len(preds_1a) != len(preds_thresh) or len(preds_1a) != len(test_records):
        raise RuntimeError(
            f"Row counts disagree: "
            f"1A={len(preds_1a)} thresh={len(preds_thresh)} "
            f"records={len(test_records)}"
        )
    # Sort by record_idx for safety, although trainer already does this.
    preds_1a = sorted(preds_1a, key=lambda r: int(r["record_idx"]))
    preds_thresh = sorted(preds_thresh, key=lambda r: int(r["record_idx"]))

    flat_1a = flatten_preds(preds_1a, test_records)
    flat_thresh = flatten_preds(preds_thresh, test_records)

    # Sanity: q2 arrays should match across preds files (same records).
    if not np.allclose(flat_1a["q2"], flat_thresh["q2"]):
        raise RuntimeError("q2 arrays mismatch across preds files")

    # Predictor + decision metrics.
    predictor_metrics = per_tau_predictor_metrics(flat_1a, flat_thresh)
    decision_metrics = per_tau_decision_metrics(preds_1a, preds_thresh, test_records)

    # Oracle from Task A.
    with open(root / "outputs/oracle_q2_ceiling/summary.json", "r") as f:
        oracle = json.load(f)
    oracle_full = oracle["full_test_split"]

    # Write artifacts.
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "predictor_metrics": {str(k): v for k, v in predictor_metrics.items()},
            "decision_metrics": {str(k): v for k, v in decision_metrics.items()},
            "oracle_full_test_split": oracle_full,
            "n_test_records": len(test_records),
            "n_survived_positions": int(len(flat_thresh["q2"])),
        }, f, indent=2, default=str)

    rep = render_report(predictor_metrics, decision_metrics, oracle_full)
    with open(out_dir / "report.md", "w") as f:
        f.write(rep)
    print(rep)
    print(f"[compare_thresh] wrote {out_dir}/report.md")
    print(f"[compare_thresh] wrote {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
