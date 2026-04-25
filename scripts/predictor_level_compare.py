"""Apples-to-apples predictor-level comparison.

Inputs: each model's `checkpoints/<run>/preds_test.pt`, which stores
per-record:
    q2_hat       (γ,)    — head's prediction
    q2_target    (γ,)    — stored record.min_pq_j (same across files)
    survived     (γ,)    — stored record.survived_j (same across files)

All three files share the same test-record ordering (same stage1_pp.pt
test split + gamma filter), so flattening by `survived == 1` gives
aligned 1-D arrays over the same positions for every model. This is the
apples-to-apples comparison.

NOTE on q2_target: joint checkpoints also evaluated Q̂ against
`record.min_pq_j` (not a live-recomputed target). This is the right
metric for predictor-level comparison because "can the head predict
the canonical Q2?" is the question — the live-Q2 training was a means,
not the evaluation target.

Metrics per model:
    continuous:   MAE, Brier (masked mean square),
    binary (y = 1[Q2 >= τ] for τ ∈ {0.5, 0.7, 0.9}):
        AUC    (ranking quality)
        ECE10  (equal-count-bin expected calibration error)
    distribution:
        Q̂ mean / std / min / max
        saturation fractions: |Q̂ < 0.01|, |Q̂ > 0.99|
        std(Q̂) / std(Q2)     — spread ratio (shrinkage indicator)

Output:
    outputs/predictor_level_compare/{report.md, summary.json}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


TAUS = (0.5, 0.7, 0.9)


def flatten(preds: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q_hat, q2_target) arrays over survived positions."""
    q_hat: List[float] = []
    q2:    List[float] = []
    for p in preds:
        qh = p["q2_hat"]
        q2_row = p["q2_target"]
        surv = p["survived"]
        # q2_hat for multi-head threshold preds is nested; this tool
        # only handles scalar-head preds (MLP / joint). If we see lists-
        # of-lists we skip the file with a clear message.
        if len(qh) > 0 and isinstance(qh[0], list):
            raise ValueError(
                "preds file has per-position multi-head outputs; "
                "predictor_level_compare expects scalar Q̂ per position."
            )
        for h, t, s in zip(qh, q2_row, surv):
            if int(s) != 1:
                continue
            q_hat.append(float(h))
            q2.append(float(t))
    return np.array(q_hat, dtype=np.float64), np.array(q2, dtype=np.float64)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-count-bin ECE. y ∈ {0,1}, p ∈ [0,1]. Mean |p - y| across
    n_bins quantile bins of p, weighted by bin count."""
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
        total += (hi - lo) / n * abs(p_sorted[lo:hi].mean() - y_sorted[lo:hi].mean())
    return float(total)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if y.sum() == 0 or y.sum() == len(y):
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def metrics(q_hat: np.ndarray, q2: np.ndarray) -> Dict:
    out: Dict = {
        "n_survived": int(len(q_hat)),
        "MAE":       float(np.abs(q_hat - q2).mean()),
        "Brier":     float(((q_hat - q2) ** 2).mean()),
        "Qhat_mean": float(q_hat.mean()),
        "Qhat_std":  float(q_hat.std(ddof=0)),
        "Qhat_min":  float(q_hat.min()),
        "Qhat_max":  float(q_hat.max()),
        "Q2_std":    float(q2.std(ddof=0)),
        "std_ratio": (
            float(q_hat.std(ddof=0) / q2.std(ddof=0))
            if q2.std(ddof=0) > 1e-12 else float("nan")
        ),
        "frac_Qhat_lt_001": float((q_hat < 0.01).mean()),
        "frac_Qhat_gt_099": float((q_hat > 0.99).mean()),
    }
    for tau in TAUS:
        y = (q2 >= float(tau)).astype(np.float64)
        out[f"AUC_tau_{tau}"] = _auc(y, q_hat)
        out[f"ECE10_tau_{tau}"] = _ece(y, q_hat, n_bins=10)
    return out


def _fmt_row(label: str, m: Dict, fields: List[Tuple[str, str]]) -> str:
    return f"  {label:<20} " + "  ".join(
        fmt.format(m[k]) if k in m and isinstance(m[k], float) and np.isfinite(m[k])
        else f"{'   —':>10}"
        for k, fmt in fields
    )


def render(rows: Dict[str, Dict]) -> str:
    lines: List[str] = []
    lines.append("# Predictor-level comparison (test split, survived positions)\n\n")
    lines.append(
        "All three (four) files share identical test records / gamma=8 filter / "
        "survived masks / stored min_pq_j targets.\n\n"
    )
    names = list(rows.keys())

    def col_widths():
        return [max(10, len(n) + 1) for n in names]

    # Table 1: continuous metrics + distribution
    lines.append("## Continuous + distribution metrics\n\n")
    lines.append("```\n")
    header = f"  {'metric':<22}" + "".join(f"{n:>14}" for n in names) + "\n"
    lines.append(header)
    row_keys = [
        ("n_survived",      "{:>14.0f}"),
        ("MAE",             "{:>14.4f}"),
        ("Brier",           "{:>14.4f}"),
        ("Qhat_mean",       "{:>14.4f}"),
        ("Qhat_std",        "{:>14.4f}"),
        ("Qhat_min",        "{:>14.4f}"),
        ("Qhat_max",        "{:>14.4f}"),
        ("Q2_std",          "{:>14.4f}"),
        ("std_ratio",       "{:>14.4f}"),
        ("frac_Qhat_lt_001","{:>14.4f}"),
        ("frac_Qhat_gt_099","{:>14.4f}"),
    ]
    for k, fmt in row_keys:
        row = f"  {k:<22}"
        for n in names:
            v = rows[n].get(k, float("nan"))
            if isinstance(v, float) and not np.isfinite(v):
                row += f"{'   —':>14}"
            else:
                row += fmt.format(float(v))
        lines.append(row + "\n")
    lines.append("```\n\n")

    # Table 2: per-τ AUC / ECE
    lines.append("## Per-τ binary classification metrics (y = 1[Q2 ≥ τ])\n\n")
    lines.append("```\n")
    lines.append(f"  {'metric':<22}" + "".join(f"{n:>14}" for n in names) + "\n")
    for tau in TAUS:
        for k_pref, lbl in [(f"AUC_tau_{tau}",   f"AUC  τ={tau}"),
                            (f"ECE10_tau_{tau}", f"ECE  τ={tau}")]:
            row = f"  {lbl:<22}"
            for n in names:
                v = rows[n].get(k_pref, float("nan"))
                if isinstance(v, float) and not np.isfinite(v):
                    row += f"{'   —':>14}"
                else:
                    row += f"{v:>14.4f}"
            lines.append(row + "\n")
    lines.append("```\n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--include_long", action="store_true",
                    help="Also include acc_jnt_1a_liveq2_long if it exists.")
    ap.add_argument("--out_dir", type=str,
                    default=str(_REPO_ROOT / "outputs/predictor_level_compare"))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        ("frozen 1A", root / "checkpoints/acc_1a/preds_test.pt"),
        ("lazy-jnt",  root / "checkpoints/acc_jnt_1a/preds_test.pt"),
        ("live-jnt",  root / "checkpoints/acc_jnt_1a_liveq2/preds_test.pt"),
    ]
    if args.include_long:
        p = root / "checkpoints/acc_jnt_1a_liveq2_long/preds_test.pt"
        if p.is_file():
            entries.append(("live-jnt-10ep", p))

    rows: Dict[str, Dict] = {}
    for name, path in entries:
        if not path.is_file():
            print(f"[compare] skip {name}: {path} missing")
            continue
        preds = torch.load(str(path), weights_only=False)
        q_hat, q2 = flatten(preds)
        rows[name] = metrics(q_hat, q2)
        print(
            f"[compare] {name:<15} n={rows[name]['n_survived']:>4d}  "
            f"MAE={rows[name]['MAE']:.4f}  "
            f"Brier={rows[name]['Brier']:.4f}  "
            f"Q̂mean={rows[name]['Qhat_mean']:.3f} "
            f"Q̂max={rows[name]['Qhat_max']:.3f}"
        )

    rep = render(rows)
    with open(out_dir / "report.md", "w") as f:
        f.write(rep)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\n" + rep)
    print(f"[compare] wrote {out_dir}/report.md")
    print(f"[compare] wrote {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
