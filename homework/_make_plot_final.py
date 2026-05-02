"""Final trade-off plot anchored at lossy l=1.0 (= Leviathan strict).

Per Leviathan/Kalman/Matias 2022 Algorithm 1 and Sahoo et al. 2025 §3,
strict speculative decoding accepts each drafted token j with probability
`min(1, p_j / q_j)` (Bernoulli on the ratio). The lossy variant replaces
this with `min(1, p_j / (l * q_j))`; at l=1.0 the two rules coincide
exactly, so `lossy l=1.0` is the methodologically correct strict
reference for any apples-to-apples comparison against the threshold and
confidence rules (which also use the ratio min(1, p/q)).

The temp=0 `strict` lane in our data uses an argmax-match shortcut
(non-standard; introduced in this codebase only). It is shown as a faint
grey marker so the difference from the canonical Leviathan rule is
visible without confusing the trade-off curves.

Each non-strict method is drawn with 3 curated points sweeping from
"more permissive" (low param) to "more strict" (high param) so the
trade-off direction is unambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (method, param) → role. Each non-strict method gets 3 trend points.
SELECTION: List[Tuple[str, str, str]] = [
    # --- Reference: lossy l=1.0 == Leviathan strict ---
    ("lossy_l",          "l=1.0",   "ref"),
    # Lossy-l trade-off curve (3 points sweeping conservative → permissive).
    ("lossy_l",          "l=2.0",   "trend"),    # more conservative than strict
    ("lossy_l",          "l=0.7",   "trend"),
    ("lossy_l",          "l=0.3",   "trend"),    # most permissive
    # Threshold trade-off curve (3 points).
    ("threshold_lossy",  "tau=0.95", "trend"),   # conservative end
    ("threshold_lossy",  "tau=0.6",  "trend"),
    ("threshold_lossy",  "tau=0.3",  "trend"),
    # Confidence trade-off curve (3 points).
    ("confidence_lossy", "tau=0.95", "trend"),
    ("confidence_lossy", "tau=0.6",  "trend"),
    ("confidence_lossy", "tau=0.3",  "trend"),
    # Footnote: temp=0 strict (argmax-match, NOT Leviathan rule).
    ("strict",           "—",       "footnote"),
]

METHOD_STYLE = {
    "lossy_l":           ("#1f77b4", "o", "Lossy-l (l=1.0 = Leviathan strict)"),
    "threshold_lossy":   ("#ff7f0e", "s", "Threshold (deterministic, r_j ≥ τ)"),
    "confidence_lossy":  ("#2ca02c", "D", "Confidence (deterministic, ∏r_j ≥ τ)"),
}


def _param_value(param: str) -> float:
    return float(param.split("=", 1)[1]) if "=" in param else 0.0


def _param_label(param: str) -> str:
    return "" if param == "—" else param.split("=", 1)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    _hw = Path(__file__).resolve().parent
    ap.add_argument("--in_dir", type=str, default=str(_hw / "results"))
    ap.add_argument("--out_dir", type=str, default=str(_hw / "figures"))
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    rows = json.loads((in_dir / "unified.json").read_text())
    by_key: Dict[Tuple[str, str], Dict] = {
        (r["method"], r["param"]): r for r in rows
    }
    missing = [(m, p) for m, p, _ in SELECTION if (m, p) not in by_key]
    if missing:
        raise SystemExit(f"missing selection rows: {missing}")

    # Per-method trend point lists (sort by param value descending so the
    # curve is drawn from most-conservative to most-permissive).
    lossy_pts = [by_key[(m, p)] for m, p, _ in SELECTION if m == "lossy_l"]
    lossy_pts.sort(key=lambda r: -_param_value(r["param"]))
    thresh_pts = [
        by_key[(m, p)] for m, p, role in SELECTION
        if m == "threshold_lossy" and role == "trend"
    ]
    thresh_pts.sort(key=lambda r: -_param_value(r["param"]))
    conf_pts = [
        by_key[(m, p)] for m, p, role in SELECTION
        if m == "confidence_lossy" and role == "trend"
    ]
    conf_pts.sort(key=lambda r: -_param_value(r["param"]))
    strict_pt = by_key[("strict", "—")]
    ref = by_key[("lossy_l", "l=1.0")]

    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    # Lossy-l trade-off curve.
    color, marker, label = METHOD_STYLE["lossy_l"]
    xs = [r["nll"] for r in lossy_pts]
    ys = [r["tok_s_mean"] for r in lossy_pts]
    ax.plot(xs, ys, color=color, marker=marker, markersize=10,
            linewidth=1.8, label=label, alpha=0.92)
    for r, x, y in zip(lossy_pts, xs, ys):
        ax.annotate(_param_label(r["param"]), (x, y),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=9, color=color, weight="bold")
    # Star overlay on l=1.0 to mark it as the strict reference.
    ax.scatter([ref["nll"]], [ref["tok_s_mean"]],
               s=350, marker="*", facecolors="none",
               edgecolors=color, linewidths=2.4, zorder=6)
    ax.annotate(
        "Strict reference\n(lossy l=1.0,\nLeviathan Alg. 1)",
        (ref["nll"], ref["tok_s_mean"]),
        xytext=(-110, -55), textcoords="offset points", fontsize=9,
        color=color, weight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
    )

    # Threshold trade-off curve.
    color, marker, label = METHOD_STYLE["threshold_lossy"]
    xs = [r["nll"] for r in thresh_pts]
    ys = [r["tok_s_mean"] for r in thresh_pts]
    ax.plot(xs, ys, color=color, marker=marker, markersize=10,
            linewidth=1.8, label=label, alpha=0.92)
    for r, x, y in zip(thresh_pts, xs, ys):
        ax.annotate(_param_label(r["param"]), (x, y),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=9, color=color, weight="bold")

    # Confidence trade-off curve.
    color, marker, label = METHOD_STYLE["confidence_lossy"]
    xs = [r["nll"] for r in conf_pts]
    ys = [r["tok_s_mean"] for r in conf_pts]
    ax.plot(xs, ys, color=color, marker=marker, markersize=10,
            linewidth=1.8, label=label, alpha=0.92)
    for r, x, y in zip(conf_pts, xs, ys):
        ax.annotate(_param_label(r["param"]), (x, y),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=9, color=color, weight="bold")

    # Direction arrow (positioned away from data points).
    ax.annotate(
        "more permissive →  faster, worse NLL",
        xy=(0.55, 11.0), xytext=(0.27, 11.0),
        fontsize=9.5, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )
    ax.set_xlim(0.15, 0.60)

    ax.set_xlabel("NLL  (lower = better quality)")
    ax.set_ylabel("tok/s  (higher = faster)")
    ax.set_title(
        "Throughput–Quality Trade-off on OpenWebText\n"
        "anchored at lossy l=1.0 (= Leviathan strict speculative decoding)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_final.png", dpi=150)
    fig.savefig(out_dir / "pareto_final.pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_dir}/pareto_final.{{png,pdf}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
