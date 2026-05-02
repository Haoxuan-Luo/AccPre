"""Curated trade-off plot from the existing 24-lane homework results.

Selects a small subset of points per method that traces the
"more-permissive → faster + worse NLL" trend, and annotates each point
with its sweep parameter so the reader can read the trade-off direction
off the figure.

Inputs:  homework/unified.json
Outputs: homework/pareto_curated.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (method, param) → label (None means skip).
# Each method gets 3 points sweeping from "most permissive" (low quality,
# fast) to "most strict" (high quality, slower). Strict is its own marker.
SELECTION: List[Tuple[str, str]] = [
    ("strict",            "—"),
    # Lossy-l: l=0.3 most permissive ... l=1.0 = stock SD baseline.
    ("lossy_l",           "l=0.3"),
    ("lossy_l",           "l=0.7"),
    ("lossy_l",           "l=1.0"),
    # Threshold: τ=0.1 most permissive ... τ=0.9 most strict.
    ("threshold_lossy",   "tau=0.1"),
    ("threshold_lossy",   "tau=0.5"),
    ("threshold_lossy",   "tau=0.9"),
    # Confidence: τ=0.2 ... τ=0.9.
    ("confidence_lossy",  "tau=0.2"),
    ("confidence_lossy",  "tau=0.6"),
    ("confidence_lossy",  "tau=0.9"),
]

METHOD_STYLE = {
    "strict":            ("#000000", "*", "Strict (Leviathan, temp=0)"),
    "lossy_l":           ("#1f77b4", "o", "Lossy-l (Bernoulli, lenience l)"),
    "threshold_lossy":   ("#ff7f0e", "s", "Threshold (r_j ≥ τ)"),
    "confidence_lossy":  ("#2ca02c", "D", "Confidence (∏r_j ≥ τ)"),
}


def _param_value(param: str) -> float:
    if "=" in param:
        return float(param.split("=", 1)[1])
    return 0.0


def _param_label(param: str) -> str:
    if param == "—":
        return ""
    return param.split("=", 1)[1]


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
    missing = [k for k in SELECTION if k not in by_key]
    if missing:
        raise SystemExit(f"missing selection rows: {missing}")

    # Group selected rows by method for plotting.
    grouped: Dict[str, List[Dict]] = {}
    for k in SELECTION:
        m = k[0]
        grouped.setdefault(m, []).append(by_key[k])
    # Sort each non-strict group by param value so the curve traces
    # most-permissive (small param for lossy_l = small l; small param for
    # threshold/conf = small τ) → most-strict.
    for m, rs in grouped.items():
        if m == "strict":
            continue
        rs.sort(key=lambda r: _param_value(r["param"]))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method, rs in grouped.items():
        color, marker, label = METHOD_STYLE[method]
        xs = [r["nll"] for r in rs]
        ys = [r["tok_s_mean"] for r in rs]
        if method == "strict":
            ax.scatter(xs, ys, s=240, c=color, marker=marker,
                       label=label, zorder=6, edgecolors="black",
                       linewidths=1.0)
            ax.annotate(
                "no other method dominates this point\n"
                "(see explanatory text)",
                (xs[0], ys[0]), xytext=(60, -12),
                textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                color="black",
            )
        else:
            ax.plot(xs, ys, color=color, marker=marker, markersize=10,
                    linewidth=1.8, label=label, alpha=0.92)
            for r, x, y in zip(rs, xs, ys):
                pv = _param_label(r["param"])
                ax.annotate(pv, (x, y), textcoords="offset points",
                            xytext=(6, 6), fontsize=8.5, color=color,
                            weight="bold")

    # Direction arrow inside the lossy-l region — clearest trade-off.
    ax.annotate(
        "more permissive  →  more tok/s, worse NLL",
        xy=(0.55, 19.6), xytext=(0.40, 19.6),
        fontsize=9, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )

    ax.set_xlabel("NLL  (lower = better quality)")
    ax.set_ylabel("tok/s  (higher = faster)")
    ax.set_title(
        "Throughput–Quality Trade-off on OpenWebText\n"
        "(20 prompts; γ=15; temp=0; curated subset)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_curated.png", dpi=150)
    fig.savefig(out_dir / "pareto_curated.pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_dir}/pareto_curated.{{png,pdf}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
