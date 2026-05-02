"""Trade-off plot anchored on `lossy l=1.0` as the strict reference.

In the Leviathan formulation, lossy speculative decoding with l=1.0 IS
strict speculative decoding — same Bernoulli accept rule `U < min(1, p/q)`,
just relabeled. So `lossy_l(l=1.0)` is the methodologically correct
"strict baseline" for any apples-to-apples comparison against the
ratio-based threshold and confidence rules. (The temp=0 `strict` lane
in our data uses a different rule — argmax-match — and is shown as a
faint reference marker only.)

Each non-reference method is sketched with 3 curated points to show the
trade-off direction (more permissive → faster, worse NLL).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (method, param, role) — role is "ref", "trend", or "footnote".
SELECTION: List[Tuple[str, str, str]] = [
    # Reference: lossy l=1.0 (= strict speculative decoding by definition).
    ("lossy_l",          "l=1.0",   "ref"),
    # Lossy-l trade-off curve (3 values total, including reference).
    ("lossy_l",          "l=0.7",   "trend"),
    ("lossy_l",          "l=0.3",   "trend"),
    # Threshold trade-off curve (3 points).
    ("threshold_lossy",  "tau=0.9", "trend"),
    ("threshold_lossy",  "tau=0.5", "trend"),
    ("threshold_lossy",  "tau=0.3", "trend"),
    # Confidence trade-off curve (3 points).
    ("confidence_lossy", "tau=0.9", "trend"),
    ("confidence_lossy", "tau=0.6", "trend"),
    ("confidence_lossy", "tau=0.3", "trend"),
    # Faint footnote: temp=0 argmax-match strict (different accept rule).
    ("strict",           "—",       "footnote"),
]

METHOD_STYLE = {
    "lossy_l":           ("#1f77b4", "o", "Lossy-l (l=1.0 = strict)"),
    "threshold_lossy":   ("#ff7f0e", "s", "Threshold (r_j ≥ τ)"),
    "confidence_lossy":  ("#2ca02c", "D", "Confidence (∏r_j ≥ τ)"),
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

    # Lossy-l curve: ref + 2 trend points, sorted by l value (1.0 → 0.7 → 0.3).
    lossy_pts = [by_key[(m, p)] for m, p, _ in SELECTION if m == "lossy_l"]
    lossy_pts.sort(key=lambda r: -_param_value(r["param"]))   # 1.0 first
    thresh_pts = [by_key[(m, p)] for m, p, r in SELECTION
                  if m == "threshold_lossy" and r == "trend"]
    thresh_pts.sort(key=lambda r: -_param_value(r["param"]))  # 0.9 first
    conf_pts = [by_key[(m, p)] for m, p, r in SELECTION
                if m == "confidence_lossy" and r == "trend"]
    conf_pts.sort(key=lambda r: -_param_value(r["param"]))    # 0.9 first
    strict_pt = by_key[("strict", "—")]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    # Lossy-l curve (with l=1.0 as the reference star).
    color, marker, label = METHOD_STYLE["lossy_l"]
    xs = [r["nll"] for r in lossy_pts]
    ys = [r["tok_s_mean"] for r in lossy_pts]
    ax.plot(xs, ys, color=color, marker=marker, markersize=10,
            linewidth=1.8, label=label, alpha=0.92)
    for r, x, y in zip(lossy_pts, xs, ys):
        pv = _param_label(r["param"])
        ax.annotate(pv, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color=color, weight="bold")
    # Highlight l=1.0 as the strict reference with a star overlay.
    ref = lossy_pts[0]
    ax.scatter([ref["nll"]], [ref["tok_s_mean"]],
               s=320, marker="*", facecolors="none",
               edgecolors=color, linewidths=2.2, zorder=6)
    ax.annotate(
        "Strict reference\n(lossy l=1.0)",
        (ref["nll"], ref["tok_s_mean"]),
        xytext=(-110, 18), textcoords="offset points", fontsize=9,
        color=color, weight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
    )

    # Threshold curve.
    color, marker, label = METHOD_STYLE["threshold_lossy"]
    xs = [r["nll"] for r in thresh_pts]
    ys = [r["tok_s_mean"] for r in thresh_pts]
    ax.plot(xs, ys, color=color, marker=marker, markersize=10,
            linewidth=1.8, label=label, alpha=0.92)
    for r, x, y in zip(thresh_pts, xs, ys):
        pv = _param_label(r["param"])
        ax.annotate(pv, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color=color, weight="bold")

    # Confidence curve.
    color, marker, label = METHOD_STYLE["confidence_lossy"]
    xs = [r["nll"] for r in conf_pts]
    ys = [r["tok_s_mean"] for r in conf_pts]
    ax.plot(xs, ys, color=color, marker=marker, markersize=10,
            linewidth=1.8, label=label, alpha=0.92)
    for r, x, y in zip(conf_pts, xs, ys):
        pv = _param_label(r["param"])
        ax.annotate(pv, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color=color, weight="bold")

    # Footnote marker: temp=0 strict (argmax-match) — different accept rule.
    ax.scatter([strict_pt["nll"]], [strict_pt["tok_s_mean"]],
               s=140, marker="*", c="#999999",
               edgecolors="#444444", linewidths=0.8,
               label="strict (temp=0, argmax-match — different rule)",
               zorder=4, alpha=0.7)

    # Direction arrow.
    ax.annotate(
        "more permissive →  more tok/s, worse NLL",
        xy=(0.55, 19.4), xytext=(0.27, 19.4),
        fontsize=9, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )

    ax.set_xlabel("NLL  (lower = better quality)")
    ax.set_ylabel("tok/s  (higher = faster)")
    ax.set_title(
        "Throughput–Quality Trade-off on OpenWebText\n"
        "anchored at lossy l=1.0 (= strict speculative decoding)"
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(19.0, 28.0)
    ax.set_xlim(0.05, 0.65)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_anchored.png", dpi=150)
    fig.savefig(out_dir / "pareto_anchored.pdf")
    plt.close(fig)
    print(f"[plot] wrote {out_dir}/pareto_anchored.{{png,pdf}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
