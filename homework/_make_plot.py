"""Pareto plot for the homework experiment.

Reads <in_dir>/unified.json and produces two figures (PNG+PDF):

  pareto_nll.{png,pdf}        — NLL  (x) vs tok/s (y)
  pareto_delta_nll.{png,pdf}  — ΔNLL (x) vs tok/s (y)

Curves: one per non-strict family (lossy_l, threshold_lossy,
confidence_lossy), with τ / l values as points along the curve.
Strict drawn as a standalone star marker (no curve).

Title: "Throughput–Quality Trade-off on OpenWebText".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_STYLE = {
    "strict":            ("#000000", "*", "Strict (Leviathan)"),
    "lossy_l":           ("#1f77b4", "o", "Lossy-l (Leviathan + lenience)"),
    "threshold_lossy":   ("#ff7f0e", "s", "Threshold (deterministic, r_j ≥ τ)"),
    "confidence_lossy":  ("#2ca02c", "D", "Confidence (deterministic, ∏r_j ≥ τ)"),
}


def _param_value(method: str, param: str) -> float:
    """Extract numeric param value for sorting (0 for strict)."""
    if method == "strict" or param in (None, "—"):
        return 0.0
    if "=" in param:
        return float(param.split("=", 1)[1])
    try:
        return float(param)
    except Exception:
        return 0.0


def split_by_method(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r["method"], []).append(r)
    for m in out:
        out[m].sort(key=lambda r: _param_value(r["method"], r.get("param", "")))
    return out


def _plot_one(
    rows_by_method: Dict[str, List[Dict]],
    x_key: str, x_label: str,
    title: str, out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for method, rows in rows_by_method.items():
        if method not in METHOD_STYLE:
            continue
        color, marker, label = METHOD_STYLE[method]
        xs = [r[x_key] for r in rows if r.get(x_key) is not None]
        ys = [r["tok_s_mean"] for r in rows if r.get(x_key) is not None]
        if not xs:
            continue
        if method == "strict":
            ax.scatter(xs, ys, s=180, c=color, marker=marker,
                       label=label, zorder=5, edgecolors="black",
                       linewidths=0.8)
        else:
            # Sort by x-axis so the connecting line does not zig-zag.
            param_vals = [_param_value(method, r.get("param", "")) for r in rows
                          if r.get(x_key) is not None]
            triples = sorted(zip(xs, ys, param_vals), key=lambda t: t[0])
            xs_s = [t[0] for t in triples]
            ys_s = [t[1] for t in triples]
            ax.plot(xs_s, ys_s, color=color, marker=marker, markersize=8,
                    linewidth=1.5, label=label, alpha=0.9)
            for x, y, pv in triples:
                ax.annotate(f"{pv:g}", (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=7, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel("tok/s (higher = faster)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[plot] wrote {out_base}.{{png,pdf}}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    _hw = Path(__file__).resolve().parent
    ap.add_argument("--in_dir", type=str, default=str(_hw / "results"))
    ap.add_argument("--out_dir", type=str, default=str(_hw / "figures"))
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((in_dir / "unified.json").read_text())
    rows_by_method = split_by_method(rows)

    title = "Throughput–Quality Trade-off on OpenWebText"
    _plot_one(
        rows_by_method, x_key="nll", x_label="NLL (lower = better)",
        title=title, out_base=out_dir / "pareto_nll",
    )
    _plot_one(
        rows_by_method, x_key="delta_nll",
        x_label="ΔNLL vs strict (lower = better)",
        title=title, out_base=out_dir / "pareto_delta_nll",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
