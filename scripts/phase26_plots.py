"""Phase 26 — Pareto figures and condensed comparison.

Reads `unified_table.json` produced by `scripts/phase26_conf_sweep.py` and
emits:

  pareto_delta_nll_tok_s.png        all rows (ΔNLL vs tok/s)
  pareto_delta_nll_tok_s_clean.png  non-dominated subset
  pareto_tok_succ_tok_s.png         all rows (tok_succ vs tok/s)
  pareto_tok_succ_tok_s_clean.png   non-dominated subset
  condensed_table.md / .json        best operating point per group

Non-dominated = Pareto front in the direction (higher tok/s, lower ΔNLL)
for Figure 1 and (higher tok/s, higher tok_succ) for Figure 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR: Dict[str, str] = {
    "strict":        "#000000",
    "lossy":         "#f58518",
    "frozen_1A":     "#3b76b3",
    "live_jnt_10ep": "#d22b2b",
    "frozen_dep":    "#9467bd",
    "live_jnt_dep":  "#e377c2",
}

MARKER: Dict[str, str] = {
    "strict":     "s",
    "lossy-l":    "D",
    "threshold":  "o",
    "confidence": "^",
}


def _pareto_front(
    pts: List[Tuple[float, float, int]],
    minimize_x: bool, maximize_y: bool,
) -> List[int]:
    """Return indices of non-dominated points.

    pts: list of (x, y, idx). A point is dominated if another point is
    no worse in both directions and strictly better in at least one.
    """
    keep: List[int] = []
    for i, (xi, yi, idx_i) in enumerate(pts):
        dominated = False
        for j, (xj, yj, _idx_j) in enumerate(pts):
            if i == j:
                continue
            x_ok = (xj <= xi) if minimize_x else (xj >= xi)
            y_ok = (yj >= yi) if maximize_y else (yj <= yi)
            x_strict = (xj < xi) if minimize_x else (xj > xi)
            y_strict = (yj > yi) if maximize_y else (yj < yi)
            if x_ok and y_ok and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            keep.append(idx_i)
    return keep


def _scatter(rows: List[Dict], out_path: Path,
             xkey: str, ykey: str, xlabel: str, ylabel: str,
             title: str, clean_indices: List[int] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    legend_seen = set()
    for i, r in enumerate(rows):
        x = r.get(xkey)
        y = r.get(ykey)
        if x is None or y is None:
            continue
        if clean_indices is not None and i not in clean_indices:
            continue
        fam = r["family"]
        rule = r["rule"]
        color = COLOR.get(fam, "#888888")
        marker = MARKER.get(rule, "x")
        key = f"{fam}|{rule}"
        label = None
        if key not in legend_seen:
            label = f"{fam}/{rule}"
            legend_seen.add(key)
        ax.scatter(
            x, y, s=80, c=color, marker=marker,
            edgecolors="black", linewidths=0.5, label=label,
        )
        txt = r.get("threshold_label") or ""
        ax.annotate(
            txt, (x, y), fontsize=7,
            textcoords="offset points", xytext=(5, 5),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    tbl_path = Path(args.table_json).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else tbl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(tbl_path) as f:
        data = json.load(f)
    rows: List[Dict] = data["rows"]

    # ---- Figure 1: ΔNLL vs tok/s ----
    pts1 = [
        (r["delta_NLL"], r["tok_s_mean"], i)
        for i, r in enumerate(rows)
        if r.get("delta_NLL") is not None
        and r.get("tok_s_mean") is not None
    ]
    front1 = _pareto_front(pts1, minimize_x=True, maximize_y=True)
    _scatter(
        rows, out_dir / "pareto_delta_nll_tok_s.png",
        xkey="delta_NLL", ykey="tok_s_mean",
        xlabel="ΔNLL (vs strict)", ylabel="tok/s",
        title="Phase 26 — ΔNLL vs tok/s (all rows)",
    )
    _scatter(
        rows, out_dir / "pareto_delta_nll_tok_s_clean.png",
        xkey="delta_NLL", ykey="tok_s_mean",
        xlabel="ΔNLL (vs strict)", ylabel="tok/s",
        title="Phase 26 — ΔNLL vs tok/s (Pareto front)",
        clean_indices=front1,
    )

    # ---- Figure 2: tok_succ vs tok/s ----
    pts2 = [
        (r["tok_succ"], r["tok_s_mean"], i)
        for i, r in enumerate(rows)
        if r.get("tok_succ") is not None
        and r.get("tok_s_mean") is not None
    ]
    front2 = _pareto_front(pts2, minimize_x=False, maximize_y=True)
    _scatter(
        rows, out_dir / "pareto_tok_succ_tok_s.png",
        xkey="tok_succ", ykey="tok_s_mean",
        xlabel="tok_succ", ylabel="tok/s",
        title="Phase 26 — tok_succ vs tok/s (all rows)",
    )
    _scatter(
        rows, out_dir / "pareto_tok_succ_tok_s_clean.png",
        xkey="tok_succ", ykey="tok_s_mean",
        xlabel="tok_succ", ylabel="tok/s",
        title="Phase 26 — tok_succ vs tok/s (Pareto front)",
        clean_indices=front2,
    )

    # ---- Condensed table: best op point per group ----
    def _best_op(subset: List[Dict]) -> Dict:
        """Pick min ΔNLL with tie break on max tok/s."""
        subset = [
            r for r in subset
            if r.get("delta_NLL") is not None
        ]
        if not subset:
            return None
        subset = sorted(
            subset,
            key=lambda r: (float(r["delta_NLL"]), -float(r["tok_s_mean"])),
        )
        return subset[0]

    groups = {
        "lossy":        [r for r in rows if r["rule"] == "lossy-l"],
        "threshold":    [r for r in rows if r["rule"] == "threshold"],
        "confidence":   [r for r in rows if r["rule"] == "confidence"],
    }
    best = {name: _best_op(gs) for name, gs in groups.items()}

    cond: List[str] = []
    cond.append("# Phase 26 — condensed comparison (best operating point per group)\n\n")
    cond.append(
        "Best = min ΔNLL with tie break on max tok/s, over all rows in the "
        "group. `tok_succ` / `rnd_mean` / `all_pass` are reported if the "
        "row has them.\n\n"
    )
    cond.append("```\n")
    cond.append(
        f"  {'group':<11}{'family':<14}{'rule':<12}{'thresh':>10}"
        f"{'tok/s':>8}{'ΔNLL':>10}{'tok_succ':>10}\n"
    )
    for name, r in best.items():
        if r is None:
            cond.append(f"  {name:<11}(no rows)\n")
            continue
        tok_succ = (
            f"{float(r['tok_succ']):>10.4f}"
            if r.get("tok_succ") is not None else f"{'—':>10}"
        )
        cond.append(
            f"  {name:<11}{r['family']:<14}{r['rule']:<12}"
            f"{r['threshold_label']:>10}"
            f"{float(r['tok_s_mean']):>8.2f}"
            f"{float(r['delta_NLL']):>+10.4f}{tok_succ}\n"
        )
    cond.append("```\n")

    cond_text = "".join(cond)
    print(cond_text)
    with open(out_dir / "condensed_table.md", "w") as f:
        f.write(cond_text)
    with open(out_dir / "condensed_table.json", "w") as f:
        json.dump({"best_per_group": best}, f, indent=2, default=str)

    print(f"[plots] wrote {out_dir}/pareto_delta_nll_tok_s.png")
    print(f"[plots] wrote {out_dir}/pareto_delta_nll_tok_s_clean.png")
    print(f"[plots] wrote {out_dir}/pareto_tok_succ_tok_s.png")
    print(f"[plots] wrote {out_dir}/pareto_tok_succ_tok_s_clean.png")
    print(f"[plots] wrote {out_dir}/condensed_table.md")
    print(f"[plots] wrote {out_dir}/condensed_table.json")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
