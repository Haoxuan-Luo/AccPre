"""Stage 6 — aggregate baselines + predictor evals; emit CSVs and Pareto plots.

Inputs:
  - results/baselines/                : 5 reused baseline JSONs (T=1)
  - results/eval/<regime>/<target>/<arch>/<rule>/<param_tag>/online_predictor_full.json

Outputs:
  - results/all_metrics.csv           : raw per-cell rows
  - results/final_table.csv           : same data, re-keyed for the paper
  - results/final_table.md            : human-readable summary
  - results/plots/pareto_nll_vs_tok_s.png
  - results/plots/pareto_tok_succ_vs_tok_s.png
  - results/plots/tokens_per_round_vs_nll.png

Plotting conventions (per user spec):
  - color    = target  (relmax / alpha_q2 / dep)
  - marker   = architecture (mlp_pos / causal_tx / bidir_tx)
  - linestyle= regime (frozen solid / joint dashed)
  - lines connect threshold points only within the same (regime, target, arch, rule).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]


# Plotting style mappings.
_TARGET_COLORS = {
    "relmax":   "tab:blue",
    "alpha_q2": "tab:orange",
    "dep":      "tab:green",
}
_ARCH_MARKERS = {
    "mlp_pos":                       "o",
    "causal_transformer_pos":        "s",
    "bidirectional_transformer_pos": "^",
}
_REGIME_LINESTYLES = {
    "frozen": "-",
    "joint":  "--",
}
# For consistency, line+marker per cell are always plotted; but the line
# itself only connects threshold points within the same (regime, target,
# arch, rule).
_BASELINE_COLOR = "black"
_BASELINE_MARKER = "x"


def _mean_tokens_per_round(d: Dict[str, Any]) -> float:
    n_total = 0
    n_rounds = 0
    for pp in d.get("per_prompt", []):
        for rd in pp.get("rounds", []):
            n_total += int(rd.get("n_committed", rd.get("n_commit", 0)))
            n_rounds += 1
    return n_total / n_rounds if n_rounds else 0.0


def _baseline_row(p: Path) -> Dict[str, Any]:
    with open(p) as f:
        d = json.load(f)
    return dict(
        method_type="baseline",
        regime="-",
        target="-",
        arch="-",
        rule="-",
        param_tag=p.stem.replace("online_", ""),
        param=None,
        fallback="verifier-aware (predecessor)",
        verifier_online="YES",
        tok_s=float(d.get("tok_s_mean", 0.0)),
        nll=float(d.get("nll_mean", float("nan"))),
        tok_succ=float(d.get("tok_succ_mean", float("nan"))),
        tokens_per_round=_mean_tokens_per_round(d),
        n_prompts=len(d.get("per_prompt", [])),
        source_json=str(p),
    )


def _predictor_row(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return None
    triple = d.get("triple") or {}
    return dict(
        method_type="predictor",
        regime=str(triple.get("regime", "?")),
        target=str(triple.get("target", "?")),
        arch=str(triple.get("arch", "?")),
        rule=str(d.get("rule", "?")),
        param_tag=str(d.get("param_tag", "?")),
        param=float(d.get("param", float("nan"))),
        fallback=str(d.get("fallback_policy", "")),
        verifier_online="NO",
        tok_s=float(d.get("tok_s_mean", 0.0)),
        nll=float(d.get("nll_mean", float("nan"))),
        tok_succ=float(d.get("tok_succ_mean", float("nan"))),
        tokens_per_round=_mean_tokens_per_round(d),
        n_prompts=len(d.get("per_prompt", [])),
        source_json=str(p),
    )


def _aggregate(baselines_dir: Path, eval_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if baselines_dir.exists():
        for p in sorted(baselines_dir.glob("online_*.json")):
            rows.append(_baseline_row(p))
    if eval_root.exists():
        for p in sorted(eval_root.rglob("online_predictor_full.json")):
            r = _predictor_row(p)
            if r is not None:
                rows.append(r)
    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    cols = ["method_type", "regime", "target", "arch", "rule", "param_tag",
            "param", "fallback", "verifier_online",
            "tok_s", "nll", "tok_succ", "tokens_per_round",
            "n_prompts", "source_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md(rows: List[Dict[str, Any]], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Final results — baselines + predictor evals")
    lines.append("")
    lines.append(f"Total rows: {len(rows)} "
                 f"(baselines={sum(1 for r in rows if r['method_type']=='baseline')}, "
                 f"predictor={sum(1 for r in rows if r['method_type']=='predictor')}).")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| name | tok/s | nll | tok_succ | tok/round | n_prompts |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["method_type"] != "baseline":
            continue
        lines.append(
            f"| {r['param_tag']} | {r['tok_s']:.2f} | {r['nll']:.4f} | "
            f"{r['tok_succ']:.4f} | {r['tokens_per_round']:.2f} | {r['n_prompts']} |"
        )
    lines.append("")
    lines.append("## Predictor cells")
    lines.append("")
    lines.append("| regime | target | arch | rule | param | tok/s | nll | tok_succ | tok/round |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["method_type"] != "predictor":
            continue
        lines.append(
            f"| {r['regime']} | {r['target']} | {r['arch']} | {r['rule']} | "
            f"{r['param_tag']} | {r['tok_s']:.2f} | {r['nll']:.4f} | "
            f"{r['tok_succ']:.4f} | {r['tokens_per_round']:.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------


def _group_predictor_lines(rows: List[Dict[str, Any]]
                           ) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    """Group predictor rows by (regime, target, arch, rule) for connected lines.

    Within each group, sort by `param` ascending so the lines connect
    threshold points monotonically.
    """
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r["method_type"] != "predictor":
            continue
        key = (r["regime"], r["target"], r["arch"], r["rule"])
        groups.setdefault(key, []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: float(r.get("param") or 0.0))
    return groups


def _plot_pareto(
    rows: List[Dict[str, Any]],
    out_path: Path,
    x_field: str,
    y_field: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[plot] WARNING matplotlib not available: {e}; skipping {out_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Predictor cells: connect by group.
    groups = _group_predictor_lines(rows)
    seen_legend: set = set()
    for (regime, target, arch, rule), pts in sorted(groups.items()):
        xs = [r[x_field] for r in pts]
        ys = [r[y_field] for r in pts]
        color = _TARGET_COLORS.get(target, "gray")
        marker = _ARCH_MARKERS.get(arch, ".")
        ls = _REGIME_LINESTYLES.get(regime, "-")
        # Only the first plotted entry for each (target, arch, regime)
        # gets a legend label; subsequent (rule) variants share the style.
        leg_key = (target, arch, regime)
        label: Optional[str] = None
        if leg_key not in seen_legend:
            label = f"{regime} {target} {arch}"
            seen_legend.add(leg_key)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                linewidth=1.0, markersize=6, alpha=0.85, label=label)

    # Baselines: scatter only.
    base_x: List[float] = []
    base_y: List[float] = []
    base_labels: List[str] = []
    for r in rows:
        if r["method_type"] != "baseline":
            continue
        base_x.append(r[x_field])
        base_y.append(r[y_field])
        base_labels.append(r["param_tag"])
    if base_x:
        ax.scatter(base_x, base_y, c=_BASELINE_COLOR, marker=_BASELINE_MARKER,
                   s=60, label="baseline (predecessor)")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _plot_tokens_per_round_vs_nll(
    rows: List[Dict[str, Any]], out_path: Path,
) -> None:
    _plot_pareto(rows, out_path,
                 x_field="nll", y_field="tokens_per_round",
                 x_label="NLL (lower=better)",
                 y_label="mean tokens / round (higher=better)",
                 title="Tokens-per-round vs NLL")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines_dir",
                    default=str(_EXP_ROOT / "results" / "baselines"))
    ap.add_argument("--eval_root",
                    default=str(_EXP_ROOT / "results" / "eval"))
    ap.add_argument("--out_root",
                    default=str(_EXP_ROOT / "results"))
    args = ap.parse_args()

    baselines_dir = Path(args.baselines_dir)
    eval_root = Path(args.eval_root)
    out_root = Path(args.out_root)

    rows = _aggregate(baselines_dir, eval_root)
    print(f"[plot] aggregated {len(rows)} rows "
          f"(baselines={sum(1 for r in rows if r['method_type']=='baseline')}, "
          f"predictor={sum(1 for r in rows if r['method_type']=='predictor')})")

    _write_csv(rows, out_root / "all_metrics.csv")
    _write_csv(rows, out_root / "final_table.csv")
    _write_md(rows, out_root / "final_table.md")

    plots = out_root / "plots"
    _plot_pareto(
        rows, plots / "pareto_nll_vs_tok_s.png",
        x_field="nll", y_field="tok_s",
        x_label="NLL (lower=better)",
        y_label="tok/s (drafter+head only; higher=better)",
        title="Pareto: NLL vs tok/s",
    )
    _plot_pareto(
        rows, plots / "pareto_tok_succ_vs_tok_s.png",
        x_field="tok_succ", y_field="tok_s",
        x_label="tok_succ (higher=better)",
        y_label="tok/s (higher=better)",
        title="Pareto: tok_succ vs tok/s",
    )
    _plot_tokens_per_round_vs_nll(rows, plots / "tokens_per_round_vs_nll.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
