"""Phase 20 follow-up — add strict and oracle-Q2 reference rows.

Consumes:
  outputs/phase20_eval_<JOB>/system_table.json    (Phase 20 main output)
  outputs/oracle_q2_ceiling/summary.json          (Phase 14 Task A)
  outputs/strict_global_idx/summary.json          (strict_refresh.py output;
                                                    written just before this
                                                    runs in the same SLURM job)
Produces:
  <out_dir>/system_table_with_refs.md
  <out_dir>/system_table_with_refs.json
  <out_dir>/pareto_with_strict.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

TAUS = (0.5, 0.7, 0.9)


def _fmt_num(x, fmt="{:.4f}") -> str:
    if x is None:
        return "  N/A   "
    if isinstance(x, float) and not np.isfinite(x):
        return "  N/A   "
    return fmt.format(float(x))


def _render(rows: List[Dict]) -> str:
    lines = ["# Table B (with strict + oracle-Q2 references)\n\n"]
    lines.append(
        "`strict` row is strict SpecDiff under the original pretrained drafter, "
        "measured on the canonical 10 test prompts (global indices 60–69) × 64 "
        "new tokens, same protocol as every method lane.\n\n"
        "`oracle-Q2` rows use `commit_threshold(record.min_pq_j, τ)` as the "
        "decision — i.e. the true per-position Q2 is fed through the exact "
        "same commit rule every method uses, evaluated on the full test "
        "split (n=1433). No executable runtime, so tok/s / NLL / ΔNLL are "
        "N/A by definition; the point of the row is the offline CF@1 "
        "ceiling.\n\n"
    )
    lines.append("```\n")
    lines.append(
        f"  {'method':<16}{'τ':>6}"
        f"{'offCF1':>8}{'onCF1':>8}"
        f"{'tok/s':>8}{'NLL':>10}{'ΔNLL':>10}"
        f"{'offU':>7}{'offO':>7}\n"
    )
    for row in rows:
        line = f"  {row['method']:<16}"
        tau = row.get("tau")
        line += f"{tau if isinstance(tau, str) else f'{float(tau):.1f}':>6}"
        for key, fmt in [
            ("offCF1", "{:>8.4f}"), ("onCF1", "{:>8.4f}"),
            ("tok_s", "{:>8.2f}"),
            ("NLL",   "{:>10.4f}"), ("delta_NLL", "{:>10.4f}"),
            ("offU",  "{:>7.3f}"), ("offO",  "{:>7.3f}"),
        ]:
            v = row.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                # Use dashes aligned to the same width.
                pad_spec = fmt.split(":")[1].rstrip("}")
                width = int(pad_spec.split(".")[0].lstrip(">"))
                line += f"{'   N/A ':>{width}}"
            else:
                line += fmt.format(float(v))
        lines.append(line + "\n")
    lines.append("```\n")
    return "".join(lines)


def _make_pareto(rows: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[phase20_addref] matplotlib unavailable ({e})")
        return
    colors = {
        "frozen_1A":     "#4c78a8",
        "lazy_jnt":      "#a0a0a0",
        "live_jnt_5ep":  "#54a24b",
        "live_jnt_10ep": "#e45756",
        "strict":        "#000000",
    }
    markers_by_tau = {0.5: "o", 0.7: "s", 0.9: "^", "—": "D"}
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for row in rows:
        name = row["method"]
        color = colors.get(name, "#888888")
        tau = row.get("tau")
        tau_key = "—" if isinstance(tau, str) else float(tau)
        marker = markers_by_tau.get(tau_key, "x")
        x = row.get("delta_NLL")
        y = row.get("tok_s")
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            continue
        ax.scatter(x, y, s=110 if name == "strict" else 90, color=color,
                   marker=marker, edgecolor="black" if name == "strict" else None,
                   linewidth=1.2 if name == "strict" else 0.0,
                   label=f"{name} τ={tau}")
        label = f"strict" if name == "strict" else f"{name}\nτ={tau}"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color=color, alpha=0.9)
    ax.axvline(0.0, linestyle="--", color="#bbbbbb", linewidth=1)
    ax.set_xlabel("Δ NLL vs strict reference  (↓ better, strict at 0)")
    ax.set_ylabel("tok/s  (↑ better)")
    ax.set_title(
        "Phase 20 Pareto: throughput vs quality gap to strict\n"
        "(+ strict reference; oracle-Q2 has no executable runtime so is not plotted)"
    )
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    legend_h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               label="τ = 0.5", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               label="τ = 0.7", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
               label="τ = 0.9", markersize=8),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
               markeredgecolor="black", label="strict", markersize=8),
    ]
    for name, color in colors.items():
        if name == "strict":
            continue
        legend_h.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color, label=name, markersize=8))
    legend_h.append(Line2D([0], [0], marker="D", color="w",
                           markerfacecolor="black", markeredgecolor="black",
                           label="strict", markersize=8))
    ax.legend(handles=legend_h, loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"[phase20_addref] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase20_dir", type=str, required=True,
                    help="Original phase20 output dir containing system_table.json")
    ap.add_argument("--oracle_json", type=str,
                    default="outputs/oracle_q2_ceiling/summary.json")
    ap.add_argument("--strict_refresh_json", type=str,
                    default="outputs/strict_global_idx/summary.json")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    root = _REPO_ROOT
    phase20_dir = Path(args.phase20_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(phase20_dir / "system_table.json") as f:
        nll = json.load(f)
    with open(phase20_dir / "predictor_table.json") as f:
        pred_rows = json.load(f)
    off = {row["method"]: row["offline_per_tau"] for row in pred_rows}

    with open(root / args.oracle_json) as f:
        oracle = json.load(f)
    oracle_full = oracle["full_test_split"]

    strict_path = root / args.strict_refresh_json
    if strict_path.is_file():
        with open(strict_path) as f:
            sr = json.load(f)
        strict_tok_s = float(sr["strict"]["tok_s_mean"])
    else:
        print(
            f"[phase20_addref] WARN: {strict_path} missing; falling back to "
            f"Phase-1 baseline 19.75 tok/s."
        )
        strict_tok_s = 19.75

    strict_nll = float(nll["strict_ref"]["nll_mean"])

    rows: List[Dict] = []

    # Order: frozen, lazy, live-5ep, live-10ep (per τ), then strict, then oracle.
    method_order = ("frozen_1A", "lazy_jnt", "live_jnt_5ep", "live_jnt_10ep")
    for name in method_order:
        if name not in nll["methods"]:
            continue
        for tau in TAUS:
            m = nll["methods"][name].get(str(tau))
            if m is None:
                continue
            off_tau = off.get(name, {}).get(str(float(tau)), {})
            rows.append({
                "method": name, "tau": float(tau),
                "offCF1": off_tau.get("offline_cf"),
                "onCF1":  m["online_cf_at_1"],
                "tok_s":  m["tok_s_mean"],
                "NLL":    m["nll_mean"],
                "delta_NLL": m["delta_nll_vs_strict"],
                "offU":   off_tau.get("offline_under"),
                "offO":   off_tau.get("offline_over"),
            })

    # Strict reference row. τ is "—" (no threshold).
    rows.append({
        "method": "strict", "tau": "—",
        "offCF1": 1.0, "onCF1": 1.0,
        "tok_s": strict_tok_s,
        "NLL":   strict_nll, "delta_NLL": 0.0,
        "offU":  0.0, "offO": 0.0,
    })

    # Oracle-Q2 rows, one per τ.
    for tau in TAUS:
        key = str(tau)
        if key not in oracle_full:
            continue
        orc = oracle_full[key]
        rows.append({
            "method": "oracle_Q2", "tau": float(tau),
            "offCF1": orc["cf_at_1"],
            "onCF1":  None,
            "tok_s":  None,
            "NLL":    None, "delta_NLL": None,
            "offU":   orc["under"], "offO": orc["over"],
        })

    rep = _render(rows)
    with open(out_dir / "system_table_with_refs.md", "w") as f:
        f.write(rep)
    with open(out_dir / "system_table_with_refs.json", "w") as f:
        json.dump({"rows": rows, "strict_tok_s_source":
                   "strict_refresh" if strict_path.is_file() else "fallback_19.75",
                   "strict_nll": strict_nll}, f, indent=2, default=str)

    _make_pareto(rows, out_dir / "pareto_with_strict.png")

    print(rep)
    print(f"[phase20_addref] wrote {out_dir}/system_table_with_refs.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
