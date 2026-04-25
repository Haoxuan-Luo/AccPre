"""Pareto plotter for v1.

Reads `summary.json` from `eval/wallclock.py` and draws:
  - Main panel: (wallclock_tok_s_mean, CF@1_aggregate) per method, with
    per-prompt bootstrap CI on CF@1 shown as horizontal error bars. The
    oracle-Q2 lane is a curve over the tau sweep; strict is one point.
  - Auxiliary panel: same x-axis lanes but y = XL-audit mean. Clearly
    labeled AUXILIARY.

No decoding logic here — pure plotting + aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def _load_summary(in_dir: str) -> Dict[str, Any]:
    path = os.path.join(in_dir, "summary.json")
    with open(path, "r") as f:
        return json.load(f)


def plot(summary: Dict[str, Any], out_fig: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_main, ax_aux) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Main panel: CF@1 vs tok/s ---
    ax_main.set_title("Pareto (primary): wall-clock tok/s vs CF@1")
    ax_main.set_xlabel("Controlled Faithfulness@1")
    ax_main.set_ylabel("wall-clock tok/s (mean over prompts)")

    s = summary["strict"]
    ax_main.errorbar(
        [s["cf_at_1_aggregate"]], [s["tok_s_mean"]],
        xerr=None, fmt="o", color="black", label="strict SpecDiff",
        markersize=8,
    )

    oq = summary["oracle_q2"]
    taus_sorted = sorted(float(t) for t in oq.keys())
    xs = [oq[str(t)]["cf_at_1_aggregate"] for t in taus_sorted]
    ys = [oq[str(t)]["tok_s_mean"] for t in taus_sorted]
    x_lo = [oq[str(t)]["cf_at_1_bootstrap_ci"][0] for t in taus_sorted]
    x_hi = [oq[str(t)]["cf_at_1_bootstrap_ci"][1] for t in taus_sorted]
    x_err_lo = [max(0.0, xs[i] - x_lo[i]) for i in range(len(xs))]
    x_err_hi = [max(0.0, x_hi[i] - xs[i]) for i in range(len(xs))]
    ax_main.errorbar(
        xs, ys, xerr=[x_err_lo, x_err_hi], fmt="s-", color="C0",
        label="oracle-Q2 threshold (tau sweep)", markersize=6,
    )
    for i, t in enumerate(taus_sorted):
        ax_main.annotate(
            f"τ={t}", (xs[i], ys[i]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.legend(loc="best")
    ax_main.grid(True, alpha=0.3)

    # --- Auxiliary panel: XL-audit ---
    ax_aux.set_title("AUXILIARY: XL-audit (NOT on Pareto axis)")
    ax_aux.set_xlabel("tau (oracle-Q2) — strict drawn at x=reference")
    ax_aux.set_ylabel("XL-audit greedy-match fraction")
    ax_aux.axhline(
        y=s["xl_audit_aux_mean"], linestyle="--", color="black",
        label=f"strict XL-audit = {s['xl_audit_aux_mean']:.3f}",
    )
    aux_ys = [oq[str(t)]["xl_audit_aux_mean"] for t in taus_sorted]
    ax_aux.plot(taus_sorted, aux_ys, "s-", color="C0",
                label="oracle-Q2 XL-audit")
    ax_aux.set_ylim(0, 1)
    ax_aux.legend(loc="best")
    ax_aux.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=120)
    print(f"[pareto] figure written to {out_fig}")


def _collect_predictor_summaries(dirs: list) -> dict:
    """Scan a list of directories for Phase-2 online_decode summaries.

    Groups by (predictor_name, kind, family, deploy_mode); each group
    becomes one τ-curve on the Pareto. Returns a dict keyed by
    `f"{name} [{family}/{deploy_mode}]"`.
    """
    import glob
    groups: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "online_*.json"))):
            with open(path, "r") as f:
                s = json.load(f)
            p = s.get("predictor", {})
            name = p.get("class", "unknown")
            fam = p.get("family", "?")
            dm = p.get("deploy_mode", "?")
            label = f"{name} [{fam}/{dm}]"
            tau = float(s["tau"])
            groups.setdefault(label, {})[tau] = s["l3"]
    return groups


def plot_phase2(
    strict_summary: Dict[str, Any],
    predictor_groups: Dict[str, Dict[float, Dict[str, Any]]],
    out_fig: str,
) -> None:
    """Two-panel Pareto: primary (tok/s vs CF@1) + auxiliary XL-audit.

    Renders strict as a single point, oracle-Q2 as a tau-curve, and each
    predictor group as its own tau-curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_main, ax_aux) = plt.subplots(1, 2, figsize=(13, 5))
    ax_main.set_title("Pareto (Phase 2): wall-clock tok/s vs CF@1")
    ax_main.set_xlabel("Controlled Faithfulness@1")
    ax_main.set_ylabel("wall-clock tok/s (mean over prompts)")

    # --- strict ---
    s = strict_summary["strict"]
    ax_main.errorbar(
        [s["cf_at_1_aggregate"]], [s["tok_s_mean"]],
        fmt="o", color="black", label="strict SpecDiff", markersize=10,
    )

    # --- oracle-Q2 ---
    oq = strict_summary.get("oracle_q2", {})
    if oq:
        taus_sorted = sorted(float(t) for t in oq.keys())
        xs = [oq[str(t)]["cf_at_1_aggregate"] for t in taus_sorted]
        ys = [oq[str(t)]["tok_s_mean"] for t in taus_sorted]
        ax_main.plot(xs, ys, "s-", color="C0",
                     label="oracle-Q2 threshold (τ)", markersize=6)
        for i, t in enumerate(taus_sorted):
            ax_main.annotate(f"τ={t}", (xs[i], ys[i]),
                             textcoords="offset points", xytext=(5, 5), fontsize=7)

    # --- predictors (Phase 2) ---
    color_cycle = [f"C{i}" for i in range(1, 10)]
    for i, (label, tau_map) in enumerate(predictor_groups.items()):
        color = color_cycle[i % len(color_cycle)]
        taus_sorted = sorted(tau_map.keys())
        xs = [tau_map[t]["cf_at_1_aggregate"] for t in taus_sorted]
        ys = [tau_map[t]["tok_s_mean"] for t in taus_sorted]
        ax_main.plot(xs, ys, "^--", color=color, label=label, markersize=6)

    ax_main.set_xlim(-0.02, 1.02)
    ax_main.legend(loc="best", fontsize=8)
    ax_main.grid(True, alpha=0.3)

    # --- Auxiliary panel: XL-audit (labeled NOT on Pareto) ---
    ax_aux.set_title("AUXILIARY: XL-audit (NOT on Pareto axis)")
    ax_aux.set_xlabel("tau")
    ax_aux.set_ylabel("XL-audit greedy-match fraction")
    ax_aux.axhline(
        y=s["xl_audit_aux_mean"], linestyle="--", color="black",
        label=f"strict XL-audit = {s['xl_audit_aux_mean']:.3f}",
    )
    if oq:
        taus_sorted = sorted(float(t) for t in oq.keys())
        aux_ys = [oq[str(t)]["xl_audit_aux_mean"] for t in taus_sorted]
        ax_aux.plot(taus_sorted, aux_ys, "s-", color="C0",
                    label="oracle-Q2 XL-audit")
    for i, (label, tau_map) in enumerate(predictor_groups.items()):
        color = color_cycle[i % len(color_cycle)]
        taus_sorted = sorted(tau_map.keys())
        aux_ys = [tau_map[t]["xl_audit_aux_mean"] for t in taus_sorted]
        ax_aux.plot(taus_sorted, aux_ys, "^--", color=color, label=label)

    ax_aux.set_ylim(0, 1)
    ax_aux.legend(loc="best", fontsize=7)
    ax_aux.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=120)
    print(f"[pareto] figure written to {out_fig}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pareto plotter")
    ap.add_argument("--in_dir", type=str, required=True,
                    help="Directory containing summary.json (strict + oracle-Q2).")
    ap.add_argument("--predictor_dirs", type=str, default="",
                    help="Comma-separated dirs of Phase-2 online_*.json summaries.")
    ap.add_argument("--out_fig", type=str, required=True)
    args = ap.parse_args()

    summary = _load_summary(args.in_dir)
    pred_dirs = [d.strip() for d in args.predictor_dirs.split(",") if d.strip()]
    if pred_dirs:
        groups = _collect_predictor_summaries(pred_dirs)
        plot_phase2(summary, groups, args.out_fig)
    else:
        plot(summary, args.out_fig)


if __name__ == "__main__":
    main()
