"""Read-only D1/D2/D3/B-lite diagnostic on acc_1a + stage1_pp.pt.

Answers:
  D1  predictor residual vs drafter-verifier mismatch on survived positions
  D2  Q2 structure / commit headroom
  D3  drafter confidence (entropy, margin, top1) vs mismatch / Q2
  B-lite  p/q ratio and clipping structure

No model changes, no training, no recollection.

Inputs  (must already exist):
  data_collected/stage1_pp.pt     canonical strict records with v2 fields
  checkpoints/acc_1a/preds_test.pt  acc_1a predictions on the test split
  configs/protocol.yaml            to enforce fingerprint consistency

Outputs:
  <out_dir>/report.md              plain-text report (the primary deliverable)
  <out_dir>/summary.json           numeric summary (for later reference)
  <out_dir>/*.png                  plots (only if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N


# ----------------------------------------------------------------------
# Loading & flattening
# ----------------------------------------------------------------------


def load_protocol(protocol_path: Path) -> ProtocolConfig:
    with open(protocol_path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def load_artifacts(
    root: Path,
) -> Tuple[List, List, ProtocolConfig]:
    """Load test-split records (post gamma filter) + acc_1a preds.

    Ordering of `test_records` matches the trainer's FrozenAcceptanceDataset
    construction, so `preds[i]['record_idx']` indexes into it.
    """
    protocol = load_protocol(root / "configs/protocol.yaml")
    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"),
        expected_protocol=protocol,
    )
    gamma = 8
    gamma_filtered = [r for r in all_recs if r.gamma == gamma]
    test_prompts = set(range(TRAIN_N + VAL_N, POOL_SIZE))
    test_records = [r for r in gamma_filtered if r.prompt_idx in test_prompts]

    preds = torch.load(
        str(root / "checkpoints/acc_1a/preds_test.pt"),
        weights_only=False,
    )
    if len(preds) != len(test_records):
        raise RuntimeError(
            f"preds_test.pt rows ({len(preds)}) != test_records ({len(test_records)}). "
            f"Have the split/filter conventions drifted?"
        )
    return test_records, preds, protocol


def flatten(
    test_records: List, preds: List,
) -> Dict[str, np.ndarray]:
    """Flatten to per-survived-position rows.

    All arrays are 1-D and same-length; NaN is used where a scalar is
    unavailable on the record.
    """
    keys_scalar = [
        "p", "q", "q2", "qhat", "de", "dm", "dt", "j",
        "accepted", "prompt_idx", "round_idx",
    ]
    acc: Dict[str, List[float]] = {k: [] for k in keys_scalar}

    for pred in preds:
        rec = test_records[int(pred["record_idx"])]
        qhat_vec = pred["q2_hat"]
        gamma = len(qhat_vec)
        for j in range(gamma):
            if int(rec.survived_j[j]) != 1:
                continue
            acc["p"].append(float(rec.p_j[j]))
            acc["q"].append(float(rec.q_j[j]))
            acc["q2"].append(float(rec.min_pq_j[j]))
            acc["qhat"].append(float(qhat_vec[j]))
            acc["de"].append(
                float(rec.drafter_entropy_j[j])
                if rec.drafter_entropy_j is not None else float("nan")
            )
            acc["dm"].append(
                float(rec.drafter_margin_j[j])
                if rec.drafter_margin_j is not None else float("nan")
            )
            acc["dt"].append(
                float(rec.drafter_top1_prob_j[j])
                if rec.drafter_top1_prob_j is not None else float("nan")
            )
            acc["j"].append(int(j))
            acc["accepted"].append(int(rec.accepted_j[j]))
            acc["prompt_idx"].append(int(rec.prompt_idx))
            acc["round_idx"].append(int(rec.round_idx))

    out: Dict[str, np.ndarray] = {k: np.array(v) for k, v in acc.items()}

    # Derived features.
    EPS = 1e-10
    p = np.maximum(out["p"], EPS)
    q = np.maximum(out["q"], EPS)
    out["abs_pq"] = np.abs(out["p"] - out["q"])
    out["log_pq"] = np.log(p / q)              # signed log-ratio
    out["abs_log_pq"] = np.abs(out["log_pq"])
    out["ratio"] = p / q                       # raw p/q
    out["res_abs"] = np.abs(out["qhat"] - out["q2"])
    out["res_sq"] = (out["qhat"] - out["q2"]) ** 2
    return out


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xs, ys = x[m], y[m]
    if xs.std() < 1e-12 or ys.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xs, ys = x[m], y[m]
    rx = np.argsort(np.argsort(xs)).astype(np.float64)
    ry = np.argsort(np.argsort(ys)).astype(np.float64)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def quantiles(x: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, float]:
    m = np.isfinite(x)
    out = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(x[m], q))
    return out


def bin_by_quantile(
    x: np.ndarray, y: np.ndarray, n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Sort by x, split into n_bins equal-count bins, return per-bin y stats."""
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[m], y[m]
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    n = len(xs)
    if n < n_bins:
        return []
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        xsi, ysi = xs[lo:hi], ys[lo:hi]
        out.append({
            "bin": i, "n": int(hi - lo),
            "x_lo": float(xsi.min()), "x_hi": float(xsi.max()),
            "y_mean": float(ysi.mean()), "y_std": float(ysi.std(ddof=0)),
        })
    return out


# ----------------------------------------------------------------------
# D1 — residual vs mismatch
# ----------------------------------------------------------------------


def d1_residual_vs_mismatch(d: Dict[str, np.ndarray]) -> Dict:
    r = {}
    r["n_survived_positions"] = int(len(d["q2"]))

    # Overall correlations.
    r["corr"] = {
        "abs_res_vs_abs_pq":        {
            "pearson": pearson(d["res_abs"], d["abs_pq"]),
            "spearman": spearman(d["res_abs"], d["abs_pq"]),
        },
        "abs_res_vs_abs_log_pq":    {
            "pearson": pearson(d["res_abs"], d["abs_log_pq"]),
            "spearman": spearman(d["res_abs"], d["abs_log_pq"]),
        },
        "abs_res_vs_ratio":         {
            "pearson": pearson(d["res_abs"], d["ratio"]),
            "spearman": spearman(d["res_abs"], d["ratio"]),
        },
        "sq_res_vs_abs_pq":         {
            "pearson": pearson(d["res_sq"], d["abs_pq"]),
            "spearman": spearman(d["res_sq"], d["abs_pq"]),
        },
    }

    # Deciles of |p-q| and |log(p/q)|; y = residual.
    r["deciles_abs_pq"] = bin_by_quantile(d["abs_pq"], d["res_abs"], n_bins=10)
    r["deciles_abs_log_pq"] = bin_by_quantile(
        d["abs_log_pq"], d["res_abs"], n_bins=10,
    )

    # Per-position (j=0..7) residual + local correlation with |log p/q|.
    per_j = []
    for j in range(8):
        mask = d["j"] == j
        if mask.sum() == 0:
            per_j.append({"j": j, "n": 0})
            continue
        per_j.append({
            "j": j,
            "n": int(mask.sum()),
            "res_mean": float(d["res_abs"][mask].mean()),
            "res_median": float(np.median(d["res_abs"][mask])),
            "q2_mean": float(d["q2"][mask].mean()),
            "abs_logpq_mean": float(d["abs_log_pq"][mask].mean()),
            "corr_res_vs_abs_log_pq_spearman":
                spearman(d["res_abs"][mask], d["abs_log_pq"][mask]),
        })
    r["per_j"] = per_j
    return r


# ----------------------------------------------------------------------
# D2 — Q2 structure / headroom
# ----------------------------------------------------------------------


def d2_q2_structure(d: Dict[str, np.ndarray]) -> Dict:
    r = {}
    q2 = d["q2"]
    r["stats"] = {
        "n": int(len(q2)),
        "mean": float(q2.mean()),
        "std": float(q2.std(ddof=0)),
        "min": float(q2.min()),
        "max": float(q2.max()),
        **quantiles(q2),
    }
    r["fraction_ge"] = {
        "0.1":  float((q2 >= 0.1).mean()),
        "0.3":  float((q2 >= 0.3).mean()),
        "0.5":  float((q2 >= 0.5).mean()),
        "0.7":  float((q2 >= 0.7).mean()),
        "0.9":  float((q2 >= 0.9).mean()),
        "eq_1": float((q2 >= 1.0 - 1e-9).mean()),
    }
    per_j = []
    for j in range(8):
        mask = d["j"] == j
        if mask.sum() == 0:
            per_j.append({"j": j, "n": 0})
            continue
        q2j = q2[mask]
        per_j.append({
            "j": j, "n": int(mask.sum()),
            "mean": float(q2j.mean()),
            "frac_ge_0.5": float((q2j >= 0.5).mean()),
            "frac_ge_0.7": float((q2j >= 0.7).mean()),
            "frac_ge_0.9": float((q2j >= 0.9).mean()),
            "frac_eq_1":   float((q2j >= 1.0 - 1e-9).mean()),
        })
    r["per_j"] = per_j
    return r


# ----------------------------------------------------------------------
# D3 — drafter confidence vs mismatch / Q2
# ----------------------------------------------------------------------


def d3_drafter_confidence(d: Dict[str, np.ndarray]) -> Dict:
    r: Dict = {}
    targets = {
        "abs_pq": d["abs_pq"],
        "abs_log_pq": d["abs_log_pq"],
        "q2": d["q2"],
        "res_abs": d["res_abs"],  # bonus — does confidence predict our error?
    }
    signals = {
        "drafter_entropy": d["de"],
        "drafter_margin":  d["dm"],
        "drafter_top1":    d["dt"],
        "q":               d["q"],
    }
    corrs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sname, sx in signals.items():
        corrs[sname] = {}
        for tname, ty in targets.items():
            corrs[sname][tname] = {
                "pearson": pearson(sx, ty),
                "spearman": spearman(sx, ty),
            }
    r["corrs"] = corrs

    # Decile binning of drafter_entropy (x) vs mismatch (y) and Q2 (y).
    r["deciles_de_vs_abs_log_pq"] = bin_by_quantile(
        d["de"], d["abs_log_pq"], n_bins=10,
    )
    r["deciles_de_vs_q2"] = bin_by_quantile(d["de"], d["q2"], n_bins=10)
    r["deciles_dt_vs_q2"] = bin_by_quantile(d["dt"], d["q2"], n_bins=10)
    return r


# ----------------------------------------------------------------------
# B-lite — clipping / ratio structure
# ----------------------------------------------------------------------


def b_lite_clipping(d: Dict[str, np.ndarray]) -> Dict:
    r = {}
    ratio = d["ratio"]
    log_ratio = d["log_pq"]
    q2 = d["q2"]
    r["ratio_stats"] = {
        "mean": float(ratio.mean()),
        "std":  float(ratio.std(ddof=0)),
        **quantiles(ratio),
    }
    r["log_ratio_stats"] = {
        "mean": float(log_ratio.mean()),
        "std":  float(log_ratio.std(ddof=0)),
        **quantiles(log_ratio),
    }
    r["fraction"] = {
        "ratio_gt_1":      float((ratio > 1.0).mean()),
        "ratio_ge_2":      float((ratio >= 2.0).mean()),
        "ratio_le_0_5":    float((ratio <= 0.5).mean()),
        "ratio_le_0_1":    float((ratio <= 0.1).mean()),
        "q2_eq_1_clipped": float((q2 >= 1.0 - 1e-9).mean()),
        "q2_lt_0_01":      float((q2 < 0.01).mean()),
        "q2_lt_0_05":      float((q2 < 0.05).mean()),
        "q2_gt_0_95_not_clipped":
            float(((q2 > 0.95) & (q2 < 1.0 - 1e-9)).mean()),
    }
    return r


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------


def _fmt_decile_table(rows: List[Dict[str, float]]) -> str:
    if not rows:
        return "  (empty)\n"
    header = (
        f"  {'dec':>3} {'n':>6} {'x_lo':>10} {'x_hi':>10} "
        f"{'y_mean':>10} {'y_std':>10}\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"  {row['bin']:>3d} {row['n']:>6d} "
            f"{row['x_lo']:>10.4f} {row['x_hi']:>10.4f} "
            f"{row['y_mean']:>10.4f} {row['y_std']:>10.4f}\n"
        )
    return "".join(lines)


def render_report(results: Dict) -> str:
    lines: List[str] = []
    add = lines.append
    d1, d2, d3, bl = results["D1"], results["D2"], results["D3"], results["B_lite"]
    n = d1["n_survived_positions"]

    add("# AccPre — D1/D2/D3/B-lite diagnostic on acc_1a + stage1_pp.pt\n")
    add(f"n_survived_positions (test split, gamma=8) = {n}\n\n")

    # ------------------ D1
    add("## D1 — Predictor residual vs drafter-verifier mismatch\n\n")
    add("Correlations (|Q̂ − Q2| against mismatch measures):\n")
    c = d1["corr"]
    add(
        f"  |Q̂−Q2| vs |p−q|     : pearson={c['abs_res_vs_abs_pq']['pearson']:+.3f}  "
        f"spearman={c['abs_res_vs_abs_pq']['spearman']:+.3f}\n"
    )
    add(
        f"  |Q̂−Q2| vs |log p/q| : pearson={c['abs_res_vs_abs_log_pq']['pearson']:+.3f}  "
        f"spearman={c['abs_res_vs_abs_log_pq']['spearman']:+.3f}\n"
    )
    add(
        f"  |Q̂−Q2| vs p/q       : pearson={c['abs_res_vs_ratio']['pearson']:+.3f}  "
        f"spearman={c['abs_res_vs_ratio']['spearman']:+.3f}\n"
    )
    add(
        f"  (Q̂−Q2)² vs |p−q|   : pearson={c['sq_res_vs_abs_pq']['pearson']:+.3f}  "
        f"spearman={c['sq_res_vs_abs_pq']['spearman']:+.3f}\n"
    )
    add("\nResidual |Q̂−Q2| binned by decile of |p − q|:\n")
    add(_fmt_decile_table(d1["deciles_abs_pq"]))
    add("\nResidual |Q̂−Q2| binned by decile of |log(p/q)|:\n")
    add(_fmt_decile_table(d1["deciles_abs_log_pq"]))
    add("\nPer-position residual + local spearman(|Q̂−Q2|, |log p/q|):\n")
    add(f"  {'j':>3} {'n':>6} {'res_mean':>10} {'q2_mean':>10} "
        f"{'|logpq|mean':>12} {'spearman':>10}\n")
    for row in d1["per_j"]:
        if row["n"] == 0:
            add(f"  {row['j']:>3d} {0:>6d}      (empty)\n")
            continue
        add(
            f"  {row['j']:>3d} {row['n']:>6d} "
            f"{row['res_mean']:>10.4f} {row['q2_mean']:>10.4f} "
            f"{row['abs_logpq_mean']:>12.4f} "
            f"{row['corr_res_vs_abs_log_pq_spearman']:>10.3f}\n"
        )

    # ------------------ D2
    add("\n## D2 — Q2 structural distribution / commit headroom\n\n")
    s = d2["stats"]
    add(
        f"Q2 stats (survived positions): n={s['n']} mean={s['mean']:.4f} "
        f"std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}\n"
    )
    add(
        f"Q2 quantiles q05={s['q05']:.4f} q25={s['q25']:.4f} q50={s['q50']:.4f} "
        f"q75={s['q75']:.4f} q95={s['q95']:.4f}\n"
    )
    add("\nFraction of survived positions with Q2 >= τ:\n")
    f = d2["fraction_ge"]
    for k, v in f.items():
        add(f"  Q2 >= {k:>4s}  :  {v:.4f}\n")
    add("\nPer-position Q2 mean + commit headroom:\n")
    add(
        f"  {'j':>3} {'n':>6} {'mean':>8} {'≥0.5':>8} {'≥0.7':>8} "
        f"{'≥0.9':>8} {'=1':>8}\n"
    )
    for row in d2["per_j"]:
        if row["n"] == 0:
            add(f"  {row['j']:>3d} {0:>6d}  (empty)\n")
            continue
        add(
            f"  {row['j']:>3d} {row['n']:>6d} {row['mean']:>8.4f} "
            f"{row['frac_ge_0.5']:>8.4f} {row['frac_ge_0.7']:>8.4f} "
            f"{row['frac_ge_0.9']:>8.4f} {row['frac_eq_1']:>8.4f}\n"
        )

    # ------------------ D3
    add("\n## D3 — Drafter confidence vs mismatch / Q2 / residual\n\n")
    add("Correlations (spearman) — does drafter confidence predict anything?\n")
    add(
        f"  {'signal':>16} {'vs |p-q|':>12} {'vs |log p/q|':>14} "
        f"{'vs Q2':>10} {'vs |res|':>10}\n"
    )
    for sname, bt in d3["corrs"].items():
        add(
            f"  {sname:>16} "
            f"{bt['abs_pq']['spearman']:>12.3f} "
            f"{bt['abs_log_pq']['spearman']:>14.3f} "
            f"{bt['q2']['spearman']:>10.3f} "
            f"{bt['res_abs']['spearman']:>10.3f}\n"
        )
    add("\n|log p/q| binned by decile of drafter_entropy:\n")
    add(_fmt_decile_table(d3["deciles_de_vs_abs_log_pq"]))
    add("\nQ2 binned by decile of drafter_entropy:\n")
    add(_fmt_decile_table(d3["deciles_de_vs_q2"]))
    add("\nQ2 binned by decile of drafter_top1_prob:\n")
    add(_fmt_decile_table(d3["deciles_dt_vs_q2"]))

    # ------------------ B-lite
    add("\n## B-lite — p/q ratio and clipping structure\n\n")
    rs = bl["ratio_stats"]
    lrs = bl["log_ratio_stats"]
    frac = bl["fraction"]
    add(
        f"p/q stats (survived): mean={rs['mean']:.4f} std={rs['std']:.4f} "
        f"q05={rs['q05']:.4f} q50={rs['q50']:.4f} q95={rs['q95']:.4f}\n"
    )
    add(
        f"log(p/q) stats      : mean={lrs['mean']:+.4f} std={lrs['std']:.4f} "
        f"q05={lrs['q05']:+.4f} q50={lrs['q50']:+.4f} q95={lrs['q95']:+.4f}\n"
    )
    add("\nFractions:\n")
    add(f"  p/q > 1                      : {frac['ratio_gt_1']:.4f}   "
        f"(these clip to Q2 = 1)\n")
    add(f"  p/q ≥ 2                      : {frac['ratio_ge_2']:.4f}\n")
    add(f"  p/q ≤ 0.5                    : {frac['ratio_le_0_5']:.4f}\n")
    add(f"  p/q ≤ 0.1                    : {frac['ratio_le_0_1']:.4f}\n")
    add(f"  Q2 == 1 (clipping saturation): {frac['q2_eq_1_clipped']:.4f}\n")
    add(f"  Q2 < 0.01                    : {frac['q2_lt_0_01']:.4f}\n")
    add(f"  Q2 < 0.05                    : {frac['q2_lt_0_05']:.4f}\n")
    add(f"  Q2 ∈ (0.95, 1)               : {frac['q2_gt_0_95_not_clipped']:.4f}\n")

    return "".join(lines)


# ----------------------------------------------------------------------
# Plots (optional)
# ----------------------------------------------------------------------


def maybe_save_plots(d: Dict[str, np.ndarray], results: Dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[diag] matplotlib unavailable ({e}); skipping plots.")
        return

    # Plot 1: residual decile bars vs |log p/q|.
    rows = results["D1"]["deciles_abs_log_pq"]
    if rows:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        xs = [r["bin"] for r in rows]
        ys = [r["y_mean"] for r in rows]
        es = [r["y_std"] for r in rows]
        ax.bar(xs, ys, yerr=es, color="#4c78a8")
        ax.set_xlabel("|log(p/q)| decile")
        ax.set_ylabel("mean |Q̂ − Q2|")
        ax.set_title("D1: residual by |log(p/q)| decile")
        fig.tight_layout()
        fig.savefig(out_dir / "d1_residual_by_log_pq_decile.png", dpi=120)
        plt.close(fig)

    # Plot 2: Q2 histogram.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(d["q2"], bins=40, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Q2 = min(1, p/q)")
    ax.set_ylabel("count")
    ax.set_title("D2: Q2 distribution (survived positions)")
    fig.tight_layout()
    fig.savefig(out_dir / "d2_q2_hist.png", dpi=120)
    plt.close(fig)

    # Plot 3: per-position fraction Q2>=0.7.
    per_j = results["D2"]["per_j"]
    js = [r["j"] for r in per_j if r.get("n", 0) > 0]
    fracs = [r["frac_ge_0.7"] for r in per_j if r.get("n", 0) > 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(js, fracs, color="#4c78a8")
    ax.set_xlabel("draft position j")
    ax.set_ylabel("P(Q2 ≥ 0.7)")
    ax.set_title("D2: commit-headroom mass by position")
    fig.tight_layout()
    fig.savefig(out_dir / "d2_q2_ge_07_by_position.png", dpi=120)
    plt.close(fig)

    # Plot 4: log(p/q) histogram (clipped x-axis for readability).
    fig, ax = plt.subplots(figsize=(6, 3.5))
    lr = d["log_pq"]
    lo, hi = float(np.quantile(lr, 0.01)), float(np.quantile(lr, 0.99))
    ax.hist(lr.clip(lo, hi), bins=60, color="#4c78a8", edgecolor="white")
    ax.axvline(0.0, color="#e45756", linestyle="--", label="p = q")
    ax.set_xlabel("log(p/q)   (clipped to 1st–99th pct)")
    ax.set_ylabel("count")
    ax.set_title("B-lite: log(p/q) distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "blite_log_pq_hist.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--out_dir", type=str,
                    default=str(_REPO_ROOT / "outputs/diag_mismatch"))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] root   = {root}")
    print(f"[diag] out_dir = {out_dir}")

    test_records, preds, protocol = load_artifacts(root)
    print(
        f"[diag] test_records={len(test_records)}  "
        f"preds={len(preds)}  protocol_fp={protocol.fingerprint()[:60]}..."
    )

    d = flatten(test_records, preds)
    print(f"[diag] flattened survived positions: {len(d['q2'])}")

    results: Dict = {}
    results["n_survived_positions"] = int(len(d["q2"]))
    results["D1"] = d1_residual_vs_mismatch(d)
    results["D2"] = d2_q2_structure(d)
    results["D3"] = d3_drafter_confidence(d)
    results["B_lite"] = b_lite_clipping(d)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[diag] wrote {summary_path}")

    report_str = render_report(results)
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"[diag] wrote {report_path}")

    maybe_save_plots(d, results, out_dir)
    print(f"[diag] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
