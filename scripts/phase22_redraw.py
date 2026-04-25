"""Re-draw Phase 22 Pareto figures from the existing unified_table.json.

No compute. Produces:
  <out_dir>/pareto_delta_nll_tok_s_clean.png      (Plot 1 full, emphasised)
  <out_dir>/pareto_tok_succ_tok_s_clean.png       (Plot 2 full, emphasised)
  <out_dir>/pareto_delta_nll_frontier.png         (Plot 1 frontier-only)

Styling rules:
  - strict: big black diamond, thick edge, zorder high, offset annotation.
  - oracle_exec: orange w/ black edge.
  - frozen_1A: clean blue.
  - live_jnt_10ep: clean red.
  - lazy_jnt, live_jnt_5ep: pale/translucent, no labels on full Plot 1
    unless the reader would miss them otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent


# Visual tiers.
TIER_PRIMARY   = {"strict", "oracle_exec", "frozen_1A", "live_jnt_10ep"}
TIER_SECONDARY = {"lazy_jnt", "live_jnt_5ep"}

COLORS: Dict[str, str] = {
    "strict":        "#000000",   # black
    "oracle_exec":   "#f58518",   # orange
    "frozen_1A":     "#3b76b3",   # blue
    "live_jnt_10ep": "#d22b2b",   # red
    "lazy_jnt":      "#a9a9a9",   # light gray
    "live_jnt_5ep":  "#7fb07a",   # pale green
}
MARKERS_BY_TAU = {0.5: "o", 0.7: "s", 0.9: "^", None: "D"}

# Labels to ALWAYS draw in Plot 1 (full).
LABEL_ALWAYS = {
    ("strict",       None),
    ("oracle_exec",  0.7),
    ("oracle_exec",  0.9),
    ("oracle_exec",  0.5),
    ("frozen_1A",    0.5),
    ("frozen_1A",    0.7),
    ("frozen_1A",    0.9),
    ("live_jnt_10ep",0.7),
    ("live_jnt_10ep",0.9),
    ("live_jnt_10ep",0.5),
}

# For Plot 2 (tok_succ × tok/s), strict is excluded (no tok_succ).
LABEL_ALWAYS_SUCC = {
    ("oracle_exec",  0.7),
    ("oracle_exec",  0.9),
    ("oracle_exec",  0.5),
    ("frozen_1A",    0.5),
    ("frozen_1A",    0.7),
    ("frozen_1A",    0.9),
    ("live_jnt_10ep",0.5),
    ("live_jnt_10ep",0.7),
    ("live_jnt_10ep",0.9),
}

# Frontier-only plot method set.
FRONTIER_ROWS = [
    ("strict",        None),
    ("oracle_exec",   0.7),
    ("frozen_1A",     0.5),
    ("frozen_1A",     0.7),
    ("live_jnt_10ep", 0.7),
    ("live_jnt_10ep", 0.9),
]


def _style(row: Dict) -> Dict:
    """Per-point visual parameters."""
    method = row["method"]
    tau = row["tau"]
    color = COLORS.get(method, "#888888")
    marker = MARKERS_BY_TAU.get(tau if isinstance(tau, (int, float)) or tau is None else None,
                                "x")
    if method == "strict":
        return dict(
            s=260, color=color, marker="D",
            edgecolor="black", linewidth=2.0,
            alpha=1.0, zorder=10,
        )
    if method in TIER_SECONDARY:
        return dict(
            s=55, color=color, marker=marker,
            edgecolor="none", alpha=0.45, zorder=2,
        )
    # Primary tier: oracle_exec / frozen_1A / live_jnt_10ep
    edge = "black" if method == "oracle_exec" else None
    lw = 1.1 if method == "oracle_exec" else 0.0
    return dict(
        s=115, color=color, marker=marker,
        edgecolor=edge, linewidth=lw,
        alpha=1.0, zorder=5,
    )


def _annotate(ax, row, xy, is_strict=False, fontsize=9.2, secondary=False, custom=None):
    method = row["method"]
    tau = row["tau"]
    if secondary:
        return
    if custom is not None:
        offset, text = custom
    else:
        if is_strict:
            offset = (22, 14)
            text = "strict"
        else:
            offset = (7, 7)
            text = f"{method}\nτ={tau}"
    color = COLORS.get(method, "#555555")
    if method == "strict":
        # Extra emphasis: boxed, black text.
        ax.annotate(
            text, xy, textcoords="offset points", xytext=offset,
            fontsize=fontsize + 1, color="black", weight="bold",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="black", linewidth=1.0),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
        )
        return
    ax.annotate(
        text, xy, textcoords="offset points", xytext=offset,
        fontsize=fontsize, color=color, alpha=0.95, zorder=6,
    )


def _legend(ax, show_strict=True, show_secondary=True):
    from matplotlib.lines import Line2D
    h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               label="τ = 0.5", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#555",
               label="τ = 0.7", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#555",
               label="τ = 0.9", markersize=8),
    ]
    if show_strict:
        h.append(Line2D(
            [0], [0], marker="D", color="w",
            markerfacecolor="black", markeredgecolor="black",
            markeredgewidth=1.5, label="strict", markersize=10,
        ))
    for name in ("oracle_exec", "frozen_1A", "live_jnt_10ep"):
        color = COLORS[name]
        h.append(Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=color,
                        markeredgecolor="black" if name == "oracle_exec" else color,
                        markeredgewidth=1.0 if name == "oracle_exec" else 0.0,
                        label=name, markersize=8))
    if show_secondary:
        for name in ("lazy_jnt", "live_jnt_5ep"):
            color = COLORS[name]
            h.append(Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=color, label=f"{name} (de-emph.)",
                            markersize=7, alpha=0.6))
    ax.legend(handles=h, loc="best", fontsize=8.5, framealpha=0.92)


def plot_full_nll(rows: List[Dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # Axis guides (lighter so markers aren't masked).
    ax.axvline(0.0, linestyle="--", color="#d8d8d8", linewidth=1, zorder=1)
    strict_rows = [r for r in rows if r["method"] == "strict"]
    if strict_rows:
        ax.axhline(strict_rows[0]["tok_s"], linestyle=":",
                   color="#e0e0e0", linewidth=1, zorder=1)

    # Annotation offsets for selected primary points (keep from colliding).
    custom_offsets = {
        ("strict",        None): (22, 14),
        ("oracle_exec",   0.5): (8, 8),
        ("oracle_exec",   0.7): (-85, 10),
        ("oracle_exec",   0.9): (-85, -5),
        ("frozen_1A",     0.5): (10, -20),
        ("frozen_1A",     0.7): (-110, 10),
        ("frozen_1A",     0.9): (-80, -20),
        ("live_jnt_10ep", 0.5): (8, 10),
        ("live_jnt_10ep", 0.7): (10, -20),
        ("live_jnt_10ep", 0.9): (12, 14),
    }

    # Plot secondary tier FIRST so primary draws on top.
    for row in rows:
        if row["method"] not in TIER_SECONDARY:
            continue
        x, y = row["delta_NLL"], row["tok_s"]
        if x is None or y is None:
            continue
        ax.scatter(x, y, **_style(row))

    # Plot primary tier.
    for row in rows:
        if row["method"] in TIER_SECONDARY:
            continue
        x, y = row["delta_NLL"], row["tok_s"]
        if x is None or y is None:
            continue
        ax.scatter(x, y, **_style(row))

    # Annotations.
    for row in rows:
        if row["method"] in TIER_SECONDARY:
            continue
        key = (row["method"], row["tau"])
        if key not in LABEL_ALWAYS:
            continue
        x, y = row["delta_NLL"], row["tok_s"]
        if x is None or y is None:
            continue
        off = custom_offsets.get(key, (7, 7))
        if row["method"] == "strict":
            text = "strict"
        else:
            text = f"{row['method']}\nτ={row['tau']}"
        _annotate(ax, row, (x, y), is_strict=(row["method"] == "strict"),
                  custom=(off, text))

    ax.set_xlabel("Δ NLL vs strict  (↓ better; strict at 0)", fontsize=11)
    ax.set_ylabel("tok/s  (↑ better)", fontsize=11)
    ax.set_title(
        "Pareto: throughput vs quality gap to strict (20 prompts)",
        fontsize=12, pad=8,
    )
    # Extra headroom so the highest label doesn't clip.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 2.5)
    ax.grid(True, alpha=0.25)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"[redraw] wrote {out_path}")


def plot_full_tok_succ(rows: List[Dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    custom_offsets = {
        ("oracle_exec",   0.5): (10, -18),
        ("oracle_exec",   0.7): (10, 6),
        ("oracle_exec",   0.9): (10, -18),
        ("frozen_1A",     0.5): (10, -18),
        ("frozen_1A",     0.7): (-105, -4),
        ("frozen_1A",     0.9): (-100, 8),
        ("live_jnt_10ep", 0.5): (10, 8),
        ("live_jnt_10ep", 0.7): (10, 8),
        ("live_jnt_10ep", 0.9): (-110, -4),
    }

    for row in rows:
        if row["method"] == "strict":
            continue
        x = row.get("tok_succ")
        y = row.get("tok_s")
        if x is None or y is None:
            continue
        # Secondary first, then primary.
    for tier_secondary in (True, False):
        for row in rows:
            if row["method"] == "strict":
                continue
            if (row["method"] in TIER_SECONDARY) != tier_secondary:
                continue
            x = row.get("tok_succ"); y = row.get("tok_s")
            if x is None or y is None:
                continue
            ax.scatter(x, y, **_style(row))
            if not tier_secondary:
                key = (row["method"], row["tau"])
                if key in LABEL_ALWAYS_SUCC:
                    off = custom_offsets.get(key, (7, 7))
                    text = f"{row['method']}\nτ={row['tau']}"
                    _annotate(ax, row, (x, y), custom=(off, text))

    ax.set_xlabel("token-level verifier success rate  (↑ better)", fontsize=11)
    ax.set_ylabel("tok/s  (↑ better)", fontsize=11)
    ax.set_title(
        "Token-level verifier success vs throughput (20 prompts)",
        fontsize=12, pad=8,
    )
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 2.5)
    ax.grid(True, alpha=0.25)
    _legend(ax, show_strict=False)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"[redraw] wrote {out_path}")


def plot_frontier_only(rows: List[Dict], out_path: Path) -> None:
    """Frontier-only Plot 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep_keys = {(m, tau) for (m, tau) in FRONTIER_ROWS}
    sub = [r for r in rows if (r["method"], r["tau"]) in keep_keys]
    sub_map = {(r["method"], r["tau"]): r for r in sub}

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.axvline(0.0, linestyle="--", color="#d8d8d8", linewidth=1, zorder=1)
    strict_row = sub_map.get(("strict", None))
    if strict_row is not None:
        ax.axhline(strict_row["tok_s"], linestyle=":",
                   color="#e0e0e0", linewidth=1, zorder=1)

    # Sort by tok/s descending for a nice reading order.
    ordered = sorted(sub, key=lambda r: -r["tok_s"])

    custom_offsets = {
        ("strict",        None): (22, 14),
        ("oracle_exec",   0.7): (10, -20),
        ("frozen_1A",     0.5): (12, -6),
        ("frozen_1A",     0.7): (12, -6),
        ("live_jnt_10ep", 0.7): (12, 8),
        ("live_jnt_10ep", 0.9): (12, -6),
    }

    for row in ordered:
        x, y = row["delta_NLL"], row["tok_s"]
        if x is None or y is None:
            continue
        ax.scatter(x, y, **_style(row))

    # Annotate all frontier points (bigger fontsize).
    for row in ordered:
        x, y = row["delta_NLL"], row["tok_s"]
        if x is None or y is None:
            continue
        key = (row["method"], row["tau"])
        off = custom_offsets.get(key, (8, 8))
        if row["method"] == "strict":
            text = "strict"
        else:
            text = f"{row['method']}  τ={row['tau']}"
        _annotate(
            ax, row, (x, y),
            is_strict=(row["method"] == "strict"),
            custom=(off, text),
            fontsize=10,
        )

    # Connect the frontier with a light dashed line to suggest the curve.
    # Order by tok/s descending (= along the frontier from fast/low-|ΔNLL| to slow/highest-quality).
    frontier_order = [
        ("frozen_1A", 0.5), ("live_jnt_10ep", 0.7),
        ("live_jnt_10ep", 0.9), ("strict", None),
    ]
    xs = []; ys = []
    for key in frontier_order:
        r = sub_map.get(key)
        if r is None:
            continue
        xs.append(r["delta_NLL"])
        ys.append(r["tok_s"])
    if len(xs) >= 2:
        ax.plot(
            xs, ys, linestyle="--", color="#aaaaaa", linewidth=1.2,
            zorder=1,
        )

    ax.set_xlabel("Δ NLL vs strict  (↓ better; strict at 0)", fontsize=11)
    ax.set_ylabel("tok/s  (↑ better)", fontsize=11)
    ax.set_title(
        "Pareto frontier (20 prompts) — selected points",
        fontsize=12, pad=8,
    )
    # Widen x range and give y headroom for labels.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - 0.04, x1 + 0.13)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 1.2, y1 + 2.5)
    ax.grid(True, alpha=0.25)
    _legend(ax, show_strict=True, show_secondary=False)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"[redraw] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input_json", type=str,
        default="outputs/phase22_unified_11872173/unified_table.json",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    input_path = (_REPO_ROOT / args.input_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)
    rows = data["rows"]
    print(f"[redraw] loaded {len(rows)} rows from {input_path}")

    plot_full_nll(rows, out_dir / "pareto_delta_nll_tok_s_clean.png")
    plot_full_tok_succ(rows, out_dir / "pareto_tok_succ_tok_s_clean.png")
    plot_frontier_only(rows, out_dir / "pareto_delta_nll_frontier.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
