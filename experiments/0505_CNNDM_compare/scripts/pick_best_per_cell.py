"""Stage 4 — pick the best (lr, max_epochs, seed) per (regime, target, arch).

Reads `train_history.json` from every cell under `runs/<regime>/<target>/<arch>/...`
and selects the cell with the lowest `best_val_weighted_mse` per
(regime, target, arch) triple. 240 cells -> 15 selections.

Tie-break order:
  1. Lower `best_val_weighted_mse`.
  2. Lower head_lr.
  3. Lower max_epochs.

Collapse filter:
  A cell is flagged "collapsed" if both `val_pred_mean < 1e-3` and
  `val_pred_std < 1e-3` at its best epoch. The collapse signal is computed
  from the last `history` record's `val_pred_mean`/`val_pred_std` whose
  `epoch == best_epoch`.

  When ranking, we PREFER non-collapsed siblings: if there is at least one
  non-collapsed sibling for a given (regime, target, arch), the selection is
  restricted to non-collapsed cells. Otherwise (all siblings collapsed), we
  still select the lowest-loss collapsed cell and tag the selection.

Outputs:
  results/best_per_cell.json
    {
      "frozen.relmax.mlp_pos": {
          "run_dir": "runs/frozen/relmax/mlp_pos/lr1e-3__ep5__s0",
          "best_val_weighted_mse": 0.0832,
          "best_epoch": 4,
          "lr": 0.001,
          "max_epochs": 5,
          "seed": 0,
          "collapsed": false,
          "n_siblings_total": 16,
          "n_siblings_collapsed": 7,
          "n_siblings_completed": 16
      },
      ...
    }
  results/best_per_cell_report.md
    Human-readable summary with: per-triple table; warnings for collapsed
    triples; warnings for missing/incomplete cells.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]


COLLAPSE_PRED_MEAN_THRESHOLD = 1e-3
COLLAPSE_PRED_STD_THRESHOLD = 1e-3


@dataclass
class CellResult:
    regime: str
    target: str
    arch: str
    lr: float
    max_epochs: int
    seed: int
    run_dir: Path
    best_val: float
    best_epoch: int
    val_pred_mean_at_best: Optional[float]
    val_pred_std_at_best: Optional[float]
    collapsed: bool
    has_history: bool = True


@dataclass
class TripleSelection:
    regime: str
    target: str
    arch: str
    selected: Optional[CellResult]
    n_total: int
    n_completed: int
    n_collapsed: int
    selected_among: str  # "non_collapsed" or "all_collapsed" or "none"
    siblings: List[CellResult] = field(default_factory=list)


def _load_train_history(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _val_pred_at_best(th: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return (val_pred_mean, val_pred_std) at the best epoch, or (None, None)."""
    best_epoch = th.get("best_epoch")
    history = th.get("history", []) or []
    if best_epoch is None:
        return None, None
    for rec in history:
        if int(rec.get("epoch", -1)) == int(best_epoch):
            m = rec.get("val_pred_mean")
            s = rec.get("val_pred_std")
            return (None if m is None else float(m),
                    None if s is None else float(s))
    return None, None


def _is_collapsed(pred_mean: Optional[float], pred_std: Optional[float]) -> bool:
    if pred_mean is None or pred_std is None:
        return False
    if not math.isfinite(pred_mean) or not math.isfinite(pred_std):
        return False
    return (
        pred_mean < COLLAPSE_PRED_MEAN_THRESHOLD
        and pred_std < COLLAPSE_PRED_STD_THRESHOLD
    )


def _scan_runs(runs_root: Path) -> List[CellResult]:
    """Scan the runs tree and return one CellResult per discovered cell.

    Discovery: any directory matching
       runs_root/<regime>/<target>/<arch>/<run_name>/train_history.json
    is treated as a cell. Run-name parsing is best-effort; missing fields
    fall back to values from the train_history.json or the materialised
    config.yaml.
    """
    results: List[CellResult] = []
    for th_path in sorted(runs_root.rglob("train_history.json")):
        run_dir = th_path.parent
        # Skip _smoke or other reserved subdirs.
        rel = run_dir.relative_to(runs_root)
        if rel.parts and rel.parts[0].startswith("_"):
            continue
        if len(rel.parts) < 4:
            continue
        regime, target, arch = rel.parts[0], rel.parts[1], rel.parts[2]
        # run_name is the last component; we read lr/ep/seed from config.
        cfg_path = run_dir / "_config_materialized.yaml"
        if not cfg_path.exists():
            cfg_path = run_dir / "config.yaml"
        lr: Optional[float] = None
        max_epochs: Optional[int] = None
        seed: Optional[int] = None
        if cfg_path.exists():
            try:
                import yaml
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                lr = float(cfg.get("head_lr"))
                max_epochs = int(cfg.get("max_epochs"))
                seed = int(cfg.get("seed", 0))
            except Exception:
                pass
        th = _load_train_history(th_path)
        if th is None:
            results.append(CellResult(
                regime=regime, target=target, arch=arch,
                lr=lr or 0.0, max_epochs=max_epochs or 0, seed=seed or 0,
                run_dir=run_dir, best_val=float("inf"), best_epoch=-1,
                val_pred_mean_at_best=None, val_pred_std_at_best=None,
                collapsed=False, has_history=False,
            ))
            continue
        bv = th.get("best_val_weighted_mse")
        be = th.get("best_epoch")
        if bv is None or be is None or not math.isfinite(float(bv)):
            continue
        m, s = _val_pred_at_best(th)
        results.append(CellResult(
            regime=regime, target=target, arch=arch,
            lr=lr or 0.0, max_epochs=max_epochs or 0, seed=seed or 0,
            run_dir=run_dir,
            best_val=float(bv), best_epoch=int(be),
            val_pred_mean_at_best=m, val_pred_std_at_best=s,
            collapsed=_is_collapsed(m, s),
        ))
    return results


def _rank_key(c: CellResult) -> Tuple[float, float, int]:
    """Sort key: (best_val asc, lr asc, max_epochs asc). seed not used for tie-break."""
    return (c.best_val, c.lr, c.max_epochs)


def _select_per_triple(cells: List[CellResult]) -> List[TripleSelection]:
    """Group cells by (regime, target, arch) and pick best per group.

    Prefers non-collapsed cells when available; otherwise falls back to the
    lowest-loss collapsed cell.
    """
    groups: Dict[Tuple[str, str, str], List[CellResult]] = {}
    for c in cells:
        if not c.has_history:
            continue
        groups.setdefault((c.regime, c.target, c.arch), []).append(c)
    selections: List[TripleSelection] = []
    for key, cs in sorted(groups.items()):
        regime, target, arch = key
        non_collapsed = [c for c in cs if not c.collapsed]
        n_total = len(cs)
        n_collapsed = sum(1 for c in cs if c.collapsed)
        if non_collapsed:
            best = min(non_collapsed, key=_rank_key)
            among = "non_collapsed"
        elif cs:
            best = min(cs, key=_rank_key)
            among = "all_collapsed"
        else:
            best = None
            among = "none"
        selections.append(TripleSelection(
            regime=regime, target=target, arch=arch,
            selected=best,
            n_total=n_total, n_completed=n_total,
            n_collapsed=n_collapsed,
            selected_among=among,
            siblings=cs,
        ))
    return selections


def _expected_triples() -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for t in ("relmax", "alpha_q2", "dep"):
        for a in ("mlp_pos", "causal_transformer_pos", "bidirectional_transformer_pos"):
            out.append(("frozen", t, a))
    for t in ("relmax", "alpha_q2", "dep"):
        for a in ("causal_transformer_pos", "bidirectional_transformer_pos"):
            out.append(("joint", t, a))
    return out


def _write_json(selections: List[TripleSelection], out_path: Path) -> None:
    obj: Dict[str, Any] = {}
    for sel in selections:
        key = f"{sel.regime}.{sel.target}.{sel.arch}"
        if sel.selected is None:
            obj[key] = {
                "run_dir": None,
                "best_val_weighted_mse": None,
                "best_epoch": None,
                "lr": None, "max_epochs": None, "seed": None,
                "collapsed": None,
                "selected_among": sel.selected_among,
                "n_siblings_total": sel.n_total,
                "n_siblings_completed": sel.n_completed,
                "n_siblings_collapsed": sel.n_collapsed,
            }
            continue
        c = sel.selected
        obj[key] = {
            "run_dir": str(c.run_dir),
            "best_val_weighted_mse": float(c.best_val),
            "best_epoch": int(c.best_epoch),
            "lr": float(c.lr),
            "max_epochs": int(c.max_epochs),
            "seed": int(c.seed),
            "collapsed": bool(c.collapsed),
            "val_pred_mean_at_best": c.val_pred_mean_at_best,
            "val_pred_std_at_best": c.val_pred_std_at_best,
            "selected_among": sel.selected_among,
            "n_siblings_total": sel.n_total,
            "n_siblings_completed": sel.n_completed,
            "n_siblings_collapsed": sel.n_collapsed,
        }
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_report(selections: List[TripleSelection], expected: List[Tuple[str, str, str]],
                  cells: List[CellResult], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# pick_best_per_cell — selection report")
    lines.append("")
    lines.append(f"Scanned {len(cells)} cells; produced {sum(1 for s in selections if s.selected)}/{len(expected)} selections.")
    lines.append("")

    found = {(s.regime, s.target, s.arch) for s in selections}
    missing = [t for t in expected if t not in found]
    if missing:
        lines.append("## ❌ Missing triples (no completed cells)")
        for t in missing:
            lines.append(f"- `{t[0]}.{t[1]}.{t[2]}`")
        lines.append("")

    all_collapsed = [s for s in selections if s.selected_among == "all_collapsed"]
    if all_collapsed:
        lines.append("## ⚠️ Triples where ALL siblings collapsed")
        lines.append("")
        lines.append("(predict-near-zero local minimum; selected cell is the lowest-loss collapsed sibling)")
        lines.append("")
        for s in all_collapsed:
            c = s.selected
            lines.append(
                f"- `{s.regime}.{s.target}.{s.arch}` — best val={c.best_val:.5f} at "
                f"`{c.run_dir.name}` (lr={c.lr:g}, ep={c.max_epochs}, "
                f"val_pred_mean={c.val_pred_mean_at_best:.2e}, "
                f"val_pred_std={c.val_pred_std_at_best:.2e})"
            )
        lines.append("")

    has_collapsed = [s for s in selections
                     if s.selected_among == "non_collapsed" and s.n_collapsed > 0]
    if has_collapsed:
        lines.append("## ℹ️ Triples with some collapsed siblings (filtered out)")
        lines.append("")
        for s in has_collapsed:
            lines.append(
                f"- `{s.regime}.{s.target}.{s.arch}`: "
                f"{s.n_collapsed}/{s.n_total} siblings collapsed; "
                f"selected non-collapsed cell `{s.selected.run_dir.name}` "
                f"(best_val={s.selected.best_val:.5f})"
            )
        lines.append("")

    # Full selection table.
    lines.append("## Selection table")
    lines.append("")
    lines.append("| triple | run_dir | best_val | best_epoch | lr | max_epochs | "
                 "val_pred_mean | val_pred_std | collapsed | selected_among |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in selections:
        if s.selected is None:
            lines.append(
                f"| `{s.regime}.{s.target}.{s.arch}` | (none) | - | - | - | - | - | - | - | {s.selected_among} |"
            )
            continue
        c = s.selected
        m = (f"{c.val_pred_mean_at_best:.3e}" if c.val_pred_mean_at_best is not None else "-")
        st = (f"{c.val_pred_std_at_best:.3e}" if c.val_pred_std_at_best is not None else "-")
        lines.append(
            f"| `{s.regime}.{s.target}.{s.arch}` | `{c.run_dir.name}` | {c.best_val:.5f} | "
            f"{c.best_epoch} | {c.lr:g} | {c.max_epochs} | {m} | {st} | "
            f"{'YES' if c.collapsed else 'no'} | {s.selected_among} |"
        )
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=str(_EXP_ROOT / "runs"),
                    help="root containing <regime>/<target>/<arch>/<cell>/")
    ap.add_argument("--out_json", default=str(_EXP_ROOT / "results" / "best_per_cell.json"))
    ap.add_argument("--out_report", default=str(_EXP_ROOT / "results" / "best_per_cell_report.md"))
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"[pick] ERROR runs_root does not exist: {runs_root}")
        return 2

    cells = _scan_runs(runs_root)
    print(f"[pick] scanned {len(cells)} cells under {runs_root}")
    if not cells:
        print("[pick] no cells found; nothing to do.")
        return 1

    selections = _select_per_triple(cells)
    expected = _expected_triples()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    _write_json(selections, Path(args.out_json))
    _write_report(selections, expected, cells, Path(args.out_report))
    print(f"[pick] wrote {args.out_json}")
    print(f"[pick] wrote {args.out_report}")
    n_sel = sum(1 for s in selections if s.selected is not None)
    n_collapsed = sum(1 for s in selections if s.selected_among == "all_collapsed")
    print(f"[pick] selections={n_sel}/{len(expected)} all-collapsed-triples={n_collapsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
