"""Summarise predictor sweeps into a leaderboard.

Reads all `<sweep_dir>/runs/*/summary.json` (or a provided JSONL) and
emits:

  leaderboard.csv   — wide table, one row per run
  leaderboard.md    — ranked markdown view (best per family × dataset)
  leaderboard.json  — machine-readable ranked entries

Also prints a compact best-per-family table to stdout.

Ranking:
  primary   : min delta_nll           (quality vs strict)
  tiebreak  : max tok_s_mean          (speed)

You can override --primary (min|max) and --primary_key to rank by any
numeric field present in the summary.json files (e.g. `best_val`,
`tok_succ`).

Usage:
    python scripts/sweep_summarize.py --sweep_dir outputs/sweeps/NAME
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_FIELDS: List[str] = [
    "family", "dataset", "run_name", "run_dir",
    "lr", "epochs", "batch_size", "seed",
    "tau", "rule", "gamma", "T",
    "n_eval_prompts", "max_new_tokens",
    # training-side
    "best_val", "test_loss", "final_train_loss", "final_val_loss",
    "n_epochs_run",
    # online-side
    "tok_s_mean", "nll", "strict_nll", "delta_nll",
    "tok_succ", "rnd_mean", "all_pass",
    "decode_s", "offline_s",
    "status",
]


def _load_summaries(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Load per-(training, tau) summaries from a sweep directory.

    Two layouts are accepted:
      (a) multi-tau: <sweep>/runs/<name>/eval/tau_*/summary.json
      (b) legacy single-tau: <sweep>/runs/<name>/summary.json
    """
    out: List[Dict[str, Any]] = []
    runs_root = sweep_dir / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(
            f"expected {runs_root} to be a directory of per-run folders"
        )
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        eval_dir = d / "eval"
        found_multi = False
        if eval_dir.is_dir():
            for td in sorted(eval_dir.iterdir()):
                sp = td / "summary.json"
                if sp.is_file():
                    with open(sp) as f:
                        row = json.load(f)
                    row.setdefault("run_name", f"{d.name}__{td.name}")
                    row.setdefault("run_dir", str(d))
                    out.append(row)
                    found_multi = True
        if not found_multi:
            sp = d / "summary.json"
            if sp.is_file():
                with open(sp) as f:
                    row = json.load(f)
                row.setdefault("run_name", d.name)
                row.setdefault("run_dir", str(d))
                out.append(row)
    return out


def _rank_key(row: Dict[str, Any], primary_key: str, primary: str) -> Tuple:
    pv = row.get(primary_key)
    tv = row.get("tok_s_mean")
    # Push missing values to the end.
    missing = pv is None
    pv_key = (float(pv) if pv is not None else float("inf"))
    if primary == "max":
        pv_key = -pv_key
    tv_key = -(float(tv) if tv is not None else float("-inf"))
    return (missing, pv_key, tv_key)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [c for c in _FIELDS if any(c in r for r in rows)] or _FIELDS
    # Include any extra keys not in _FIELDS so nothing is lost.
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v: Any, w: int = 8, p: int = 4) -> str:
    if v is None:
        return f"{'—':>{w}}"
    if isinstance(v, (int,)):
        return f"{v:>{w}d}"
    if isinstance(v, float):
        return f"{v:>{w}.{p}f}"
    return f"{str(v):>{w}}"


def _render_markdown(
    rows_sorted: List[Dict[str, Any]],
    primary_key: str, primary: str,
    sweep_dir: Path,
) -> str:
    out: List[str] = []
    out.append(f"# Sweep leaderboard — {sweep_dir.name}\n\n")
    out.append(
        f"Ranked by **{primary} {primary_key}**; tiebreak max tok_s_mean. "
        f"{len(rows_sorted)} runs.\n\n"
    )
    # Group heading: best per (family, dataset).
    best_per_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows_sorted:
        key = (r.get("family", "?"), r.get("dataset", "?"))
        if key not in best_per_pair:
            best_per_pair[key] = r
    out.append("## Best run per (family, dataset)\n\n")
    out.append("```\n")
    out.append(
        f"  {'family':<14}{'dataset':<10}{'tau':>6}{'lr':>10}{'ep':>5}"
        f"{'bs':>5}{'seed':>5}{'best_val':>10}{'delta_nll':>11}"
        f"{'tok/s':>8}{'tok_succ':>10}\n"
    )
    for (fam, ds), r in best_per_pair.items():
        out.append(
            f"  {str(fam):<14}{str(ds):<10}"
            f"{_fmt(r.get('tau'), 6, 2)}"
            f"{_fmt(r.get('lr'), 10, 6)}"
            f"{_fmt(r.get('epochs'), 5, 0)}"
            f"{_fmt(r.get('batch_size') or 0, 5, 0)}"
            f"{_fmt(r.get('seed'), 5, 0)}"
            f"{_fmt(r.get('best_val'), 10, 5)}"
            f"{_fmt(r.get('delta_nll'), 11, 4)}"
            f"{_fmt(r.get('tok_s_mean'), 8, 2)}"
            f"{_fmt(r.get('tok_succ'), 10, 4)}\n"
        )
    out.append("```\n\n")

    # Best per (family, dataset, tau) — what the user specifically asks for.
    best_per_triple: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for r in rows_sorted:
        key = (r.get("family", "?"), r.get("dataset", "?"),
               float(r.get("tau", float("nan"))))
        if key not in best_per_triple:
            best_per_triple[key] = r
    out.append("## Best run per (family, dataset, tau)\n\n")
    out.append("```\n")
    out.append(
        f"  {'family':<14}{'dataset':<10}{'tau':>6}{'lr':>10}{'ep':>5}"
        f"{'bs':>5}{'seed':>5}{'best_val':>10}{'test_loss':>11}"
        f"{'delta_nll':>11}{'tok/s':>8}{'tok_succ':>10}"
        f"{'rnd_mean':>10}{'all_pass':>10}\n"
    )
    for (fam, ds, tau), r in sorted(best_per_triple.items()):
        out.append(
            f"  {str(fam):<14}{str(ds):<10}"
            f"{_fmt(tau, 6, 2)}"
            f"{_fmt(r.get('lr'), 10, 6)}"
            f"{_fmt(r.get('epochs'), 5, 0)}"
            f"{_fmt(r.get('batch_size') or 0, 5, 0)}"
            f"{_fmt(r.get('seed'), 5, 0)}"
            f"{_fmt(r.get('best_val'), 10, 5)}"
            f"{_fmt(r.get('test_loss'), 11, 5)}"
            f"{_fmt(r.get('delta_nll'), 11, 4)}"
            f"{_fmt(r.get('tok_s_mean'), 8, 2)}"
            f"{_fmt(r.get('tok_succ'), 10, 4)}"
            f"{_fmt(r.get('rnd_mean'), 10, 4)}"
            f"{_fmt(r.get('all_pass'), 10, 4)}\n"
        )
    out.append("```\n\n")

    out.append("## All runs (ranked)\n\n")
    out.append("```\n")
    out.append(
        f"  {'rank':>4}  {'family':<14}{'dataset':<10}"
        f"{'tau':>6}{'lr':>10}{'ep':>5}{'bs':>5}{'seed':>5}"
        f"{'best_val':>10}{'test_loss':>11}"
        f"{'delta_nll':>11}{'tok/s':>8}{'tok_succ':>10}"
        f"{'rnd_mean':>10}{'all_pass':>10}{'status':>10}\n"
    )
    for i, r in enumerate(rows_sorted, 1):
        out.append(
            f"  {i:>4}  {str(r.get('family', '?')):<14}"
            f"{str(r.get('dataset', '?')):<10}"
            f"{_fmt(r.get('tau'), 6, 2)}"
            f"{_fmt(r.get('lr'), 10, 6)}"
            f"{_fmt(r.get('epochs'), 5, 0)}"
            f"{_fmt(r.get('batch_size') or 0, 5, 0)}"
            f"{_fmt(r.get('seed'), 5, 0)}"
            f"{_fmt(r.get('best_val'), 10, 5)}"
            f"{_fmt(r.get('test_loss'), 11, 5)}"
            f"{_fmt(r.get('delta_nll'), 11, 4)}"
            f"{_fmt(r.get('tok_s_mean'), 8, 2)}"
            f"{_fmt(r.get('tok_succ'), 10, 4)}"
            f"{_fmt(r.get('rnd_mean'), 10, 4)}"
            f"{_fmt(r.get('all_pass'), 10, 4)}"
            f"  {str(r.get('status', '?')):<10}\n"
        )
    out.append("```\n")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep_dir", required=True, type=str)
    ap.add_argument("--primary_key", type=str, default="delta_nll",
                    help="Metric to rank by (column in summary.json).")
    ap.add_argument("--primary", type=str, default="min",
                    choices=("min", "max"),
                    help="Whether the primary metric is minimised or "
                         "maximised.")
    ap.add_argument("--out_md", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    rows = _load_summaries(sweep_dir)
    if not rows:
        print(f"[sum] no runs under {sweep_dir}/runs/*/summary.json", file=sys.stderr)
        return 1

    rows_sorted = sorted(
        rows, key=lambda r: _rank_key(r, args.primary_key, args.primary),
    )

    md = _render_markdown(rows_sorted, args.primary_key, args.primary, sweep_dir)
    print(md)

    md_out = Path(args.out_md) if args.out_md else (sweep_dir / "leaderboard.md")
    csv_out = Path(args.out_csv) if args.out_csv else (sweep_dir / "leaderboard.csv")
    json_out = Path(args.out_json) if args.out_json else (sweep_dir / "leaderboard.json")
    md_out.write_text(md)
    _write_csv(csv_out, rows_sorted)
    with open(json_out, "w") as f:
        json.dump({
            "sweep_dir": str(sweep_dir),
            "primary": args.primary,
            "primary_key": args.primary_key,
            "runs": rows_sorted,
        }, f, indent=2, default=str)
    print(f"[sum] wrote {md_out}")
    print(f"[sum] wrote {csv_out}")
    print(f"[sum] wrote {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
