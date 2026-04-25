"""Grid driver for predictor sweeps.

Expands a YAML grid file into grid points and sequentially invokes
`scripts/sweep_train_eval.py` for each. The sweep directory layout:

    <sweep_dir>/
        runs/
            <run_name>/           # one per grid point; see sweep_train_eval
        expanded_grid.yaml        # the expansion used
        runs.jsonl                # one-line-per-run summary index

Run-name convention (no spaces, collision-free within a sweep):
    <family>__lr<lr>__ep<epochs>__bs<bs>__s<seed>

Usage:
    python scripts/sweep_grid.py \\
        --family frozen_alpha --dataset owt \\
        --grid configs/sweep/default_grid.yaml \\
        --sweep_dir outputs/sweeps/$(date +%Y%m%d)_frozen_alpha_owt

Defaults in the grid file can be overridden via --lr_grid / --epochs_grid
/ --batch_size_grid / --seeds on the CLI; see --help.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_grid(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _format_lr(lr: float) -> str:
    # 1e-3 -> "1e-3", 3e-4 -> "3e-4"
    mant, exp = f"{lr:.0e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp = str(int(exp))
    return f"{mant}e{exp}"


def _run_name(family: str, lr: float, epochs: int,
              bs: Optional[int], seed: int) -> str:
    bs_s = "def" if bs is None else str(bs)
    return (
        f"{family}__lr{_format_lr(lr)}__ep{epochs}__bs{bs_s}__s{seed}"
    )


def _expand(
    lr_grid: List[float], ep_grid: List[int],
    bs_grid: List[Optional[int]], seeds: List[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lr, ep, bs, s in itertools.product(lr_grid, ep_grid, bs_grid, seeds):
        out.append({"lr": lr, "epochs": ep, "batch_size": bs, "seed": s})
    return out


def _run_one(
    family: str, dataset: str, run_dir: Path, point: Dict[str, Any],
    eval_cfg: Dict[str, Any], decouple_schedule: bool,
    wandb_project: Optional[str], wandb_entity: Optional[str],
) -> int:
    taus = eval_cfg.get("taus")
    if taus is None:
        # Back-compat: accept a single `tau`.
        taus = [eval_cfg.get("tau", 0.5)]
    taus = [float(x) for x in taus]
    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts/sweep_train_eval.py"),
        "--family", family, "--dataset", dataset,
        "--run_dir", str(run_dir),
        "--run_name", run_dir.name,
        "--lr", str(float(point["lr"])),
        "--epochs", str(int(point["epochs"])),
        "--seed", str(int(point["seed"])),
        "--n_eval_prompts", str(int(eval_cfg.get("n_prompts", 10))),
        "--max_new_tokens", str(int(eval_cfg.get("max_new_tokens", 1024))),
        "--gamma", str(int(eval_cfg.get("gamma", 8))),
        "--T", str(int(eval_cfg.get("T", 2))),
        "--taus", *(str(t) for t in taus),
        "--rule", str(eval_cfg.get("rule", "threshold")),
    ]
    if point.get("batch_size") is not None:
        cmd.extend(["--batch_size", str(int(point["batch_size"]))])
    if decouple_schedule:
        cmd.append("--decouple_schedule")
    if wandb_project:
        cmd.extend(["--wandb_project", wandb_project])
    if wandb_entity:
        cmd.extend(["--wandb_entity", wandb_entity])
    print(f"[grid] {'='*60}\n[grid] → {run_dir.name}")
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sweep_dir", required=True, type=str)
    ap.add_argument("--grid", type=str,
                    default="configs/sweep/default_grid.yaml")

    # CLI-level grid overrides (optional). Space-separated lists.
    ap.add_argument("--lr_grid", nargs="+", type=float, default=None)
    ap.add_argument("--epochs_grid", nargs="+", type=int, default=None)
    ap.add_argument("--batch_size_grid", nargs="+", type=int, default=None,
                    help="Use explicit integers; omit to use the grid file.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)

    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip grid points whose run_dir/summary.json exists.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the grid without running anything.")

    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    grid_path = args.grid
    if not Path(grid_path).is_absolute():
        grid_path = _REPO_ROOT / grid_path
    grid = _load_grid(Path(grid_path))

    lr_grid = args.lr_grid or [float(x) for x in grid.get("lr", [])]
    ep_grid = args.epochs_grid or [int(x) for x in grid.get("epochs", [])]
    seeds = args.seeds or [int(x) for x in grid.get("seed", [0])]
    # batch_size: grid file uses `null` sentinels for "use family default".
    raw_bs = (args.batch_size_grid
              if args.batch_size_grid is not None
              else grid.get("batch_size", [None]))
    bs_grid: List[Optional[int]] = [
        None if (x is None) else int(x) for x in raw_bs
    ]
    if not lr_grid or not ep_grid:
        raise SystemExit(
            "grid expansion empty — set at least one lr and one epochs value."
        )

    eval_cfg = grid.get("eval", {})
    decouple = bool(grid.get("decouple_schedule", True))

    points = _expand(lr_grid, ep_grid, bs_grid, seeds)
    print(
        f"[grid] {len(points)} grid points: lr={lr_grid} "
        f"epochs={ep_grid} batch_size={bs_grid} seeds={seeds}"
    )

    # Persist the expanded grid for traceability.
    with open(sweep_dir / "expanded_grid.yaml", "w") as f:
        yaml.safe_dump({
            "family": args.family, "dataset": args.dataset,
            "lr": lr_grid, "epochs": ep_grid,
            "batch_size": bs_grid, "seeds": seeds,
            "eval": eval_cfg, "decouple_schedule": decouple,
            "n_points": len(points),
        }, f, sort_keys=False)

    if args.dry_run:
        for p in points:
            print(_run_name(args.family, p["lr"], p["epochs"],
                            p["batch_size"], p["seed"]))
        return 0

    t_start = time.time()
    taus = [float(x) for x in (eval_cfg.get("taus") or [eval_cfg.get("tau", 0.5)])]
    tau_subdirs = [f"tau_{int(t)}p{int(round((t - int(t)) * 10))}" for t in taus]

    def _run_complete(run_dir: Path) -> bool:
        """A run is complete when every requested tau has a summary.json."""
        return all(
            (run_dir / "eval" / sd / "summary.json").exists()
            for sd in tau_subdirs
        )

    results: List[Dict[str, Any]] = []
    for i, p in enumerate(points):
        rn = _run_name(args.family, p["lr"], p["epochs"],
                       p["batch_size"], p["seed"])
        run_dir = runs_dir / rn
        print(
            f"\n[grid] [{i + 1}/{len(points)}] elapsed={time.time() - t_start:.0f}s "
            f"{rn}"
        )
        if args.skip_existing and _run_complete(run_dir):
            print(f"[grid] skipping (all tau summaries exist): {run_dir}")
        else:
            _run_one(
                args.family, args.dataset, run_dir, p, eval_cfg,
                decouple_schedule=decouple,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
            )
        # Collect per-tau summaries into the sweep-level index.
        for sd in tau_subdirs:
            sp = run_dir / "eval" / sd / "summary.json"
            if sp.exists():
                with open(sp) as f:
                    results.append(json.load(f))
            else:
                results.append({
                    "run_name": rn, "tau_subdir": sd,
                    "status": "no_summary",
                })

    with open(sweep_dir / "runs.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(
        f"\n[grid] DONE in {time.time() - t_start:.0f}s. "
        f"{len(results)} (run × tau) rows → {sweep_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
