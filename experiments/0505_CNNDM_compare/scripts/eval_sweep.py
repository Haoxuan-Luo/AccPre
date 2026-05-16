"""Stage 5 — eval sweep launcher.

Reads `results/best_per_cell.json`, expands the 15 selected checkpoints
across {hard_threshold, prefix_product} × 6 thresholds = 180 cells, and
dispatches `eval_predictor.py` per cell.

Output layout (matches `eval_plan.md`):
  results/eval/<regime>/<target>/<arch>/<rule>/<param_tag>/online_predictor_full.json

Skip-on-exists per cell. Sharding via --shard / --n_shards or SLURM env.

Per-target fallback policy (T=1 plan; same as predecessor):
  relmax   -> confidence_gated_sampled_first  (eta=0.5)
  alpha_q2 -> confidence_gated_sampled_first  (eta=0.5)
  dep      -> drafter_argmax_first

`relative_max` is FORBIDDEN — `eval_predictor.py` raises if it ever sees it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]


# Allowed rules (mirrors eval_predictor.py + eval_plan.md §3).
_ALLOWED_RULES = ("hard_threshold", "prefix_product")
_RULE_GRIDS: Dict[str, Tuple[float, ...]] = {
    "hard_threshold": (0.3, 0.5, 0.6, 0.7, 0.8, 0.9),
    "prefix_product": (0.3,  0.45, 0.6, 0.75, 0.9, 0.95),
}

_FALLBACK_BY_TARGET = {
    "relmax":   ("confidence_gated_sampled_first", 0.5),
    "alpha_q2": ("confidence_gated_sampled_first", 0.5),
    "dep":      ("drafter_argmax_first",           0.5),  # eta unused
}


def _param_tag(rule: str, param: float) -> str:
    prefix = "tau" if rule == "hard_threshold" else "tauc"
    s = f"{float(param):.3f}".rstrip("0").rstrip(".")
    return f"{prefix}_{s.replace('.', 'p')}"


def _resolve_shard(args_shard: Optional[int],
                   args_n_shards: Optional[int]) -> Tuple[int, int]:
    if args_shard is not None and args_n_shards is not None:
        return int(args_shard), int(args_n_shards)
    sj = os.environ.get("SLURM_ARRAY_TASK_ID")
    sn = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if sj is not None and sn is not None:
        return int(sj), int(sn)
    return 0, 1


def _build_cells(best_per_cell: Dict[str, Any], rules: Tuple[str, ...]
                 ) -> List[Dict[str, Any]]:
    """Expand 15 selected ckpts × len(rules) × 6 thresholds = 180 cells.

    Each cell carries the source run_dir + (rule, param) + fallback derived
    from the target.
    """
    cells: List[Dict[str, Any]] = []
    for triple_key, sel in sorted(best_per_cell.items()):
        if not sel or not sel.get("run_dir"):
            continue
        regime, target, arch = triple_key.split(".", 2)
        if target not in _FALLBACK_BY_TARGET:
            raise ValueError(f"unknown target {target!r} for fallback policy")
        fb, eta = _FALLBACK_BY_TARGET[target]
        for rule in rules:
            for p in _RULE_GRIDS[rule]:
                cells.append(dict(
                    triple=triple_key,
                    regime=regime, target=target, arch=arch,
                    run_dir=sel["run_dir"],
                    rule=rule,
                    param=float(p),
                    param_tag=_param_tag(rule, p),
                    fallback=fb,
                    eta=float(eta),
                ))
    return cells


def _cell_out_path(eval_root: Path, c: Dict[str, Any]) -> Path:
    return (
        eval_root
        / c["regime"] / c["target"] / c["arch"] / c["rule"] / c["param_tag"]
        / "online_predictor_full.json"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_per_cell", default=str(_EXP_ROOT / "results" / "best_per_cell.json"))
    ap.add_argument("--eval_root", default=str(_EXP_ROOT / "results" / "eval"))
    ap.add_argument("--rules", default=",".join(_ALLOWED_RULES),
                    help=f"comma-separated subset of {_ALLOWED_RULES}")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--gamma", type=int, default=15)
    ap.add_argument("--T", type=int, default=1,
                    help="MDLM denoising steps; T=1 matches the stage-1 records")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--dataset", default="owt_300")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n_shards", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rules_req = tuple(r.strip() for r in args.rules.split(",") if r.strip())
    for r in rules_req:
        if r not in _ALLOWED_RULES:
            raise ValueError(f"forbidden rule {r!r}; must be in {_ALLOWED_RULES}")

    bpc_path = Path(args.best_per_cell)
    if not bpc_path.exists():
        print(f"[eval-sweep] ERROR best_per_cell.json missing: {bpc_path}")
        return 2
    with open(bpc_path) as f:
        best = json.load(f)

    cells = _build_cells(best, rules_req)
    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    shard, n_shards = _resolve_shard(args.shard, args.n_shards)
    if n_shards <= 0:
        raise ValueError("n_shards must be >= 1")
    my_cells = [c for i, c in enumerate(cells) if i % n_shards == shard]
    if args.limit is not None:
        my_cells = my_cells[: int(args.limit)]

    print(f"[eval-sweep] total_cells={len(cells)} shard={shard}/{n_shards} "
          f"my_cells={len(my_cells)}")
    if shard == 0:
        with open(eval_root.parent / "_eval_grid.json", "w") as f:
            json.dump(cells, f, indent=2)

    if args.dry_run:
        for c in my_cells[:20]:
            print(f"  {c}")
        if len(my_cells) > 20:
            print(f"  ... ({len(my_cells) - 20} more)")
        return 0

    eval_script = _THIS.parent / "eval_predictor.py"
    n_skip = 0
    n_run = 0
    n_fail = 0

    for c in my_cells:
        out_path = _cell_out_path(eval_root, c)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"[eval-sweep] SKIP {out_path.relative_to(eval_root)}")
            n_skip += 1
            continue
        cmd = [
            sys.executable, "-u", str(eval_script),
            "--run_dir", str(c["run_dir"]),
            "--rule", c["rule"], "--param", str(c["param"]),
            "--fallback", c["fallback"], "--eta", str(c["eta"]),
            "--out_path", str(out_path),
            "--protocol", args.protocol,
            "--gamma", str(args.gamma), "--T", str(args.T),
            "--max_new_tokens", str(args.max_new_tokens),
            "--n_prompts", str(args.n_prompts),
            "--dataset", args.dataset,
        ]
        print(f"[eval-sweep] === RUN {c['triple']} / {c['rule']} / {c['param_tag']} ===")
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        elapsed = time.time() - t0
        n_run += 1
        if rc != 0:
            n_fail += 1
            print(f"[eval-sweep] FAIL rc={rc} cell={c['triple']}/{c['rule']}/{c['param_tag']} "
                  f"elapsed={elapsed:.1f}s")

    print(f"[eval-sweep] shard {shard}/{n_shards}: skip={n_skip} run={n_run} fail={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
