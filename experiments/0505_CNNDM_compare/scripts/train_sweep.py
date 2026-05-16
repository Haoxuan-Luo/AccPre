"""Stage E — full training sweep launcher (240 cells).

Materialises one config per cell and dispatches to `train_frozen.main` /
`train_joint.main` in-process. Skip-on-exists by `(run_dir/model.pt and
run_dir/train_history.json)` per cell.

Grid (per `training_grid.md`):
  Frozen: 3 targets × 3 archs × 4 lrs × 4 epochs × 1 seed = 144 cells
  Joint:  3 targets × 2 archs × 4 lrs × 4 epochs × 1 seed = 96 cells

Stable run-dir naming:
    runs/<regime>/<target>/<arch>/lr<lr>__ep<E>__s<seed>/

`<lr>` is rendered as `1e-3` -> `1e-3`, `3e-4` -> `3e-4`, etc., with `.`
replaced by `p` (e.g. `1.5e-4` -> `1p5e-4`). Stable for any future lr added.

Sharding (job-array friendly):
  --shard k --n_shards K  (k in [0, K-1])
filters cells to those with `index % K == k`.
Or: read SLURM_ARRAY_TASK_ID + SLURM_ARRAY_TASK_COUNT directly.

This driver does NOT modify accpre/.

Protocol policy: T=1 throughout (matches the stage-1 records and the 5
reused baseline JSONs at T=1).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXP_ROOT))


# Grid axes (single source of truth). Keep in sync with training_grid.md.
# CNN/DailyMail variant on the T=1, gamma=15 records.
_FROZEN_TARGETS = ("relmax", "alpha_q2", "dep")
_FROZEN_ARCHS = ("mlp_pos", "causal_transformer_pos", "bidirectional_transformer_pos")
_JOINT_TARGETS = ("relmax", "alpha_q2", "dep")
_JOINT_ARCHS = ("causal_transformer_pos", "bidirectional_transformer_pos")
_LRS: Tuple[float, ...] = (1e-3, 3e-4, 1e-4, 1e-5)
_EPOCHS: Tuple[int, ...] = (1, 3, 5, 10)
_SEEDS: Tuple[int, ...] = (0,)


_FROZEN_DEP_PATHS = {
    "alpha_q2": None,
    "relmax": "data_collected/stage1_pp_cnn_dm_300_g15_T1_relmax.pt",
    "dep":    "data_collected/stage1_pp_cnn_dm_300_g15_T1_dep.pt",
}
_JOINT_LIVE_FLAG = {
    "alpha_q2": "live_q2_target",
    "relmax": "live_relmax_target",
    "dep": "live_dep_target",
}


def _format_lr(lr: float) -> str:
    """Render lr as a stable, path-safe tag.

    Always renders in scientific form with the mantissa's leading-zero-trimmed
    integer part (where possible), so the standard grid {1e-3, 3e-4, 1e-4, 1e-5}
    maps to {lr1e-03, lr3e-04, lr1e-04, lr1e-05}. Non-integer mantissas (e.g.
    1.5e-4) render with `p` for the decimal point: lr1p5e-04. The mapping is
    injective for any fixed-precision grid.
    """
    # `:.6g` keeps up to 6 significant digits, then we coerce to scientific.
    s_sig = f"{lr:.6g}"
    # Convert any non-scientific form to scientific.
    if "e" not in s_sig:
        # `:.1e` is enough for our grid (mantissas are integers); `.6g`
        # already truncated trailing zeros so we don't lose precision here.
        if float(f"{lr:.1e}") == lr:
            s = f"{lr:.1e}"
            # Strip trailing ".0" before "e" for compact output (e.g.
            # '1.0e-03' -> '1e-03').
            mant, _, exp = s.partition("e")
            mant = mant.rstrip("0").rstrip(".") or "0"
            s = f"{mant}e{exp}"
        else:
            # Fall back to `.6g`-style with `p` for the decimal.
            s = s_sig
    else:
        s = s_sig
    s = s.replace(".", "p")
    return f"lr{s}"


def _cell_run_dir(runs_root: Path, regime: str, target: str, arch: str,
                  lr: float, ep: int, seed: int) -> Path:
    return runs_root / regime / target / arch / f"{_format_lr(lr)}__ep{ep}__s{seed}"


def _cell_done(cell_dir: Path) -> bool:
    """A cell is "done" if its model.pt + train_history.json both exist."""
    return (cell_dir / "model.pt").exists() and (cell_dir / "train_history.json").exists()


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Read a recovery manifest CSV.

    Required columns: regime, target, arch, lr, max_epochs, seed.
    Extra columns (e.g. status, run_dir, recovery_idx) are ignored.
    """
    import csv
    cells: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            try:
                cells.append(dict(
                    regime=str(row["regime"]).strip(),
                    target=str(row["target"]).strip(),
                    arch=str(row["arch"]).strip(),
                    lr=float(row["lr"]),
                    max_epochs=int(row["max_epochs"]),
                    seed=int(row["seed"]),
                ))
            except (KeyError, ValueError) as e:
                raise ValueError(f"manifest {path} row {i}: {e}") from e
    return cells


def _move_aside_partial(cell_dir: Path, runs_root: Path) -> None:
    """If a cell has model.pt but no train_history.json, stash stale artifacts.

    Moves model.pt, preds_*.pt, config*.yaml, ckpt/ into
    <cell>/_partial_<UTC-timestamp>/. Idempotent for cells with no stale files.
    """
    has_model = (cell_dir / "model.pt").exists()
    has_hist  = (cell_dir / "train_history.json").exists()
    if not has_model or has_hist:
        return
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    stash = cell_dir / f"_partial_{ts}"
    stash.mkdir(parents=True, exist_ok=True)
    moved: List[str] = []
    for name in ("model.pt", "preds_val.pt", "preds_test.pt",
                 "config.yaml", "_config_materialized.yaml"):
        p = cell_dir / name
        if p.exists():
            p.rename(stash / name)
            moved.append(name)
    ckpt = cell_dir / "ckpt"
    if ckpt.exists() and ckpt.is_dir():
        ckpt.rename(stash / "ckpt")
        moved.append("ckpt/")
    rel = cell_dir.relative_to(runs_root) if runs_root in cell_dir.parents else cell_dir
    print(f"[sweep] MOVED-PARTIAL {rel} -> _partial_{ts} ({moved})")


# ----------------------------------------------------------------------
# Cell-list construction.
# ----------------------------------------------------------------------


def build_cell_list(
    regimes: Tuple[str, ...] = ("frozen", "joint"),
    lrs: Tuple[float, ...] = _LRS,
    epochs: Tuple[int, ...] = _EPOCHS,
    seeds: Tuple[int, ...] = _SEEDS,
) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    if "frozen" in regimes:
        for target in _FROZEN_TARGETS:
            for arch in _FROZEN_ARCHS:
                for lr in lrs:
                    for ep in epochs:
                        for s in seeds:
                            cells.append(dict(
                                regime="frozen",
                                target=target,
                                arch=arch,
                                lr=float(lr),
                                max_epochs=int(ep),
                                seed=int(s),
                            ))
    if "joint" in regimes:
        for target in _JOINT_TARGETS:
            for arch in _JOINT_ARCHS:
                for lr in lrs:
                    for ep in epochs:
                        for s in seeds:
                            cells.append(dict(
                                regime="joint",
                                target=target,
                                arch=arch,
                                lr=float(lr),
                                max_epochs=int(ep),
                                seed=int(s),
                            ))
    return cells


# ----------------------------------------------------------------------
# Per-cell config materialisation.
# ----------------------------------------------------------------------


# Defaults match training_grid.md §3.
_FROZEN_DEFAULTS: Dict[str, Any] = {
    "protocol_path": "configs/protocol.yaml",
    "mode": "frozen",
    "family": "hidden_per_pos_v2",
    "gamma": 15,
    "T": 1,                                  # T=1 protocol
    "dataset": "cnn_dm_300",
    "hidden_dim": 128,
    "d_model": 256,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.1,
    "token_emb_dim": 64,
    "batch_size": 128,
    "weight_decay": 1.0e-4,
    "warmup_ratio": 0.05,
    "early_stop_patience": 2,
    "loss": "expected_survival_weighted_mse",
    "lambda_pred": 1.0,
    "lambda_mdlm": 0.0,
    "lambda_kl": 0.0,
    "lambda_mono": 0.0,
    "data_path": "data_collected/stage1_pp_cnn_dm_300_g15_T1.pt",
}

_JOINT_DEFAULTS: Dict[str, Any] = {
    "protocol_path": "configs/protocol.yaml",
    "mode": "joint",
    "family": "hidden_per_pos_v2",
    "gamma": 15,
    "T": 1,                                  # T=1 protocol
    "dataset": "cnn_dm_300",
    "d_model": 256,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.1,
    "token_emb_dim": 64,
    "drafter_train_mode": "frozen",
    "lambda_pred": 1.0,
    "lambda_mdlm": 0.0,
    "lambda_kl": 0.0,
    "lambda_mono": 0.0,
    "batch_size": 32,
    "grad_accum": 4,
    "weight_decay": 1.0e-4,
    "warmup_ratio": 0.05,
    "early_stop_patience": 2,
    "loss": "expected_survival_weighted_mse",
    "data_path": "data_collected/stage1_pp_cnn_dm_300_g15_T1.pt",
}


def materialize_config(cell: Dict[str, Any], runs_root: Path,
                       configs_root: Path,
                       prompt_overrides: Optional[Dict[str, List[int]]] = None
                       ) -> Tuple[Path, Dict[str, Any]]:
    """Build the per-cell config and write it to `configs_root/.../cell.yaml`.

    Returns (config_path, cfg_dict).

    `prompt_overrides`, if given, is merged into the cfg before write — used
    by the dry-run to substitute tiny prompt subsets via `--n_prompts_per_split`.
    """
    regime = cell["regime"]
    target = cell["target"]
    arch = cell["arch"]
    lr = float(cell["lr"])
    ep = int(cell["max_epochs"])
    seed = int(cell["seed"])

    if regime == "frozen":
        cfg = copy.deepcopy(_FROZEN_DEFAULTS)
        cfg["dep_targets_path"] = _FROZEN_DEP_PATHS[target]
    elif regime == "joint":
        cfg = copy.deepcopy(_JOINT_DEFAULTS)
        # Joint MUST NOT load a fixed-target file. Set live flags instead.
        cfg.pop("dep_targets_path", None)
        for f in ("live_relmax_target", "live_q2_target", "live_dep_target"):
            cfg[f] = False
        cfg[_JOINT_LIVE_FLAG[target]] = True
    else:
        raise ValueError(f"unknown regime {regime!r}")

    cfg["target"] = target
    cfg["arch"] = arch
    cfg["head_lr"] = lr
    cfg["max_epochs"] = ep
    cfg["seed"] = seed
    if prompt_overrides:
        cfg.update(prompt_overrides)

    out_dir = _cell_run_dir(runs_root, regime, target, arch, lr, ep, seed)
    cfg["out_dir"] = str(out_dir)
    cfg["run_name"] = f"{regime}_{target}_{arch}_{_format_lr(lr)}__ep{ep}__s{seed}"

    cfg_dir = configs_root / regime / target / arch
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{_format_lr(lr)}__ep{ep}__s{seed}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, cfg


# ----------------------------------------------------------------------
# Cell dispatch.
# ----------------------------------------------------------------------


def _run_cell(cell: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    if cell["regime"] == "frozen":
        from scripts.train_frozen import main as train_frozen_main
        return int(train_frozen_main(cfg))
    elif cell["regime"] == "joint":
        from scripts.train_joint import main as train_joint_main
        return int(train_joint_main(cfg))
    raise ValueError(f"unknown regime {cell['regime']!r}")


def _resolve_shard(args_shard: Optional[int],
                   args_n_shards: Optional[int]) -> Tuple[int, int]:
    """Resolve sharding from CLI or SLURM env. Defaults to (0, 1) (no shard)."""
    if args_shard is not None and args_n_shards is not None:
        return int(args_shard), int(args_n_shards)
    sj = os.environ.get("SLURM_ARRAY_TASK_ID")
    sn = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if sj is not None and sn is not None:
        return int(sj), int(sn)
    return 0, 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all", choices=("all", "frozen", "joint"))
    ap.add_argument("--target", default=None,
                    help="restrict to one target (relmax|alpha_q2|dep); default=all")
    ap.add_argument("--arch", default=None,
                    help="restrict to one arch; default=all valid for the regime")
    ap.add_argument("--lr", type=float, default=None,
                    help="restrict to one lr; default=all 4")
    ap.add_argument("--max_epochs", type=int, default=None,
                    help="restrict to one max_epochs; default=all 4")
    ap.add_argument("--seed", type=int, default=None,
                    help="restrict to one seed; default=all (currently {0})")
    ap.add_argument("--shard", type=int, default=None,
                    help="this task's shard index in [0, n_shards-1]")
    ap.add_argument("--n_shards", type=int, default=None,
                    help="total number of shards in the job array")
    ap.add_argument("--runs_root", default=str(_EXP_ROOT / "runs"),
                    help="dir under which per-cell run dirs are written")
    ap.add_argument("--configs_root",
                    default=str(_EXP_ROOT / "configs/_full"),
                    help="dir under which per-cell materialised configs are written")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the cell list and exit; do not train")
    ap.add_argument("--limit", type=int, default=None,
                    help="hard cap on number of cells to run (after sharding)")
    ap.add_argument("--n_prompts_per_split", type=int, default=None,
                    help=("DRY-RUN ONLY: shrink each split to N prompts (uses "
                          "the leading N indices in each canonical owt_300 split). "
                          "Skips the canonical split entirely; do not use for "
                          "real evaluation."))
    ap.add_argument("--manifest", default=None,
                    help=("CSV with columns regime,target,arch,lr,max_epochs,seed. "
                          "If given, REPLACES the canonical grid build; --regime/"
                          "--target/--arch/--lr/--max_epochs/--seed filters are "
                          "ignored. Sharding (--shard/--n_shards) and --limit still "
                          "apply on the manifest list (in CSV row order)."))
    ap.add_argument("--move_aside_partial", action="store_true",
                    help=("Before running each cell, if its dir contains model.pt "
                          "but no train_history.json (partial / timed-out training), "
                          "move stale artifacts (model.pt, preds_*.pt, config*.yaml, "
                          "ckpt/) into <cell>/_partial_<UTC-timestamp>/ so the "
                          "rerun starts from a clean slate. Has no effect on cells "
                          "that are already complete (those are still skipped) or "
                          "on cells with no prior artifacts."))
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    configs_root = Path(args.configs_root)
    configs_root.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        cells = _load_manifest(manifest_path)
        regimes = tuple(sorted({c["regime"] for c in cells}))
        print(f"[sweep] MANIFEST mode: loaded {len(cells)} cells from "
              f"{manifest_path} (regimes={regimes}); grid filters ignored")
    else:
        # Build full grid then filter by CLI args.
        if args.regime == "all":
            regimes: Tuple[str, ...] = ("frozen", "joint")
        else:
            regimes = (args.regime,)
        cells = build_cell_list(regimes=regimes)
        if args.target is not None:
            cells = [c for c in cells if c["target"] == args.target]
        if args.arch is not None:
            cells = [c for c in cells if c["arch"] == args.arch]
        if args.lr is not None:
            cells = [c for c in cells if abs(c["lr"] - float(args.lr)) < 1e-12]
        if args.max_epochs is not None:
            cells = [c for c in cells if c["max_epochs"] == int(args.max_epochs)]
        if args.seed is not None:
            cells = [c for c in cells if c["seed"] == int(args.seed)]

    # Sharding (after filtering — keeps round-robin balance).
    shard, n_shards = _resolve_shard(args.shard, args.n_shards)
    if n_shards <= 0:
        raise ValueError("n_shards must be >= 1")
    my_cells = [c for i, c in enumerate(cells) if i % n_shards == shard]

    if args.limit is not None:
        my_cells = my_cells[: int(args.limit)]

    print(f"[sweep] regimes={regimes} total_cells={len(cells)} "
          f"shard={shard}/{n_shards} my_cells={len(my_cells)}")

    # Materialise the canonical full grid once for documentation /
    # downstream coordination. Idempotent. Skipped in manifest mode so the
    # 240-cell canonical grid is not clobbered by a 27-cell recovery list.
    grid_path = runs_root / "_train_grid.json"
    if shard == 0 and args.manifest is None:
        with open(grid_path, "w") as f:
            json.dump(cells, f, indent=2)
        print(f"[sweep] wrote canonical grid to {grid_path}")

    if args.dry_run:
        for c in my_cells[:20]:
            print(f"  {c}")
        if len(my_cells) > 20:
            print(f"  ... ({len(my_cells) - 20} more)")
        return 0

    n_skip = 0
    n_run = 0
    n_fail = 0
    failures: List[Tuple[Dict[str, Any], str]] = []
    summary: List[Dict[str, Any]] = []

    # Build prompt overrides for the dry-run flag. The owt_300 canonical split
    # is train=0..159, val=160..199, test=200..299; we keep the first N from
    # each. WARNING: this bypasses the canonical split — only useful for
    # smoke / dry-run validation.
    prompt_overrides: Optional[Dict[str, List[int]]] = None
    if args.n_prompts_per_split is not None:
        n = int(args.n_prompts_per_split)
        prompt_overrides = {
            "train_prompt_indices": list(range(0, n)),
            "val_prompt_indices":   list(range(160, 160 + n)),
            "test_prompt_indices":  list(range(200, 200 + n)),
        }
        print(f"[sweep] DRY-RUN: using prompt subset n={n} per split "
              f"(train={prompt_overrides['train_prompt_indices']}, "
              f"val={prompt_overrides['val_prompt_indices']}, "
              f"test={prompt_overrides['test_prompt_indices']})")

    for c in my_cells:
        cell_dir = _cell_run_dir(runs_root, c["regime"], c["target"], c["arch"],
                                 c["lr"], c["max_epochs"], c["seed"])
        cell_dir.mkdir(parents=True, exist_ok=True)
        if _cell_done(cell_dir):
            print(f"[sweep] SKIP {cell_dir.relative_to(runs_root)} (model.pt + train_history.json present)")
            n_skip += 1
            summary.append(dict(**c, run_dir=str(cell_dir), rc=0, status="skip"))
            continue
        if args.move_aside_partial:
            _move_aside_partial(cell_dir, runs_root)
        cfg_path, cfg = materialize_config(c, runs_root, configs_root,
                                           prompt_overrides=prompt_overrides)
        print(f"[sweep] === RUN {c} ===")
        t0 = time.time()
        try:
            rc = _run_cell(c, cfg)
        except Exception as e:
            rc = -1
            failures.append((c, str(e)))
        elapsed = time.time() - t0
        n_run += 1
        if rc != 0:
            n_fail += 1
            failures.append((c, f"rc={rc}"))
        summary.append(dict(**c, run_dir=str(cell_dir), rc=int(rc),
                            elapsed_s=float(elapsed),
                            status="ok" if rc == 0 else "fail"))

    summary_path = runs_root / f"_sweep_summary_shard{shard}_of{n_shards}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sweep] wrote shard summary to {summary_path}")
    print(f"[sweep] shard {shard}/{n_shards}: skip={n_skip} run={n_run} fail={n_fail}")
    if failures:
        for c, msg in failures[:10]:
            print(f"  FAIL {c}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
