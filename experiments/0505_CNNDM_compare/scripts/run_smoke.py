"""Stage C smoke runner.

Materialises and runs (in-process) a small grid of frozen + joint cells:
  - Frozen: 9 cells = 3 targets × 3 archs (mlp_pos, causal_tx, bidir_tx)
  - Joint:  6 cells = 3 targets × 2 transformer archs (causal_tx, bidir_tx)
  - All cells: max_epochs=1, tiny prompt subsets, batch_size=4 (joint) / 32 (frozen)

The smoke runner is meant for end-to-end shape / schema verification:
each cell trains for one epoch on a handful of records and writes the
same artifact set (model.pt, train_history.json, etc.) the full chain will.

USAGE
-----

  # Local (CPU is sufficient for frozen smoke; joint smoke needs GPU).
  python scripts/run_smoke.py --frozen_only            # 9 frozen cells
  python scripts/run_smoke.py --joint_only             # 6 joint cells (GPU)
  python scripts/run_smoke.py                          # all 15

  # Limit to a single cell for quick iteration:
  python scripts/run_smoke.py --regime frozen --target alpha_q2 --arch mlp_pos

ARTIFACTS
---------

Materialised configs land in:
  experiments/0505_OWT_compare/configs/_smoke/<regime>_<target>_<arch>.yaml
Run output lands in:
  experiments/0505_OWT_compare/runs/_smoke/<regime>/<target>/<arch>/

This script does NOT submit SLURM jobs; for cluster runs, call it from
`jobs/job_smoke.slurm`.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXP_ROOT))


# CNN/DailyMail smoke with the T=1, gamma=15 record + target files.
_FROZEN_TARGETS = ("relmax", "alpha_q2", "dep")
_FROZEN_ARCHS = ("mlp_pos", "causal_transformer_pos", "bidirectional_transformer_pos")
_JOINT_TARGETS = ("relmax", "alpha_q2", "dep")
_JOINT_ARCHS = ("causal_transformer_pos", "bidirectional_transformer_pos")


_FROZEN_DEP_PATHS = {
    "alpha_q2": None,
    "relmax": "data_collected/stage1_pp_cnn_dm_g15_T1_relmax.pt",
    "dep":    "data_collected/stage1_pp_cnn_dm_g15_T1_dep.pt",
}
_JOINT_LIVE_FLAG = {
    "alpha_q2": "live_q2_target",
    "relmax":   "live_relmax_target",
    "dep":      "live_dep_target",
}


def _load_template(regime: str) -> Dict[str, Any]:
    path = _EXP_ROOT / "configs/_smoke" / f"_template_{regime}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _materialize_frozen(target: str, arch: str) -> Tuple[Path, Dict[str, Any]]:
    cfg = copy.deepcopy(_load_template("frozen"))
    cfg["target"] = target
    cfg["arch"] = arch
    cfg["dep_targets_path"] = _FROZEN_DEP_PATHS[target]
    cfg["run_name"] = f"smoke_frozen_{target}_{arch}"
    out_dir = _EXP_ROOT / f"runs/_smoke_g15_T1/frozen/{target}/{arch}"
    cfg["out_dir"] = str(out_dir)
    cfg_path = _EXP_ROOT / f"configs/_smoke/frozen_{target}_{arch}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, cfg


def _materialize_joint(target: str, arch: str) -> Tuple[Path, Dict[str, Any]]:
    cfg = copy.deepcopy(_load_template("joint"))
    cfg["target"] = target
    cfg["arch"] = arch
    # Reset all live-flags then set only the one we need.
    for f in ("live_relmax_target", "live_q2_target", "live_dep_target"):
        cfg[f] = False
    cfg[_JOINT_LIVE_FLAG[target]] = True
    cfg.pop("dep_targets_path", None)   # ensure it stays absent
    cfg["run_name"] = f"smoke_joint_{target}_{arch}"
    out_dir = _EXP_ROOT / f"runs/_smoke_g15_T1/joint/{target}/{arch}"
    cfg["out_dir"] = str(out_dir)
    cfg_path = _EXP_ROOT / f"configs/_smoke/joint_{target}_{arch}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, cfg


def _run_cell(regime: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Call train_frozen.main or train_joint.main in-process and return summary."""
    t0 = time.time()
    if regime == "frozen":
        from scripts.train_frozen import main as train_frozen_main
        rc = train_frozen_main(cfg)
    else:
        from scripts.train_joint import main as train_joint_main
        rc = train_joint_main(cfg)
    elapsed = time.time() - t0
    return {"regime": regime, "rc": int(rc), "elapsed_s": elapsed,
            "out_dir": cfg["out_dir"]}


def _verify_outputs(regime: str, cfg: Dict[str, Any]) -> List[str]:
    """Verify the cell wrote the expected artifacts. Returns list of issues."""
    out_dir = Path(cfg["out_dir"])
    issues: List[str] = []
    for fname in ("model.pt", "train_history.json", "config.yaml",
                  "_config_materialized.yaml"):
        if not (out_dir / fname).exists():
            issues.append(f"missing {fname}")
    if not (out_dir / "ckpt/model.pt").exists():
        issues.append("missing ckpt/model.pt symlink")
    if not (out_dir / "ckpt/config.yaml").exists():
        issues.append("missing ckpt/config.yaml symlink")
    th_path = out_dir / "train_history.json"
    if th_path.exists():
        with open(th_path) as f:
            th = json.load(f)
        if "best_val_weighted_mse" not in th:
            issues.append("train_history.json missing best_val_weighted_mse")
        if regime == "joint":
            audit = th.get("live_target_audit", {})
            if audit.get("joint_dep_targets_path") is not None:
                issues.append(f"joint cell wrote joint_dep_targets_path != None: {audit}")
            if regime == "joint" and not any(
                audit.get(k) for k in ("live_relmax_target", "live_q2_target", "live_dep_target")
            ):
                issues.append(f"joint cell has no live_*_target flag set: {audit}")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen_only", action="store_true")
    ap.add_argument("--joint_only", action="store_true")
    ap.add_argument("--regime", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()

    cells: List[Tuple[str, str, str]] = []
    if not args.joint_only:
        for t in _FROZEN_TARGETS:
            for a in _FROZEN_ARCHS:
                if args.regime and args.regime != "frozen": continue
                if args.target and args.target != t: continue
                if args.arch and args.arch != a: continue
                cells.append(("frozen", t, a))
    if not args.frozen_only:
        for t in _JOINT_TARGETS:
            for a in _JOINT_ARCHS:
                if args.regime and args.regime != "joint": continue
                if args.target and args.target != t: continue
                if args.arch and args.arch != a: continue
                cells.append(("joint", t, a))

    print(f"[smoke] running {len(cells)} cells")
    summary: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str, str, str]] = []
    for (regime, target, arch) in cells:
        print(f"\n[smoke] === {regime} / {target} / {arch} ===")
        if regime == "frozen":
            cfg_path, cfg = _materialize_frozen(target, arch)
        else:
            cfg_path, cfg = _materialize_joint(target, arch)
        try:
            rec = _run_cell(regime, cfg)
        except Exception as e:
            rec = {"regime": regime, "rc": -1, "error": str(e),
                   "out_dir": cfg["out_dir"]}
            failures.append((regime, target, arch, str(e)))
        rec["target"] = target
        rec["arch"] = arch
        rec["config"] = str(cfg_path)
        # Verify artifacts (only if rc was 0 / no exception).
        if rec.get("rc", -1) == 0:
            issues = _verify_outputs(regime, cfg)
            if issues:
                rec["artifact_issues"] = issues
                failures.append((regime, target, arch, "; ".join(issues)))
        summary.append(rec)

    summary_path = _EXP_ROOT / "runs/_smoke_g15_T1/_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[smoke] wrote summary to {summary_path}")
    if failures:
        print(f"[smoke] FAILED ({len(failures)} cells):")
        for fr in failures:
            print(f"  - {fr}")
        return 1
    print(f"[smoke] PASS ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
