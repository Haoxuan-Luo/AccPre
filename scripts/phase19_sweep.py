"""Phase 19 hyperparameter sweep: Vlite-N and Vlite-H.

2 variants × 2 head_lr × 2 dropout = 8 runs. max_epochs=10 with
early_stop_patience=2. For each run we write a temp yaml into
`outputs/<sweep_dir>/configs/` and invoke the standard trainer, which
saves checkpoints to `checkpoints/<run_name>/`.

At the end we print a summary table of (variant, lr, dropout, best_val,
test_loss) and also emit a JSON with the same data so downstream
comparison/online decode jobs can pick the best per variant.

Kept simple on purpose: no process pool, one run at a time on a single
GPU. Total budget estimated at ~70 minutes on a 10GB MIG slice.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


BASE_CONFIGS = {
    "vlite_n": "configs/predictors/acc_vlite_n.yaml",
    "vlite_h": "configs/predictors/acc_vlite_h.yaml",
}


def _run_name(variant: str, lr: float, dropout: float) -> str:
    # Stable run-name schema so checkpoints are easy to find.
    lr_tag = f"lr{lr:.0e}".replace("-0", "m").replace("+0", "p")
    drop_tag = f"drop{dropout:.2f}".replace(".", "p")
    return f"acc_{variant}__{lr_tag}__{drop_tag}"


def _make_config(
    base_yaml: Path, run_name: str, head_lr: float, dropout: float,
    out_yaml: Path,
) -> Dict:
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["run_name"] = run_name
    cfg["head_lr"] = float(head_lr)
    cfg["dropout"] = float(dropout)
    cfg["max_epochs"] = 10
    cfg["early_stop_patience"] = 2
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--sweep_dir", type=str, required=True,
                    help="Output directory for generated configs + summary.")
    ap.add_argument("--lrs", type=str, default="1e-3,5e-4")
    ap.add_argument("--dropouts", type=str, default="0.1,0.2")
    ap.add_argument("--variants", type=str, default="vlite_n,vlite_h")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sweep_dir = Path(args.sweep_dir).resolve()
    cfg_dir = sweep_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    drops = [float(x) for x in args.dropouts.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    runs: List[Tuple[str, float, float, str, Path]] = []
    for v in variants:
        base_yaml = root / BASE_CONFIGS[v]
        for lr in lrs:
            for drop in drops:
                rn = _run_name(v, lr, drop)
                out_yaml = cfg_dir / f"{rn}.yaml"
                _make_config(base_yaml, rn, lr, drop, out_yaml)
                runs.append((v, lr, drop, rn, out_yaml))

    print(f"[phase19_sweep] {len(runs)} runs to execute")
    for v, lr, drop, rn, yml in runs:
        print(f"  - {rn} (variant={v} lr={lr} dropout={drop})")

    summary: List[Dict] = []
    for v, lr, drop, rn, yml in runs:
        print(f"\n[phase19_sweep] >>> {rn}")
        cmd = [
            "python3", "-m", "accpre.train.cli",
            "--config", str(yml),
        ]
        rc = subprocess.call(cmd, cwd=str(root))
        if rc != 0:
            print(f"[phase19_sweep] !! training failed rc={rc} for {rn}")
            summary.append({
                "variant": v, "head_lr": lr, "dropout": drop,
                "run_name": rn, "status": "FAIL", "rc": rc,
            })
            continue
        # Read training history to pull best_val and test_loss.
        hist_path = root / "checkpoints" / rn / "train_history.json"
        entry: Dict = {
            "variant": v, "head_lr": lr, "dropout": drop,
            "run_name": rn, "status": "OK",
            "checkpoint_dir": str((root / "checkpoints" / rn).relative_to(root)),
        }
        if hist_path.is_file():
            with open(hist_path) as f:
                h = json.load(f)
            entry["best_val"] = float(h.get("best_val", float("nan")))
            entry["test_loss"] = float(h.get("test_loss", float("nan")))
            entry["n_epochs"] = len(h.get("history", []))
        else:
            entry["best_val"] = float("nan")
            entry["test_loss"] = float("nan")
        summary.append(entry)

    # Pick best (lowest best_val) per variant.
    best_by_variant: Dict[str, Dict] = {}
    for row in summary:
        if row.get("status") != "OK":
            continue
        v = row["variant"]
        bv = row.get("best_val", float("inf"))
        prev = best_by_variant.get(v)
        if prev is None or bv < prev.get("best_val", float("inf")):
            best_by_variant[v] = row

    print("\n[phase19_sweep] summary\n" + "=" * 80)
    hdr = f"  {'run_name':<34} {'variant':<10} {'lr':>10} {'dropout':>8} {'best_val':>10} {'test':>10}"
    print(hdr)
    for row in summary:
        mark = " "
        if best_by_variant.get(row["variant"], {}).get("run_name") == row["run_name"]:
            mark = "*"
        print(
            f"{mark} {row['run_name']:<34} {row['variant']:<10} "
            f"{row['head_lr']:>10.1e} {row['dropout']:>8.2f} "
            f"{row.get('best_val', float('nan')):>10.4f} "
            f"{row.get('test_loss', float('nan')):>10.4f}"
        )
    print("\nbest per variant:")
    for v, row in best_by_variant.items():
        print(f"  {v}: {row['run_name']} (best_val={row['best_val']:.4f})")

    out_path = sweep_dir / "sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {"runs": summary, "best_by_variant": best_by_variant},
            f, indent=2, default=str,
        )
    print(f"\n[phase19_sweep] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
