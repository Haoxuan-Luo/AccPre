"""Single predictor sweep run: train ONE config, then evaluate at 1+ taus.

Usage:
    python scripts/sweep_train_eval.py \\
        --family frozen_alpha --dataset owt \\
        --lr 1e-3 --epochs 10 --batch_size 128 --seed 0 \\
        --taus 0.5 0.7 0.9 --gamma 15 --max_new_tokens 1024 \\
        --run_dir outputs/sweeps/<root>/runs/<train_tag>

Writes:
    <run_dir>/
      ckpt/                       # accpre.train.cli output
        model.pt, config.yaml, train_history.json, preds_*.pt,
        drafter.pt (joint only), config.sweep.yaml
      training_summary.json       # train-side metrics (shared across taus)
      metrics.jsonl               # events for all taus
      eval/
        tau_0p5/
          online_predictor.json
          online_predictor_full.json
          summary.json            # one row per (train, tau) — consumed by summariser
        tau_0p7/ ...
        tau_0p9/ ...

Strict NLL reference discipline (Phase 26 convention):
    Strict is ALWAYS computed with the PRETRAINED MDLM drafter,
    regardless of whether the predictor is joint or frozen. This keeps
    ΔNLL a consistent reference across families. The strict-NLL cache
    keys on the pretrained drafter's signature; joint runs therefore
    reuse the same cached value as their frozen peers on the same
    dataset × horizon × gamma × T setting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.sweep.datasets import get_dataset
from accpre.sweep.eval_lane import (
    evaluate_predictor, load_greedy_protocol, strict_nll,
)
from accpre.sweep.logging import RunLogger


FAMILY_TO_BASE_CONFIG = {
    "frozen_alpha": "configs/sweep/frozen_alpha.yaml",
    "joint_alpha":  "configs/sweep/joint_alpha.yaml",
    "frozen_dep":   "configs/sweep/frozen_dep.yaml",
    "joint_dep":    "configs/sweep/joint_dep.yaml",
}

FAMILIES = tuple(FAMILY_TO_BASE_CONFIG)


def _tau_tag(tau: float) -> str:
    # 0.5 -> '0p5', 0.9 -> '0p9'. For values outside [0,1] we still
    # produce a readable tag.
    whole = int(tau)
    frac = int(round((tau - whole) * 10))
    return f"{whole}p{frac}"


def _load_base_config(family: str) -> Dict[str, Any]:
    if family not in FAMILY_TO_BASE_CONFIG:
        raise KeyError(
            f"unknown family {family!r}; known: {sorted(FAMILY_TO_BASE_CONFIG)}"
        )
    with open(_REPO_ROOT / FAMILY_TO_BASE_CONFIG[family]) as f:
        return yaml.safe_load(f)


def _resolve_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_base_config(args.family)
    cfg["run_name"] = args.run_name
    cfg["seed"] = int(args.seed)
    if args.lr is not None:
        cfg["head_lr"] = float(args.lr)
    if args.epochs is not None:
        cfg["max_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.backbone_lr is not None:
        cfg["backbone_lr"] = float(args.backbone_lr)

    bundle = get_dataset(args.dataset)
    if not bundle.records_path.exists():
        raise FileNotFoundError(
            f"records file for dataset {args.dataset!r} does not exist: "
            f"{bundle.records_path}. Run collection first."
        )
    cfg["data_path"] = str(bundle.records_path)
    if cfg.get("target") == "dependence" and cfg.get("mode") == "frozen":
        if bundle.dep_targets_path is None or not bundle.dep_targets_path.exists():
            raise FileNotFoundError(
                f"dep-targets file required for frozen_dep on "
                f"{args.dataset!r}: {bundle.dep_targets_path}. "
                f"Run scripts/collect_dep_targets.py first."
            )
        cfg["dep_targets_path"] = str(bundle.dep_targets_path)

    if args.decouple_schedule and cfg.get("mode") == "joint":
        cfg["total_steps_override"] = int(args.total_steps_override or 3700)
        cfg["warmup_steps_override"] = int(
            args.warmup_steps_override
            or round(0.05 * cfg["total_steps_override"])
        )
    return cfg


def _write_training_config(cfg: Dict[str, Any], ckpt_dir: Path) -> Path:
    path = ckpt_dir / "config.sweep.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _run_trainer(training_cfg_path: Path, ckpt_dir: Path) -> int:
    cmd = [
        sys.executable, "-m", "accpre.train.cli",
        "--config", str(training_cfg_path),
        "--out_dir", str(ckpt_dir),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[sweep] trainer: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=env)


def _load_trainer_artifacts(ckpt_dir: Path) -> Dict[str, Any]:
    with open(ckpt_dir / "train_history.json") as f:
        hist = json.load(f)
    with open(ckpt_dir / "config.yaml") as f:
        saved_cfg = yaml.safe_load(f)
    return {
        "run_name":  hist.get("run_name"),
        "best_val":  float(hist.get("best_val", float("nan"))),
        "test_loss": float(hist.get("test_loss", float("nan"))),
        "history":   hist.get("history", []),
        "final_train_loss":
            float(hist["history"][-1]["train_loss"]) if hist.get("history") else None,
        "final_val_loss":
            float(hist["history"][-1]["val_loss"]) if hist.get("history") else None,
        "n_epochs_run": len(hist.get("history", [])),
        "saved_cfg": saved_cfg,
    }


def _build_models(ckpt_dir: Path, protocol, dtype, device: str):
    """Build (predictor, pretrained_drafter, verifier, joint_drafter_state).

    Returns the PRETRAINED drafter up-front so the caller can compute
    strict NLL against the Phase 26 reference, and the path to the
    joint drafter.pt (or None) so the caller can swap weights in before
    running the predictor lane.
    """
    from accpre.eval.online_decode import _load_predictor_from_checkpoint
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    predictor, _cfg, drafter_state_path = _load_predictor_from_checkpoint(
        str(ckpt_dir), gamma=8,
    )
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )
    # The pretrained MDLM state is what the just-constructed drafter
    # already holds. We stash it on CPU so we can restore it later if
    # we ever want to re-run strict inside the same process.
    pretrained_cpu_state = {
        k: v.detach().cpu().clone() for k, v in drafter.model.state_dict().items()
    }
    return predictor, drafter, verifier, drafter_state_path, pretrained_cpu_state


def _swap_to_joint_drafter(drafter, drafter_state_path: str, device: str):
    print(f"[sweep] swapping drafter state <- {drafter_state_path}")
    drafter.model.load_state_dict(
        torch.load(drafter_state_path, map_location=device),
    )
    drafter.model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--run_name", type=str, default=None)

    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--backbone_lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--total_steps_override", type=int, default=None)
    ap.add_argument("--warmup_steps_override", type=int, default=None)
    ap.add_argument("--decouple_schedule", action="store_true", default=True)

    # Eval knobs.
    ap.add_argument("--n_eval_prompts", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5],
                    help="One or more thresholds; each produces its own "
                         "summary.json under eval/tau_XpY/.")
    ap.add_argument("--rule", type=str, default="threshold",
                    choices=("threshold", "confidence"))
    ap.add_argument("--strict_cache_dir", type=str,
                    default="outputs/sweeps/_strict_cache")

    # Logging.
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)

    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "ckpt"
    eval_root = run_dir / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or run_dir.name

    training_cfg = _resolve_training_config(args)
    training_cfg["run_name"] = run_name
    training_cfg_path = _write_training_config(training_cfg, ckpt_dir)

    base_fields = {
        "family": args.family,
        "dataset": args.dataset,
        "lr": training_cfg["head_lr"],
        "epochs": training_cfg["max_epochs"],
        "batch_size": training_cfg["batch_size"],
        "seed": args.seed,
        "rule": args.rule,
        "gamma": args.gamma, "T": args.T,
        "n_eval_prompts": args.n_eval_prompts,
        "max_new_tokens": args.max_new_tokens,
    }
    logger = RunLogger(
        run_dir=run_dir, run_id=run_name,
        base_fields=base_fields,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=run_name,
    )

    # --- TRAIN ---
    if not args.skip_train:
        t0 = time.time()
        rc = _run_trainer(training_cfg_path, ckpt_dir)
        train_s = time.time() - t0
        logger.log("train_final", return_code=rc, train_s=train_s)
        if rc != 0:
            payload = {
                **base_fields,
                "status": "train_failed", "return_code": rc,
                "train_s": train_s, "run_dir": str(run_dir),
            }
            with open(run_dir / "training_summary.json", "w") as f:
                json.dump(payload, f, indent=2)
            logger.close()
            return rc
    else:
        print("[sweep] --skip_train: using existing checkpoint")

    train_arts = _load_trainer_artifacts(ckpt_dir)
    logger.log(
        "train_history_summary",
        best_val=train_arts["best_val"],
        test_loss=train_arts["test_loss"],
        final_train_loss=train_arts["final_train_loss"],
        final_val_loss=train_arts["final_val_loss"],
        n_epochs_run=train_arts["n_epochs_run"],
    )
    training_summary = {
        **base_fields,
        "run_name": run_name,
        "run_dir":  str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "best_val":         train_arts["best_val"],
        "test_loss":        train_arts["test_loss"],
        "final_train_loss": train_arts["final_train_loss"],
        "final_val_loss":   train_arts["final_val_loss"],
        "n_epochs_run":     train_arts["n_epochs_run"],
        "status":           "trained",
    }
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    if args.skip_eval:
        logger.close()
        return 0

    # --- EVAL (shared setup) ---
    protocol = load_greedy_protocol()
    assert float(protocol.temperature) == 0.0, (
        "sweep eval expects greedy (temp=0) protocol."
    )
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor, drafter, verifier, joint_state_path, _pretrained = _build_models(
        ckpt_dir, protocol, dtype, device,
    )

    bundle = get_dataset(args.dataset)
    prompts, prompt_indices = bundle.prompt_loader(args.n_eval_prompts)
    print(
        f"[sweep] {len(prompts)} eval prompts from {bundle.name} "
        f"(indices {prompt_indices[0]}..{prompt_indices[-1]}), "
        f"gamma={args.gamma}, T={args.T}, max_new_tokens={args.max_new_tokens}"
    )

    # STRICT is computed with the PRETRAINED drafter (Phase 26 reference).
    # We haven't touched the drafter state yet, so this drafter IS the
    # pretrained MDLM; the strict-cache key will hash accordingly.
    strict_cache_dir = Path(args.strict_cache_dir)
    if not strict_cache_dir.is_absolute():
        strict_cache_dir = _REPO_ROOT / strict_cache_dir
    s_nll = strict_nll(
        drafter, verifier, prompts, prompt_indices, protocol,
        gamma=args.gamma, T=args.T, max_new_tokens=args.max_new_tokens,
        dataset_name=args.dataset, cache_dir=strict_cache_dir,
    )

    # Swap in the joint drafter (if applicable) for the predictor lane.
    if joint_state_path is not None:
        _swap_to_joint_drafter(drafter, joint_state_path, device)

    # --- EVAL per tau ---
    status_all = "ok"
    for tau in args.taus:
        tag = _tau_tag(float(tau))
        tau_dir = eval_root / f"tau_{tag}"
        tau_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[sweep] evaluating tau={tau} (rule={args.rule}) -> {tau_dir}")
        try:
            eval_res = evaluate_predictor(
                predictor, drafter, verifier, prompts, prompt_indices, protocol,
                tau=float(tau), rule=args.rule,
                gamma=args.gamma, T=args.T, max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"predictor eval failed at tau={tau}: {type(e).__name__}: {e}"
            print(f"[sweep] ERROR {msg}")
            status_all = "eval_failed"
            logger.log("eval_failed", tau=float(tau), error=msg)
            with open(tau_dir / "summary.json", "w") as f:
                json.dump({
                    **training_summary,
                    "tau": float(tau),
                    "status": "eval_failed",
                    "error": msg,
                }, f, indent=2)
            continue

        # Persist trimmed + full lane.
        with open(tau_dir / "online_predictor.json", "w") as f:
            trimmed = {k: v for k, v in eval_res.items() if k != "lane"}
            json.dump(trimmed, f, indent=2, default=str)
        with open(tau_dir / "online_predictor_full.json", "w") as f:
            json.dump(eval_res["lane"], f, indent=2, default=str)

        delta_nll = float(eval_res["nll"]) - float(s_nll)
        logger.log(
            "eval_final",
            tau=float(tau),
            tok_s_mean=eval_res["tok_s_mean"],
            nll=eval_res["nll"], strict_nll=float(s_nll), delta_nll=delta_nll,
            tok_succ=eval_res["tok_succ"], rnd_mean=eval_res["rnd_mean"],
            all_pass=eval_res["all_pass"],
            decode_s=eval_res["decode_s"], offline_s=eval_res["offline_s"],
        )

        tau_summary = {
            **training_summary,
            "tau":          float(tau),
            "tok_s_mean":   eval_res["tok_s_mean"],
            "nll":          eval_res["nll"],
            "strict_nll":   float(s_nll),
            "delta_nll":    delta_nll,
            "tok_succ":     eval_res["tok_succ"],
            "rnd_mean":     eval_res["rnd_mean"],
            "all_pass":     eval_res["all_pass"],
            "decode_s":     eval_res["decode_s"],
            "offline_s":    eval_res["offline_s"],
            "status":       "ok",
        }
        with open(tau_dir / "summary.json", "w") as f:
            json.dump(tau_summary, f, indent=2)

    # Final training_summary — overwrite with eval status signal.
    training_summary["status"] = (
        "ok" if status_all == "ok" else status_all
    )
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    logger.close()
    print(f"[sweep] DONE. training_summary → {run_dir / 'training_summary.json'}")
    print(f"[sweep] per-tau summaries under → {eval_root}/tau_*/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
