"""0505_OWT_compare — frozen training driver.

Trains one head architecture against one fixed-target file using the
expected-survival weighted MSE loss. Drafter weights are not used at
training time (records carry pre-computed drafter hidden states and
scalars). Verifier weights are not used at all in this driver — the
verifier is required only by the joint trainer and by the offline NLL
pass in the eval driver.

This driver does NOT modify accpre/. It reuses:
  - `accpre.core.schema.load_records` for the stage-1 record list.
  - `accpre.core.protocol.ProtocolConfig` for the protocol fingerprint.
  - `accpre.train.dataset.FrozenAcceptanceDataset` for per-record items;
    we wrap it locally to additionally yield `alpha = record.min_pq_j`,
    which the loss needs as the survival weight regardless of target.
  - `accpre.train.dataset.split_records_by_prompt` for the train/val/test
    partition.
  - `accpre.collect.features.extract_features` (indirectly, via the dataset)
    for the canonical `hidden_per_pos_v2` feature bundle.

It uses the new heads from `scripts.heads.build_head` and the loss from
`scripts.losses.expected_survival_weighted_mse`.

Output artifacts (written to `out_dir`):
    config.yaml                  pristine cfg + the resolved defaults
    _config_materialized.yaml    same content; redundant alias kept for
                                 compatibility with the existing
                                 OWT_Frozen_0429 conventions
    model.pt                     head state_dict
    train_history.json           per-epoch train/val metrics + best_epoch
    preds_val.pt / preds_test.pt (gamma,)-tensor predictions over each split
    ckpt/model.pt, ckpt/config.yaml  symlinks to the parent files; matches
                                 the predecessor's eval-driver convention
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

# Pin the repo root so we can run the script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXP_ROOT))

from accpre.core.schema import RoundRecord, load_records
from accpre.core.protocol import ProtocolConfig
from accpre.train.dataset import FrozenAcceptanceDataset, split_records_by_prompt
from accpre.data.splits import get_split_config

from scripts.heads import build_head
from scripts.losses import expected_survival_weighted_mse


_VALID_TARGETS = {"relmax", "alpha_q2", "dep"}
_VALID_ARCHS = {"mlp_pos", "causal_transformer_pos", "bidirectional_transformer_pos"}


def _load_protocol(path: str) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


# ----------------------------------------------------------------------
# Local dataset wrapper that adds the per-record alpha (= min_pq_j).
# ----------------------------------------------------------------------


class _WeightedFrozenDataset(Dataset):
    """Wraps `FrozenAcceptanceDataset` and adds an `alpha` field equal to
    `record.min_pq_j` (per-position Leviathan ratio in [0, 1]).

    Required because the survival weight w_j = prod_{i<j} alpha_i is
    independent of the *target* (relmax, alpha_q2, dep) — alpha is always
    the canonical Leviathan ratio. The wrapped dataset's `q2_target` is
    the regression target (depends on `dep_targets_path`); `alpha` is
    the weight source.
    """

    def __init__(
        self,
        records: List[RoundRecord],
        family: str,
        dep_targets_path: str | None,
        expected_protocol_fp: str | None,
    ) -> None:
        self.records = list(records)
        self._inner = FrozenAcceptanceDataset(
            records=self.records,
            family=family,
            dep_targets_path=dep_targets_path,
            expected_protocol_fp=expected_protocol_fp,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._inner[idx]
        r = self.records[idx]
        item["alpha"] = torch.tensor(r.min_pq_j, dtype=torch.float32)
        return item


# ----------------------------------------------------------------------
# Optimizer + LR schedule.
# ----------------------------------------------------------------------


def _build_optimizer(head: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg["head_lr"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup -> cosine decay to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------
# Per-loader epoch step.
# ----------------------------------------------------------------------


def _move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _train_one_epoch(
    head: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    device: str,
    step_counter: List[int],
    warmup_steps: int,
    total_steps: int,
) -> Dict[str, float]:
    head.train()
    head_lr = float(cfg["head_lr"])
    accum = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        # LR schedule per optimizer step (not per micro-batch).
        scale = _lr_scale(step_counter[0], warmup_steps, total_steps)
        for g in optim.param_groups:
            g["lr"] = head_lr * scale
        y_hat = head(batch["features"], token_ids=batch["token_ids"])
        loss = expected_survival_weighted_mse(
            y_hat, batch["q2_target"], batch["alpha"],
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step_counter[0] += 1
        accum += float(loss.item())
        n_batches += 1
    return {"train_loss": accum / max(1, n_batches), "n_batches": n_batches}


@torch.no_grad()
def _eval_one_epoch(
    head: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    head.eval()
    accum_loss = 0.0
    n_batches = 0
    target_sum = 0.0
    target_sq_sum = 0.0
    pred_sum = 0.0
    pred_sq_sum = 0.0
    n_positions = 0
    n_w_positions = 0.0
    for batch in loader:
        batch = _move_batch(batch, device)
        y_hat = head(batch["features"], token_ids=batch["token_ids"])
        loss = expected_survival_weighted_mse(
            y_hat, batch["q2_target"], batch["alpha"],
        )
        accum_loss += float(loss.item())
        n_batches += 1
        # Diagnostics: target/pred mean/std.
        t = batch["q2_target"].flatten()
        p = y_hat.flatten()
        target_sum += float(t.sum().item())
        target_sq_sum += float((t * t).sum().item())
        pred_sum += float(p.sum().item())
        pred_sq_sum += float((p * p).sum().item())
        n_positions += int(t.numel())
        n_w_positions += float(batch["alpha"].sum().item())
    if n_positions == 0:
        return {"val_loss": float("nan"), "n_batches": 0}
    target_mean = target_sum / n_positions
    target_var = max(0.0, target_sq_sum / n_positions - target_mean * target_mean)
    pred_mean = pred_sum / n_positions
    pred_var = max(0.0, pred_sq_sum / n_positions - pred_mean * pred_mean)
    return {
        "val_loss": accum_loss / max(1, n_batches),
        "n_batches": n_batches,
        "n_positions": n_positions,
        "target_mean": target_mean,
        "target_std": math.sqrt(target_var),
        "pred_mean": pred_mean,
        "pred_std": math.sqrt(pred_var),
        "alpha_sum_mean": n_w_positions / max(1, n_positions),
    }


@torch.no_grad()
def _gather_predictions(
    head: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> torch.Tensor:
    head.eval()
    preds: List[torch.Tensor] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        y_hat = head(batch["features"], token_ids=batch["token_ids"])
        preds.append(y_hat.detach().cpu())
    if not preds:
        return torch.zeros(0)
    return torch.cat(preds, dim=0)


# ----------------------------------------------------------------------
# Entry.
# ----------------------------------------------------------------------


def _resolve_split_indices(cfg: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:
    """Return (train, val, test) prompt-index lists.

    If `cfg["train_prompt_indices"]` etc. are present, use them directly.
    Otherwise fall back to `cfg["dataset"]` and `accpre.data.splits`.
    """
    keys = ("train_prompt_indices", "val_prompt_indices", "test_prompt_indices")
    if all(k in cfg and cfg[k] is not None for k in keys):
        return (
            [int(i) for i in cfg["train_prompt_indices"]],
            [int(i) for i in cfg["val_prompt_indices"]],
            [int(i) for i in cfg["test_prompt_indices"]],
        )
    ds = str(cfg["dataset"])
    sc = get_split_config(ds)
    train = list(range(0, sc.train_n))
    val = list(range(sc.train_n, sc.train_n + sc.val_n))
    test = list(range(sc.train_n + sc.val_n, sc.pool_size))
    return train, val, test


def _validate_cfg(cfg: Dict[str, Any]) -> None:
    if cfg["mode"] != "frozen":
        raise ValueError(f"frozen trainer requires mode='frozen', got {cfg['mode']!r}")
    if cfg["target"] not in _VALID_TARGETS:
        raise ValueError(f"target must be in {_VALID_TARGETS}, got {cfg['target']!r}")
    if cfg["arch"] not in _VALID_ARCHS:
        raise ValueError(f"arch must be in {_VALID_ARCHS}, got {cfg['arch']!r}")
    if cfg["family"] != "hidden_per_pos_v2":
        raise ValueError(
            f"family must be 'hidden_per_pos_v2'; got {cfg['family']!r}"
        )
    if cfg.get("loss", "expected_survival_weighted_mse") != "expected_survival_weighted_mse":
        raise ValueError(
            f"loss must be 'expected_survival_weighted_mse'; got {cfg.get('loss')!r}"
        )


def _resolve_dep_targets_path(cfg: Dict[str, Any]) -> str | None:
    """Frozen target file path; depends on the target family.

    For target='alpha_q2', returns None (FrozenAcceptanceDataset uses
    record.min_pq_j directly, which is bit-identical to min(1, p/q)).
    For target='relmax' / 'dep', returns the configured fixed-target file.
    """
    if cfg["target"] == "alpha_q2":
        return None
    p = cfg.get("dep_targets_path")
    if p is None:
        raise ValueError(
            f"target={cfg['target']!r} requires `dep_targets_path` in the config"
        )
    return str(p)


def main(cfg: Dict[str, Any]) -> int:
    _validate_cfg(cfg)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    # --- protocol ---
    protocol = _load_protocol(str(cfg["protocol_path"]))
    expected_fp = protocol.fingerprint()
    print(f"[frozen] protocol fp: {expected_fp}")
    print(f"[frozen] temperature: {protocol.temperature}")

    # --- records + split ---
    records = load_records(str(cfg["data_path"]))
    target_gamma = int(cfg["gamma"])
    total = len(records)
    records = [r for r in records if r.gamma == target_gamma]
    print(f"[frozen] loaded {total} records; {len(records)} at gamma={target_gamma}")

    train_idx, val_idx, test_idx = _resolve_split_indices(cfg)
    parts = split_records_by_prompt(records, train_idx, val_idx, test_idx)
    print(f"[frozen] split: train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")

    dep_path = _resolve_dep_targets_path(cfg)
    train_ds = _WeightedFrozenDataset(parts["train"], cfg["family"], dep_path, expected_fp)
    val_ds = _WeightedFrozenDataset(parts["val"], cfg["family"], dep_path, expected_fp)
    test_ds = _WeightedFrozenDataset(parts["test"], cfg["family"], dep_path, expected_fp)

    g = torch.Generator().manual_seed(seed)
    bs = int(cfg.get("batch_size", 128))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, generator=g, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)

    # --- head ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(
        cfg["arch"],
        gamma=target_gamma,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        d_model=int(cfg.get("d_model", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        token_emb_dim=int(cfg.get("token_emb_dim", 64)),
    )
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[frozen] head={cfg['arch']} family={cfg['family']} "
          f"deploy_mode={head.deploy_mode} trainable_params={n_params}")
    head.to(device)
    optim = _build_optimizer(head, cfg)

    max_epochs = int(cfg.get("max_epochs", 1))
    n_train_batches = max(1, len(train_loader))
    total_steps = max_epochs * n_train_batches
    warmup_steps = int(round(float(cfg.get("warmup_ratio", 0.05)) * total_steps))
    print(f"[frozen] total_steps={total_steps} warmup_steps={warmup_steps}")

    best_val = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    patience = int(cfg.get("early_stop_patience", 2))
    no_improve = 0
    step_counter = [0]
    t_start = time.time()

    for epoch in range(max_epochs):
        tr = _train_one_epoch(
            head, train_loader, optim, cfg, device, step_counter, warmup_steps, total_steps,
        )
        va = _eval_one_epoch(head, val_loader, device)
        record = {"epoch": int(epoch), **tr, **{f"val_{k}": v for k, v in va.items()}}
        history.append(record)
        print(
            f"[frozen] epoch {epoch}: train_loss={tr['train_loss']:.5f} "
            f"val_loss={va.get('val_loss', float('nan')):.5f} "
            f"target_mean={va.get('target_mean', float('nan')):.4f} "
            f"pred_mean={va.get('pred_mean', float('nan')):.4f}"
        )
        if va["val_loss"] < best_val - 1e-6:
            best_val = float(va["val_loss"])
            best_epoch = int(epoch)
            no_improve = 0
            # Save best ckpt only.
            torch.save(head.state_dict(), out_dir / "model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[frozen] early stop at epoch {epoch} (best epoch={best_epoch})")
                break

    # If no epoch ran (max_epochs=0) or none improved, still save the final state
    # so downstream eval can load *something*.
    if not (out_dir / "model.pt").exists():
        torch.save(head.state_dict(), out_dir / "model.pt")
        if best_epoch < 0:
            best_epoch = 0
            best_val = float(history[-1]["val_loss"]) if history else float("nan")

    # Test loss + predictions on val/test for downstream calibration / Pareto plot.
    test_metrics = _eval_one_epoch(head, test_loader, device)
    val_preds = _gather_predictions(head, val_loader, device)
    test_preds = _gather_predictions(head, test_loader, device)

    elapsed = time.time() - t_start
    summary = {
        "run_name": str(cfg.get("run_name", "")),
        "regime": "frozen",
        "target": cfg["target"],
        "arch": cfg["arch"],
        "family": cfg["family"],
        "best_val_weighted_mse": best_val,
        "best_epoch": best_epoch,
        "test_loss": float(test_metrics.get("val_loss", float("nan"))),
        "history": history,
        "elapsed_s": elapsed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "trainable_params": n_params,
    }
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(val_preds, out_dir / "preds_val.pt")
    torch.save(test_preds, out_dir / "preds_test.pt")
    with open(out_dir / "_config_materialized.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    if not (out_dir / "config.yaml").exists():
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)
    # ckpt/ symlinks for compatibility with the legacy eval driver.
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    for src, dst in (("model.pt", "model.pt"), ("config.yaml", "config.yaml")):
        link = ckpt_dir / dst
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / src)

    print(f"[frozen] best_val={best_val:.5f} best_epoch={best_epoch} "
          f"test_loss={test_metrics.get('val_loss', float('nan')):.5f} "
          f"elapsed={elapsed:.1f}s")
    print(f"[frozen] wrote artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to yaml cfg")
    ap.add_argument("--out_dir", default=None, help="override cfg out_dir")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    sys.exit(main(cfg))
