"""0505_OWT_compare — joint live-target training driver.

Per-batch, the trainer:
  1. Re-runs the drafter on the record's reconstructed full prefix under the
     stored `round_rng_seed` (matched-randomness, identical to stage-1 collection).
     Weights are frozen by default; we use `drafter.draft_with_features` which
     is `@torch.no_grad`, so the drafter contributes no autograd graph.
  2. Computes the live target y_j and the live alpha_j according to
     `cfg["target"]` (relmax / alpha_q2 / dep). Verifier and drafter are both
     frozen and run under no_grad.
  3. Assembles the `hidden_per_pos_v2` feature bundle from the freshly
     produced drafter hidden + 5 scalars (matches the offline feature
     extractor exactly).
  4. Forwards the head; computes
       L = expected_survival_weighted_mse(y_hat, y_target, alpha)
     and backprops into head parameters.

Verifier-free invariant: the verifier is called only inside the live-target
helpers in `scripts.live_targets`; it is never used at *inference* time. The
predictor head consumes only drafter-side features.

Joint must NEVER load fixed target files. The trainer enforces this with a
hard assertion at start; sanity_no_fixed_targets_in_joint.py also static-greps
every joint config.

Backbone-adaptation interface (drafter_train_mode != 'frozen', lambda_mdlm,
lambda_kl, lambda_mono) is staged but inactive in this experiment. The
trainer rejects any non-default value with an explicit error so a future
follow-on must extend the trainer rather than silently flip a switch.

This driver does NOT modify accpre/.
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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXP_ROOT))

from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import RoundRecord, load_records
from accpre.data.splits import get_split_config
from accpre.train.dataset import (
    JointAcceptanceDataset,
    joint_collate_fn,
    split_records_by_prompt,
)

from scripts.heads import build_head
from scripts.live_targets import compute_live_target
from scripts.losses import expected_survival_weighted_mse


_VALID_TARGETS = {"relmax", "alpha_q2", "dep"}
_VALID_JOINT_ARCHS = {"causal_transformer_pos", "bidirectional_transformer_pos"}


def _load_protocol(path: str) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


# ----------------------------------------------------------------------
# Hard guarantee: joint trainer must not load any fixed-target file.
# ----------------------------------------------------------------------


def _validate_cfg_joint(cfg: Dict[str, Any]) -> None:
    if cfg["mode"] != "joint":
        raise ValueError(f"joint trainer requires mode='joint', got {cfg['mode']!r}")
    if cfg["target"] not in _VALID_TARGETS:
        raise ValueError(f"target must be in {_VALID_TARGETS}, got {cfg['target']!r}")
    if cfg["arch"] not in _VALID_JOINT_ARCHS:
        raise ValueError(
            f"joint arch must be in {_VALID_JOINT_ARCHS} "
            f"(mlp_pos is frozen-only); got {cfg['arch']!r}"
        )
    if cfg["family"] != "hidden_per_pos_v2":
        raise ValueError(
            f"family must be 'hidden_per_pos_v2'; got {cfg['family']!r}"
        )

    # Joint must never load a fixed-target file.
    if cfg.get("dep_targets_path") is not None:
        raise AssertionError(
            "joint training MUST NOT load fixed-target files; "
            f"got dep_targets_path={cfg.get('dep_targets_path')!r}. "
            "Remove the field from the config."
        )
    target = cfg["target"]
    flag_for = {
        "relmax": "live_relmax_target",
        "alpha_q2": "live_q2_target",
        "dep": "live_dep_target",
    }[target]
    if not bool(cfg.get(flag_for, False)):
        raise AssertionError(
            f"joint training for target={target!r} requires "
            f"`{flag_for}: true` in the config; got "
            f"{cfg.get(flag_for, False)!r}."
        )
    # The OTHER live flags must be absent or false.
    for other_flag in {"live_relmax_target", "live_q2_target", "live_dep_target"} - {flag_for}:
        if bool(cfg.get(other_flag, False)):
            raise AssertionError(
                f"joint config has `{other_flag}: true` but target={target!r} "
                f"requires `{flag_for}: true` exclusively."
            )

    # Backbone-adaptation interface must be in default-frozen state for this experiment.
    mode = str(cfg.get("drafter_train_mode", "frozen"))
    if mode != "frozen":
        raise NotImplementedError(
            f"drafter_train_mode={mode!r} is staged but not implemented in this experiment. "
            "Set drafter_train_mode: frozen."
        )
    for key in ("lambda_mdlm", "lambda_kl", "lambda_mono"):
        v = float(cfg.get(key, 0.0))
        if v != 0.0:
            raise NotImplementedError(
                f"{key}={v} is staged but not implemented in this experiment. "
                f"Leave it at 0."
            )

    if cfg.get("loss", "expected_survival_weighted_mse") != "expected_survival_weighted_mse":
        raise ValueError(
            f"loss must be 'expected_survival_weighted_mse'; got {cfg.get('loss')!r}"
        )


# ----------------------------------------------------------------------
# Models.
# ----------------------------------------------------------------------


def _build_drafter(protocol: ProtocolConfig, device: str):
    """Load MDLM drafter on `device` in the protocol's dtype.

    The drafter is frozen for this experiment; we do NOT call .train() on it.
    `draft_with_features` is @torch.no_grad-decorated by accpre, so any tensor
    it returns is detached from the autograd graph.
    """
    from accpre.models.drafter_mdlm import MDLMDrafter
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[protocol.dtype]
    return MDLMDrafter(model_name=protocol.drafter_model, device=device, dtype=dtype)


def _build_verifier(protocol: ProtocolConfig, device: str):
    """Load GPT-2 XL verifier on `device` in the protocol's dtype.

    The verifier is used at training time only for live-target construction.
    Inference is V-free.
    """
    from accpre.models.verifier_gpt2 import GPT2Verifier
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[protocol.dtype]
    return GPT2Verifier(model_name=protocol.verifier_model, device=device, dtype=dtype)


# ----------------------------------------------------------------------
# Live feature assembly (mirrors `accpre.collect.features`'s v2 layout).
# ----------------------------------------------------------------------


def _assemble_v2_features(
    drafter_hidden_per_pos: torch.Tensor,    # (gamma, H_d) fp32
    draft_log_probs: torch.Tensor,           # (gamma, V_d)
    draft_tokens: torch.Tensor,              # (gamma,) long
) -> torch.Tensor:
    """Build a (gamma, H_d + 5) feature tensor matching extract_features('hidden_per_pos_v2', ...).

    Five scalars per position: [q, log q, drafter_entropy, drafter_margin, drafter_top1_prob].
    Computed from the drafter's (refreshed) draft_log_probs row at each position.
    All inputs assumed detached / no-grad already; output is fp32.
    """
    EPS = 1e-10
    h = drafter_hidden_per_pos.to(torch.float32)
    log_p = draft_log_probs.to(torch.float32)
    p = log_p.exp()
    entropy = -(p * log_p).sum(dim=-1)
    top2 = log_p.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    top1 = p.max(dim=-1).values
    idx = torch.arange(int(draft_tokens.shape[0]), device=draft_tokens.device)
    q_t = p[idx, draft_tokens.long()].clamp(min=EPS)
    log_q_t = torch.log(q_t)
    scalars = torch.stack([q_t, log_q_t, entropy, margin, top1], dim=1)  # (gamma, 5)
    return torch.cat([h, scalars], dim=1)                                # (gamma, H_d+5)


# ----------------------------------------------------------------------
# Per-item forward.
# ----------------------------------------------------------------------


def _forward_item(
    head: torch.nn.Module,
    drafter,
    verifier,
    protocol: ProtocolConfig,
    target_name: str,
    prefix_ids: torch.Tensor,
    round_rng_seed: int,
    target_gamma: int,
    T: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one record's forward through the drafter (no-grad) + head.

    Returns (y_hat, y_target, alpha) of shape (gamma,) each, on `device`.
    """
    # Drafter forward (no_grad inside the wrapper).
    draft_rng = _make_generator(device, int(round_rng_seed) ^ protocol.draft_salt)
    draft_tokens, draft_log_probs, drafter_hidden_per_pos = drafter.draft_with_features(
        prefix_ids=prefix_ids.to(device),
        gamma=int(target_gamma),
        T=int(T),
        temperature=protocol.temperature,
        q_mode=protocol.q_mode,
        generator=draft_rng,
        pool="per_position",
        layers=(-1,),
    )
    # All three returned tensors are detached (no_grad). Move to head device.
    draft_tokens = draft_tokens.to(device)
    draft_log_probs = draft_log_probs.to(device)
    drafter_hidden_per_pos = drafter_hidden_per_pos.to(device)

    # Live target + alpha (no_grad). Helpers handle device alignment internally.
    out = compute_live_target(
        target_name,
        drafter=drafter,
        verifier=verifier,
        prefix_ids=prefix_ids,
        draft_tokens=draft_tokens,
        draft_log_probs=draft_log_probs,
        gamma=int(target_gamma),
    )
    y_target = out.y_target.to(device).to(torch.float32)
    alpha = out.alpha.to(device).to(torch.float32)

    # Feature assembly (matches accpre.collect.features.extract_features('hidden_per_pos_v2', ...)).
    features = _assemble_v2_features(drafter_hidden_per_pos, draft_log_probs, draft_tokens)
    # Add a leading batch axis to match the head's (B, gamma, F) interface.
    features_b = features.unsqueeze(0)
    tokens_b = draft_tokens.unsqueeze(0).long()

    y_hat = head(features_b, token_ids=tokens_b).squeeze(0)        # (gamma,)
    return y_hat, y_target, alpha


# ----------------------------------------------------------------------
# Schedule.
# ----------------------------------------------------------------------


def _lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------
# Train / val loops.
# ----------------------------------------------------------------------


def _train_one_epoch(
    head: torch.nn.Module,
    drafter,
    verifier,
    protocol: ProtocolConfig,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    device: str,
    step_counter: List[int],
    warmup_steps: int,
    total_steps: int,
) -> Dict[str, float]:
    head.train()
    target_name = cfg["target"]
    target_gamma = int(cfg["gamma"])
    T = int(cfg.get("T", 2))
    grad_accum = int(cfg.get("grad_accum", 1))
    head_lr = float(cfg["head_lr"])

    accum_loss = 0.0
    n_items = 0
    optim.zero_grad(set_to_none=True)
    micro_in_step = 0

    for batch in loader:
        prefix_ids_list = batch["prefix_ids"]                # list of (L_i,) tensors
        seeds = batch["round_rng_seed"]                       # (B,)
        B = len(prefix_ids_list)
        for i in range(B):
            y_hat, y_target, alpha = _forward_item(
                head, drafter, verifier, protocol, target_name,
                prefix_ids_list[i], int(seeds[i].item()),
                target_gamma, T, device,
            )
            loss = expected_survival_weighted_mse(
                y_hat.unsqueeze(0), y_target.unsqueeze(0), alpha.unsqueeze(0),
            )
            (loss / grad_accum).backward()
            accum_loss += float(loss.item())
            n_items += 1
            micro_in_step += 1
            if micro_in_step >= grad_accum:
                # LR schedule per optimizer step.
                scale = _lr_scale(step_counter[0], warmup_steps, total_steps)
                for g in optim.param_groups:
                    g["lr"] = head_lr * scale
                # Optional: clip head grad norm (light value; documented in resource_plan).
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step_counter[0] += 1
                micro_in_step = 0

    # Flush trailing micro-batches.
    if micro_in_step > 0:
        scale = _lr_scale(step_counter[0], warmup_steps, total_steps)
        for g in optim.param_groups:
            g["lr"] = head_lr * scale
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step_counter[0] += 1

    return {"train_loss": accum_loss / max(1, n_items), "n_items": n_items}


@torch.no_grad()
def _eval_one_epoch(
    head: torch.nn.Module,
    drafter,
    verifier,
    protocol: ProtocolConfig,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    head.eval()
    target_name = cfg["target"]
    target_gamma = int(cfg["gamma"])
    T = int(cfg.get("T", 2))

    accum_loss = 0.0
    n_items = 0
    target_sum = 0.0
    target_sq_sum = 0.0
    pred_sum = 0.0
    pred_sq_sum = 0.0
    n_positions = 0
    alpha_sum = 0.0

    for batch in loader:
        prefix_ids_list = batch["prefix_ids"]
        seeds = batch["round_rng_seed"]
        B = len(prefix_ids_list)
        for i in range(B):
            y_hat, y_target, alpha = _forward_item(
                head, drafter, verifier, protocol, target_name,
                prefix_ids_list[i], int(seeds[i].item()),
                target_gamma, T, device,
            )
            loss = expected_survival_weighted_mse(
                y_hat.unsqueeze(0), y_target.unsqueeze(0), alpha.unsqueeze(0),
            )
            accum_loss += float(loss.item())
            n_items += 1
            n_positions += int(y_target.numel())
            target_sum += float(y_target.sum().item())
            target_sq_sum += float((y_target ** 2).sum().item())
            pred_sum += float(y_hat.sum().item())
            pred_sq_sum += float((y_hat ** 2).sum().item())
            alpha_sum += float(alpha.sum().item())

    if n_items == 0:
        return {"val_loss": float("nan"), "n_items": 0}
    target_mean = target_sum / max(1, n_positions)
    pred_mean = pred_sum / max(1, n_positions)
    target_var = max(0.0, target_sq_sum / max(1, n_positions) - target_mean ** 2)
    pred_var = max(0.0, pred_sq_sum / max(1, n_positions) - pred_mean ** 2)
    return {
        "val_loss": accum_loss / n_items,
        "n_items": n_items,
        "n_positions": n_positions,
        "target_mean": target_mean,
        "target_std": math.sqrt(target_var),
        "pred_mean": pred_mean,
        "pred_std": math.sqrt(pred_var),
        "alpha_mean": alpha_sum / max(1, n_positions),
    }


@torch.no_grad()
def _gather_predictions(
    head: torch.nn.Module,
    drafter,
    verifier,
    protocol: ProtocolConfig,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: str,
) -> torch.Tensor:
    """Run V-free head forward over `loader` items and concatenate predictions.

    Each item produces one (gamma,) prediction vector via the SAME live-target
    forward path used during eval; only the head output is retained. The
    drafter is used to regenerate features (matched-randomness from the stored
    record seed); the verifier is NOT consulted here (we only need predictions,
    not target/alpha). However we keep the verifier argument for signature
    parity with `_eval_one_epoch` and because `_forward_item` requires it for
    live target/alpha — those are computed but discarded.

    Returns a (n_items, gamma) tensor on CPU.
    """
    head.eval()
    target_name = cfg["target"]
    target_gamma = int(cfg["gamma"])
    T = int(cfg.get("T", 2))
    preds: List[torch.Tensor] = []
    for batch in loader:
        prefix_ids_list = batch["prefix_ids"]
        seeds = batch["round_rng_seed"]
        B = len(prefix_ids_list)
        for i in range(B):
            y_hat, _y_target, _alpha = _forward_item(
                head, drafter, verifier, protocol, target_name,
                prefix_ids_list[i], int(seeds[i].item()),
                target_gamma, T, device,
            )
            preds.append(y_hat.detach().cpu())
    if not preds:
        return torch.zeros(0, target_gamma)
    return torch.stack(preds, dim=0)


# ----------------------------------------------------------------------
# Entry.
# ----------------------------------------------------------------------


def _resolve_split_indices(cfg: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:
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


def main(cfg: Dict[str, Any]) -> int:
    _validate_cfg_joint(cfg)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    protocol = _load_protocol(str(cfg["protocol_path"]))
    expected_fp = protocol.fingerprint()
    print(f"[joint] protocol fp: {expected_fp}")
    print(f"[joint] temperature: {protocol.temperature}")

    target_gamma = int(cfg["gamma"])
    records = load_records(str(cfg["data_path"]))
    total = len(records)
    records = [r for r in records if r.gamma == target_gamma]
    print(f"[joint] loaded {total} records; {len(records)} at gamma={target_gamma}")

    train_idx, val_idx, test_idx = _resolve_split_indices(cfg)
    parts = split_records_by_prompt(records, train_idx, val_idx, test_idx)
    print(f"[joint] split: train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")

    # JointAcceptanceDataset: dep_targets_path INTENTIONALLY None.
    train_ds = JointAcceptanceDataset(
        records=parts["train"], family=cfg["family"],
        dep_targets_path=None,
        dataset=str(cfg["dataset"]),
        expected_protocol_fp=expected_fp,
    )
    val_ds = JointAcceptanceDataset(
        records=parts["val"], family=cfg["family"],
        dep_targets_path=None,
        dataset=str(cfg["dataset"]),
        expected_protocol_fp=expected_fp,
    )
    test_ds = JointAcceptanceDataset(
        records=parts["test"], family=cfg["family"],
        dep_targets_path=None,
        dataset=str(cfg["dataset"]),
        expected_protocol_fp=expected_fp,
    )

    g = torch.Generator().manual_seed(seed)
    bs = int(cfg.get("batch_size", 32))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, generator=g, drop_last=False,
        collate_fn=joint_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, drop_last=False,
        collate_fn=joint_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, drop_last=False,
        collate_fn=joint_collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    drafter = _build_drafter(protocol, device)
    verifier = _build_verifier(protocol, device)
    drafter.model.eval()
    verifier.model.eval()
    # Drafter and verifier are frozen.
    for p in drafter.model.parameters():
        p.requires_grad_(False)
    for p in verifier.model.parameters():
        p.requires_grad_(False)

    head = build_head(
        cfg["arch"],
        gamma=target_gamma,
        d_model=int(cfg.get("d_model", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        token_emb_dim=int(cfg.get("token_emb_dim", 64)),
    )
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[joint] head={cfg['arch']} family={cfg['family']} "
          f"deploy_mode={head.deploy_mode} trainable_params={n_params}")
    head.to(device)
    optim = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg["head_lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    max_epochs = int(cfg.get("max_epochs", 1))
    grad_accum = int(cfg.get("grad_accum", 1))
    n_items_per_epoch = max(1, len(train_ds))
    total_steps = max(1, (max_epochs * n_items_per_epoch) // max(1, grad_accum))
    warmup_steps = int(round(float(cfg.get("warmup_ratio", 0.05)) * total_steps))
    print(f"[joint] total_steps={total_steps} warmup_steps={warmup_steps} "
          f"batch_size={bs} grad_accum={grad_accum}")

    # Evidence that we are NOT consulting any fixed-target file. Recorded
    # in the train history and the materialized config.
    no_fixed_targets_audit = {
        "joint_dep_targets_path": cfg.get("dep_targets_path"),
        "live_relmax_target": bool(cfg.get("live_relmax_target", False)),
        "live_q2_target": bool(cfg.get("live_q2_target", False)),
        "live_dep_target": bool(cfg.get("live_dep_target", False)),
        "drafter_train_mode": str(cfg.get("drafter_train_mode", "frozen")),
    }
    print(f"[joint] live-target audit: {no_fixed_targets_audit}")

    best_val = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    patience = int(cfg.get("early_stop_patience", 2))
    no_improve = 0
    step_counter = [0]
    t_start = time.time()

    for epoch in range(max_epochs):
        tr = _train_one_epoch(
            head, drafter, verifier, protocol,
            train_loader, optim, cfg, device,
            step_counter, warmup_steps, total_steps,
        )
        va = _eval_one_epoch(
            head, drafter, verifier, protocol,
            val_loader, cfg, device,
        )
        record = {"epoch": int(epoch), **tr, **{f"val_{k}": v for k, v in va.items()}}
        history.append(record)
        print(
            f"[joint] epoch {epoch}: train_loss={tr['train_loss']:.5f} "
            f"val_loss={va.get('val_loss', float('nan')):.5f} "
            f"target_mean={va.get('target_mean', float('nan')):.4f} "
            f"pred_mean={va.get('pred_mean', float('nan')):.4f} "
            f"alpha_mean={va.get('alpha_mean', float('nan')):.4f}"
        )
        if va["val_loss"] < best_val - 1e-6:
            best_val = float(va["val_loss"])
            best_epoch = int(epoch)
            no_improve = 0
            torch.save(head.state_dict(), out_dir / "model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[joint] early stop at epoch {epoch} (best epoch={best_epoch})")
                break

    if not (out_dir / "model.pt").exists():
        torch.save(head.state_dict(), out_dir / "model.pt")
        if best_epoch < 0:
            best_epoch = 0
            best_val = float(history[-1]["val_loss"]) if history else float("nan")

    # Reload the best checkpoint before gathering predictions so that what
    # we save matches `best_val_weighted_mse` (consistent with frozen trainer).
    head.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    head.eval()

    # Test-set evaluation under best checkpoint.
    test_metrics = _eval_one_epoch(
        head, drafter, verifier, protocol, test_loader, cfg, device,
    )

    # Gather predictions on val and test (V-free head forward; live targets
    # are computed inside `_forward_item` but we only retain the head output).
    val_preds = _gather_predictions(
        head, drafter, verifier, protocol, val_loader, cfg, device,
    )
    test_preds = _gather_predictions(
        head, drafter, verifier, protocol, test_loader, cfg, device,
    )

    elapsed = time.time() - t_start
    summary = {
        "run_name": str(cfg.get("run_name", "")),
        "regime": "joint",
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
        "live_target_audit": no_fixed_targets_audit,
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
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    for src, dst in (("model.pt", "model.pt"), ("config.yaml", "config.yaml")):
        link = ckpt_dir / dst
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / src)

    print(f"[joint] best_val={best_val:.5f} best_epoch={best_epoch} "
          f"elapsed={elapsed:.1f}s")
    print(f"[joint] wrote artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    sys.exit(main(cfg))
