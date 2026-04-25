"""Phase 18 — drafter-only token-slot alignment trainer.

Minimal direct alignment probe. Trains ONLY the drafter to match
verifier probability on the STORED drafted-token slot. No acceptance
head in the loop.

For each batch item (a stored RoundRecord):
  - Reconstruct full prefix via the Phase-15 cache.
  - Replay the drafter with grad (final-step gradients), matched RNG,
    to get current `draft_log_probs` (γ, V_drafter).
  - Run the verifier once on `concat(prefix, record.draft_tokens)` (no
    grad, bf16) to get `target_log_probs` (prefix_len+γ, V_verifier).
  - log_q[j] = draft_log_probs[j, record.draft_tokens[j]]
  - log_p[j] = target_log_probs[prefix_len-1+j, record.draft_tokens[j]]
  - L_align = mean_j (log_q[j] - log_p[j])²        masked by survived

Numerically safe clamps: both log_q and log_p are clamped below at
log(1e-10) ≈ -23 to prevent -inf. Gradients are clipped at norm 1.0.

Deliberately minimal:
  - Fixed token slot, not row KL.
  - Stored draft_tokens, not replayed tokens (lazy-label-stable under
    matched RNG: Phase 15 sanity confirmed tok_match=True at init).
  - Same full-prefix reconstruction path as Phase 16 live-Q2 joint.
  - No head, no multi-task, no additional auxiliary.

Saves `drafter.pt` + `config.yaml` + `train_history.json` to `out_dir`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N
from accpre.train.dataset import (
    JointAcceptanceDataset, joint_collate_fn, split_records_by_prompt,
)


LOG_EPS: float = math.log(1e-10)   # ≈ −23.03; prevent log-space −inf.


def _warmup_linear_lr_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return float(step + 1) / float(warmup_steps)


def _align_loss(
    log_q: torch.Tensor,        # (γ,)  grad-carrying drafter log-prob
    log_p: torch.Tensor,        # (γ,)  no-grad verifier log-prob
    survived: torch.Tensor,     # (γ,)  0/1 mask
) -> torch.Tensor:
    """Mean-squared log-prob gap on the drafted-token slot, masked by
    stored `survived_j`. Positive-only by construction.
    """
    lq = log_q.clamp(min=LOG_EPS)
    lp = log_p.clamp(min=LOG_EPS)
    se = (lq - lp) ** 2
    denom = survived.sum().clamp(min=1e-9)
    return (se * survived).sum() / denom


def load_config(path: str) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "mode": "drafter_align",
        "loss": "log_q_log_p_mse",
        "family": "hidden_per_pos_v2",   # purely for dataset compatibility
        "gamma": 8,
        "T": 2,
        "batch_size": 16,
        "backbone_lr": 5.0e-5,
        "weight_decay": 1.0e-4,
        "warmup_ratio": 0.05,
        "max_epochs": 3,
        "early_stop_patience": 2,
        "grad_clip_norm": 1.0,
        "seed": 0,
        "data_path": "data_collected/stage1_pp.pt",
    }
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a dict, got {type(cfg).__name__}")
    out = dict(defaults)
    out.update(cfg)
    if "run_name" not in out:
        raise KeyError("config missing required key: run_name")
    if out["mode"] != "drafter_align":
        raise ValueError(
            f"align_cli expects mode='drafter_align'; got {out['mode']!r}"
        )
    return out


def _load_protocol(path: str) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_name = cfg["run_name"]
    out_dir = args.out_dir or os.path.join("checkpoints", run_name)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    protocol_path = cfg.get("protocol_path", "configs/protocol.yaml")
    protocol = _load_protocol(protocol_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    print(f"[align] loading records from {cfg['data_path']}")
    records = load_records(cfg["data_path"], expected_protocol=protocol)
    target_gamma = int(cfg["gamma"])
    records = [r for r in records if r.gamma == target_gamma]
    tr = list(range(TRAIN_N))
    va = list(range(TRAIN_N, TRAIN_N + VAL_N))
    te = list(range(TRAIN_N + VAL_N, POOL_SIZE))
    parts = split_records_by_prompt(records, tr, va, te)
    family = str(cfg["family"])
    train_ds = JointAcceptanceDataset(parts["train"], family=family)
    val_ds   = JointAcceptanceDataset(parts["val"],   family=family)
    test_ds  = JointAcceptanceDataset(parts["test"],  family=family)
    print(
        f"[align] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    g = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=g, collate_fn=joint_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["batch_size"]), collate_fn=joint_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg["batch_size"]), collate_fn=joint_collate_fn,
    )

    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.train()
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device,
        dtype=torch.bfloat16,      # eval-only, bf16 to leave headroom
    )
    verifier.model.eval()
    print(
        f"[align] drafter fp32 (trained) + verifier bf16 (eval) loaded"
    )

    backbone_lr = float(cfg["backbone_lr"])
    optim = torch.optim.AdamW(
        drafter.model.parameters(),
        lr=backbone_lr, weight_decay=float(cfg["weight_decay"]),
    )
    total_steps = max(1, int(cfg["max_epochs"]) * len(train_loader))
    warmup_steps = int(float(cfg["warmup_ratio"]) * total_steps)
    grad_clip = float(cfg.get("grad_clip_norm", 1.0))

    def _step_lr(step: int) -> None:
        scale = _warmup_linear_lr_scale(step, warmup_steps)
        for pg in optim.param_groups:
            pg["lr"] = backbone_lr * scale

    def _forward_item(batch, i):
        """Single-step grad-carrying drafter forward + verifier forward.

        We do NOT re-enter `draft_with_features_grad` here because that
        uses `grad_at_final_only=True`, which means positions committed
        at an intermediate DDPM step have no grad path. For a handful
        of records (rare, ~0.4 % at γ=8, T=2) ALL positions are
        intermediate-committed, and the loss has no grad_fn — backward
        fails.

        Instead we run one forward pass at sigma=1 (fully-noised input,
        all γ draft positions MASK'd) through the drafter, apply the
        same SUBS parameterization, and index log_p_x0 at the γ draft
        positions. This yields grad on every position every time and
        is semantically valid for alignment: the drafter's all-masked
        prediction is the base distribution the DDPM loop samples
        from; pushing it toward verifier-p on drafted tokens is the
        same thing whichever step actually committed that token.
        """
        prefix = batch["prefix_ids"][i].to(device)
        tokens = batch["token_ids"][i].to(device).long()      # stored
        survived = batch["survived"][i].to(device)
        prefix_len = int(prefix.shape[0])
        gamma = int(tokens.shape[0])

        x = torch.full(
            (1, prefix_len + gamma),
            drafter.mask_index, dtype=torch.long, device=device,
        )
        x[0, :prefix_len] = prefix
        sigma = torch.ones(1, device=device)       # fully-noised timestep
        out = drafter.model(input_ids=x, timesteps=sigma)
        logits_all = out.logits if hasattr(out, "logits") else out
        log_p_x0 = drafter._subs_parameterization(logits_all[0], x[0])

        idx = torch.arange(gamma, device=device)
        log_q = log_p_x0[prefix_len + idx, tokens]            # (γ,) grad

        # Verifier on stored draft_tokens (no grad, bf16 → fp32 for arith).
        candidate = torch.cat([prefix, tokens])
        with torch.no_grad():
            target_log_probs = verifier.score(candidate).to(torch.float32)
        log_p = target_log_probs[prefix_len - 1 + idx, tokens]  # (γ,) no grad
        return log_q, log_p, survived

    @torch.no_grad()
    def _eval(loader):
        drafter.model.eval()
        tot, n = 0.0, 0
        for batch in loader:
            bs = len(batch["prefix_ids"])
            for i in range(bs):
                log_q, log_p, surv = _forward_item(batch, i)
                tot += float(_align_loss(log_q, log_p, surv).item())
                n += 1
        drafter.model.train()
        return tot / max(n, 1)

    best_val = float("inf")
    best_state = None
    patience_left = int(cfg["early_stop_patience"])
    step = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(int(cfg["max_epochs"])):
        drafter.model.train()
        running_loss, ep_n = 0.0, 0
        for batch in train_loader:
            _step_lr(step)
            optim.zero_grad()
            bs = len(batch["prefix_ids"])
            for i in range(bs):
                log_q, log_p, surv = _forward_item(batch, i)
                loss = _align_loss(log_q, log_p, surv)
                (loss / bs).backward()
                running_loss += float(loss.item())
                ep_n += 1
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    drafter.model.parameters(), max_norm=grad_clip,
                )
            optim.step()
            step += 1
        train_loss = running_loss / max(ep_n, 1)
        val_loss = _eval(val_loader)
        history.append({
            "epoch": epoch, "train": train_loss, "val": val_loss,
        })
        print(f"[align] epoch {epoch}: train={train_loss:.5f}  val={val_loss:.5f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in drafter.model.state_dict().items()
            }
            patience_left = int(cfg["early_stop_patience"])
        else:
            patience_left -= 1
            if patience_left < 0:
                print(f"[align] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        drafter.model.load_state_dict(best_state)

    test_loss = _eval(test_loader)
    print(f"[align] best_val={best_val:.5f}  test_loss={test_loss:.5f}")

    torch.save(
        drafter.model.state_dict(), os.path.join(out_dir, "drafter.pt"),
    )
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(
            {"run_name": run_name, "best_val": best_val,
             "test_loss": test_loss, "history": history},
            f, indent=2,
        )
    print(f"[align] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
