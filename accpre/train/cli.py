"""One canonical predictor trainer.

Reads a yaml config naming target + input family + mode + hyperparams,
builds the dataset / predictor / loss, runs the training loop, saves
the checkpoint, predictions, and a summary JSON.

Config keys (all mandatory unless noted):

  run_name:           str          # used for output paths
  data_path:          str          # path to stage1.pt
  target:             str          # "acceptance" | "committed_length"
  family:             str          # "numeric" | "hidden" | "hidden_numeric" | "upper_bound"
  mode:               str          # "frozen" | "joint"  (v1: frozen only)
  gamma:              int
  hidden_dim:         int = 128
  dropout:            float = 0.1
  batch_size:         int = 128
  head_lr:            float = 1e-3
  weight_decay:       float = 1e-4
  warmup_ratio:       float = 0.05
  max_epochs:         int = 10
  early_stop_patience: int = 2
  seed:               int = 0
  aux_bce:            bool = false        # acceptance only, off by default
  tau_grid:           list[float]         # length only; default [0.1,0.3,0.5,0.7,0.9]
  out_dir:            str                 # checkpoints/{run_name}
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from accpre.core.schema import RoundRecord, load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N
from accpre.predictors.acceptance_mlp import AcceptanceMLP
from accpre.train.dataset import (
    DEFAULT_TAU_GRID,
    FrozenAcceptanceDataset,
    FrozenLengthDataset,
    JointAcceptanceDataset,
    joint_collate_fn,
    split_records_by_prompt,
)
from accpre.train.losses import (
    ce_length,
    compute_length_class_weights,
    masked_bce_accepted,
    masked_mse_q2,
    masked_multi_thresh_bce,
    masked_soft_bce_q2,
    masked_threshold_bce_q2,
    masked_weighted_mse_q2,
)


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------


DEFAULTS: Dict[str, Any] = {
    "mode": "frozen",
    "gamma": 8,
    "hidden_dim": 128,
    "dropout": 0.1,
    "batch_size": 128,
    "head_lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.05,
    "max_epochs": 10,
    "early_stop_patience": 2,
    "seed": 0,
    "aux_bce": False,
    "loss": "mse",   # "mse" | "soft_bce"  (acceptance target only)
    "tau_grid": list(DEFAULT_TAU_GRID),
    # Phase 16: joint-only flag. When true, the joint loop recomputes
    # Q2_j per step from the CURRENT drafter's q and a live verifier
    # forward on the replayed draft_tokens, rather than using the
    # stored `record.min_pq_j`. Requires loading the GPT-2 verifier
    # alongside the drafter during training.
    "live_q2_target": False,
    # Phase 23: LR-schedule overrides. Leave `None` to keep the legacy
    # behavior (total_steps = max_epochs * n_train_batches,
    # warmup_steps = warmup_ratio * total_steps). Set either or both to
    # decouple the LR schedule from `max_epochs` so two runs that differ
    # only in `max_epochs` share a comparable warmup trajectory.
    "total_steps_override": None,
    "warmup_steps_override": None,
    # Phase 23 branch (a): 1-TV / dependence target.
    # `target: "dependence"` reuses the same 1A head architecture and the
    # same masked-MSE loss, but swaps the regression target from Q2_j
    # (the Leviathan ratio) to s_j = 1 - TV(Q^prefix_j, Q^revealed_j)
    # per position — see `accpre/core/dependence.py`.
    #   - Frozen:  provide `dep_targets_path` pointing to a
    #     `data_collected/stage1_pp_dep.pt` produced by
    #     `scripts/collect_dep_targets.py` on the pretrained drafter.
    #   - Joint:   set `live_dep_target: true` to recompute s_j per
    #     step from the CURRENT drafter (no precomputed file needed).
    #     `live_dep_target` is exclusive with `live_q2_target`.
    "dep_targets_path": None,
    "live_dep_target": False,
}


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a dict, got {type(cfg).__name__}")
    out = dict(DEFAULTS)
    out.update(cfg)
    # Required keys.
    for k in ("run_name", "data_path", "target", "family"):
        if k not in out:
            raise KeyError(f"config missing required key: {k!r}")
    if out["target"] not in ("acceptance", "committed_length", "dependence"):
        raise ValueError(
            f"target must be acceptance / committed_length / dependence, "
            f"got {out['target']!r}"
        )
    if out["target"] == "dependence":
        # 1-TV dependence uses the 1A frozen + joint-live machinery. It
        # reuses the acceptance-predictor heads (same MSE on [0, 1]) with
        # the regression target swapped for s_j.
        if out["mode"] == "joint_multitask":
            raise ValueError(
                "target='dependence' is not supported in joint_multitask "
                "mode (would need a parallel path for the length head)."
            )
        if out["mode"] == "frozen" and not out.get("dep_targets_path"):
            raise ValueError(
                "target='dependence' + mode='frozen' requires a "
                "`dep_targets_path` pointing to a dep-targets .pt file "
                "(produced by scripts/collect_dep_targets.py)."
            )
        if out["mode"] == "joint":
            if not bool(out.get("live_dep_target", False)) and not out.get("dep_targets_path"):
                raise ValueError(
                    "target='dependence' + mode='joint' requires either "
                    "`live_dep_target: true` (recompute s_j per step) or "
                    "a precomputed `dep_targets_path` (lazy dep target)."
                )
        if bool(out.get("live_q2_target", False)) and bool(out.get("live_dep_target", False)):
            raise ValueError(
                "Cannot set both live_q2_target=True and "
                "live_dep_target=True; targets are mutually exclusive."
            )
        if bool(out.get("live_q2_target", False)) and out["target"] == "dependence":
            raise ValueError(
                "target='dependence' is incompatible with "
                "live_q2_target=True (Q2 live target is for "
                "target='acceptance')."
            )
    else:
        if bool(out.get("live_dep_target", False)):
            raise ValueError(
                "live_dep_target=True requires target='dependence'."
            )
        if out.get("dep_targets_path") is not None:
            raise ValueError(
                "dep_targets_path is only meaningful with "
                "target='dependence'."
            )
    if out["mode"] not in ("frozen", "joint", "joint_multitask"):
        raise ValueError(
            f"mode must be frozen/joint/joint_multitask, got {out['mode']!r}"
        )
    if out["mode"] in ("joint", "joint_multitask"):
        if out["target"] not in ("acceptance", "dependence"):
            raise NotImplementedError(
                f"mode={out['mode']!r} for target={out['target']!r} is not "
                f"implemented yet. v1 supports joint for "
                f"acceptance / dependence; joint_multitask for acceptance only."
            )
        if out["mode"] == "joint_multitask" and out["target"] == "dependence":
            raise ValueError(
                "target='dependence' is not supported under joint_multitask "
                "(would need a parallel path for the length head)."
            )
        if out["family"] not in (
            "hidden", "hidden_numeric", "hidden_per_pos", "hidden_per_pos_v2",
        ):
            raise ValueError(
                f"mode={out['mode']!r} requires family in "
                f"{{hidden, hidden_numeric, hidden_per_pos, "
                f"hidden_per_pos_v2}}; got {out['family']!r}."
            )
        # Multitask joint pipes the same feature tensor into BOTH the
        # acceptance head and the τ-conditioned length head. The length
        # head currently supports only pooled 1-D features; any
        # per-position family would silently break its pack/forward
        # shapes. Refuse at config load rather than at runtime.
        if (
            out["mode"] == "joint_multitask"
            and out["family"] in ("hidden_per_pos", "hidden_per_pos_v2")
        ):
            raise ValueError(
                f"mode='joint_multitask' with family={out['family']!r} is not "
                "supported: the length head in this path expects pooled input. "
                "Use mode='joint' (acceptance-only) for per-position joint "
                "training, or a pooled family for multitask."
            )
        # head_arch gating in joint: token-embedding heads are only
        # supported for hidden_per_pos_v2 joint at present (Phase 15).
        head_arch = str(out.get("head_arch", "base"))
        if out["family"] == "hidden_per_pos_v2" and head_arch != "tokemb_mlp":
            raise ValueError(
                f"mode={out['mode']!r} + family='hidden_per_pos_v2' requires "
                f"head_arch='tokemb_mlp'; got head_arch={head_arch!r}."
            )
        if out["family"] != "hidden_per_pos_v2" and head_arch == "tokemb_mlp":
            raise ValueError(
                f"head_arch='tokemb_mlp' in joint mode requires "
                f"family='hidden_per_pos_v2'; got family={out['family']!r}."
            )
        # Phase 16: live-Q2 target requires the joint-multitask path to
        # ALSO recompute per-item Q2, which it currently doesn't. Refuse
        # that combination explicitly.
        if bool(out.get("live_q2_target", False)) and out["mode"] == "joint_multitask":
            raise ValueError(
                "live_q2_target=True is only wired through mode='joint' "
                "(acceptance-only) at present; joint_multitask would need "
                "its own branch."
            )
    elif bool(out.get("live_q2_target", False)):
        raise ValueError(
            "live_q2_target=True has no effect under mode='frozen'; "
            "the frozen target is record.min_pq_j by definition."
        )
    return out


# -----------------------------------------------------------------------
# Predictor factory
# -----------------------------------------------------------------------


def build_predictor(cfg: Dict[str, Any]):
    # Dependence target reuses the same regression-head architectures as
    # acceptance (sigmoid per position, in [0, 1]), so dispatch falls
    # into the acceptance branch by design.
    if cfg["target"] in ("acceptance", "dependence"):
        # Phase 5: all "hidden_per_pos*" families use a per-position head.
        if str(cfg["family"]).startswith("hidden_per_pos"):
            head_arch = str(cfg.get("head_arch", "base"))
            if head_arch == "base":
                from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPos
                return AcceptanceMLPPerPos(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                )
            if head_arch == "wide":
                from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPosWide
                return AcceptanceMLPPerPosWide(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                )
            if head_arch == "attn":
                from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPosAttn
                return AcceptanceMLPPerPosAttn(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                )
            if head_arch == "tx":
                # Phase 9: per-position transformer acceptance head (A1..A4).
                from accpre.predictors.pp_transformer import AcceptanceTx
                return AcceptanceTx(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    d_model=int(cfg.get("d_model", 128)),
                    num_layers=int(cfg.get("num_layers", 2)),
                    num_heads=int(cfg.get("num_heads", 4)),
                    dropout=cfg["dropout"],
                )
            if head_arch == "tokemb_mlp":
                # Phase 12 / 1A: per-position MLP + drafted-token embedding.
                from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmb
                return AcceptanceMLPPerPosTokEmb(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            if head_arch == "tokemb_mlp_sepln":
                # Phase 13 / 1A fusion-C ablation: separate-LN fusion.
                from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmbSepLN
                return AcceptanceMLPPerPosTokEmbSepLN(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            if head_arch == "tokemb_tx":
                # Phase 12 / 1B: per-position transformer + drafted-token embedding.
                from accpre.predictors.pp_tokemb import AcceptanceTxTokEmb
                return AcceptanceTxTokEmb(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    d_model=int(cfg.get("d_model", 128)),
                    num_layers=int(cfg.get("num_layers", 2)),
                    num_heads=int(cfg.get("num_heads", 4)),
                    dropout=cfg["dropout"],
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            if head_arch == "thresh_multihead":
                # Phase 14: direct threshold classifier. Same 1A body,
                # K=len(TAU_GRID) sigmoid heads, BCE on `1[Q2 >= τ]`.
                from accpre.predictors.pp_threshold import AcceptanceThresholdMultiHead
                return AcceptanceThresholdMultiHead(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            if head_arch == "tokemb_mlp_vhproj":
                # Phase 19 Vlite-H: 1A + projected verifier-hidden prefix.
                from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmbVHProj
                return AcceptanceMLPPerPosTokEmbVHProj(
                    family=cfg["family"],
                    gamma=cfg["gamma"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                    vh_proj_dim=int(cfg.get("vh_proj_dim", 64)),
                )
            raise ValueError(f"unknown head_arch {head_arch!r}")
        return AcceptanceMLP(
            family=cfg["family"],
            gamma=cfg["gamma"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        )
    if cfg["target"] == "committed_length":
        head_arch = str(cfg.get("head_arch", "base"))
        if head_arch == "tx":
            # Phase 9: per-position transformer length head (L1..L4).
            from accpre.predictors.pp_transformer import LengthTx
            return LengthTx(
                family=cfg["family"],
                gamma=cfg["gamma"],
                d_model=int(cfg.get("d_model", 128)),
                num_layers=int(cfg.get("num_layers", 2)),
                num_heads=int(cfg.get("num_heads", 4)),
                dropout=cfg["dropout"],
            )
        # Default: pooled LengthMLP (Stage 2A.2).
        from accpre.predictors.length_mlp import LengthMLP
        return LengthMLP(
            family=cfg["family"],
            gamma=cfg["gamma"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        )
    raise AssertionError(f"unknown target {cfg['target']!r}")


# -----------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------


def _prompt_index_sets() -> Tuple[List[int], List[int], List[int]]:
    """Return (train, val, test) global prompt index lists."""
    train = list(range(0, TRAIN_N))
    val = list(range(TRAIN_N, TRAIN_N + VAL_N))
    test = list(range(TRAIN_N + VAL_N, POOL_SIZE))
    return train, val, test


def _build_loaders(
    cfg: Dict[str, Any], records: List[RoundRecord],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Filter to records matching the configured gamma. Tail rounds with
    # cur_gamma < gamma (context-room or remaining-budget truncation)
    # cannot be batched against full-gamma records without padding; the
    # cleanest v1 choice is to drop them.
    target_gamma = int(cfg["gamma"])
    total = len(records)
    records = [r for r in records if r.gamma == target_gamma]
    dropped = total - len(records)
    if dropped > 0:
        print(
            f"[train] dropped {dropped} / {total} records with gamma != "
            f"{target_gamma}; {len(records)} remain"
        )

    tr, va, te = _prompt_index_sets()
    parts = split_records_by_prompt(records, tr, va, te)

    family = cfg["family"]
    if cfg["target"] in ("acceptance", "dependence"):
        # Dependence target reuses the acceptance pipeline; the only
        # change is the regression target's source. Pass through
        # `dep_targets_path` — `FrozenAcceptanceDataset` swaps in s_j for
        # Q2_j in the `q2_target` slot when it's set.
        dep_path = cfg.get("dep_targets_path") if cfg["target"] == "dependence" else None
        ds_cls = lambda recs: FrozenAcceptanceDataset(
            recs, family=family, dep_targets_path=dep_path,
        )
    else:
        ds_cls = lambda recs, seed=0: FrozenLengthDataset(
            recs, family=family, tau_grid=cfg["tau_grid"], seed=seed,
        )

    g = torch.Generator().manual_seed(int(cfg["seed"]))
    if cfg["target"] in ("acceptance", "dependence"):
        train_ds = ds_cls(parts["train"])
        val_ds = ds_cls(parts["val"])
        test_ds = ds_cls(parts["test"])
    else:
        train_ds = ds_cls(parts["train"], seed=cfg["seed"])
        val_ds = ds_cls(parts["val"], seed=cfg["seed"] + 1)
        test_ds = ds_cls(parts["test"], seed=cfg["seed"] + 2)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False,
    )
    return train_loader, val_loader, test_loader


def _compute_loss(
    predictor,
    batch: Dict[str, torch.Tensor],
    target: str,
    aux_bce: bool,
    loss_kind: str = "mse",
    length_class_weights: torch.Tensor | None = None,
    q_weight_lambda: float = 0.0,
    threshold_aux_weight: float = 0.0,
    threshold_aux_taus=(0.5, 0.7, 0.9),
    threshold_aux_sharpness: float = 10.0,
) -> torch.Tensor:
    if target in ("acceptance", "dependence"):
        # Dependence target flows through the same regression path as
        # acceptance; `batch["q2_target"]` carries s_j instead of Q2_j
        # when target='dependence'.
        # Pass token_ids when the predictor consumes them (tokemb heads);
        # other predictors accept **_kwargs and ignore this.
        fwd_kwargs = {}
        if "token_ids" in batch and hasattr(predictor, "token_emb"):
            fwd_kwargs["token_ids"] = batch["token_ids"]
        q_hat = predictor(batch["features"], **fwd_kwargs)
        if loss_kind == "mse":
            loss = masked_mse_q2(q_hat, batch["q2_target"], batch["survived"])
        elif loss_kind == "weighted_mse":
            loss = masked_weighted_mse_q2(
                q_hat, batch["q2_target"], batch["survived"],
                q_weight_lambda=q_weight_lambda,
            )
        elif loss_kind == "soft_bce":
            loss = masked_soft_bce_q2(q_hat, batch["q2_target"], batch["survived"])
        elif loss_kind == "multi_thresh_bce":
            # Phase 14: direct threshold classifier head — q_hat is (B, γ, K).
            _tau_grid = tuple(
                float(t) for t in getattr(predictor, "TAU_GRID", (0.5, 0.7, 0.9))
            )
            loss = masked_multi_thresh_bce(
                q_hat, batch["q2_target"], batch["survived"], taus=_tau_grid,
            )
        else:
            raise ValueError(
                f"unknown loss_kind {loss_kind!r}; use "
                f"'mse' | 'weighted_mse' | 'soft_bce' | 'multi_thresh_bce'"
            )
        if aux_bce:
            loss = loss + 0.1 * masked_bce_accepted(
                q_hat, batch["accepted"], batch["survived"]
            )
        if float(threshold_aux_weight) > 0.0:
            loss = loss + float(threshold_aux_weight) * masked_threshold_bce_q2(
                q_hat, batch["q2_target"], batch["survived"],
                taus=tuple(float(t) for t in threshold_aux_taus),
                sharpness=float(threshold_aux_sharpness),
            )
        return loss
    # committed_length — cross-entropy, optionally class-weighted.
    logits = predictor(batch["features"], batch["tau"])
    return ce_length(logits, batch["L_target"], class_weights=length_class_weights)


@torch.no_grad()
def _eval_loss(
    predictor,
    loader,
    target: str,
    aux_bce: bool,
    loss_kind: str = "mse",
    length_class_weights: torch.Tensor | None = None,
    q_weight_lambda: float = 0.0,
    threshold_aux_weight: float = 0.0,
    threshold_aux_taus=(0.5, 0.7, 0.9),
    threshold_aux_sharpness: float = 10.0,
) -> float:
    predictor.eval()
    tot, n = 0.0, 0
    for batch in loader:
        l = _compute_loss(
            predictor, batch, target, aux_bce, loss_kind, length_class_weights,
            q_weight_lambda=q_weight_lambda,
            threshold_aux_weight=threshold_aux_weight,
            threshold_aux_taus=threshold_aux_taus,
            threshold_aux_sharpness=threshold_aux_sharpness,
        )
        tot += float(l.item()) * int(batch["features"].shape[0])
        n += int(batch["features"].shape[0])
    predictor.train()
    return tot / max(n, 1)


def _warmup_linear_lr_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return float(step + 1) / float(warmup_steps)


def _compute_schedule(cfg: Dict[str, Any], n_train_batches: int) -> Tuple[int, int]:
    """Derive (total_steps, warmup_steps) from config with optional overrides.

    Default behavior (no overrides, matches pre-Phase-23 semantics):
        total_steps  = max(1, max_epochs * n_train_batches)
        warmup_steps = int(warmup_ratio * total_steps)

    Optional overrides (use when you want the LR warmup schedule to stay
    comparable across runs that differ in `max_epochs` — e.g. comparing
    live_jnt_5ep vs live_jnt_10ep where the 2× max_epochs silently
    doubled warmup and changed the LR trajectory from step 0):

        total_steps_override:  fixed total_steps; warmup_steps is still
                               derived as `warmup_ratio * total_steps`
                               unless warmup_steps_override is also set.
        warmup_steps_override: fixed warmup_steps, bypasses warmup_ratio
                               entirely.

    Both overrides are optional and both default to `None` in DEFAULTS.
    Return values are always non-negative integers.
    """
    if cfg.get("total_steps_override") is not None:
        total_steps = max(1, int(cfg["total_steps_override"]))
    else:
        total_steps = max(1, int(cfg["max_epochs"]) * int(n_train_batches))
    if cfg.get("warmup_steps_override") is not None:
        warmup_steps = max(0, int(cfg["warmup_steps_override"]))
    else:
        warmup_steps = int(float(cfg["warmup_ratio"]) * total_steps)
    return total_steps, warmup_steps


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------


def _joint_loss(
    q_hat: torch.Tensor,
    q2_target: torch.Tensor,
    survived: torch.Tensor,
    loss_kind: str,
) -> torch.Tensor:
    """Per-example acceptance loss for joint training."""
    if loss_kind == "mse":
        se = (q_hat - q2_target) ** 2 * survived
        return se.sum() / survived.sum().clamp(min=1e-9)
    if loss_kind == "soft_bce":
        q = q_hat.clamp(1e-7, 1.0 - 1e-7)
        bce = -(q2_target * torch.log(q) + (1.0 - q2_target) * torch.log(1.0 - q))
        return (bce * survived).sum() / survived.sum().clamp(min=1e-9)
    raise ValueError(f"unknown loss_kind {loss_kind!r}")


def _joint_assemble_features(
    drafter_hidden: torch.Tensor,
    family: str,
    numeric: torch.Tensor | None,
) -> torch.Tensor:
    """Build a feature vector from drafter hidden + optional numeric.

    Shape-matches `accpre.collect.features.extract_features` / the online
    builder, so the same head architecture applies.

    For `hidden_per_pos`, `drafter_hidden` is shaped `(γ, H)` (not pooled);
    the per-position head consumes it directly with no numeric concat.
    The `hidden_per_pos_v2` case lives in `_joint_assemble_v2` instead,
    because it needs the fresh `draft_log_probs` and `draft_tokens` (for
    the 5 drafter-side scalars), which this signature doesn't carry.
    """
    if family == "hidden":
        return drafter_hidden
    if family == "hidden_per_pos":
        return drafter_hidden  # (γ, H), fed straight to AcceptanceMLPPerPos
    if family == "hidden_numeric":
        if numeric is None:
            raise ValueError("hidden_numeric family requires numeric features")
        return torch.cat([drafter_hidden, numeric.to(drafter_hidden.device)])
    if family == "hidden_per_pos_v2":
        raise ValueError(
            "hidden_per_pos_v2 joint features are built via "
            "`_joint_assemble_v2` (needs draft_log_probs + draft_tokens)."
        )
    raise ValueError(f"unsupported joint family {family!r}")


@torch.no_grad()
def _compute_live_q2(
    verifier,
    prefix_ids: torch.Tensor,        # (prefix_len,) long, on device
    draft_tokens: torch.Tensor,      # (γ,)  long, on device
    draft_log_probs: torch.Tensor,   # (γ, V_drafter) — grad-carrying OK
    device,
) -> torch.Tensor:
    """Live regression target for the acceptance head: the canonical
    Leviathan ratio computed under the CURRENT drafter + verifier states.

    This helper does NOT implement the accept test itself (there is no
    Bernoulli draw, no survived tracking, no accept_rng — all of that
    lives in `accpre.core.accept.per_position_accept`). It only forms
    the real-valued target that the acceptance predictor regresses onto,
    evaluated on the drafter's replayed tokens under the current model
    states. Returns a detached (γ,) tensor.

    - `p` comes from `verifier.score(concat(prefix, draft_tokens))` at
      position `prefix_len - 1 + j`, evaluated on `draft_tokens[j]`.
    - `q` comes from the CURRENT `draft_log_probs[j, draft_tokens[j]]`.
      We detach the drafter's log-probs before forming the target so
      gradients from the training loss do NOT flow through the target —
      only through the prediction q_hat. The EPS=1e-10 clamp on both
      sides mirrors accept.py so the ratio is well-defined even at
      near-zero probability mass.
    - We don't backprop through the verifier at all, hence the
      `@torch.no_grad()` decorator on the helper.
    """
    prefix_len = int(prefix_ids.shape[0])
    gamma = int(draft_tokens.shape[0])
    candidate = torch.cat([prefix_ids, draft_tokens.long()]).to(device)
    target_log_probs = verifier.score(candidate)           # (L+γ, V_v)
    # Cast target log-probs to fp32 before exp/clamp: when the verifier
    # is loaded in bf16 (training path), the raw log-probs are bf16,
    # but we want the downstream ratio + min to be in fp32 so the target
    # matches the drafter's fp32 numerical regime and can be compared
    # apples-to-apples against the sanity/frozen fp32 path.
    target_log_probs = target_log_probs.to(torch.float32)
    idx = torch.arange(gamma, device=device)
    target_pos = prefix_len - 1 + idx                       # (γ,)
    EPS = 1e-10
    p_j = target_log_probs[target_pos, draft_tokens.long()].exp().clamp(min=EPS)
    q_j = draft_log_probs.detach().to(torch.float32)[
        idx, draft_tokens.long()
    ].exp().clamp(min=EPS)
    q2_live = torch.minimum(torch.ones_like(p_j), p_j / q_j)
    return q2_live.detach()


def _joint_assemble_v2(
    drafter_hidden: torch.Tensor,       # (γ, H) — final layer, unpooled
    draft_log_probs: torch.Tensor,      # (γ, V) — refreshed SUBS rows
    draft_tokens: torch.Tensor,         # (γ,)  — refreshed committed tokens
) -> torch.Tensor:
    """Assemble the `hidden_per_pos_v2` feature block (γ, H+5).

    Mirrors the online path's per-position scalar computation exactly
    (`accpre.eval.online_decode._run_single_prompt`, the `v2` branch),
    so joint training sees the same scalar layout the frozen and online
    paths do: `[q, log q, entropy, margin, top1_prob]`.

    The q / log q here are taken from the REFRESHED drafter's
    `draft_log_probs[j, tok_j]`, where `tok_j` is the token committed
    under matched RNG at this round. Under an unmodified drafter +
    matched RNG, these reproduce `record.q_j` up to floating-point
    tolerance; as the drafter fine-tunes, they drift.
    """
    probs = draft_log_probs.exp()                                   # (γ, V)
    entropy_t = -(probs * draft_log_probs).sum(dim=-1)              # (γ,)
    top2 = draft_log_probs.topk(2, dim=-1).values                   # (γ, 2)
    margin_t = top2[:, 0] - top2[:, 1]                              # (γ,)
    top1_t = probs.max(dim=-1).values                               # (γ,)
    idx = torch.arange(
        int(draft_tokens.shape[0]), device=draft_tokens.device,
    )
    q_t = probs[idx, draft_tokens.long()].clamp(min=1e-10)          # (γ,)
    log_q_t = torch.log(q_t)                                        # (γ,)
    scalars = torch.stack(
        [q_t, log_q_t, entropy_t, margin_t, top1_t], dim=1,
    )                                                               # (γ, 5)
    return torch.cat([drafter_hidden, scalars], dim=1)              # (γ, H+5)


def _train_joint_acceptance(cfg: Dict[str, Any], out_dir: str) -> None:
    """Stage 2B.1 joint-acceptance trainer.

    Semantics (DESIGN_PHASE2.md §2B):
      - Backbone: MDLM (fine-tuned via gradients through the final DDPM
        step only; intermediate forwards no_grad).
      - Head: AcceptanceMLP (or the per-position / tokemb_mlp variant).
      - Target:
          live_q2_target=False (lazy, default):
            stored Stage-1 Q2 (`record.min_pq_j`). The drafter's output
            q_j evolves during fine-tuning but the target does not; the
            documented trade-off is that we fine-tune the drafter to
            produce features that predict the ORIGINAL Q2 labels.
          live_q2_target=True (Phase 16):
            Q2_j = min(1, p/q) recomputed each step from the CURRENT
            drafter's q and a no-grad verifier forward on the replayed
            (prefix, draft_tokens) — see `_compute_live_q2`.
      - Loss: `cfg["loss"]` ("mse" or "soft_bce"), masked by `survived_j`.
      - Training context: FULL prefix per record, reconstructed by
        `JointAcceptanceDataset._build_full_prefix_cache` (Phase 15 fix)
        from the parent prompt's round chain. Exactly matches the prefix
        that Stage 2A.0 collection saw; no last-32-token approximation.
        (`record.prefix_tail` is still stored for audit/diagnostics but
        is NOT used as the training context.)
      - Randomness: each batch item's `round_rng_seed` is used to
        derive `draft_rng`, so the drafter samples the same trajectory
        it would sample at inference under the same seed.

    Saves `checkpoints/<run_name>/model.pt` (head state) +
    `drafter.pt` (fine-tuned MDLM state_dict) + `config.yaml`.
    `preds_{val,test}.pt` written against the fine-tuned predictor.
    """
    import yaml

    from accpre.core.draft_verify import _make_generator
    from accpre.core.protocol import ProtocolConfig
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.predictors.acceptance_mlp import AcceptanceMLP

    run_name = cfg["run_name"]
    os.makedirs(out_dir, exist_ok=True)

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    # --- protocol ---
    protocol_path = cfg.get("protocol_path", "configs/protocol.yaml")
    with open(protocol_path, "r") as f:
        proto_dict = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in proto_dict and isinstance(proto_dict[k], str):
            proto_dict[k] = int(proto_dict[k], 0)
    protocol = ProtocolConfig(**proto_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[protocol.dtype]

    # --- data ---
    print(f"[joint] loading records from {cfg['data_path']}")
    records = load_records(cfg["data_path"], expected_protocol=protocol)
    target_gamma = int(cfg["gamma"])
    records = [r for r in records if r.gamma == target_gamma]
    tr, va, te = _prompt_index_sets()
    parts = split_records_by_prompt(records, tr, va, te)
    # Optional precomputed dep-targets path (lazy dep target — used when
    # target='dependence' without live_dep_target). When live_dep_target
    # is true this is ignored; the trainer recomputes s_j per step.
    _dep_path = (
        cfg.get("dep_targets_path")
        if cfg.get("target") == "dependence"
        and not bool(cfg.get("live_dep_target", False))
        else None
    )
    train_ds = JointAcceptanceDataset(
        parts["train"], family=cfg["family"], dep_targets_path=_dep_path,
    )
    val_ds = JointAcceptanceDataset(
        parts["val"], family=cfg["family"], dep_targets_path=_dep_path,
    )
    test_ds = JointAcceptanceDataset(
        parts["test"], family=cfg["family"], dep_targets_path=_dep_path,
    )
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]),
                              shuffle=True, generator=g,
                              collate_fn=joint_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]),
                            collate_fn=joint_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]),
                             collate_fn=joint_collate_fn)
    print(f"[joint] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # --- models ---
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.train()   # fine-tune mode

    # Phase 16: live-Q2 target loads the verifier once here and keeps
    # it in eval mode for no-grad forwards inside the training loop.
    # The verifier is loaded in bf16 (not protocol.dtype) to leave GPU
    # headroom for the drafter's grad/optim buffers on a 10GB MIG slice:
    #   MDLM fp32 (170M) ≈ 680MB weights + 680MB grads + 2×680MB Adam
    #     state ≈ 2.7GB, + activations ~1GB ≈ 3.7GB.
    #   GPT-2-XL fp32 (1.5B) ≈ 6GB weights → total ≈ 10GB => OOM.
    #   GPT-2-XL bf16 halves to ≈ 3GB → total ≈ 6.7GB => fits.
    # The verifier is eval-only (no grad), so bf16 numeric noise (~1e-3)
    # is on par with MDLM's own internal bf16 autocast noise — within the
    # tolerance we already accepted in Phase 15 sanity.
    use_live_q2 = bool(cfg.get("live_q2_target", False))
    use_live_dep = bool(cfg.get("live_dep_target", False))
    verifier = None
    if use_live_q2:
        from accpre.models.verifier_gpt2 import GPT2Verifier
        verifier = GPT2Verifier(
            model_name=protocol.verifier_model,
            device=device,
            dtype=torch.bfloat16,
        )
        verifier.model.eval()
        print(
            f"[joint] live_q2_target=True: loaded verifier "
            f"{protocol.verifier_model!r} (bf16) for per-step Q2 recomputation."
        )
    if use_live_dep:
        # Phase 23 branch (a): dependence target recomputed per step from
        # the current drafter. No verifier is needed — the target is
        # purely a property of the drafter's distributions at the γ
        # positions given progressively more revealed tokens.
        print(
            "[joint] live_dep_target=True: per-step s_j = 1 - TV("
            "Q^prefix_j, Q^revealed_j) from the current drafter."
        )

    head_arch = str(cfg.get("head_arch", "base"))
    if cfg["family"] == "hidden_per_pos_v2" and head_arch == "tokemb_mlp":
        # Phase 15: 1A-style head (token-embedding MLP) on top of the
        # refreshed drafter features (H + 5 scalars + learned token_emb).
        from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmb
        head = AcceptanceMLPPerPosTokEmb(
            family=cfg["family"], gamma=target_gamma,
            hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
            token_emb_dim=int(cfg.get("token_emb_dim", 64)),
        ).to(device)
    elif cfg["family"] == "hidden_per_pos":
        from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPos
        head = AcceptanceMLPPerPos(
            family=cfg["family"], gamma=target_gamma,
            hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
        ).to(device)
    else:
        head = AcceptanceMLP(
            family=cfg["family"], gamma=target_gamma,
            hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
        ).to(device)
    head.train()
    print(f"[joint] predictor: {head.describe()}")

    # --- optimizer: separate LRs for backbone and head ---
    head_lr = float(cfg["head_lr"])
    backbone_lr = float(cfg.get("backbone_lr", 5.0e-5))
    optim = torch.optim.AdamW(
        [
            {"params": list(head.parameters()), "lr": head_lr},
            {"params": list(drafter.model.parameters()), "lr": backbone_lr},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps, warmup_steps = _compute_schedule(cfg, len(train_loader))
    print(
        f"[joint] schedule: total_steps={total_steps} warmup_steps={warmup_steps}"
        + (
            " (from total_steps_override)"
            if cfg.get("total_steps_override") is not None
            else ""
        )
        + (
            " (warmup_steps_override)"
            if cfg.get("warmup_steps_override") is not None
            else ""
        )
    )

    def _step_lr(step: int) -> None:
        scale = _warmup_linear_lr_scale(step, warmup_steps)
        for group, lr in zip(optim.param_groups, (head_lr, backbone_lr)):
            group["lr"] = lr * scale

    # --- helper: one forward pass for a single batch item ---
    def _forward_item(batch, i):
        prefix = batch["prefix_ids"][i].to(device)
        seed = int(batch["round_rng_seed"][i].item())
        draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
        # Per-position pool required for any `hidden_per_pos*` family;
        # all others (hidden, hidden_numeric) remain mean-pooled.
        _pool = (
            "per_position"
            if str(cfg["family"]).startswith("hidden_per_pos")
            else "mean"
        )
        # Joint mode is currently restricted (see load_config whitelist) to
        # {hidden, hidden_numeric, hidden_per_pos, hidden_per_pos_v2} —
        # all final-layer-only families. Pass layers=(-1,) explicitly so
        # future extensions to multi-layer joint families will require an
        # explicit code change here rather than silently relying on a
        # kwarg default.
        draft_tokens, draft_log_probs, drafter_hidden = drafter.draft_with_features_grad(
            prefix_ids=prefix,
            gamma=target_gamma,
            T=int(cfg.get("T", 2)),
            temperature=protocol.temperature,
            q_mode=protocol.q_mode,
            generator=draft_rng,
            pool=_pool,
            layers=(-1,),
        )
        # Use REPLAYED tokens for tok-emb input under any *live* target
        # (live_q2 OR live_dep). See Phase 23 fix rationale: live targets
        # are computed on replayed tokens, feats come from replayed
        # drafter outputs, so the head's tok-emb input must track the
        # same tokens. Lazy (stored-target) modes keep stored tok-ids.
        _use_replayed_tok = use_live_q2 or use_live_dep
        if cfg["family"] == "hidden_per_pos_v2":
            feats = _joint_assemble_v2(
                drafter_hidden, draft_log_probs, draft_tokens,
            )                                                      # (γ, H+5)
            tok = (
                draft_tokens.detach().to(device).long()
                if _use_replayed_tok
                else batch["token_ids"][i].to(device)
            )                                                      # (γ,)
            q_hat = head.forward(
                feats.unsqueeze(0), token_ids=tok.unsqueeze(0),
            ).squeeze(0)
        else:
            numeric = (
                batch["numeric"][i] if "numeric" in batch else None
            )
            feats = _joint_assemble_features(
                drafter_hidden, cfg["family"], numeric,
            )
            q_hat = head.forward(feats.unsqueeze(0)).squeeze(0)
        if use_live_q2:
            # Phase 16: recompute Q2 under the CURRENT drafter + verifier.
            q2_target = _compute_live_q2(
                verifier, prefix, draft_tokens,
                draft_log_probs, device,
            )
        elif use_live_dep:
            # Phase 23 branch (a): recompute s_j = 1 - TV(Q^prefix_j,
            # Q^revealed_j) from the CURRENT drafter on the REPLAYED
            # draft_tokens. No verifier. No gradients flow through the
            # target — `compute_dependence_target` is @torch.no_grad.
            from accpre.core.dependence import compute_dependence_target
            q2_target = compute_dependence_target(
                drafter, prefix, draft_tokens.detach(), int(target_gamma),
            ).to(device)
        else:
            q2_target = batch["q2_target"][i].to(device)
        # `survived` stays the stored record.survived_j mask, as in
        # Phase 15. See PHASE16 report: keeping the mask consistent
        # across lazy/live variants makes the comparison apples-to-
        # apples; a dynamic mask would be a separate variable.
        survived = batch["survived"][i].to(device)
        return q_hat, q2_target, survived

    # --- eval helper (no_grad) ---
    @torch.no_grad()
    def _eval(loader):
        head.eval()
        drafter.model.eval()
        tot, n = 0.0, 0
        for batch in loader:
            bs = len(batch["prefix_ids"])
            for i in range(bs):
                q_hat, q2_target, survived = _forward_item(batch, i)
                loss = _joint_loss(q_hat, q2_target, survived, cfg["loss"])
                tot += float(loss.item())
                n += 1
        head.train()
        drafter.model.train()
        return tot / max(n, 1)

    # --- training loop ---
    best_val = float("inf")
    best_head = None
    best_drafter = None
    patience_left = int(cfg["early_stop_patience"])
    step = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(int(cfg["max_epochs"])):
        head.train()
        drafter.model.train()
        epoch_loss_sum, epoch_n = 0.0, 0
        for batch in train_loader:
            _step_lr(step)
            optim.zero_grad()
            bs = int(len(batch["prefix_ids"]))
            for i in range(bs):
                q_hat, q2_target, survived = _forward_item(batch, i)
                loss = _joint_loss(q_hat, q2_target, survived, cfg["loss"])
                # Accumulate grad, divide by bs for correct mean gradient.
                (loss / bs).backward()
                epoch_loss_sum += float(loss.item())
                epoch_n += 1
            optim.step()
            step += 1

        train_loss = epoch_loss_sum / max(epoch_n, 1)
        val_loss = _eval(val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[joint] epoch {epoch}: train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_drafter = {k: v.detach().cpu().clone()
                            for k, v in drafter.model.state_dict().items()}
            patience_left = int(cfg["early_stop_patience"])
        else:
            patience_left -= 1
            if patience_left < 0:
                print(f"[joint] early stopping at epoch {epoch}")
                break

    if best_head is not None:
        head.load_state_dict(best_head)
    if best_drafter is not None:
        drafter.model.load_state_dict(best_drafter)

    test_loss = _eval(test_loader)
    print(f"[joint] best_val={best_val:.5f} test_loss={test_loss:.5f}")

    # --- save artifacts ---
    torch.save(head.state_dict(), os.path.join(out_dir, "model.pt"))
    torch.save(drafter.model.state_dict(), os.path.join(out_dir, "drafter.pt"))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump({"run_name": run_name, "best_val": best_val,
                   "test_loss": test_loss, "history": history}, f, indent=2)

    # Save per-record Q̂ on val + test using the fine-tuned drafter + head.
    _save_joint_predictions(head, drafter, protocol, cfg, val_loader,
                            os.path.join(out_dir, "preds_val.pt"))
    _save_joint_predictions(head, drafter, protocol, cfg, test_loader,
                            os.path.join(out_dir, "preds_test.pt"))
    print(f"[joint] wrote artifacts to {out_dir}")


@torch.no_grad()
def _save_joint_predictions(head, drafter, protocol, cfg, loader, path):
    from accpre.core.draft_verify import _make_generator

    head.eval()
    drafter.model.eval()
    device = drafter.device
    target_gamma = int(cfg["gamma"])
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        bs = int(len(batch["prefix_ids"]))
        for i in range(bs):
            prefix = batch["prefix_ids"][i].to(device)
            seed = int(batch["round_rng_seed"][i].item())
            draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
            _pool = (
                "per_position"
                if str(cfg["family"]).startswith("hidden_per_pos")
                else "mean"
            )
            # layers=(-1,) explicit: joint mode is restricted to
            # final-layer-only families; see _train_joint_acceptance.
            draft_tokens, draft_log_probs, h = drafter.draft_with_features(
                prefix_ids=prefix, gamma=target_gamma,
                T=int(cfg.get("T", 2)),
                temperature=protocol.temperature, q_mode=protocol.q_mode,
                generator=draft_rng,
                pool=_pool,
                layers=(-1,),
            )
            if h is None:
                if _pool == "per_position":
                    h = torch.zeros(target_gamma, drafter.hidden_size, device=device)
                else:
                    h = torch.zeros(drafter.hidden_size, device=device)
            else:
                h = h.to(device)
            # Phase 23: live targets (Q2 or dependence) use replayed
            # tokens for the tok-emb input; lazy modes keep stored tokens.
            _use_live_q2 = bool(cfg.get("live_q2_target", False))
            _use_live_dep = bool(cfg.get("live_dep_target", False))
            _use_replayed_tok = _use_live_q2 or _use_live_dep
            if cfg["family"] == "hidden_per_pos_v2":
                feats = _joint_assemble_v2(
                    h, draft_log_probs.to(device), draft_tokens.to(device),
                )
                tok = (
                    draft_tokens.to(device).long()
                    if _use_replayed_tok
                    else batch["token_ids"][i].to(device)
                )
                q = head.forward(
                    feats.unsqueeze(0), token_ids=tok.unsqueeze(0),
                ).squeeze(0).detach().cpu()
            else:
                numeric = batch["numeric"][i] if "numeric" in batch else None
                feats = _joint_assemble_features(h, cfg["family"], numeric)
                q = head.forward(feats.unsqueeze(0)).squeeze(0).detach().cpu()
            # When the training target is live-dep, store the LIVE s_j
            # (under the fine-tuned drafter) as the preds' target field
            # so MAE matches what training was optimizing against. Other
            # modes keep the stored lazy target (Q2 or precomputed s).
            if _use_live_dep:
                from accpre.core.dependence import compute_dependence_target
                tgt_tensor = compute_dependence_target(
                    drafter, prefix, draft_tokens.detach(), int(target_gamma),
                )
                tgt_list = tgt_tensor.tolist()
            else:
                tgt_list = batch["q2_target"][i].tolist()
            rows.append({
                "record_idx": int(batch["record_idx"][i].item()),
                "q2_hat": q.tolist(),
                "q2_target": tgt_list,
                "survived": batch["survived"][i].tolist(),
                "accepted": batch["accepted"][i].tolist(),
            })
    torch.save(rows, path)


def _train_joint_multitask_acceptance(cfg: Dict[str, Any], out_dir: str) -> None:
    """Stage 2B.2 joint multitask trainer, acceptance-focused.

    Shared MDLM backbone + two heads:
      - acceptance head (AcceptanceMLP) — used at inference.
      - τ-conditioned length head (LengthMLP) — auxiliary supervision
        only, its outputs are NOT used by online decode.

    Per-item loss: `w_acc * L_acc + w_len * L_len`, with fixed weights
    from config (defaults `w_acc = 1.0`, `w_len = 0.3` — soft-BCE
    acceptance converges near 0.6, weighted-CE length near 1.7 in
    Stage 2A, so `0.3 * 1.7 ≈ 0.5` keeps the two contributions on
    comparable scale without introducing learnable weights).

    L_acc uses `cfg["loss"]` ("mse" or "soft_bce"). L_len uses weighted
    cross-entropy over `{0, ..., γ}` with inverse-frequency class
    weights computed from the training split (Stage 2A fix).

    Saves:
      - `model.pt`          — acceptance head state (online_decode uses this)
      - `length_head.pt`    — aux length head state (diagnostic)
      - `drafter.pt`        — fine-tuned MDLM state
      - `config.yaml`, `train_history.json`, `preds_{val,test}.pt`
    """
    import random as _random
    import yaml

    from accpre.core.commit import commit_threshold
    from accpre.core.draft_verify import _make_generator
    from accpre.core.protocol import ProtocolConfig
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.predictors.acceptance_mlp import AcceptanceMLP
    from accpre.predictors.length_mlp import LengthMLP

    run_name = cfg["run_name"]
    os.makedirs(out_dir, exist_ok=True)

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    # Separate RNG for per-item τ sampling (so it does not perturb
    # PyTorch's global RNG, which drives optimizer init / dropout etc.).
    tau_rng = _random.Random(int(cfg["seed"]) ^ 0xABCDEF)
    tau_grid = tuple(float(t) for t in cfg.get("tau_grid", DEFAULT_TAU_GRID))

    # --- protocol ---
    protocol_path = cfg.get("protocol_path", "configs/protocol.yaml")
    with open(protocol_path, "r") as f:
        proto_dict = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in proto_dict and isinstance(proto_dict[k], str):
            proto_dict[k] = int(proto_dict[k], 0)
    protocol = ProtocolConfig(**proto_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[protocol.dtype]

    # --- data ---
    print(f"[mtjoint] loading records from {cfg['data_path']}")
    records = load_records(cfg["data_path"], expected_protocol=protocol)
    target_gamma = int(cfg["gamma"])
    records = [r for r in records if r.gamma == target_gamma]
    tr, va, te = _prompt_index_sets()
    parts = split_records_by_prompt(records, tr, va, te)
    train_ds = JointAcceptanceDataset(parts["train"], family=cfg["family"])
    val_ds = JointAcceptanceDataset(parts["val"], family=cfg["family"])
    test_ds = JointAcceptanceDataset(parts["test"], family=cfg["family"])
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]),
                              shuffle=True, generator=g,
                              collate_fn=joint_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]),
                            collate_fn=joint_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]),
                             collate_fn=joint_collate_fn)
    print(f"[mtjoint] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # --- length class weights from the TRAIN split (Stage 2A fix) ---
    length_class_weights = compute_length_class_weights(
        parts["train"], tau_grid, target_gamma,
    ).to(device)
    print(
        f"[mtjoint] length_class_weights = "
        f"{[round(float(w), 3) for w in length_class_weights.tolist()]}"
    )

    # --- models ---
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.train()

    # Per-position families require AcceptanceMLPPerPos (shared MLP applied
    # position-wise over `(γ, H)`); pooled families use AcceptanceMLP.
    # Mirror the dispatch logic used by `_train_joint_acceptance`.
    if cfg["family"] == "hidden_per_pos":
        from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPos
        acc_head = AcceptanceMLPPerPos(
            family=cfg["family"], gamma=target_gamma,
            hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
        ).to(device)
    else:
        acc_head = AcceptanceMLP(
            family=cfg["family"], gamma=target_gamma,
            hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
        ).to(device)
    # Length head currently supports only pooled families; load_config's
    # joint-mode whitelist excludes hidden_per_pos_* for length, but if
    # someone wires it up later, LengthMLP would need a per-pos variant.
    len_head = LengthMLP(
        family=cfg["family"], gamma=target_gamma,
        hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
    ).to(device)
    acc_head.train()
    len_head.train()
    print(f"[mtjoint] acc head: {acc_head.describe()}")
    print(f"[mtjoint] len head: {len_head.describe()}")

    # --- optimizer ---
    head_lr = float(cfg["head_lr"])
    backbone_lr = float(cfg.get("backbone_lr", 5.0e-5))
    optim = torch.optim.AdamW(
        [
            {"params": list(acc_head.parameters()), "lr": head_lr},
            {"params": list(len_head.parameters()), "lr": head_lr},
            {"params": list(drafter.model.parameters()), "lr": backbone_lr},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps, warmup_steps = _compute_schedule(cfg, len(train_loader))

    # Loss weights (fixed; see DESIGN_PHASE2 §2B.2 / docstring above).
    w_acc = float(cfg.get("w_acc", 1.0))
    w_len = float(cfg.get("w_len", 0.3))
    print(f"[mtjoint] fixed loss weights: w_acc={w_acc}  w_len={w_len}")

    def _step_lr(step: int) -> None:
        scale = _warmup_linear_lr_scale(step, warmup_steps)
        for group, lr in zip(optim.param_groups,
                             (head_lr, head_lr, backbone_lr)):
            group["lr"] = lr * scale

    def _acc_loss(q_hat, q2_target, survived):
        if cfg["loss"] == "mse":
            se = (q_hat - q2_target) ** 2 * survived
            return se.sum() / survived.sum().clamp(min=1e-9)
        if cfg["loss"] == "soft_bce":
            q = q_hat.clamp(1e-7, 1.0 - 1e-7)
            bce = -(q2_target * torch.log(q)
                    + (1.0 - q2_target) * torch.log(1.0 - q))
            return (bce * survived).sum() / survived.sum().clamp(min=1e-9)
        raise ValueError(f"unknown loss_kind {cfg['loss']!r}")

    def _forward_item(batch, i):
        """One training-item forward. Returns (loss_acc, loss_len, total_loss)."""
        prefix = batch["prefix_ids"][i].to(device)
        seed = int(batch["round_rng_seed"][i].item())
        draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
        _pool = "per_position" if cfg["family"] == "hidden_per_pos" else "mean"
        # layers=(-1,) explicit: joint_multitask mode is restricted to
        # final-layer-only families (load_config excludes hidden_per_pos_ml).
        _, _, drafter_hidden = drafter.draft_with_features_grad(
            prefix_ids=prefix, gamma=target_gamma,
            T=int(cfg.get("T", 2)),
            temperature=protocol.temperature, q_mode=protocol.q_mode,
            generator=draft_rng,
            pool=_pool,
            layers=(-1,),
        )
        numeric = batch["numeric"][i] if "numeric" in batch else None
        feats = _joint_assemble_features(drafter_hidden, cfg["family"], numeric)

        # --- acceptance branch ---
        q_hat = acc_head.forward(feats.unsqueeze(0)).squeeze(0)
        q2_target = batch["q2_target"][i].to(device)
        survived = batch["survived"][i].to(device)
        l_acc = _acc_loss(q_hat, q2_target, survived)

        # --- length branch (aux supervision) ---
        tau_val = tau_rng.choice(tau_grid)
        L_target_scalar = commit_threshold(
            batch["min_pq_j"][i].tolist(), float(tau_val),
        )
        tau_tensor = torch.tensor(float(tau_val),
                                  dtype=torch.float32, device=device)
        logits = len_head.forward(feats.unsqueeze(0),
                                  tau_tensor.unsqueeze(0))
        target_tensor = torch.tensor([int(L_target_scalar)],
                                     dtype=torch.long, device=device)
        l_len = torch.nn.functional.cross_entropy(
            logits, target_tensor, weight=length_class_weights,
        )

        total = w_acc * l_acc + w_len * l_len
        return l_acc, l_len, total

    @torch.no_grad()
    def _eval(loader):
        acc_head.eval(); len_head.eval(); drafter.model.eval()
        tot_acc, tot_len, tot_total, n = 0.0, 0.0, 0.0, 0
        for batch in loader:
            for i in range(int(len(batch["prefix_ids"]))):
                la, ll, tl = _forward_item(batch, i)
                tot_acc += float(la.item())
                tot_len += float(ll.item())
                tot_total += float(tl.item())
                n += 1
        acc_head.train(); len_head.train(); drafter.model.train()
        return (tot_acc / max(n, 1), tot_len / max(n, 1), tot_total / max(n, 1))

    # --- training loop ---
    best_val_total = float("inf")
    best_states = None
    patience_left = int(cfg["early_stop_patience"])
    step = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(int(cfg["max_epochs"])):
        acc_head.train(); len_head.train(); drafter.model.train()
        sum_acc, sum_len, sum_total, n_items = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            _step_lr(step)
            optim.zero_grad()
            bs = int(len(batch["prefix_ids"]))
            for i in range(bs):
                la, ll, tl = _forward_item(batch, i)
                (tl / bs).backward()
                sum_acc += float(la.item())
                sum_len += float(ll.item())
                sum_total += float(tl.item())
                n_items += 1
            optim.step()
            step += 1

        train_acc = sum_acc / max(n_items, 1)
        train_len = sum_len / max(n_items, 1)
        train_total = sum_total / max(n_items, 1)
        val_acc, val_len, val_total = _eval(val_loader)
        history.append({
            "epoch": epoch,
            "train_acc": train_acc, "train_len": train_len, "train_total": train_total,
            "val_acc": val_acc, "val_len": val_len, "val_total": val_total,
        })
        print(
            f"[mtjoint] epoch {epoch}: "
            f"train acc={train_acc:.4f} len={train_len:.4f} tot={train_total:.4f}  "
            f"val acc={val_acc:.4f} len={val_len:.4f} tot={val_total:.4f}"
        )

        if val_total < best_val_total - 1e-6:
            best_val_total = val_total
            best_states = {
                "acc_head": {k: v.detach().cpu().clone()
                             for k, v in acc_head.state_dict().items()},
                "len_head": {k: v.detach().cpu().clone()
                             for k, v in len_head.state_dict().items()},
                "drafter": {k: v.detach().cpu().clone()
                            for k, v in drafter.model.state_dict().items()},
            }
            patience_left = int(cfg["early_stop_patience"])
        else:
            patience_left -= 1
            if patience_left < 0:
                print(f"[mtjoint] early stopping at epoch {epoch}")
                break

    if best_states is not None:
        acc_head.load_state_dict(best_states["acc_head"])
        len_head.load_state_dict(best_states["len_head"])
        drafter.model.load_state_dict(best_states["drafter"])

    test_acc, test_len, test_total = _eval(test_loader)
    print(
        f"[mtjoint] best_val_total={best_val_total:.4f} | "
        f"test acc={test_acc:.4f} len={test_len:.4f} total={test_total:.4f}"
    )

    # --- save artifacts ---
    torch.save(acc_head.state_dict(), os.path.join(out_dir, "model.pt"))
    torch.save(len_head.state_dict(), os.path.join(out_dir, "length_head.pt"))
    torch.save(drafter.model.state_dict(), os.path.join(out_dir, "drafter.pt"))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump({
            "run_name": run_name,
            "best_val_total": best_val_total,
            "test_acc": test_acc, "test_len": test_len, "test_total": test_total,
            "history": history,
        }, f, indent=2)

    # Save per-record predictions (acceptance head only — what online uses).
    _save_joint_predictions(acc_head, drafter, protocol, cfg, val_loader,
                            os.path.join(out_dir, "preds_val.pt"))
    _save_joint_predictions(acc_head, drafter, protocol, cfg, test_loader,
                            os.path.join(out_dir, "preds_test.pt"))
    print(f"[mtjoint] wrote artifacts to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2A/2B predictor trainer.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory; defaults to checkpoints/{run_name}.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_name = cfg["run_name"]
    out_dir = args.out_dir or os.path.join("checkpoints", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Joint-mode dispatch.
    if cfg["mode"] == "joint":
        _train_joint_acceptance(cfg, out_dir)
        return
    if cfg["mode"] == "joint_multitask":
        _train_joint_multitask_acceptance(cfg, out_dir)
        return

    # Seeds.
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    # Load the frozen protocol so `load_records` can enforce BOTH
    # within-file consistency AND consistency with the configured
    # protocol. Joint paths already do this; frozen training had been
    # running without an explicit protocol check (only within-file
    # consistency was enforced).
    import yaml
    from accpre.core.protocol import ProtocolConfig
    protocol_path = cfg.get("protocol_path", "configs/protocol.yaml")
    with open(protocol_path, "r") as f:
        proto_dict = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in proto_dict and isinstance(proto_dict[k], str):
            proto_dict[k] = int(proto_dict[k], 0)
    protocol = ProtocolConfig(**proto_dict)

    # Data.
    print(f"[train] loading records from {cfg['data_path']}")
    records = load_records(cfg["data_path"], expected_protocol=protocol)
    print(f"[train] {len(records)} records loaded")

    train_loader, val_loader, test_loader = _build_loaders(cfg, records)
    print(
        f"[train] loaders: "
        f"train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    # Class-weighted CE for length predictors (optional).
    length_class_weights = None
    if (
        cfg["target"] == "committed_length"
        and str(cfg.get("loss", "ce")) == "weighted_ce"
    ):
        # Compute weights from the TRAIN split only. Under `FrozenLengthDataset`
        # each record yields one (features, τ, L_τ) triple per epoch (τ is
        # freshly sampled each __getitem__). Using the record × full-tau_grid
        # product gives the expected per-class frequency.
        train_records = train_loader.dataset.records
        length_class_weights = compute_length_class_weights(
            train_records, tuple(cfg["tau_grid"]), int(cfg["gamma"]),
        )
        print(
            f"[train] length_class_weights = "
            f"{[round(float(w), 3) for w in length_class_weights.tolist()]}"
        )

    # Predictor.
    predictor = build_predictor(cfg)
    print(f"[train] predictor: {predictor.describe()}")

    # Optimizer.
    optim = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(cfg["head_lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps, warmup_steps = _compute_schedule(cfg, len(train_loader))

    # Training loop with early stopping.
    best_val = float("inf")
    best_state = None
    patience_left = int(cfg["early_stop_patience"])
    step = 0
    history = []

    for epoch in range(int(cfg["max_epochs"])):
        predictor.train()
        epoch_loss_sum, epoch_n = 0.0, 0
        for batch in train_loader:
            for g in optim.param_groups:
                g["lr"] = float(cfg["head_lr"]) * _warmup_linear_lr_scale(step, warmup_steps)
            optim.zero_grad()
            loss = _compute_loss(
                predictor, batch, cfg["target"], cfg["aux_bce"],
                cfg["loss"], length_class_weights,
                q_weight_lambda=float(cfg.get("q_weight_lambda", 0.0)),
                threshold_aux_weight=float(cfg.get("threshold_aux_weight", 0.0)),
                threshold_aux_taus=tuple(cfg.get("threshold_aux_taus", (0.5, 0.7, 0.9))),
                threshold_aux_sharpness=float(cfg.get("threshold_aux_sharpness", 10.0)),
            )
            loss.backward()
            optim.step()
            bs = int(batch["features"].shape[0])
            epoch_loss_sum += float(loss.item()) * bs
            epoch_n += bs
            step += 1

        train_loss = epoch_loss_sum / max(epoch_n, 1)
        val_loss = _eval_loss(
            predictor, val_loader, cfg["target"], cfg["aux_bce"],
            cfg["loss"], length_class_weights,
            q_weight_lambda=float(cfg.get("q_weight_lambda", 0.0)),
            threshold_aux_weight=float(cfg.get("threshold_aux_weight", 0.0)),
            threshold_aux_taus=tuple(cfg.get("threshold_aux_taus", (0.5, 0.7, 0.9))),
            threshold_aux_sharpness=float(cfg.get("threshold_aux_sharpness", 10.0)),
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[train] epoch {epoch}: train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
            patience_left = int(cfg["early_stop_patience"])
        else:
            patience_left -= 1
            if patience_left < 0:
                print(f"[train] early stopping at epoch {epoch}")
                break

    # Restore best state.
    if best_state is not None:
        predictor.load_state_dict(best_state)

    test_loss = _eval_loss(
        predictor, test_loader, cfg["target"], cfg["aux_bce"],
        cfg["loss"], length_class_weights,
        q_weight_lambda=float(cfg.get("q_weight_lambda", 0.0)),
        threshold_aux_weight=float(cfg.get("threshold_aux_weight", 0.0)),
        threshold_aux_taus=tuple(cfg.get("threshold_aux_taus", (0.5, 0.7, 0.9))),
        threshold_aux_sharpness=float(cfg.get("threshold_aux_sharpness", 10.0)),
    )
    print(f"[train] best_val={best_val:.5f}  test_loss={test_loss:.5f}")

    # Save checkpoint + config + history.
    torch.save(predictor.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump({
            "run_name": run_name,
            "best_val": best_val,
            "test_loss": test_loss,
            "history": history,
        }, f, indent=2)

    # Save predictions for val + test — consumed by L1/L2 evaluators.
    _save_predictions(predictor, cfg, val_loader, os.path.join(out_dir, "preds_val.pt"))
    _save_predictions(predictor, cfg, test_loader, os.path.join(out_dir, "preds_test.pt"))
    print(f"[train] wrote artifacts to {out_dir}")


@torch.no_grad()
def _save_predictions(predictor, cfg, loader, path):
    predictor.eval()
    target = cfg["target"]
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        features = batch["features"]
        record_idxs = batch["record_idx"].tolist()
        if target in ("acceptance", "dependence"):
            fwd_kwargs = {}
            if "token_ids" in batch and hasattr(predictor, "token_emb"):
                fwd_kwargs["token_ids"] = batch["token_ids"]
            q = predictor(features, **fwd_kwargs).detach().cpu()
            survived = batch["survived"]
            q2 = batch["q2_target"]  # carries s_j when target='dependence'
            accepted = batch["accepted"]
            for i, r_idx in enumerate(record_idxs):
                rows.append({
                    "record_idx": int(r_idx),
                    "q2_hat": q[i].tolist(),
                    "q2_target": q2[i].tolist(),
                    "survived": survived[i].tolist(),
                    "accepted": accepted[i].tolist(),
                })
        else:  # committed_length
            logits = predictor(features, batch["tau"]).detach().cpu()
            probs = logits.softmax(dim=-1)
            L_hat = logits.argmax(dim=-1)
            tau = batch["tau"]
            L_target = batch["L_target"]
            for i, r_idx in enumerate(record_idxs):
                rows.append({
                    "record_idx": int(r_idx),
                    "tau": float(tau[i].item()),
                    "L_hat": int(L_hat[i].item()),
                    "L_target": int(L_target[i].item()),
                    "probs": probs[i].tolist(),
                })
    torch.save(rows, path)


if __name__ == "__main__":
    main()
