"""L1 — predictor-level accuracy metrics (offline oracle replay).

Operates on saved `preds_{val,test}.pt` outputs from the trainer
(`accpre/train/cli.py::_save_predictions`). Never calls a model.

Acceptance predictors are scored against the canonical `Q2_j` target
(logged in every RoundRecord as `min_pq_j`), masked by `survived_j == 1`.

Length predictors are scored against `L_τ^oracle =
commit_threshold(record.min_pq_j, τ)` where τ is the τ sampled at
training/eval time (also logged in the preds file).

Primary metric for both: MAE.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch


# ------------------------------------------------------------------
# Acceptance predictor metrics
# ------------------------------------------------------------------


def _flatten_survived(
    q2_hat: torch.Tensor,
    q2_target: torch.Tensor,
    survived: torch.Tensor,
    accepted: torch.Tensor,
):
    """Keep only (batch-row, position) entries where survived == 1."""
    mask = survived.bool()
    return q2_hat[mask], q2_target[mask], accepted[mask]


def acceptance_metrics(pred_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute L1 metrics for an acceptance predictor.

    Args:
        pred_rows: list of dicts as written by trainer. Each row has
            keys q2_hat (list len γ), q2_target (list len γ),
            survived (list len γ), accepted (list len γ).

    Returns:
        dict with keys: q2_mae, q2_bias, q2_brier, q2_ece, acc_auc,
            n_positions (survived count used).
    """
    q_hat = torch.tensor([r["q2_hat"] for r in pred_rows], dtype=torch.float32)
    q_tgt = torch.tensor([r["q2_target"] for r in pred_rows], dtype=torch.float32)
    survived = torch.tensor([r["survived"] for r in pred_rows], dtype=torch.float32)
    accepted = torch.tensor([r["accepted"] for r in pred_rows], dtype=torch.float32)

    qh, qt, ac = _flatten_survived(q_hat, q_tgt, survived, accepted)
    n = int(qh.shape[0])
    if n == 0:
        return {"q2_mae": 0.0, "q2_bias": 0.0, "q2_brier": 0.0,
                "q2_ece": 0.0, "acc_auc": float("nan"), "n_positions": 0}

    diff = qh - qt
    q2_mae = float(diff.abs().mean().item())
    q2_bias = float(diff.mean().item())
    q2_brier = float((diff ** 2).mean().item())

    # 10-bin ECE treating qh as a calibrated probability for `accepted`.
    bins = torch.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bins[b], bins[b + 1]
        # Inclusive on right for the last bin.
        if b < 9:
            mask = (qh >= lo) & (qh < hi)
        else:
            mask = (qh >= lo) & (qh <= hi)
        cnt = int(mask.sum().item())
        if cnt == 0:
            continue
        conf = float(qh[mask].mean().item())
        acc = float(ac[mask].mean().item())
        ece += (cnt / n) * abs(conf - acc)
    q2_ece = float(ece)

    # AUROC of qh against the binary accepted label.
    # Simple Mann-Whitney-U computation; O(n log n).
    pos = qh[ac > 0.5]
    neg = qh[ac <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        acc_auc = float("nan")
    else:
        all_scores = torch.cat([pos, neg])
        all_labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        order = torch.argsort(all_scores)
        sorted_labels = all_labels[order]
        # Ranks: 1..N.
        ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float64)
        pos_rank_sum = float(ranks[sorted_labels > 0.5].sum().item())
        n_pos = int(len(pos))
        n_neg = int(len(neg))
        auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        acc_auc = float(auc)

    return {
        "q2_mae": q2_mae,
        "q2_bias": q2_bias,
        "q2_brier": q2_brier,
        "q2_ece": q2_ece,
        "acc_auc": acc_auc,
        "n_positions": n,
    }


# ------------------------------------------------------------------
# Committed-length predictor metrics
# ------------------------------------------------------------------


def length_metrics(pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute L1 metrics for a committed-length predictor.

    Per-τ and aggregated metrics (MAE / bias / exact / over / under).

    Args:
        pred_rows: rows with keys tau, L_hat, L_target.
    """
    by_tau: Dict[float, List[Dict[str, int]]] = {}
    for r in pred_rows:
        tau = float(r["tau"])
        by_tau.setdefault(tau, []).append({
            "L_hat": int(r["L_hat"]),
            "L_target": int(r["L_target"]),
        })

    per_tau: Dict[float, Dict[str, float]] = {}
    all_diffs = []
    for tau in sorted(by_tau.keys()):
        entries = by_tau[tau]
        diffs = [e["L_hat"] - e["L_target"] for e in entries]
        abs_diffs = [abs(d) for d in diffs]
        n = len(entries)
        per_tau[tau] = {
            "len_mae": float(sum(abs_diffs) / max(n, 1)),
            "len_bias": float(sum(diffs) / max(n, 1)),
            "len_exact": float(sum(1 for d in diffs if d == 0) / max(n, 1)),
            "len_over": float(sum(1 for d in diffs if d > 0) / max(n, 1)),
            "len_under": float(sum(1 for d in diffs if d < 0) / max(n, 1)),
            "n": n,
        }
        all_diffs.extend(diffs)

    if all_diffs:
        len_mae_avg = sum(abs(d) for d in all_diffs) / len(all_diffs)
    else:
        len_mae_avg = 0.0

    return {
        "per_tau": {str(k): v for k, v in per_tau.items()},
        "len_mae_avg": float(len_mae_avg),
        "n_rows_total": len(pred_rows),
    }
