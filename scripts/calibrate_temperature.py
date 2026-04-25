"""Post-hoc temperature scaling for acceptance predictors.

Reads `<ckpt>/preds_val.pt` (q2_hat, q2_target, survived already saved
by `accpre.train.cli._save_predictions`), fits a single scalar T > 0 to
minimise survived-masked Brier on the validation split:

  z = logit(q_hat)        (logit inversion of sigmoid)
  q_cal = sigmoid(z / T)
  L(T) = sum_{j: survived} (q_cal_j - q2_target_j)^2  /  sum(survived)

Saves the fitted T to `<ckpt>/temperature.json`. `online_decode` detects
this file at load time and wraps the predictor so that `predict_q2`
returns calibrated Q̂ (no architecture change).

Usage:
  python scripts/calibrate_temperature.py <ckpt_dir> [--init_T 1.0] [--steps 300]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def _fit_temperature(
    q_hat: torch.Tensor,          # (N, γ) in (0, 1)
    q_target: torch.Tensor,       # (N, γ)
    survived: torch.Tensor,       # (N, γ) 0/1
    init_T: float = 1.0,
    steps: int = 300,
    lr: float = 0.05,
) -> float:
    """Fit a scalar temperature T that minimises survived-masked Brier.

    Returns T (positive float). Uses Adam on `log_T` for stability.
    """
    eps = 1e-7
    q = q_hat.clamp(eps, 1.0 - eps)
    z = torch.log(q / (1.0 - q))                                  # (N, γ)
    mask = survived.float()
    denom = mask.sum().clamp(min=1.0)

    log_T = torch.nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_T)))))
    opt = torch.optim.Adam([log_T], lr=lr)

    for _ in range(steps):
        T = log_T.exp()
        q_cal = torch.sigmoid(z / T)
        se = (q_cal - q_target) ** 2
        loss = (se * mask).sum() / denom
        opt.zero_grad()
        loss.backward()
        opt.step()

    return float(log_T.exp().item())


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-hoc temperature scaling.")
    ap.add_argument("ckpt_dir", type=str, help="Checkpoint directory with preds_val.pt.")
    ap.add_argument("--init_T", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    val_path = os.path.join(args.ckpt_dir, "preds_val.pt")
    if not os.path.isfile(val_path):
        print(f"[calibrate] ERROR: {val_path} not found", file=sys.stderr)
        return 1

    rows = torch.load(val_path, weights_only=False)
    q_hat = torch.tensor([r["q2_hat"] for r in rows], dtype=torch.float32)
    q_target = torch.tensor([r["q2_target"] for r in rows], dtype=torch.float32)
    survived = torch.tensor([r["survived"] for r in rows], dtype=torch.float32)

    # Pre-fit Brier.
    eps = 1e-7
    q = q_hat.clamp(eps, 1 - eps)
    mask = survived.float()
    denom = mask.sum().clamp(min=1.0)
    brier_pre = ((q - q_target) ** 2 * mask).sum().item() / denom.item()

    T = _fit_temperature(
        q_hat, q_target, survived,
        init_T=args.init_T, steps=args.steps, lr=args.lr,
    )

    z = torch.log(q / (1 - q))
    q_cal = torch.sigmoid(z / T)
    brier_post = ((q_cal - q_target) ** 2 * mask).sum().item() / denom.item()

    out = {
        "T": float(T),
        "brier_val_pre": float(brier_pre),
        "brier_val_post": float(brier_post),
        "n_survived": int(mask.sum().item()),
        "init_T": float(args.init_T),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "loss": "masked_brier_val",
    }
    out_path = os.path.join(args.ckpt_dir, "temperature.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] ckpt={args.ckpt_dir}")
    print(f"[calibrate] fitted T = {T:.4f}")
    print(f"[calibrate] val Brier  pre = {brier_pre:.5f}  post = {brier_post:.5f}  "
          f"Δ = {brier_post - brier_pre:+.5f}")
    print(f"[calibrate] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
