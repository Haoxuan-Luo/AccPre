"""Stage 2A debug / sanity pass.

Loads the 5 frozen predictor checkpoints + saved train-time predictions +
online-decode JSONs and reports the five diagnostic sections from the
Phase-2A review request:

  §1 acceptance predictor quality beyond raw MSE
  §2 tau sensitivity of predictor outputs
  §3 length predictor debugging
  §4 online decode diagnostics (L_hat distribution, progress guarantee)
  §5 verdict

CPU-only. No model loads beyond the small frozen heads.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accpre.collect.features import extract_features
from accpre.core.commit import commit_threshold
from accpre.core.schema import RoundRecord, load_records
from accpre.data.splits import TRAIN_N, VAL_N
from accpre.eval.predictor_metrics import acceptance_metrics, length_metrics
from accpre.predictors.acceptance_mlp import AcceptanceMLP
from accpre.predictors.length_mlp import LengthMLP
from accpre.train.dataset import split_records_by_prompt


# ---------- global paths (adjust here if layouts change) ----------
CKPT_ROOT = "checkpoints"
DATA = "data_collected/stage1.pt"
ONLINE_ROOT = "outputs/phase2_online_11791962"  # latest Phase 2A online run
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _load_ckpt(name: str):
    """Return (predictor, cfg) from `checkpoints/<name>/`."""
    d = os.path.join(CKPT_ROOT, name)
    with open(os.path.join(d, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    if cfg["target"] == "acceptance":
        p = AcceptanceMLP(
            family=cfg["family"], gamma=int(cfg["gamma"]),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    else:
        p = LengthMLP(
            family=cfg["family"], gamma=int(cfg["gamma"]),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    p.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
    p.eval()
    return p, cfg


def _prompt_index_sets():
    return (
        list(range(0, TRAIN_N)),
        list(range(TRAIN_N, TRAIN_N + VAL_N)),
        list(range(TRAIN_N + VAL_N, TRAIN_N + VAL_N + 20)),
    )


def _header(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


# ======================================================================
# §1. Acceptance predictor quality beyond raw MSE
# ======================================================================

def section_1() -> None:
    _header("§1  Acceptance predictor quality beyond raw MSE (test split)")
    for name in ("acc_frz_num", "acc_frz_hid", "acc_frz_hidnum"):
        preds = torch.load(
            os.path.join(CKPT_ROOT, name, "preds_test.pt"), weights_only=False,
        )
        m = acceptance_metrics(preds)
        print(
            f"  {name:<16}  "
            f"MAE={m['q2_mae']:.4f}  "
            f"Brier(MSE)={m['q2_brier']:.4f}  "
            f"bias={m['q2_bias']:+.4f}  "
            f"ECE={m['q2_ece']:.4f}  "
            f"AUC={m['acc_auc']:.3f}  "
            f"n={m['n_positions']}"
        )

    print()
    print("  Q̂ vs Q2 histogram (10 bins on [0, 1]); masked by survived==1:")
    for name in ("acc_frz_num", "acc_frz_hid", "acc_frz_hidnum"):
        preds = torch.load(
            os.path.join(CKPT_ROOT, name, "preds_test.pt"), weights_only=False,
        )
        q_hat, q_tgt = [], []
        for r in preds:
            for j in range(len(r["q2_hat"])):
                if r["survived"][j] > 0.5:
                    q_hat.append(float(r["q2_hat"][j]))
                    q_tgt.append(float(r["q2_target"][j]))
        qh = np.array(q_hat)
        qt = np.array(q_tgt)
        print(
            f"\n  [{name}]  Q̂  mean={qh.mean():.3f}  "
            f"median={np.median(qh):.3f}  min={qh.min():.3f}  max={qh.max():.3f}"
        )
        print(
            f"  [{name}]  Q2 mean={qt.mean():.3f}  "
            f"median={np.median(qt):.3f}  min={qt.min():.3f}  max={qt.max():.3f}"
        )
        print("    bin      Q̂ count    Q2 count")
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            h_count = int(((qh >= lo) & (qh < hi)).sum())
            t_count = int(((qt >= lo) & (qt < hi)).sum())
            print(f"    [{lo:.1f}–{hi:.1f}]   {h_count:>7}    {t_count:>7}")


# ======================================================================
# §2  Tau sensitivity (acceptance predictors — same Q̂, vary τ)
# ======================================================================

def section_2(records: List[RoundRecord]) -> None:
    _header("§2  τ sensitivity of acceptance predictors")
    _, _, te = _prompt_index_sets()
    parts = split_records_by_prompt(records, [], [], te)
    test_records = [r for r in parts["test"] if r.gamma == 8]
    print(f"  Using {len(test_records)} test records (γ=8).")

    for name in ("acc_frz_num", "acc_frz_hid", "acc_frz_hidnum"):
        p, _ = _load_ckpt(name)
        L_per_tau: Dict[float, List[int]] = {t: [] for t in TAUS}
        for r in test_records:
            feats = extract_features(r, p.family)
            with torch.no_grad():
                qh = p.predict_q2(feats).detach().cpu().numpy().tolist()
            for t in TAUS:
                L_per_tau[t].append(commit_threshold(qh, t))

        print(f"\n  [{name}]  (same Q̂; commit_threshold applied per τ)")
        for t in TAUS:
            arr = np.array(L_per_tau[t])
            zero_rate = float((arr == 0).mean())
            mean_L = float(arr.mean())
            mean_nc = float(np.maximum(arr, 1).mean())
            dist = {int(L): int((arr == L).sum()) for L in range(9) if (arr == L).any()}
            print(
                f"    τ={t:.1f}  mean L̂={mean_L:.2f}  L̂=0 rate={zero_rate:.2%}  "
                f"mean max(1,L̂)={mean_nc:.2f}  dist={dist}"
            )


# ======================================================================
# §3  Length predictor debugging
# ======================================================================

def section_3(records: List[RoundRecord]) -> None:
    _header("§3  Length predictor debugging")

    # 3a. Oracle label variance across τ.
    _, _, te = _prompt_index_sets()
    parts = split_records_by_prompt(records, [], [], te)
    test_records = [r for r in parts["test"] if r.gamma == 8]
    print(f"  Using {len(test_records)} test records (γ=8).")
    print("\n  §3a  Oracle label L_τ^oracle variance across τ:")
    for t in TAUS:
        arr = np.array([commit_threshold(r.min_pq_j, t) for r in test_records])
        dist = {int(L): int((arr == L).sum()) for L in range(9) if (arr == L).any()}
        print(
            f"    τ={t:.1f}  mean={arr.mean():.2f}  "
            f"unique values used: {sorted(dist.keys())}  dist={dist}"
        )

    # 3b. Does the model's output change with τ? Same record, vary τ.
    print("\n  §3b  Same record, varying τ — does L̂ actually move?")
    sample_rec = test_records[0]
    for name in ("len_frz_num", "len_frz_hidnum"):
        p, _ = _load_ckpt(name)
        feats = extract_features(sample_rec, p.family)
        print(f"    [{name}] record prompt_idx={sample_rec.prompt_idx} round_idx={sample_rec.round_idx}")
        prev_logits = None
        for t in TAUS:
            tau_t = torch.tensor(float(t), dtype=torch.float32)
            with torch.no_grad():
                logits = p.forward(feats, tau_t).squeeze(0).cpu().numpy()
            L_hat = int(np.argmax(logits))
            probs = torch.tensor(logits).softmax(dim=-1).cpu().numpy()
            print(
                f"      τ={t:.1f}  L̂={L_hat}  "
                f"logits[0:3]={np.round(logits[:3], 3).tolist()}  "
                f"logits std across classes={logits.std():.3f}"
            )
        # Variance of logits across τ for this record.
        stacked = []
        for t in TAUS:
            tau_t = torch.tensor(float(t), dtype=torch.float32)
            with torch.no_grad():
                stacked.append(p.forward(feats, tau_t).squeeze(0).cpu().numpy())
        stacked = np.stack(stacked, axis=0)   # (|τ|, γ+1)
        per_class_std = stacked.std(axis=0)
        print(f"      logit std across τ (per class) = {np.round(per_class_std, 4).tolist()}")

    # 3c. Distribution of predicted L_hat across the test set.
    print("\n  §3c  Predicted L̂ distribution across the test set, per τ:")
    for name in ("len_frz_num", "len_frz_hidnum"):
        p, _ = _load_ckpt(name)
        for t in TAUS:
            L_hats = []
            for r in test_records:
                feats = extract_features(r, p.family)
                L_hats.append(int(p.predict_L(feats, float(t))))
            arr = np.array(L_hats)
            dist = {int(L): int((arr == L).sum()) for L in range(9) if (arr == L).any()}
            print(f"    [{name}] τ={t:.1f}: dist={dist}  mean={arr.mean():.2f}")

    # 3d. Confusion matrix on held-out preds_test.pt (τ was sampled at eval
    # time; each row has a different τ). Check per-class accuracy.
    print("\n  §3d  Confusion (predicted vs target) on saved preds_test.pt:")
    for name in ("len_frz_num", "len_frz_hidnum"):
        preds = torch.load(
            os.path.join(CKPT_ROOT, name, "preds_test.pt"), weights_only=False,
        )
        # 9×9 confusion
        M = np.zeros((9, 9), dtype=int)
        for r in preds:
            M[int(r["L_target"]), int(r["L_hat"])] += 1
        print(f"\n    [{name}]")
        hdr = "  tgt \\ pred " + "".join(f"{i:>5}" for i in range(9))
        print("   ", hdr)
        for i in range(9):
            row = "".join(f"{int(M[i, j]):>5}" for j in range(9))
            row_sum = int(M[i].sum())
            print(f"    tgt={i} ({row_sum:>4}): {row}")


# ======================================================================
# §4  Online decode diagnostics
# ======================================================================

def section_4() -> None:
    _header("§4  Online decode diagnostics")
    per_pred = sorted(glob.glob(os.path.join(ONLINE_ROOT, "*") + "/"))
    for pred_dir in per_pred:
        name = os.path.basename(pred_dir.rstrip("/"))
        print(f"\n  [{name}]")
        for path in sorted(glob.glob(os.path.join(pred_dir, "online_*.json"))):
            with open(path, "r") as f:
                s = json.load(f)
            tau = s["tau"]
            L_hats: List[int] = []
            n_committed: List[int] = []
            for pp in s["online_lane"]["per_prompt"]:
                for rd in pp["rounds"]:
                    L_hats.append(int(rd["L_hat"]))
                    n_committed.append(int(rd["n_committed"]))
            L_arr = np.array(L_hats)
            NC = np.array(n_committed)
            n = len(L_arr)
            guard_rate = float((L_arr == 0).mean())
            dist = {int(L): int((L_arr == L).sum()) for L in range(9) if (L_arr == L).any()}
            print(
                f"    τ={tau:.1f}  n_rounds={n}  "
                f"L̂ mean={L_arr.mean():.2f}  L̂=0 rate={guard_rate:.2%}  "
                f"n_committed mean={NC.mean():.2f}"
            )
            print(f"      L̂ dist: {dist}")


# ======================================================================
# §5  Verdict
# ======================================================================

def section_5() -> None:
    _header("§5  Verdict (manual — see script output above)")
    print("  This script emits the numbers; conclusions are written back by the")
    print("  reviewer. Typical classifications to pick from:")
    print("    - likely code / wiring bug")
    print("    - likely target / modeling issue")
    print("    - likely predictor underfitting / overfitting")
    print("    - likely deployment-rule issue (max(1, L̂) dominating τ signal)")


def main() -> None:
    records = load_records(DATA)
    print(f"[debug] loaded {len(records)} records from {DATA}")

    section_1()
    section_2(records)
    section_3(records)
    section_4()
    section_5()


if __name__ == "__main__":
    main()
