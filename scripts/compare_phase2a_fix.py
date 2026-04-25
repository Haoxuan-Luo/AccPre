"""Stage 2A before/after comparison.

Compares the OLD (MSE acceptance + buggy LengthMLP) predictors against
the NEW (soft-BCE acceptance + fixed LengthMLP) predictors. Reads:

  - checkpoints/                      (all trained predictors)
  - data_collected/stage1.pt          (test records)
  - outputs/phase2_online_11791962/   (OLD online outputs)
  - outputs/phase2_online_fix_11794789/ (NEW online outputs)

Emits one report with §1-§3 from the original debug pass, replicated
per-predictor for direct comparison.
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
from accpre.eval.predictor_metrics import acceptance_metrics
from accpre.predictors.acceptance_mlp import AcceptanceMLP
from accpre.predictors.length_mlp import LengthMLP
from accpre.train.dataset import split_records_by_prompt


CKPT_ROOT = "checkpoints"
DATA = "data_collected/stage1.pt"
ONLINE_OLD = "outputs/phase2_online_11791962"
ONLINE_NEW = "outputs/phase2_online_fix_11794789"
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _load_ckpt(name: str):
    d = os.path.join(CKPT_ROOT, name)
    with open(os.path.join(d, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    if cfg["target"] == "acceptance":
        p = AcceptanceMLP(family=cfg["family"], gamma=int(cfg["gamma"]),
                          hidden_dim=int(cfg.get("hidden_dim", 128)),
                          dropout=float(cfg.get("dropout", 0.1)))
    else:
        p = LengthMLP(family=cfg["family"], gamma=int(cfg["gamma"]),
                      hidden_dim=int(cfg.get("hidden_dim", 128)),
                      dropout=float(cfg.get("dropout", 0.1)))
    p.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
    p.eval()
    return p, cfg


def _header(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def _test_records(records: List[RoundRecord]) -> List[RoundRecord]:
    te = list(range(TRAIN_N + VAL_N, TRAIN_N + VAL_N + 20))
    parts = split_records_by_prompt(records, [], [], te)
    return [r for r in parts["test"] if r.gamma == 8]


# ======================================================================
# §1 acceptance: MSE vs soft-BCE, offline + Q̂ range
# ======================================================================

def section_1(test_records) -> None:
    _header("§1  Acceptance predictor — OFFLINE metrics + Q̂ range")
    for base in ("acc_frz_num", "acc_frz_hid", "acc_frz_hidnum"):
        sbce = base + "_sbce"
        print(f"\n  --- {base} (OLD, MSE)  vs  {sbce} (NEW, soft-BCE) ---")
        for name in (base, sbce):
            preds = torch.load(
                os.path.join(CKPT_ROOT, name, "preds_test.pt"),
                weights_only=False,
            )
            m = acceptance_metrics(preds)

            q_hat = [float(x)
                     for r in preds for j, x in enumerate(r["q2_hat"])
                     if r["survived"][j] > 0.5]
            qh = np.array(q_hat)
            print(
                f"    [{name:<22}]  "
                f"MAE={m['q2_mae']:.4f}  "
                f"Brier={m['q2_brier']:.4f}  "
                f"bias={m['q2_bias']:+.4f}  "
                f"ECE={m['q2_ece']:.4f}  "
                f"AUC={m['acc_auc']:.3f}"
            )
            print(
                f"    [{name:<22}]  "
                f"Q̂ min={qh.min():.3f}  max={qh.max():.3f}  "
                f"median={np.median(qh):.3f}  mean={qh.mean():.3f}  "
                f"frac≥0.7={(qh >= 0.7).mean():.2%}"
            )


# ======================================================================
# §2 acceptance: τ-sensitivity of L_hat distribution
# ======================================================================

def section_2(test_records) -> None:
    _header("§2  Acceptance predictor — τ sensitivity of L̂ (from Q̂)")
    for base in ("acc_frz_num", "acc_frz_hid", "acc_frz_hidnum"):
        for name in (base, base + "_sbce"):
            p, _ = _load_ckpt(name)
            per = {t: [] for t in TAUS}
            for r in test_records:
                feats = extract_features(r, p.family)
                with torch.no_grad():
                    qh = p.predict_q2(feats).detach().cpu().numpy().tolist()
                for t in TAUS:
                    per[t].append(commit_threshold(qh, t))
            print(f"\n  [{name}]")
            for t in TAUS:
                arr = np.array(per[t])
                dist = {int(L): int((arr == L).sum())
                        for L in range(9) if (arr == L).any()}
                print(
                    f"    τ={t:.1f}  mean L̂={arr.mean():.2f}  "
                    f"L̂=0 rate={(arr == 0).mean():.2%}  dist={dist}"
                )


# ======================================================================
# §3 length: τ sensitivity on same record, then batch dist
# ======================================================================

def section_3(test_records) -> None:
    _header("§3  Length predictor — τ sensitivity (fixed LayerNorm arch)")

    sample = test_records[0]
    for name in ("len_frz_num", "len_frz_hidnum"):
        p, _ = _load_ckpt(name)
        feats = extract_features(sample, p.family)
        print(f"\n  [{name}] same record (prompt {sample.prompt_idx}, round {sample.round_idx}):")
        stacked = []
        for t in TAUS:
            tau_t = torch.tensor(float(t), dtype=torch.float32)
            with torch.no_grad():
                logits = p.forward(feats, tau_t).squeeze(0).cpu().numpy()
            stacked.append(logits)
            L_hat = int(np.argmax(logits))
            print(
                f"    τ={t:.1f}  L̂={L_hat}  "
                f"logits[0:3]={np.round(logits[:3], 3).tolist()}"
            )
        stacked = np.stack(stacked, axis=0)
        per_class_std = stacked.std(axis=0)
        print(f"    logit std across τ (per class): {np.round(per_class_std, 3).tolist()}")

        # Batch distribution of L̂ per τ.
        print(f"  [{name}] L̂ distribution across {len(test_records)} test records:")
        for t in TAUS:
            L_hats = []
            for r in test_records:
                fe = extract_features(r, p.family)
                L_hats.append(int(p.predict_L(fe, float(t))))
            arr = np.array(L_hats)
            dist = {int(L): int((arr == L).sum())
                    for L in range(9) if (arr == L).any()}
            print(
                f"    τ={t:.1f}  mean L̂={arr.mean():.2f}  "
                f"L̂=0 rate={(arr == 0).mean():.2%}  dist={dist}"
            )


# ======================================================================
# §4 online: tok/s + CF@1 before vs after
# ======================================================================

def _read_online(root: str) -> Dict[str, Dict[float, Dict]]:
    out: Dict[str, Dict[float, Dict]] = {}
    for d in sorted(glob.glob(os.path.join(root, "*") + "/")):
        name = os.path.basename(d.rstrip("/"))
        out[name] = {}
        for path in sorted(glob.glob(os.path.join(d, "online_*.json"))):
            with open(path, "r") as f:
                s = json.load(f)
            out[name][float(s["tau"])] = s["l3"]
    return out


def section_4() -> None:
    _header("§4  Online decode — tok/s × CF@1 (before/after)")
    old = _read_online(ONLINE_OLD)
    new = _read_online(ONLINE_NEW)

    acc_base = ("acc_frz_num", "acc_frz_hid", "acc_frz_hidnum")
    for base in acc_base:
        sbce = base + "_sbce"
        if base not in old or sbce not in new:
            continue
        print(f"\n  --- acceptance: {base} (OLD, MSE) vs {sbce} (NEW, soft-BCE) ---")
        print(f"    {'τ':<5}{'OLD tok/s':<12}{'OLD CF@1':<11}{'NEW tok/s':<12}{'NEW CF@1':<11}")
        for tau in sorted(old[base].keys()):
            o = old[base][tau]
            n = new.get(sbce, {}).get(tau)
            n_tps = f"{n['tok_s_mean']:.2f}" if n else "—"
            n_cf = f"{n['cf_at_1_aggregate']:.3f}" if n else "—"
            print(
                f"    {tau:<5}{o['tok_s_mean']:<12.2f}{o['cf_at_1_aggregate']:<11.3f}"
                f"{n_tps:<12}{n_cf:<11}"
            )

    for name in ("len_frz_num", "len_frz_hidnum"):
        if name not in old or name not in new:
            continue
        print(f"\n  --- length: {name} (OLD, buggy arch)  vs  {name} (NEW, fixed arch) ---")
        print(f"    {'τ':<5}{'OLD tok/s':<12}{'OLD CF@1':<11}{'NEW tok/s':<12}{'NEW CF@1':<11}")
        for tau in sorted(old[name].keys()):
            o = old[name][tau]
            nn = new[name].get(tau)
            n_tps = f"{nn['tok_s_mean']:.2f}" if nn else "—"
            n_cf = f"{nn['cf_at_1_aggregate']:.3f}" if nn else "—"
            print(
                f"    {tau:<5}{o['tok_s_mean']:<12.2f}{o['cf_at_1_aggregate']:<11.3f}"
                f"{n_tps:<12}{n_cf:<11}"
            )


# ======================================================================
# §5 online: L_hat distribution (before vs after)
# ======================================================================

def _online_L_hats(root: str, name: str) -> Dict[float, np.ndarray]:
    d = os.path.join(root, name)
    out: Dict[float, np.ndarray] = {}
    for path in sorted(glob.glob(os.path.join(d, "online_*.json"))):
        with open(path, "r") as f:
            s = json.load(f)
        tau = float(s["tau"])
        xs = []
        for pp in s["online_lane"]["per_prompt"]:
            xs.extend(int(rd["L_hat"]) for rd in pp["rounds"])
        out[tau] = np.array(xs)
    return out


def section_5() -> None:
    _header("§5  Online decode — L̂ distribution before/after")
    for base in ("acc_frz_num", "acc_frz_hid", "acc_frz_hidnum"):
        sbce = base + "_sbce"
        old_Ls = _online_L_hats(ONLINE_OLD, base)
        new_Ls = _online_L_hats(ONLINE_NEW, sbce)
        print(f"\n  [{base}] OLD:")
        for tau in sorted(old_Ls.keys()):
            arr = old_Ls[tau]
            dist = {int(L): int((arr == L).sum())
                    for L in range(9) if (arr == L).any()}
            print(f"    τ={tau:.1f}  n={len(arr)}  mean={arr.mean():.2f}  "
                  f"zero={np.mean(arr == 0):.2%}  dist={dist}")
        print(f"  [{sbce}] NEW:")
        for tau in sorted(new_Ls.keys()):
            arr = new_Ls[tau]
            dist = {int(L): int((arr == L).sum())
                    for L in range(9) if (arr == L).any()}
            print(f"    τ={tau:.1f}  n={len(arr)}  mean={arr.mean():.2f}  "
                  f"zero={np.mean(arr == 0):.2%}  dist={dist}")

    for name in ("len_frz_num", "len_frz_hidnum"):
        print(f"\n  [{name}] OLD (buggy arch):")
        for tau, arr in sorted(_online_L_hats(ONLINE_OLD, name).items()):
            dist = {int(L): int((arr == L).sum())
                    for L in range(9) if (arr == L).any()}
            print(f"    τ={tau:.1f}  n={len(arr)}  mean={arr.mean():.2f}  "
                  f"zero={np.mean(arr == 0):.2%}  dist={dist}")
        print(f"  [{name}] NEW (fixed arch):")
        for tau, arr in sorted(_online_L_hats(ONLINE_NEW, name).items()):
            dist = {int(L): int((arr == L).sum())
                    for L in range(9) if (arr == L).any()}
            print(f"    τ={tau:.1f}  n={len(arr)}  mean={arr.mean():.2f}  "
                  f"zero={np.mean(arr == 0):.2%}  dist={dist}")


def main() -> None:
    records = load_records(DATA)
    print(f"[cmp] loaded {len(records)} records from {DATA}")
    test_records = _test_records(records)
    print(f"[cmp] {len(test_records)} test records (γ=8)")

    section_1(test_records)
    section_2(test_records)
    section_3(test_records)
    section_4()
    section_5()


if __name__ == "__main__":
    main()
