"""Post-training joint evaluation — recollect + offline CF@1 + Q2 variance diag.

Runs end-to-end against a joint checkpoint (drafter.pt + model.pt):

  1. Recollect strict records across the full test split (prompts
     60..79) using the fine-tuned drafter. Saves
     `<out_dir>/recollected_test.pt`.
  2. Compute offline CF@1 at τ in {0.5, 0.7, 0.9} with the joint 1A
     head on the recollected records. Baseline for comparison comes
     from `checkpoints/acc_1a/preds_test.pt` evaluated against the
     ORIGINAL `data_collected/stage1_pp.pt` (frozen 1A already
     reported; we re-read it here and do not retrain).
  3. Conditional-variance diagnostic: split survived positions into
     drafter_top1 deciles, compute mean/std of Q2 within each decile,
     on BOTH original and recollected records, and print side-by-side.

Not re-measured here:
  - Online tok/s (separate `online_decode` invocation in the SLURM
    job; this script doesn't load models).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.collect.features import extract_features
from accpre.core.commit import commit_threshold
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records, save_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N, test_split


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


# ------------------------------------------------------------------
# Step 1 — recollect full test split
# ------------------------------------------------------------------


def recollect_full_test_split(
    checkpoint_dir: Path, protocol: ProtocolConfig, gamma: int, T: int,
    max_new_tokens: int,
) -> List:
    """Run strict SpecDiff across all test prompts (60..79) using the
    fine-tuned drafter from `<checkpoint_dir>/drafter.pt`. Returns a
    fresh list of `RoundRecord`s with full v2+ feature fields.
    """
    import torch
    from accpre.eval.online_decode import recollect_strict_with_drafter
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter_state = torch.load(
        str(checkpoint_dir / "drafter.pt"), map_location=device,
    )
    drafter.model.load_state_dict(drafter_state)
    drafter.model.eval()
    print(f"[eval_joint] loaded fine-tuned drafter state")

    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompts = test_split()   # 20 prompts
    prompt_indices = list(range(TRAIN_N + VAL_N, POOL_SIZE))  # 60..79
    print(
        f"[eval_joint] recollecting {len(prompts)} test prompts "
        f"(global indices {prompt_indices[0]}..{prompt_indices[-1]})"
    )
    recs = recollect_strict_with_drafter(
        prompts=prompts, prompt_indices=prompt_indices, protocol=protocol,
        drafter=drafter, verifier=verifier,
        gamma=gamma, T=T, max_new_tokens=max_new_tokens,
    )
    return recs


# ------------------------------------------------------------------
# Step 2 — offline CF@1 for acc_jnt_1a on recollected records
# ------------------------------------------------------------------


def offline_cf_joint(
    checkpoint_dir: Path, recollected: List, taus=(0.5, 0.7, 0.9),
) -> Dict:
    """Load the 1A-style head from a joint checkpoint and compute CF@1
    at each τ on the recollected records.

    The joint checkpoint's `model.pt` is an `AcceptanceMLPPerPosTokEmb`
    state dict — same arch as frozen acc_1a.
    """
    from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmb
    with open(checkpoint_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    head = AcceptanceMLPPerPosTokEmb(
        family=cfg["family"], gamma=int(cfg.get("gamma", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        token_emb_dim=int(cfg.get("token_emb_dim", 64)),
    )
    head.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location="cpu"),
    )
    head.eval()

    per_tau: Dict[float, Dict] = {}
    for tau in taus:
        tau = float(tau)
        exact = under = over = 0
        L_hat_counter: Counter = Counter()
        L_strict_counter: Counter = Counter()
        abs_diffs = []
        per_prompt_flags: Dict[int, List[int]] = defaultdict(list)
        for r in recollected:
            feats = extract_features(r, "hidden_per_pos_v2")
            tok = torch.tensor(r.draft_tokens, dtype=torch.long)
            L_hat = int(head.predict_L(feats, tau, token_ids=tok))
            L_hat = max(0, min(L_hat, r.gamma))
            L_strict = int(r.L)
            L_hat_counter[L_hat] += 1
            L_strict_counter[L_strict] += 1
            abs_diffs.append(abs(L_hat - L_strict))
            if L_hat == L_strict:
                exact += 1
                per_prompt_flags[int(r.prompt_idx)].append(1)
            elif L_hat < L_strict:
                under += 1
                per_prompt_flags[int(r.prompt_idx)].append(0)
            else:
                over += 1
                per_prompt_flags[int(r.prompt_idx)].append(0)
        n = max(len(recollected), 1)
        per_tau[tau] = {
            "tau": tau, "n": n,
            "cf_at_1": exact / n,
            "under": under / n,
            "over": over / n,
            "mean_abs_diff": sum(abs_diffs) / n,
            "L_hat_distribution": {
                str(k): L_hat_counter[k] / n for k in sorted(L_hat_counter)
            },
            "L_strict_distribution": {
                str(k): L_strict_counter[k] / n for k in sorted(L_strict_counter)
            },
            "per_prompt_cf": {
                p: sum(v) / len(v) for p, v in per_prompt_flags.items()
            },
        }
    return per_tau


# ------------------------------------------------------------------
# Step 3 — Q2 conditional variance diagnostic (before vs after)
# ------------------------------------------------------------------


def conditional_variance_table(records: List, n_bins: int = 10) -> Dict:
    """For each decile of drafter_top1_prob, compute Q2 mean/std/count
    restricted to survived positions. Returns a dict with the per-decile
    row list + an overall mean/std.
    """
    top1s: List[float] = []
    q2s: List[float] = []
    for r in records:
        for j in range(int(r.gamma)):
            if int(r.survived_j[j]) != 1:
                continue
            top1s.append(float(r.drafter_top1_prob_j[j]))
            q2s.append(float(r.min_pq_j[j]))
    if len(top1s) == 0:
        return {"rows": [], "overall": {"n": 0}}
    top1s_arr = np.array(top1s)
    q2s_arr = np.array(q2s)
    order = np.argsort(top1s_arr, kind="stable")
    top1s_arr = top1s_arr[order]
    q2s_arr = q2s_arr[order]
    n = len(top1s_arr)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        t = top1s_arr[lo:hi]
        q = q2s_arr[lo:hi]
        rows.append({
            "bin": i, "n": int(hi - lo),
            "top1_lo": float(t.min()), "top1_hi": float(t.max()),
            "q2_mean": float(q.mean()), "q2_std": float(q.std(ddof=0)),
        })
    return {
        "rows": rows,
        "overall": {
            "n": int(n),
            "q2_mean": float(q2s_arr.mean()),
            "q2_std": float(q2s_arr.std(ddof=0)),
        },
    }


# ------------------------------------------------------------------
# Report rendering
# ------------------------------------------------------------------


def fmt_cf_table(frozen_cf, joint_cf, oracle_cf=None) -> str:
    lines = []
    taus = sorted(joint_cf.keys())
    lines.append("          " + "".join(f"{'τ='+str(t):>12}" for t in taus) + "\n")
    for label, get_fn in [
        ("frozen 1A ", lambda t: frozen_cf.get(t, {}).get("cf_at_1", float("nan"))),
        ("acc_jnt_1a", lambda t: joint_cf[t]["cf_at_1"]),
        ("Δ vs frozen",
         lambda t: joint_cf[t]["cf_at_1"]
         - frozen_cf.get(t, {}).get("cf_at_1", float("nan"))),
    ]:
        row = f"{label:<12}"
        for t in taus:
            v = get_fn(t)
            if isinstance(v, float) and np.isfinite(v):
                row += f"{v:>12.4f}"
            else:
                row += f"{'   —   ':>12}"
        lines.append(row + "\n")
    if oracle_cf is not None:
        row = f"{'oracle':<12}"
        for t in taus:
            row += f"{oracle_cf.get(t, {}).get('cf_at_1', float('nan')):>12.4f}"
        lines.append(row + "\n")
    return "".join(lines)


def fmt_under_over(joint_cf, frozen_cf) -> str:
    lines = []
    taus = sorted(joint_cf.keys())
    lines.append("          " + "".join(f"{'τ='+str(t):>12}" for t in taus) + "\n")
    for label, get_fn in [
        ("under jnt1a ", lambda t: joint_cf[t]["under"]),
        ("under 1A    ", lambda t: frozen_cf.get(t, {}).get("under", float("nan"))),
        ("over  jnt1a ", lambda t: joint_cf[t]["over"]),
        ("over  1A    ", lambda t: frozen_cf.get(t, {}).get("over", float("nan"))),
    ]:
        row = f"{label:<12}"
        for t in taus:
            v = get_fn(t)
            row += f"{v:>12.4f}" if isinstance(v, float) and np.isfinite(v) else f"{'   —   ':>12}"
        lines.append(row + "\n")
    return "".join(lines)


def fmt_var_table(before: Dict, after: Dict) -> str:
    lines = []
    lines.append(
        f"  {'dec':>3} {'n_bef':>6} {'top1_bef':>18} {'mean_bef':>10} {'std_bef':>10}"
        f"  {'n_aft':>6} {'top1_aft':>18} {'mean_aft':>10} {'std_aft':>10}\n"
    )
    nb, na = len(before["rows"]), len(after["rows"])
    for i in range(max(nb, na)):
        b = before["rows"][i] if i < nb else None
        a = after["rows"][i] if i < na else None
        row = f"  {i:>3}"
        if b is not None:
            row += (
                f" {b['n']:>6d} [{b['top1_lo']:6.3f}, {b['top1_hi']:6.3f}] "
                f"{b['q2_mean']:>10.4f} {b['q2_std']:>10.4f}"
            )
        else:
            row += f" {'-':>6} {'-':>18} {'-':>10} {'-':>10}"
        if a is not None:
            row += (
                f"  {a['n']:>6d} [{a['top1_lo']:6.3f}, {a['top1_hi']:6.3f}] "
                f"{a['q2_mean']:>10.4f} {a['q2_std']:>10.4f}"
            )
        else:
            row += f"  {'-':>6} {'-':>18} {'-':>10} {'-':>10}"
        lines.append(row + "\n")
    ob = before["overall"]; oa = after["overall"]
    lines.append(
        f"  OVERALL before: n={ob.get('n', 0)} mean={ob.get('q2_mean', float('nan')):.4f} "
        f"std={ob.get('q2_std', float('nan')):.4f}\n"
        f"  OVERALL after : n={oa.get('n', 0)} mean={oa.get('q2_mean', float('nan')):.4f} "
        f"std={oa.get('q2_std', float('nan')):.4f}\n"
    )
    return "".join(lines)


# ------------------------------------------------------------------
# Frozen-1A CF@1 reference (re-computed against stage1_pp.pt)
# ------------------------------------------------------------------


def frozen_acc_1a_cf_reference(
    root: Path, protocol: ProtocolConfig, taus=(0.5, 0.7, 0.9),
) -> Dict:
    """Reproduce acc_1a's offline CF@1 on the original stage1_pp.pt test
    split (full test, not the online subset). Used as the baseline for
    comparison against joint.
    """
    test_p = set(range(TRAIN_N + VAL_N, POOL_SIZE))
    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"),
        expected_protocol=protocol,
    )
    test_records = [
        r for r in all_recs if r.prompt_idx in test_p and r.gamma == 8
    ]
    preds = torch.load(
        str(root / "checkpoints/acc_1a/preds_test.pt"),
        weights_only=False,
    )
    # Sort both in record_idx order just in case.
    preds = sorted(preds, key=lambda r: int(r["record_idx"]))
    assert len(preds) == len(test_records), (
        f"1A preds count {len(preds)} != test records {len(test_records)}"
    )
    out: Dict[float, Dict] = {}
    for tau in taus:
        tau = float(tau)
        exact = under = over = 0
        L_hat_counter: Counter = Counter()
        for p, r in zip(preds, test_records):
            q_list = [float(x) for x in p["q2_hat"]]
            L_hat = commit_threshold(q_list, tau)
            L_hat = max(0, min(L_hat, r.gamma))
            L_strict = int(r.L)
            L_hat_counter[L_hat] += 1
            if L_hat == L_strict:
                exact += 1
            elif L_hat < L_strict:
                under += 1
            else:
                over += 1
        n = len(test_records)
        out[tau] = {
            "tau": tau, "n": n,
            "cf_at_1": exact / n, "under": under / n, "over": over / n,
            "L_hat_distribution": {
                str(k): L_hat_counter[k] / n for k in sorted(L_hat_counter)
            },
        }
    return out


def original_records_cond_var(
    root: Path, protocol: ProtocolConfig,
) -> Dict:
    test_p = set(range(TRAIN_N + VAL_N, POOL_SIZE))
    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"),
        expected_protocol=protocol,
    )
    test_records = [
        r for r in all_recs if r.prompt_idx in test_p and r.gamma == 8
    ]
    return conditional_variance_table(test_records)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--checkpoint_dir", type=str,
                    default="checkpoints/acc_jnt_1a")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(root / "configs/protocol.yaml")
    print(f"[eval_joint] ckpt={ckpt_dir}")
    print(f"[eval_joint] out_dir={out_dir}")

    # Step 1 — recollect.
    recollected = recollect_full_test_split(
        ckpt_dir, protocol, args.gamma, args.T, args.max_new_tokens,
    )
    recoll_path = out_dir / "recollected_test.pt"
    save_records(recollected, str(recoll_path))
    print(
        f"[eval_joint] recollected {len(recollected)} records -> {recoll_path}"
    )

    # Step 2 — offline CF@1.
    joint_cf = offline_cf_joint(ckpt_dir, recollected)
    frozen_cf = frozen_acc_1a_cf_reference(root, protocol)

    # Step 3 — conditional variance diagnostic.
    cv_before = original_records_cond_var(root, protocol)
    cv_after = conditional_variance_table(recollected)

    # Load oracle summary if available.
    oracle_cf = {}
    oracle_path = root / "outputs/oracle_q2_ceiling/summary.json"
    if oracle_path.is_file():
        with open(oracle_path) as f:
            orc = json.load(f)
        for k, v in orc.get("full_test_split", {}).items():
            oracle_cf[float(k)] = v

    # Persist.
    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "recollected_test_records": len(recollected),
        "joint_cf": {str(k): v for k, v in joint_cf.items()},
        "frozen_cf": {str(k): v for k, v in frozen_cf.items()},
        "oracle_cf_full_test": {
            str(k): v for k, v in oracle_cf.items()
        },
        "cond_var_before": cv_before,
        "cond_var_after": cv_after,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Report.
    rep: List[str] = []
    rep.append(
        "# Phase 15 — joint (acc_jnt_1a) vs frozen (acc_1a), test split\n\n"
    )
    rep.append(
        f"Frozen 1A: evaluated on ORIGINAL stage1_pp.pt test records "
        f"(n=1433 expected). "
        f"acc_jnt_1a: evaluated on recollected records under the "
        f"fine-tuned drafter (n={len(recollected)}).\n\n"
    )
    rep.append("## Offline CF@1 (full test split, post-recollection)\n\n```\n")
    rep.append(fmt_cf_table(frozen_cf, joint_cf, oracle_cf if oracle_cf else None))
    rep.append("```\n\n")
    rep.append("## Under / over rates\n\n```\n")
    rep.append(fmt_under_over(joint_cf, frozen_cf))
    rep.append("```\n\n")
    rep.append(
        "## Q2 conditional variance by drafter_top1 decile  "
        "(before = original stage1_pp.pt; after = recollected under "
        "fine-tuned drafter)\n\n```\n"
    )
    rep.append(fmt_var_table(cv_before, cv_after))
    rep.append("```\n\n")
    rep.append("## Induced L̂ distribution per τ (joint)\n\n")
    for tau in sorted(joint_cf.keys()):
        rep.append(f"τ = {tau}:  L̂ dist = "
                   f"{joint_cf[tau]['L_hat_distribution']}\n")
    rep.append("\n")
    with open(out_dir / "report.md", "w") as f:
        f.writelines(rep)
    print("".join(rep))
    print(f"[eval_joint] wrote {out_dir}/report.md")
    print(f"[eval_joint] wrote {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
