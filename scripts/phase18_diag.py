"""Phase 18 diagnostic — Q2 structure and conditional variance before
vs after drafter alignment, and frozen-1A offline CF@1 on aligned
records.

Reads two record files (original + aligned) + two 1A head checkpoints
(frozen acc_1a + retrained acc_1a_aligned), computes:
  - Q2 distribution: mean / std / fraction >= τ for τ in {0.5,0.7,0.9} / frac == 1
  - Conditional variance by drafter_top1 decile (reuse of Phase 16 code)
  - Strict L distribution
  - Offline CF@1 for BOTH heads on BOTH datasets, at τ ∈ {0.5, 0.7, 0.9}.

This script is read-only w.r.t. data and checkpoints; it only writes
its own `report.md` + `summary.json` to `--out_dir`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N


def _load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def filter_test_gamma(records, gamma: int):
    test_prompts = set(range(TRAIN_N + VAL_N, POOL_SIZE))
    return [r for r in records if r.prompt_idx in test_prompts and r.gamma == gamma]


def q2_stats(records) -> Dict:
    q2s: List[float] = []
    L_counter: Counter = Counter()
    for r in records:
        L_counter[int(r.L)] += 1
        for j in range(int(r.gamma)):
            if int(r.survived_j[j]) == 1:
                q2s.append(float(r.min_pq_j[j]))
    q2 = np.array(q2s)
    n_rounds = len(records)
    return {
        "n_records": int(n_rounds),
        "n_survived_positions": int(len(q2)),
        "Q2_mean": float(q2.mean()) if len(q2) else float("nan"),
        "Q2_std":  float(q2.std(ddof=0)) if len(q2) else float("nan"),
        "frac_Q2_ge_0.5": float((q2 >= 0.5).mean()) if len(q2) else float("nan"),
        "frac_Q2_ge_0.7": float((q2 >= 0.7).mean()) if len(q2) else float("nan"),
        "frac_Q2_ge_0.9": float((q2 >= 0.9).mean()) if len(q2) else float("nan"),
        "frac_Q2_eq_1":   float((q2 >= 1.0 - 1e-9).mean()) if len(q2) else float("nan"),
        "L_mean": (
            float(sum(k * v for k, v in L_counter.items()) / max(n_rounds, 1))
            if n_rounds else float("nan")
        ),
        "L_distribution_fraction": {
            str(k): L_counter[k] / max(n_rounds, 1) for k in sorted(L_counter)
        },
    }


def cond_var_table(records, n_bins: int = 10) -> Dict:
    top1s: List[float] = []
    q2s:   List[float] = []
    for r in records:
        for j in range(int(r.gamma)):
            if int(r.survived_j[j]) != 1:
                continue
            top1s.append(float(r.drafter_top1_prob_j[j]))
            q2s.append(float(r.min_pq_j[j]))
    if len(top1s) == 0:
        return {"rows": [], "overall": {}}
    t = np.array(top1s); q = np.array(q2s)
    order = np.argsort(t, kind="stable")
    t = t[order]; q = q[order]
    n = len(t)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        ts = t[lo:hi]; qs = q[lo:hi]
        rows.append({
            "bin": i, "n": int(hi - lo),
            "top1_lo": float(ts.min()), "top1_hi": float(ts.max()),
            "q2_mean": float(qs.mean()), "q2_std": float(qs.std(ddof=0)),
        })
    return {
        "rows": rows,
        "overall": {
            "n": int(n),
            "q2_mean": float(q.mean()),
            "q2_std":  float(q.std(ddof=0)),
        },
    }


def offline_cf_frozen_head(head_ckpt_dir: Path, records, taus=(0.5, 0.7, 0.9)) -> Dict:
    from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmb
    with open(head_ckpt_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    head = AcceptanceMLPPerPosTokEmb(
        family=cfg["family"], gamma=int(cfg.get("gamma", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        token_emb_dim=int(cfg.get("token_emb_dim", 64)),
    )
    head.load_state_dict(
        torch.load(head_ckpt_dir / "model.pt", map_location="cpu"),
    )
    head.eval()
    out: Dict[float, Dict] = {}
    for tau in taus:
        tau = float(tau)
        exact = under = over = 0
        Lhat_counter: Counter = Counter()
        for r in records:
            feats = extract_features(r, cfg["family"])
            tok = torch.tensor(r.draft_tokens, dtype=torch.long)
            Lhat = int(head.predict_L(feats, tau, token_ids=tok))
            Lhat = max(0, min(Lhat, r.gamma))
            Ls = int(r.L)
            Lhat_counter[Lhat] += 1
            if Lhat == Ls: exact += 1
            elif Lhat < Ls: under += 1
            else: over += 1
        n = max(len(records), 1)
        out[tau] = {
            "tau": tau, "n": n,
            "cf_at_1": exact / n, "under": under / n, "over": over / n,
            "Lhat_eq_0_frac":
                Lhat_counter[0] / n if 0 in Lhat_counter else 0.0,
            "Lhat_distribution":
                {str(k): Lhat_counter[k] / n for k in sorted(Lhat_counter)},
        }
    return out


def render(summary: Dict) -> str:
    lines = []
    lines.append("# Phase 18 — drafter alignment diagnostic\n\n")
    lines.append("Drafter: token-slot alignment (log q − log p)² on stored drafted tokens.\n\n")

    lines.append("## Q2 structure (survived positions in test split)\n\n```\n")
    lines.append(f"  {'metric':<22} {'before':>12} {'after':>12}\n")
    b = summary["q2_before"]; a = summary["q2_after"]
    for k in ("n_records", "n_survived_positions",
              "Q2_mean", "Q2_std",
              "frac_Q2_ge_0.5", "frac_Q2_ge_0.7",
              "frac_Q2_ge_0.9", "frac_Q2_eq_1",
              "L_mean"):
        bv = b.get(k); av = a.get(k)
        def fmt(v):
            if v is None: return "   —   "
            if isinstance(v, float):
                return f"{v:>12.4f}"
            return f"{v:>12d}" if isinstance(v, int) else f"{v:>12}"
        lines.append(f"  {k:<22} {fmt(bv)} {fmt(av)}\n")
    lines.append("```\n\n")

    lines.append("## Conditional variance (Q2 | drafter_top1 decile)\n\n```\n")
    lines.append(
        f"  {'bin':>3} {'n_bef':>6} {'top1_bef':>18} {'meanB':>8} {'stdB':>8}"
        f"   {'n_aft':>6} {'top1_aft':>18} {'meanA':>8} {'stdA':>8}\n"
    )
    rb = summary["cv_before"]["rows"]; ra = summary["cv_after"]["rows"]
    for i in range(max(len(rb), len(ra))):
        b = rb[i] if i < len(rb) else None
        a = ra[i] if i < len(ra) else None
        row = f"  {i:>3}"
        if b is not None:
            row += (
                f" {b['n']:>6d} [{b['top1_lo']:6.3f}, {b['top1_hi']:6.3f}] "
                f"{b['q2_mean']:>8.4f} {b['q2_std']:>8.4f}"
            )
        else:
            row += f" {'-':>6} {'-':>18} {'-':>8} {'-':>8}"
        if a is not None:
            row += (
                f"  {a['n']:>6d} [{a['top1_lo']:6.3f}, {a['top1_hi']:6.3f}] "
                f"{a['q2_mean']:>8.4f} {a['q2_std']:>8.4f}"
            )
        else:
            row += f"  {'-':>6} {'-':>18} {'-':>8} {'-':>8}"
        lines.append(row + "\n")
    ob = summary["cv_before"]["overall"]; oa = summary["cv_after"]["overall"]
    lines.append(
        f"  OVERALL before: n={ob.get('n', 0)} mean={ob.get('q2_mean', float('nan')):.4f} "
        f"std={ob.get('q2_std', float('nan')):.4f}\n"
    )
    lines.append(
        f"  OVERALL after : n={oa.get('n', 0)} mean={oa.get('q2_mean', float('nan')):.4f} "
        f"std={oa.get('q2_std', float('nan')):.4f}\n"
    )
    lines.append("```\n\n")

    lines.append("## Offline CF@1 — full test split (frozen heads on respective records)\n\n")
    lines.append("```\n")
    taus = (0.5, 0.7, 0.9)
    lines.append("  " + " " * 28 + "".join(f"{'τ='+str(t):>12}" for t in taus) + "\n")
    for label, key in [
        ("frozen 1A  (orig → orig)", "cf_frozen_on_orig"),
        ("frozen 1A  (orig → aligned)", "cf_frozen_on_aligned"),
        ("aligned 1A (aligned → aligned)", "cf_aligned_on_aligned"),
        ("aligned 1A (aligned → orig)",    "cf_aligned_on_orig"),
    ]:
        row = f"  CF@1  {label:<28}"
        tbl = summary.get(key, {})
        for tau in taus:
            v = tbl.get(str(tau), {}).get("cf_at_1", float("nan"))
            row += f"{v:>12.4f}" if isinstance(v, float) and np.isfinite(v) else f"{'   —':>12}"
        lines.append(row + "\n")
    lines.append("\n  (L̂=0 fraction)\n")
    for label, key in [
        ("frozen 1A  (orig → orig)", "cf_frozen_on_orig"),
        ("aligned 1A (aligned → aligned)", "cf_aligned_on_aligned"),
    ]:
        row = f"  L̂=0  {label:<28}"
        tbl = summary.get(key, {})
        for tau in taus:
            v = tbl.get(str(tau), {}).get("Lhat_eq_0_frac", float("nan"))
            row += f"{v:>12.4f}" if isinstance(v, float) and np.isfinite(v) else f"{'   —':>12}"
        lines.append(row + "\n")
    lines.append("```\n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--data_before", type=str,
                    default="data_collected/stage1_pp.pt")
    ap.add_argument("--data_after", type=str,
                    default="data_collected/stage1_pp_aligned.pt")
    ap.add_argument("--frozen_ckpt", type=str,
                    default="checkpoints/acc_1a")
    ap.add_argument("--aligned_ckpt", type=str,
                    default="checkpoints/acc_1a_aligned")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--gamma", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = _load_protocol(root / "configs/protocol.yaml")
    before_all = load_records(
        str(root / args.data_before), expected_protocol=protocol,
    )
    after_all = load_records(
        str(root / args.data_after), expected_protocol=protocol,
    )
    before_test = filter_test_gamma(before_all, int(args.gamma))
    after_test  = filter_test_gamma(after_all,  int(args.gamma))

    summary: Dict = {}
    summary["q2_before"] = q2_stats(before_test)
    summary["q2_after"]  = q2_stats(after_test)
    summary["cv_before"] = cond_var_table(before_test)
    summary["cv_after"]  = cond_var_table(after_test)

    frozen_dir = (root / args.frozen_ckpt).resolve()
    aligned_dir = (root / args.aligned_ckpt).resolve()

    # Frozen 1A head on original records (sanity: should match acc_1a's stored test metrics).
    summary["cf_frozen_on_orig"] = {
        str(k): v for k, v in
        offline_cf_frozen_head(frozen_dir, before_test).items()
    }
    # Frozen 1A head on ALIGNED records — cross-eval: does old head degrade on new drafter?
    summary["cf_frozen_on_aligned"] = {
        str(k): v for k, v in
        offline_cf_frozen_head(frozen_dir, after_test).items()
    }
    if aligned_dir.is_dir() and (aligned_dir / "model.pt").is_file():
        # Aligned 1A head on aligned records — the real test.
        summary["cf_aligned_on_aligned"] = {
            str(k): v for k, v in
            offline_cf_frozen_head(aligned_dir, after_test).items()
        }
        # Cross-check: aligned head on original records.
        summary["cf_aligned_on_orig"] = {
            str(k): v for k, v in
            offline_cf_frozen_head(aligned_dir, before_test).items()
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    rep = render(summary)
    with open(out_dir / "report.md", "w") as f:
        f.write(rep)
    print(rep)
    print(f"[phase18_diag] wrote {out_dir}/report.md")
    print(f"[phase18_diag] wrote {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
