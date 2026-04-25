"""Phase 23 — sanity for the 1-TV dependence target.

Checks, on a spread of stage1_pp.pt records under the unmodified
pretrained MDLM drafter:

  (A)  s_j ∈ [0, 1] for every j and record.
  (B)  s_0 == 1.0 exactly (by construction — identical inputs).
  (C)  s is NOT collinear with Q2: |ρ(s_flat, Q2_flat)| < 0.95 over the
       pooled position values, AND at least 25% of positions show
       |s_j − Q2_j| > 0.10 (per-position disagreement at a meaningful
       magnitude). Global means can coincidentally match between two
       different targets, so mean-gap alone is a bad test; we use
       correlation + per-position disagreement instead.
  (D)  Deterministic reproducibility: computing s twice on the same
       record (pretrained drafter, same draft_tokens) yields the same
       result bit-for-bit.

Exit 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.dependence import compute_dependence_target
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N
from accpre.train.dataset import _build_full_prefix_cache


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in (
        "draft_salt", "accept_salt", "fallback_salt",
        "schema_version", "max_verifier_ctx",
    ):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _pick(records) -> List[int]:
    picks = [0, 1, 2]
    seen = {0, 1, 2}
    by_prompt = {}
    for pos, r in enumerate(records):
        prev = by_prompt.get(int(r.prompt_idx))
        if prev is None or int(r.round_idx) > int(records[prev].round_idx):
            by_prompt[int(r.prompt_idx)] = pos
    for pos in sorted(by_prompt.values()):
        if pos not in seen:
            picks.append(pos)
            seen.add(pos)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--records", type=str,
                    default="data_collected/stage1_pp.pt")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--n_records", type=int, default=12)
    ap.add_argument("--gamma", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    protocol = load_protocol(root / "configs/protocol.yaml")
    recs = load_records(str(root / args.records), expected_protocol=protocol)
    recs = [r for r in recs if int(r.gamma) == int(args.gamma)]
    ranges = {
        "train": (0, TRAIN_N),
        "val":   (TRAIN_N, TRAIN_N + VAL_N),
        "test":  (TRAIN_N + VAL_N, POOL_SIZE),
    }
    lo, hi = ranges[args.split]
    recs = [r for r in recs if lo <= int(r.prompt_idx) < hi]
    picks = _pick(recs)[: int(args.n_records)]
    print(
        f"[dep_sanity] {args.split}: n={len(recs)}  "
        f"picks={picks}"
    )

    prefixes = _build_full_prefix_cache(recs)

    from accpre.models.drafter_mdlm import MDLMDrafter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()

    s_flat: List[float] = []
    q2_flat: List[float] = []
    s0_all_exact = True
    s_in_range = True
    det_ok = True
    rows = []
    for pos in picks:
        r = recs[pos]
        prefix = prefixes[pos].to(device)
        draft_tokens = torch.tensor(
            r.draft_tokens, dtype=torch.long, device=device,
        )
        s = compute_dependence_target(
            drafter, prefix, draft_tokens, int(args.gamma),
        )
        s2 = compute_dependence_target(
            drafter, prefix, draft_tokens, int(args.gamma),
        )
        # (D) determinism.
        if (s - s2).abs().max().item() > 1e-7:
            det_ok = False

        # (A) range.
        if float(s.min().item()) < -1e-6 or float(s.max().item()) > 1.0 + 1e-6:
            s_in_range = False

        # (B) s_0 == 1.0.
        if abs(float(s[0].item()) - 1.0) > 1e-4:
            s0_all_exact = False

        q2 = torch.tensor(r.min_pq_j, dtype=torch.float32)
        s_flat.extend(float(x) for x in s.tolist())
        q2_flat.extend(float(x) for x in q2.tolist())

        rows.append({
            "pos": int(pos),
            "prompt_idx": int(r.prompt_idx),
            "round_idx": int(r.round_idx),
            "s": [float(x) for x in s.tolist()],
            "s_min": float(s.min().item()),
            "s_max": float(s.max().item()),
            "s_mean": float(s.mean().item()),
            "s0": float(s[0].item()),
            "q2_mean": float(q2.mean().item()),
        })
        print(
            f"[dep_sanity]  pos={pos:4d}  prompt={r.prompt_idx:2d} "
            f"round={r.round_idx:2d}  "
            f"s[0..{args.gamma-1}]=[{', '.join(f'{x:.3f}' for x in s.tolist())}]  "
            f"mean(s)={float(s.mean()):.3f}  mean(Q2)={float(q2.mean()):.3f}"
        )

    # (C) s vs Q2 — distinctness.
    s_t = torch.tensor(s_flat, dtype=torch.float32)
    q_t = torch.tensor(q2_flat, dtype=torch.float32)
    if s_t.numel() >= 2:
        s_centered = s_t - s_t.mean()
        q_centered = q_t - q_t.mean()
        denom = (s_centered.norm() * q_centered.norm()).clamp(min=1e-9)
        rho = float((s_centered * q_centered).sum() / denom)
    else:
        rho = 0.0
    mean_gap = float((s_t.mean() - q_t.mean()).abs())
    # Per-position disagreement fraction: how often does |s_j − Q2_j| > 0.10?
    disagree_frac = float(((s_t - q_t).abs() > 0.10).float().mean().item()) \
        if s_t.numel() > 0 else 0.0
    print(
        f"\n[dep_sanity] overall:  n_vals={s_t.numel()}  "
        f"mean(s)={float(s_t.mean()):.3f}  mean(Q2)={float(q_t.mean()):.3f}  "
        f"ρ(s, Q2)={rho:+.3f}  |Δmean|={mean_gap:.3f}  "
        f"disagree(|s-Q2|>0.10)={disagree_frac:.2f}"
    )

    fails = []
    if not s_in_range:
        fails.append("s out of [0, 1]")
    if not s0_all_exact:
        fails.append("s_0 != 1")
    if abs(rho) > 0.95:
        fails.append(f"s collinear with Q2 (|ρ|>0.95: got {rho:+.3f})")
    if disagree_frac < 0.25:
        fails.append(
            f"s ≈ Q2 at the per-position level: only "
            f"{disagree_frac:.2f} of positions differ by more than 0.10"
        )
    if not det_ok:
        fails.append("non-deterministic recompute")

    summary = {
        "split": args.split,
        "n_picks": len(picks),
        "overall": {
            "mean_s": float(s_t.mean()),
            "mean_q2": float(q_t.mean()),
            "rho_s_q2": rho,
            "mean_gap": mean_gap,
            "disagree_frac_gt_0p10": disagree_frac,
        },
        "rows": rows,
        "fails": fails,
    }
    out_path = root / "outputs/sanity_dep_target.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[dep_sanity] wrote {out_path}")
    if not fails:
        print("[dep_sanity] PASS")
        return 0
    for msg in fails:
        print(f"[dep_sanity] FAIL: {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
