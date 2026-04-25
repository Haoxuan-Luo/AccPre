"""Phase 16 sanity A — live Q2 must match stored record.min_pq_j under
the unmodified pretrained drafter + verifier + matched RNG.

Identical logic to the full-prefix sanity test (Phase 15) plus a
verifier forward: we compute `Q2_live = min(1, p_live / q_live)` and
compare to `record.min_pq_j`.

Rationale: before trusting any live-Q2 training run, we must verify
that the live-Q2 target reduces to the stored target when the drafter
is unmodified. If it does not, any gain/loss from live-Q2 training is
confounded by a replay bug, not by the target change.

Tolerance: observed MDLM q_diff noise is ~1.5e-3 (Phase 15 sanity), and
GPT-2 verifier is also in bf16 autocast internally, so we expect Q2
noise up to ~5e-3 in typical ranges and higher where p or q is small.
Set `atol_q2=2e-2` (≈ 13× floor) to avoid false alarms; a real bug
would blow past that.

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

from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N
from accpre.train.dataset import JointAcceptanceDataset


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--n_records", type=int, default=16)
    ap.add_argument("--atol_q2", type=float, default=2e-2,
                    help="atol for |Q2_live - record.min_pq_j| check.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    protocol = load_protocol(root / "configs/protocol.yaml")
    print(f"[live_q2_sanity] protocol fp: {protocol.fingerprint()[:60]}...")

    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"),
        expected_protocol=protocol,
    )
    gamma_filtered = [r for r in all_recs if r.gamma == 8]
    ranges = {
        "train": (0, TRAIN_N),
        "val": (TRAIN_N, TRAIN_N + VAL_N),
        "test": (TRAIN_N + VAL_N, POOL_SIZE),
    }
    lo, hi = ranges[args.split]
    split_prompts = set(range(lo, hi))
    split_records = [r for r in gamma_filtered if r.prompt_idx in split_prompts]
    print(
        f"[live_q2_sanity] {args.split} split: n={len(split_records)} records "
        f"across prompts {lo}..{hi - 1}"
    )

    ds = JointAcceptanceDataset(split_records, family="hidden_per_pos_v2")

    # Pick a spread: early rounds + one deep-round record per prompt.
    picks: List[int] = [0, 1, 2]
    by_prompt = {}
    for pos, r in enumerate(split_records):
        prev = by_prompt.get(int(r.prompt_idx))
        if prev is None or int(r.round_idx) > int(split_records[prev].round_idx):
            by_prompt[int(r.prompt_idx)] = pos
    for pos in sorted(by_prompt.values()):
        picks.append(pos)
    seen = set(); dedup = []
    for p in picks:
        if p in seen: continue
        seen.add(p); dedup.append(p)
    picks = dedup[: args.n_records]
    print(f"[live_q2_sanity] checking {len(picks)} records at positions {picks}")

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
    drafter.model.eval()
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )
    verifier.model.eval()

    EPS = 1e-10
    n_fail = 0
    worst_q2 = 0.0
    mean_q2_list: List[float] = []
    rows_out = []
    for pos in picks:
        item = ds[pos]
        r = split_records[pos]
        prefix = item["prefix_ids"].to(device)
        seed = int(item["round_rng_seed"].item())
        draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
        draft_tokens, draft_log_probs, _ = drafter.draft_with_features(
            prefix_ids=prefix, gamma=int(r.gamma), T=int(r.T),
            temperature=protocol.temperature, q_mode=protocol.q_mode,
            generator=draft_rng, pool="per_position", layers=(-1,),
        )

        # Live verifier forward
        candidate = torch.cat([prefix, draft_tokens.long()])
        with torch.no_grad():
            target_lp = verifier.score(candidate)
        prefix_len = int(prefix.shape[0])
        gamma = int(draft_tokens.shape[0])
        idx = torch.arange(gamma, device=device)
        target_pos = prefix_len - 1 + idx
        p_live = target_lp[target_pos, draft_tokens.long()].exp().clamp(min=EPS)
        q_live = draft_log_probs[idx, draft_tokens.long()].exp().clamp(min=EPS)
        q2_live = torch.minimum(torch.ones_like(p_live), p_live / q_live).cpu()

        q2_stored = torch.tensor(r.min_pq_j, dtype=torch.float32)
        diffs = (q2_live - q2_stored).abs()
        q2_abs_max = float(diffs.max().item())
        q2_abs_mean = float(diffs.mean().item())

        worst_q2 = max(worst_q2, q2_abs_max)
        mean_q2_list.append(q2_abs_mean)

        # Also report q-diff and p-diff to distinguish which side of
        # the ratio drifts more.
        q_stored = torch.tensor(r.q_j, dtype=torch.float32)
        p_stored = torch.tensor(r.p_j, dtype=torch.float32)
        q_abs_max = float((q_live.cpu() - q_stored).abs().max().item())
        p_abs_max = float((p_live.cpu() - p_stored).abs().max().item())

        status = "OK" if q2_abs_max <= args.atol_q2 else "FAIL"
        print(
            f"[live_q2_sanity] {status}  prompt={r.prompt_idx:2d} round={r.round_idx:2d} "
            f"pref={prefix_len:3d}  "
            f"|Q2| max={q2_abs_max:.2e} mean={q2_abs_mean:.2e}  "
            f"|q| max={q_abs_max:.2e}  |p| max={p_abs_max:.2e}"
        )
        rows_out.append({
            "pos": int(pos), "prompt_idx": int(r.prompt_idx),
            "round_idx": int(r.round_idx), "prefix_len": int(prefix_len),
            "q2_abs_max": q2_abs_max, "q2_abs_mean": q2_abs_mean,
            "q_abs_max": q_abs_max, "p_abs_max": p_abs_max,
            "status": status,
        })
        if status == "FAIL":
            n_fail += 1

    summary = {
        "split": args.split, "n_checked": len(picks), "n_fail": n_fail,
        "worst_q2_abs": worst_q2,
        "mean_of_means": (
            sum(mean_q2_list) / max(len(mean_q2_list), 1)
        ),
        "atol_q2": args.atol_q2,
        "rows": rows_out,
    }
    out_path = root / "outputs/sanity_live_q2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[live_q2_sanity] wrote {out_path}")
    if n_fail == 0:
        print(
            f"[live_q2_sanity] PASS  n={len(picks)}  worst_q2={worst_q2:.3e}  "
            f"mean_of_means={summary['mean_of_means']:.3e}"
        )
        return 0
    print(f"[live_q2_sanity] FAIL  {n_fail}/{len(picks)} records failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
