"""Build the continuous verifier-relative target file (relmax).

Per record, per drafted position j:

    score_j = p_v(draft_j) / p_v(argmax_v_j)   ∈ [0, 1]

where `p_v(draft_j)` is the verifier softmax probability assigned to the
drafted token at position j, and `p_v(argmax_v_j)` is the maximum
verifier softmax probability at position j (the verifier's own greedy
choice). This score is the verifier-relative confidence in the drafted
token; it is well-defined under any protocol temperature.

Inputs
------
- A stage-1 record file (e.g.
  `data_collected/stage1_pp_owt_300_g15_T1.pt`).
- The protocol file used to collect the records (`configs/protocol.yaml`
  for the T=1 OWT_Frozen_0429 experiment).
- The OWT prefix-pool (via `accpre.train.dataset.JointAcceptanceDataset`
  which reconstructs the actual prefix at each round from the strict
  trajectory).

Outputs
-------
A torch.save dict in the same layout as the dep target file so that
`accpre/train/dataset.py:_load_dep_targets` can ingest it via the
existing `dep_targets_path` mechanism.

    {
      "protocol_fp":      str,
      "drafter_fp":       str,
      "sigma":            0.0,
      "gamma":            int,
      "targets":          dict[(p, r) -> list[float γ]],
      "_label_semantics": "p_verifier(draft) / p_verifier(argmax_verifier) under temperature=1",
      "_summary":         {n_records, score_mean/std/min/max,
                           n_accepted_positions, accepted_score_hist_5bin, ...},
    }

Sanity diagnostic
-----------------
For each accepted position (`record.accepted_j[j] == 1`), bin the
relmax score into 5 equal-width bins over `[0, 1]`. Under T=0 (greedy)
this histogram concentrates entirely in the top bin; under T=1
(Leviathan stochastic accept) the histogram has positive mass in all
bins by design, because the accept rule `U < min(1, p_v/q)` accepts
positions where the draft is not the verifier's argmax. **No hard
assert that accepted positions have score=1.0 — that property is
T=0-specific.**

CLI
---
    python build_target_relmax.py \\
        --records data_collected/stage1_pp_owt_300_g15_T1.pt \\
        --protocol configs/protocol.yaml \\
        --gamma 15 \\
        --dataset owt_300 \\
        --out data_collected/stage1_pp_owt_300_g15_T1_relmax.pt

Smoke flag (`--n_smoke N`) runs only the first N records and writes a
summary JSON without overwriting the production file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.train.dataset import JointAcceptanceDataset


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--gamma", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_smoke", type=int, default=0,
                    help="If > 0, process only first N records and write smoke summary.")
    ap.add_argument("--smoke_summary",
                    default="experiments/OWT_Frozen_0429/results/relmax_smoke_summary.json")
    ap.add_argument("--dataset", default="owt_300")
    args = ap.parse_args()

    protocol = load_protocol(Path(args.protocol))
    print(f"[relmax] protocol fp: {protocol.fingerprint()[:60]}...")
    print(f"[relmax] temperature: {protocol.temperature}")

    all_recs = load_records(args.records, expected_protocol=protocol)
    gamma_filtered = [r for r in all_recs if int(r.gamma) == int(args.gamma)]
    print(f"[relmax] loaded {len(all_recs)} records; γ={args.gamma} → {len(gamma_filtered)} kept")

    if args.n_smoke > 0:
        gamma_filtered = gamma_filtered[:args.n_smoke]
        print(f"[relmax] SMOKE mode: limiting to {len(gamma_filtered)} records")

    # JointAcceptanceDataset reconstructs the running prefix at each round
    # from the strict trajectory (record.L + record.bonus_or_fallback_token).
    ds = JointAcceptanceDataset(
        gamma_filtered, family="hidden_per_pos_v2",
        dataset=str(args.dataset),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[protocol.dtype]
    print(f"[relmax] device={device}, dtype={protocol.dtype}")

    from accpre.models.verifier_gpt2 import GPT2Verifier
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )
    verifier.model.eval()

    EPS = 1e-10
    targets: Dict = {}
    n_processed = 0
    n_acc_total = 0
    accepted_hist = [0, 0, 0, 0, 0]   # 5 equal-width bins over [0, 1]
    score_sum = 0.0
    score_sumsq = 0.0
    score_count = 0
    score_min, score_max = 1e9, -1e9
    score_le_eps_count = 0
    t_start = time.time()

    print(f"[relmax] processing {len(gamma_filtered)} records ...")
    for idx, record in enumerate(gamma_filtered):
        item = ds[idx]
        prefix = item["prefix_ids"].to(device)
        prefix_len = int(prefix.shape[0])
        gamma = int(record.gamma)
        draft_tokens = torch.tensor(record.draft_tokens, dtype=torch.long, device=device)

        candidate = torch.cat([prefix, draft_tokens])  # (L+γ,)
        if candidate.shape[0] > protocol.max_verifier_ctx:
            raise RuntimeError(
                f"verifier ctx exceeded at p={record.prompt_idx} r={record.round_idx}: "
                f"{candidate.shape[0]} > {protocol.max_verifier_ctx}"
            )

        target_lp = verifier.score(candidate)              # (L+γ, V_v)
        target_lp = target_lp.to(torch.float32)

        idx_pos = torch.arange(gamma, device=device)
        target_pos = (prefix_len - 1) + idx_pos              # (γ,)

        rows = target_lp[target_pos, :]                       # (γ, V_v)
        V_v = int(rows.shape[-1])
        draft_for_lookup = draft_tokens.clamp(max=V_v - 1)    # (γ,) long
        log_p_draft = rows.gather(1, draft_for_lookup.unsqueeze(1)).squeeze(1)  # (γ,)
        log_p_argmax = rows.max(dim=-1).values                 # (γ,)

        p_draft = log_p_draft.exp().clamp(min=EPS)
        p_argmax = log_p_argmax.exp().clamp(min=EPS)
        score = (p_draft / p_argmax).clamp(min=0.0, max=1.0)   # (γ,) in [0, 1]

        score_cpu = score.cpu().tolist()
        targets[(int(record.prompt_idx), int(record.round_idx))] = [
            float(x) for x in score_cpu
        ]

        # Per-position score stats.
        for j in range(gamma):
            s = float(score_cpu[j])
            score_sum += s
            score_sumsq += s * s
            score_count += 1
            if s < score_min: score_min = s
            if s > score_max: score_max = s
            if s < EPS * 10: score_le_eps_count += 1

        # Histogram of relmax score at accepted positions only (T=1 diagnostic).
        for j, a in enumerate(record.accepted_j):
            if int(a) == 1:
                n_acc_total += 1
                bin_idx = min(int(float(score_cpu[j]) * 5.0), 4)
                accepted_hist[bin_idx] += 1

        n_processed += 1
        if (n_processed % 200 == 0) or (n_processed == len(gamma_filtered)):
            el = time.time() - t_start
            rate = n_processed / max(el, 1e-9)
            eta_s = (len(gamma_filtered) - n_processed) / max(rate, 1e-9)
            print(f"[relmax] {n_processed}/{len(gamma_filtered)}  "
                  f"({rate:.1f} rec/s, ETA {eta_s/60:.1f} min)")

    elapsed = time.time() - t_start
    mean_score = score_sum / max(score_count, 1)
    var_score = score_sumsq / max(score_count, 1) - mean_score * mean_score
    std_score = max(var_score, 0.0) ** 0.5

    summary = {
        "n_records": int(n_processed),
        "n_total_positions": int(score_count),
        "n_accepted_positions": int(n_acc_total),
        "accepted_score_hist_5bin": [int(x) for x in accepted_hist],
        "score_mean": float(mean_score),
        "score_std": float(std_score),
        "score_min": float(score_min),
        "score_max": float(score_max),
        "score_count": int(score_count),
        "score_le_eps_count": int(score_le_eps_count),
        "elapsed_s": float(elapsed),
        "throughput_rec_per_s": float(n_processed / max(elapsed, 1e-9)),
    }
    print(f"[relmax] sanity:\n  {json.dumps(summary, indent=2)}")

    if args.n_smoke > 0:
        out_path = Path(args.smoke_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[relmax] smoke summary written to {out_path}")
        return 0

    blob = {
        "protocol_fp":      protocol.fingerprint(),
        "drafter_fp":       protocol.drafter_model,
        "sigma":            0.0,
        "gamma":            int(args.gamma),
        "targets":          targets,
        "_label_semantics": "p_verifier(draft) / p_verifier(argmax_verifier) under temperature=1",
        "_summary":         summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, str(out_path))
    print(f"[relmax] saved {len(targets)} targets to {out_path}")
    print(f"[relmax] DONE in {elapsed/60:.2f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
