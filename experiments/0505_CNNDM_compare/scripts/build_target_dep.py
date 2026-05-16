"""Build the dep target file (1 - TV self-consistency) from stage-1 records.

Drafter-side computation, temperature-invariant. Takes `--protocol` so
the saved blob's `protocol_fp` matches the records (T=1 for the main
OWT_Frozen_0429 plan; configs/protocol.yaml).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.dependence import compute_dependence_target
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import get_split_config
from accpre.train.dataset import _build_full_prefix_cache


def load_protocol(path: str) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=int, required=True)
    ap.add_argument("--dataset", default="owt")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    print(f"[dep_temp0] protocol fp: {protocol.fingerprint()[:60]}...")
    print(f"[dep_temp0] temperature: {protocol.temperature}")

    records = load_records(args.records, expected_protocol=protocol)
    gamma_records = [r for r in records if int(r.gamma) == int(args.gamma)]
    print(f"[dep_temp0] loaded {len(records)} records; kept {len(gamma_records)} at gamma={args.gamma}")

    cfg = get_split_config(args.dataset)
    ranges = {
        "train": (0, cfg.train_n),
        "val":   (cfg.train_n, cfg.train_n + cfg.val_n),
        "test":  (cfg.train_n + cfg.val_n, cfg.pool_size),
    }
    keep_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    keep_prompts = set()
    for s in keep_splits:
        keep_prompts |= set(range(*ranges[s]))
    filtered = [r for r in gamma_records if int(r.prompt_idx) in keep_prompts]
    print(f"[dep_temp0] after split filter: {len(filtered)} records")

    print("[dep_temp0] building full-prefix cache...")
    prefixes: List[torch.Tensor] = _build_full_prefix_cache(filtered, dataset=args.dataset)

    from accpre.models.drafter_mdlm import MDLMDrafter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    drafter = MDLMDrafter(model_name=protocol.drafter_model, device=device, dtype=dtype_map[protocol.dtype])
    drafter.model.eval()

    out: Dict = {}
    t0 = time.time()
    for pos, r in enumerate(filtered):
        prefix = prefixes[pos].to(device)
        draft_tokens = torch.tensor(r.draft_tokens, dtype=torch.long, device=device)
        s = compute_dependence_target(drafter, prefix, draft_tokens, int(args.gamma), sigma=float(args.sigma))
        key = (int(r.prompt_idx), int(r.round_idx))
        out[key] = [float(x) for x in s.tolist()]
        if (pos + 1) % int(args.log_every) == 0:
            el = time.time() - t0
            eta = el / (pos + 1) * (len(filtered) - pos - 1)
            print(f"[dep_temp0] {pos + 1:5d}/{len(filtered)} elapsed={el:6.1f}s eta={eta:6.1f}s")

    print(f"[dep_temp0] wrote {len(out)} dep targets in {time.time()-t0:.1f}s")
    blob = {
        "protocol_fp": protocol.fingerprint(),
        "drafter_fp":  protocol.drafter_model,
        "sigma": float(args.sigma),
        "gamma": int(args.gamma),
        "targets": out,
        "_label_semantics": (
            f"1 - TV(Q_prefix, Q_revealed) under T={protocol.temperature} "
            f"(sigma={float(args.sigma)}); drafter-side, temperature-invariant"
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, str(out_path))
    print(f"[dep_temp0] saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
