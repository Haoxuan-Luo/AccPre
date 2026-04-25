"""Phase 23 — offline collection of 1-TV dependence targets s_j.

For each `RoundRecord` in `data_collected/stage1_pp.pt` with the
target γ, reconstruct the FULL prefix (same as `JointAcceptanceDataset`
does — see `accpre/train/dataset.py::_build_full_prefix_cache`), and
compute s_j = 1 - TV(Q^prefix_j, Q^revealed_j) on the PRETRAINED MDLM
drafter. The revealed chain uses the stored `record.draft_tokens`.

Output (torch.save):
    {
      "protocol_fp":       str,                   # for consistency checks
      "drafter_fp":        str,                   # pretrained drafter name
      "sigma":             float,                 # 1.0 (always)
      "gamma":             int,                   # target γ (8)
      "targets":           dict[(int prompt_idx, int round_idx)
                                -> list[float γ]] # s_j per record
    }

Only records with `record.gamma == target_gamma` are included; the rest
are skipped so the file lines up with the Phase-15 joint dataset filter.

Invoked from `jobs/phase23_collect_dep.slurm` on a GPU node.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--records", type=str,
                    default="data_collected/stage1_pp.pt")
    ap.add_argument("--out", type=str,
                    default="data_collected/stage1_pp_dep.pt")
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--splits", type=str, default="train,val,test",
                    help="Which splits to include (comma-separated).")
    ap.add_argument("--drafter_state", type=str, default=None,
                    help="Optional drafter .pt to load before collection. "
                         "When omitted, the canonical pretrained MDLM is "
                         "used (the frozen-target setting).")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    protocol = load_protocol(root / "configs/protocol.yaml")
    print(f"[collect_dep] protocol fp: {protocol.fingerprint()[:60]}...")

    records = load_records(
        str(root / args.records), expected_protocol=protocol,
    )
    gamma_records = [r for r in records if int(r.gamma) == int(args.gamma)]
    print(
        f"[collect_dep] loaded {len(records)} records; "
        f"kept {len(gamma_records)} at gamma={args.gamma}"
    )

    ranges = {
        "train": (0, TRAIN_N),
        "val": (TRAIN_N, TRAIN_N + VAL_N),
        "test": (TRAIN_N + VAL_N, POOL_SIZE),
    }
    keep_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    keep_prompts: set = set()
    for s in keep_splits:
        if s not in ranges:
            raise ValueError(f"unknown split {s!r}")
        keep_prompts |= set(range(*ranges[s]))

    filtered = [r for r in gamma_records if int(r.prompt_idx) in keep_prompts]
    print(
        f"[collect_dep] splits={keep_splits}  "
        f"n_records_after_split_filter={len(filtered)}"
    )

    # Full-prefix cache — same reconstruction the joint dataset uses.
    print("[collect_dep] building full-prefix cache...")
    prefixes: List[torch.Tensor] = _build_full_prefix_cache(filtered)

    from accpre.models.drafter_mdlm import MDLMDrafter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    if args.drafter_state is not None:
        path = str(root / args.drafter_state)
        print(f"[collect_dep] loading drafter override: {path}")
        drafter.model.load_state_dict(torch.load(path, map_location=device))
        drafter.model.eval()

    out: Dict = {}
    t0 = time.time()
    for pos, r in enumerate(filtered):
        prefix = prefixes[pos].to(device)
        draft_tokens = torch.tensor(
            r.draft_tokens, dtype=torch.long, device=device,
        )
        s = compute_dependence_target(
            drafter, prefix, draft_tokens, int(args.gamma),
            sigma=float(args.sigma),
        )
        key = (int(r.prompt_idx), int(r.round_idx))
        out[key] = [float(x) for x in s.tolist()]
        if (pos + 1) % int(args.log_every) == 0:
            el = time.time() - t0
            eta = el / (pos + 1) * (len(filtered) - pos - 1)
            print(
                f"[collect_dep] {pos + 1:5d}/{len(filtered)} "
                f"elapsed={el:6.1f}s  eta={eta:6.1f}s  "
                f"last_key={key}  last_s_mean={sum(out[key])/len(out[key]):.3f}"
            )
    total_el = time.time() - t0
    print(
        f"[collect_dep] wrote {len(out)} entries in {total_el:.1f}s "
        f"({total_el / max(len(out), 1):.3f}s/record)"
    )

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "protocol_fp": protocol.fingerprint(),
        "drafter_fp":  (
            f"{protocol.drafter_model}+state={args.drafter_state}"
            if args.drafter_state is not None
            else protocol.drafter_model
        ),
        "sigma": float(args.sigma),
        "gamma": int(args.gamma),
        "targets": out,
    }
    torch.save(blob, str(out_path))
    print(f"[collect_dep] saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
