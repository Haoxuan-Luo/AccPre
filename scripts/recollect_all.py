"""Recollect strict SpecDiff records across all 80 canonical prompts
using a given drafter state.

Wraps `accpre.eval.online_decode.recollect_strict_with_drafter` but
scoped to the full pool (train + val + test). Saves a new stage-1-
equivalent .pt file the downstream trainer can read as
`data_collected/...`.

Required for Phase 18: after drafter alignment, every split's training
records must be recollected under the new drafter state before we can
train a fresh frozen head and compare apples-to-apples against the
original.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import save_records
from accpre.data.prompts import load_owt_prompts
from accpre.data.splits import POOL_SIZE, PREFIX_LEN, PROMPT_SEED
from accpre.eval.online_decode import recollect_strict_with_drafter


def _load_protocol(path: str) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drafter_state", type=str, default=None,
                    help="Optional drafter .pt state_dict to load. "
                         "If omitted, uses the pretrained drafter as-is "
                         "(useful as a round-trip sanity check).")
    ap.add_argument("--protocol_config", type=str, default="configs/protocol.yaml")
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n_prompts", type=int, default=POOL_SIZE,
                    help="Number of canonical prompts to recollect "
                         "(default: all 80; lower for smoke tests).")
    ap.add_argument("--out", type=str, required=True,
                    help="Output .pt path.")
    args = ap.parse_args()

    protocol = _load_protocol(args.protocol_config)

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
    if args.drafter_state is not None:
        state = torch.load(args.drafter_state, map_location=device)
        drafter.model.load_state_dict(state)
        print(f"[recollect_all] loaded drafter state from {args.drafter_state}")
    else:
        print("[recollect_all] using pretrained drafter as-is (no override)")
    drafter.model.eval()

    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    # Canonical full pool (train+val+test, global indices 0..POOL_SIZE-1).
    pool = load_owt_prompts(
        n_prompts=int(args.n_prompts),
        prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )
    prompt_indices = list(range(int(args.n_prompts)))
    print(
        f"[recollect_all] recollecting {len(pool)} prompts "
        f"(global indices 0..{len(pool) - 1}) max_new_tokens={args.max_new_tokens}"
    )

    records = recollect_strict_with_drafter(
        prompts=pool, prompt_indices=prompt_indices, protocol=protocol,
        drafter=drafter, verifier=verifier,
        gamma=int(args.gamma), T=int(args.T),
        max_new_tokens=int(args.max_new_tokens),
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_records(records, str(out_path))
    print(f"[recollect_all] wrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
