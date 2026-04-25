"""Strict tok/s refresh on global test-split prompt indices.

The Phase 1 wallclock run (`accpre.eval.wallclock`) produced strict
tok/s = 19.75 using LOCAL `enumerate` indices 0..9 over
`test_split()[:10]`. Every Phase 3+ predictor lane (e.g.
`accpre.eval.online_decode`) seeds with GLOBAL test-split indices
[TRAIN_N + VAL_N .. TRAIN_N + VAL_N + n_prompts-1] = [60..69]. The
two RNG trajectories diverge even though the prompts are identical,
so `× strict` reported by the compare scripts is not perfectly
randomness-matched.

This script re-measures strict tok/s on the SAME global indices used
by the predictor lanes, so the ratio is apples-to-apples. Output is
a small JSON read by `accpre.eval.strict_ref.load_strict_tok_s`,
which the Phase 5/6/7 comparison scripts call to resolve `STRICT`.

No new experiment beyond the refresh; semantics unchanged.

Usage:
    python -m accpre.eval.strict_refresh \\
        --config configs/protocol.yaml \\
        --n_prompts 10 --max_new_tokens 64 --gamma 8 --T 2 \\
        --out outputs/strict_global_idx/summary.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from accpre.data.splits import TRAIN_N, VAL_N, test_split
from accpre.eval.wallclock import _load_protocol_from_yaml, run_strict_lane


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure strict tok/s on global test-split prompt indices."
    )
    ap.add_argument("--config", type=str, required=True,
                    help="Path to configs/protocol.yaml.")
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--out", type=str,
                    default="outputs/strict_global_idx/summary.json")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    protocol = _load_protocol_from_yaml(args.config)

    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[protocol.dtype]
    print(f"[strict_refresh] device={device} dtype={protocol.dtype}")

    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompts = test_split()[: args.n_prompts]
    global_offset = TRAIN_N + VAL_N   # 60 under the canonical split
    prompt_indices = [global_offset + i for i in range(len(prompts))]
    print(
        f"[strict_refresh] {len(prompts)} prompts, global indices "
        f"{prompt_indices[0]}..{prompt_indices[-1]}"
    )

    _, tps, _ = run_strict_lane(
        drafter=drafter,
        verifier=verifier,
        prompts=prompts,
        protocol=protocol,
        gamma=args.gamma,
        T=args.T,
        max_new_tokens=args.max_new_tokens,
        prompt_indices=prompt_indices,
    )
    mean_tok_s = sum(tps) / max(len(tps), 1)

    summary = {
        "protocol_fingerprint": protocol.fingerprint(),
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "gamma": args.gamma,
        "T": args.T,
        "prompt_indices": prompt_indices,
        "strict": {
            "tok_s_per_prompt": tps,
            "tok_s_mean": mean_tok_s,
        },
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[strict_refresh] strict tok/s mean = {mean_tok_s:.4f}")
    print(f"[strict_refresh] wrote {args.out}")


if __name__ == "__main__":
    main()
