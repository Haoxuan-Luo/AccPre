"""Stage 2A.0 collection CLI.

Runs the canonical strict-SpecDiff pipeline with feature extraction
enabled and writes `data_collected/stage1.pt` as a list of
`RoundRecord`s whose optional feature fields are populated.

Every round is produced by `draft_verify_round_with_features`; there
is no separate DDPM re-implementation here. The canonical
`ProtocolConfig` pins temperature, q_mode, schema_version, salts; the
record layout is the same one Phase 1 already uses.

Usage:
    python -m accpre.collect.cli \\
        --config configs/protocol.yaml \\
        --out data_collected/stage1.pt \\
        --splits train,val,test \\
        --gamma 8 --T 2 --max_new_tokens 128

Splits are defined in `accpre.data.splits`. One `RoundRecord` is
written per (prompt, round) — the full trajectory for each prompt up
to `max_new_tokens`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import torch

from accpre.core.draft_verify import draft_verify_round_with_features
from accpre.core.protocol import ProtocolConfig, derive_seed
from accpre.core.schema import RoundRecord, save_records


def _load_protocol_from_yaml(path: str) -> ProtocolConfig:
    import yaml
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _load_split_prompts(
    splits: List[str],
    dataset: str = "owt",
) -> List[Tuple[int, torch.Tensor, str, str]]:
    """Return (global_prompt_idx, prefix_ids, prefix_text, split_name) tuples.

    Global prompt indices are the pool indices for the chosen dataset.
    They are used for `round_rng_seed` derivation so the same prompt
    always seeds identically no matter which split(s) we collect on.
    """
    from accpre.data.prompts import load_prompts
    from accpre.data.splits import get_split_config

    cfg = get_split_config(dataset)
    pool = load_prompts(
        dataset=cfg.name, n_prompts=cfg.pool_size,
        prefix_len=cfg.prefix_len, seed=cfg.seed,
    )
    ranges = {
        "train": (0, cfg.train_n),
        "val":   (cfg.train_n, cfg.train_n + cfg.val_n),
        "test":  (cfg.train_n + cfg.val_n, cfg.pool_size),
    }
    out: List[Tuple[int, torch.Tensor, str, str]] = []
    for s in splits:
        if s not in ranges:
            raise ValueError(f"Unknown split: {s!r}")
        lo, hi = ranges[s]
        for g in range(lo, hi):
            ids, text = pool[g]
            out.append((g, ids, text, s))
    return out


def run_collection(
    protocol: ProtocolConfig,
    drafter,
    verifier,
    prompts: List[Tuple[int, torch.Tensor, str, str]],
    gamma: int,
    T: int,
    max_new_tokens: int,
) -> List[RoundRecord]:
    """Collect RoundRecords across all given prompts.

    For each prompt, runs the canonical strict trajectory up to
    `max_new_tokens`, calling `draft_verify_round_with_features` each
    round. Extends `running` using strict's committed draft prefix +
    bonus/fallback token (Phase 1 semantics).
    """
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    all_records: List[RoundRecord] = []

    for p_idx, prefix_ids, _text, split_name in prompts:
        prefix_ids = prefix_ids.to(device)
        running = prefix_ids.clone()
        round_idx = 0

        while (running.shape[0] - prefix_ids.shape[0]) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_ids.shape[0])
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(int(gamma), int(remaining), int(ctx_room))
            if cur_gamma <= 0:
                break

            seed = derive_seed(protocol, p_idx, round_idx)
            rec = draft_verify_round_with_features(
                prefix_ids=running,
                drafter=drafter, verifier=verifier,
                gamma=cur_gamma, T=T, protocol=protocol,
                prompt_idx=p_idx, round_idx=round_idx,
                round_rng_seed=seed,
            )
            all_records.append(rec)

            # Advance running context per Phase 1 strict semantics.
            draft_prefix = torch.tensor(
                rec.draft_tokens[: rec.L], dtype=torch.long, device=device,
            )
            extra = torch.tensor(
                [rec.bonus_or_fallback_token], dtype=torch.long, device=device,
            )
            running = torch.cat([running, draft_prefix, extra])
            round_idx += 1

        print(
            f"[collect] prompt {p_idx:3d} ({split_name}): "
            f"{round_idx} rounds, {running.shape[0] - prefix_ids.shape[0]} new tokens"
        )

    return all_records


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2A.0 — offline collection.")
    ap.add_argument("--config", type=str, required=True,
                    help="Path to protocol.yaml.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output records path. Defaults to "
                         "data_collected/stage1_pp_<dataset>.pt.")
    ap.add_argument("--dataset", type=str, default="owt",
                    choices=("owt", "cnn_dm", "mt_bench",
                             "owt_300", "owt_smoke", "cnn_dm_300"),
                    help="Which prompt pool to use; per-dataset split "
                         "sizes come from accpre.data.splits.DATASETS.")
    ap.add_argument("--splits", type=str, default="train,val,test",
                    help="Comma-separated: train,val,test.")
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_path = args.out or f"data_collected/stage1_pp_{args.dataset}.pt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    protocol = _load_protocol_from_yaml(args.config)

    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    print(f"[collect] device={device} dtype={protocol.dtype} dataset={args.dataset}")

    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompts = _load_split_prompts(splits, dataset=args.dataset)
    print(f"[collect] loaded {len(prompts)} prompts across splits {splits}")

    t_start = time.time()
    records = run_collection(
        protocol=protocol, drafter=drafter, verifier=verifier,
        prompts=prompts, gamma=args.gamma, T=args.T,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t_start

    save_records(records, out_path)
    print(
        f"[collect] wrote {len(records)} records to {out_path} "
        f"in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
