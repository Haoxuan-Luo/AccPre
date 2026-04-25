"""Wall-clock lane runners for strict SpecDiff and the oracle-Q2 baseline.

One timer pattern, applied identically to every lane: timer starts
immediately before the decode loop, stops immediately after the last
commit. Model load, tokenization, post-hoc audits (XL-audit, CF@1) are
all OUTSIDE the timer. This is the only accepted timer discipline in v1.

Two lanes in Phase 1:
  - strict: canonical Leviathan SpecDiff (`draft_verify_round`).
  - oracle_q2(tau): same draft/verify/accept scaffold as strict, but
    commits via `commit_threshold(min_pq_j, tau)` instead of
    `commit_strict(accepted_j)`. Uses the SAME draft_rng and the SAME
    fallback_rng as strict — so any mismatch in outputs is purely a
    function of the commit rule, not randomness drift.

CLI entry (`python -m accpre.eval.wallclock ...`):
  Loads protocol.yaml, runs both lanes (strict + oracle_q2 for each
  tau), computes CF@1 and XL-audit side-table, writes outputs to a JSON
  plus saved records to .pt.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from accpre.core.commit import commit_strict, commit_threshold
from accpre.core.draft_verify import (
    build_record,
    draft_verify_accept,
    draft_verify_round,
    sample_fallback_or_bonus,
)
from accpre.core.protocol import ProtocolConfig, derive_seed
from accpre.core.schema import RoundRecord, save_records


# ---------- lane runners ----------


def _append_committed(
    running: torch.Tensor, rec: RoundRecord
) -> torch.Tensor:
    """Extend the running context with the committed prefix + extra token."""
    device = running.device
    draft_prefix = torch.tensor(
        rec.draft_tokens[: rec.L], dtype=torch.long, device=device
    )
    extra = torch.tensor(
        [rec.bonus_or_fallback_token], dtype=torch.long, device=device
    )
    return torch.cat([running, draft_prefix, extra])


def run_strict_lane(
    drafter,
    verifier,
    prompts: List[Tuple[torch.Tensor, str]],
    protocol: ProtocolConfig,
    gamma: int,
    T: int,
    max_new_tokens: int,
    prompt_indices: Optional[List[int]] = None,
) -> Tuple[List[List[RoundRecord]], List[float], List[torch.Tensor]]:
    """Run strict SpecDiff on every prompt. Returns per-prompt records,
    wall-clock tok/s, and final generated sequences.

    `prompt_indices` optionally overrides the default local `enumerate`
    indices. Supplying the **global** pool indices (matching Phase 2+
    collection and the predictor-lane online decode convention) is
    required for randomness-matched comparison with those lanes. If
    None (default), local 0..N-1 indices are used — this preserves the
    Phase 1 baseline behavior (19.75 tok/s reference).
    """
    if prompt_indices is not None and len(prompt_indices) != len(prompts):
        raise ValueError(
            f"prompt_indices length {len(prompt_indices)} != "
            f"prompts length {len(prompts)}"
        )

    records_per_prompt: List[List[RoundRecord]] = []
    tok_s_per_prompt: List[float] = []
    generated_per_prompt: List[torch.Tensor] = []

    MAX_CTX = protocol.max_verifier_ctx

    for i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[i]) if prompt_indices is not None else i
        prefix_ids = prefix_ids.to(drafter.device)
        running = prefix_ids.clone()
        records: List[RoundRecord] = []
        round_idx = 0

        # Timer: decode loop only.
        t_start = time.time()
        while (running.shape[0] - prefix_ids.shape[0]) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_ids.shape[0])
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(int(gamma), int(remaining), int(ctx_room))
            if cur_gamma <= 0:
                break

            seed = derive_seed(protocol, p_idx, round_idx)
            rec = draft_verify_round(
                prefix_ids=running,
                drafter=drafter,
                verifier=verifier,
                gamma=cur_gamma,
                T=T,
                protocol=protocol,
                prompt_idx=p_idx,
                round_idx=round_idx,
                round_rng_seed=seed,
            )
            running = _append_committed(running, rec)
            records.append(rec)
            round_idx += 1
        elapsed = time.time() - t_start

        n_new = int(running.shape[0] - prefix_ids.shape[0])
        tok_s = n_new / max(elapsed, 1e-9)
        records_per_prompt.append(records)
        tok_s_per_prompt.append(tok_s)
        generated_per_prompt.append(running)

    return records_per_prompt, tok_s_per_prompt, generated_per_prompt


def run_oracle_q2_lane(
    drafter,
    verifier,
    prompts: List[Tuple[torch.Tensor, str]],
    protocol: ProtocolConfig,
    gamma: int,
    T: int,
    max_new_tokens: int,
    tau: float,
) -> Tuple[List[List[RoundRecord]], List[float], List[torch.Tensor]]:
    """Run the oracle-Q2 threshold baseline ONLINE.

    Uses the same draft+verify+accept scaffold as strict (same draft_rng,
    same accept_rng, same fallback_rng), but replaces `commit_strict`
    with `commit_threshold(min_pq_j, tau)`. Still runs the full verifier
    pass, so this lane is NOT a throughput win — it is a quality-vs-
    throughput reference for what a threshold rule on the ideal
    per-position signal can achieve.

    The records stored by this lane have `L == L_tau` (not the strict L);
    `accepted_j`, `min_pq_j`, `U_j`, etc. are identical to what strict
    would have logged on the same seed.
    """
    records_per_prompt: List[List[RoundRecord]] = []
    tok_s_per_prompt: List[float] = []
    generated_per_prompt: List[torch.Tensor] = []

    MAX_CTX = protocol.max_verifier_ctx

    for p_idx, (prefix_ids, _text) in enumerate(prompts):
        prefix_ids = prefix_ids.to(drafter.device)
        running = prefix_ids.clone()
        records: List[RoundRecord] = []
        round_idx = 0

        t_start = time.time()
        while (running.shape[0] - prefix_ids.shape[0]) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_ids.shape[0])
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(int(gamma), int(remaining), int(ctx_room))
            if cur_gamma <= 0:
                break

            seed = derive_seed(protocol, p_idx, round_idx)
            artifacts = draft_verify_accept(
                prefix_ids=running,
                drafter=drafter,
                verifier=verifier,
                gamma=cur_gamma,
                T=T,
                protocol=protocol,
                round_rng_seed=seed,
            )
            # Oracle-Q2 rule: threshold on min_pq_j (== Q2_j).
            L_tau = commit_threshold(artifacts.outcome.min_pq_j, tau)
            bonus = sample_fallback_or_bonus(
                L=L_tau,
                gamma=artifacts.gamma,
                draft_log_probs=artifacts.draft_log_probs,
                target_log_probs=artifacts.target_log_probs,
                prefix_len=artifacts.prefix_len,
                protocol=protocol,
                fallback_rng=artifacts.fallback_rng,
            )
            rec = build_record(
                artifacts=artifacts,
                L=L_tau,
                bonus_or_fallback_token=bonus,
                protocol=protocol,
                prompt_idx=p_idx,
                round_idx=round_idx,
            )
            running = _append_committed(running, rec)
            records.append(rec)
            round_idx += 1
        elapsed = time.time() - t_start

        n_new = int(running.shape[0] - prefix_ids.shape[0])
        tok_s = n_new / max(elapsed, 1e-9)
        records_per_prompt.append(records)
        tok_s_per_prompt.append(tok_s)
        generated_per_prompt.append(running)

    return records_per_prompt, tok_s_per_prompt, generated_per_prompt


# ---------- CLI entry ----------


def _load_protocol_from_yaml(path: str) -> ProtocolConfig:
    import yaml

    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # `yaml.safe_load` parses 0xDEADBEEF-style literals as strings if quoted;
    # we declared them unquoted so they arrive as ints. Coerce defensively.
    for k in ("draft_salt", "accept_salt", "fallback_salt", "schema_version",
              "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _flatten_records(records_per_prompt: List[List[RoundRecord]]) -> List[RoundRecord]:
    return [r for rs in records_per_prompt for r in rs]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 wall-clock: strict + oracle-Q2 threshold."
    )
    ap.add_argument("--config", type=str, required=True,
                    help="Path to protocol.yaml.")
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--taus", type=str, default="0.3,0.5,0.7,0.9")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    protocol = _load_protocol_from_yaml(args.config)

    # Lazy imports so the module is importable without HF installed.
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier
    from accpre.data.splits import test_split
    from accpre.eval.faithfulness import controlled_faithfulness_at_1
    from accpre.eval.quality import xl_audit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[protocol.dtype]
    print(f"[phase1] device={device} dtype={protocol.dtype}")

    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompts = test_split()[: args.n_prompts]
    print(f"[phase1] {len(prompts)} prompts loaded")

    # --- Lane 1: strict SpecDiff ---
    print("[phase1] Running STRICT lane...")
    s_recs, s_tps, s_gen = run_strict_lane(
        drafter, verifier, prompts, protocol, args.gamma, args.T,
        args.max_new_tokens,
    )
    save_records(_flatten_records(s_recs), os.path.join(args.out_dir, "strict_records.pt"))
    print(f"[phase1]   strict mean tok/s = {sum(s_tps)/max(len(s_tps),1):.2f}")

    # --- Lane 2+: oracle-Q2 threshold sweep ---
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    oracle_results: Dict[str, Dict[str, object]] = {}
    for tau in taus:
        print(f"[phase1] Running ORACLE-Q2 lane (tau={tau})...")
        o_recs, o_tps, o_gen = run_oracle_q2_lane(
            drafter, verifier, prompts, protocol, args.gamma, args.T,
            args.max_new_tokens, tau,
        )
        save_records(
            _flatten_records(o_recs),
            os.path.join(args.out_dir, f"oracle_q2_tau_{tau}_records.pt"),
        )

        # L3 primary quality: CF@1 — method_commit_fn is the oracle-Q2
        # commit applied to each strict record.
        strict_flat = _flatten_records(s_recs)
        cf1 = controlled_faithfulness_at_1(
            strict_flat,
            method_commit_fn=(lambda r, t=tau: commit_threshold(r.min_pq_j, t)),
        )

        # Auxiliary: XL-audit on the oracle-Q2 generated sequences.
        xl = [
            xl_audit(g, int(prefix.shape[0]), verifier)
            for (prefix, _), g in zip(prompts, o_gen)
        ]
        oracle_results[str(tau)] = {
            "tok_s_per_prompt": o_tps,
            "tok_s_mean": sum(o_tps) / max(len(o_tps), 1),
            "cf_at_1_aggregate": cf1["aggregate"],
            "cf_at_1_bootstrap_ci": cf1["bootstrap_ci"],
            "cf_at_1_per_prompt": cf1["per_prompt"],
            "xl_audit_aux_per_prompt": xl,
            "xl_audit_aux_mean": sum(xl) / max(len(xl), 1),
        }

    # --- Auxiliary XL-audit on STRICT (sanity check) ---
    xl_strict = [
        xl_audit(g, int(prefix.shape[0]), verifier)
        for (prefix, _), g in zip(prompts, s_gen)
    ]
    strict_results = {
        "tok_s_per_prompt": s_tps,
        "tok_s_mean": sum(s_tps) / max(len(s_tps), 1),
        "cf_at_1_aggregate": 1.0,   # strict is its own reference
        "xl_audit_aux_per_prompt": xl_strict,
        "xl_audit_aux_mean": sum(xl_strict) / max(len(xl_strict), 1),
    }

    summary = {
        "protocol_fingerprint": protocol.fingerprint(),
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "gamma": args.gamma,
        "T": args.T,
        "taus": taus,
        "strict": strict_results,
        "oracle_q2": oracle_results,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase1] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
