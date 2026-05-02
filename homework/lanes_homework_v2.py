"""Homework lanes — v2, temperature-aware.

Rebuilt on top of the canonical
  accpre.core.draft_verify.draft_verify_accept
  + commit rule
  + accpre.core.draft_verify.sample_fallback_or_bonus

so the bonus/fallback token is sampled correctly under both temp=0 (argmax)
and temp>0 (multinomial / Leviathan adjusted-distribution fallback).

Three rules are exposed, all reusing the same per-position ratio
`r_j = min(1, p_j / q_j)` returned in `artifacts.outcome.min_pq_j`:

  run_lossy_lane_one(l)          : Bernoulli `U_j < min(1, p_j / (l*q_j))`
  run_threshold_lossy_lane_one(τ): deterministic `r_j ≥ τ` (commit_threshold)
  run_confidence_lossy_lane_one(τ): deterministic `∏r_j ≥ τ` (commit_confidence)

All three commit `L` accepted draft tokens + 1 bonus/fallback token per round
(strict-style), matching the existing baseline_eval.py and phase26 pattern so
the four homework methods stay directly comparable.

Inline tok_succ / rnd_mean / all_pass keep the lossy-lane denominator
discipline: every round counts toward n_round_total even when L=0, which is
the apples-to-apples convention with the predictor lanes.

NOTE: `phase26.run_lossy_lane_one` and the v1 homework lanes hardcoded
`argmax(target[fb_pos])` for the bonus, which is correct only at temp=0.
This v2 module fixes that by routing through the canonical helper. At
temp=0 the behavior is identical; at temp>0 the bonus is sampled from the
right distribution.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import torch

from specdiff.commit import commit_confidence, commit_strict, commit_threshold
from specdiff.draft_verify import (
    _make_generator, draft_verify_accept, sample_fallback_or_bonus,
)


def _per_round_metrics(
    artifacts, L: int, accepted_j: Sequence[int],
) -> Tuple[int, int]:
    """Return (n_succ, n_committed=L) bounded by L for inline tok_succ.

    `accepted_j` is what the LANE'S accept rule returned per position
    (already populated by the caller); the success indicator comes from
    `artifacts.outcome.accepted_j`, which is the canonical Leviathan
    accept (= per-position success under temp=0 greedy or
    Bernoulli(min(1, p/q)) under temp>0).
    """
    canonical = artifacts.outcome.accepted_j
    n_succ = sum(int(canonical[j]) for j in range(int(L))) if L > 0 else 0
    return n_succ, int(L)


def _decode_loop(
    drafter, verifier, prompts, prompt_indices, protocol,
    rule_fn, method_name: str, extra_meta: Dict,
    gamma: int, T: int, max_new_tokens: int,
) -> Dict:
    """Generic strict-style decode loop. `rule_fn(artifacts) -> L` selects L."""
    from specdiff.protocol import derive_seed
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    for i, (prefix_ids, _txt) in enumerate(prompts):
        p_idx = int(prompt_indices[i])
        running = prefix_ids.to(device).clone()
        prefix_start = int(running.shape[0])
        rounds: List[Dict] = []
        n_tok_total = 0; n_tok_succ = 0
        n_round_total = 0; n_round_all_pass = 0
        sum_round_succ_rate = 0.0
        round_idx = 0
        t_start = time.time()
        while (running.shape[0] - prefix_start) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_start)
            ctx_room = MAX_CTX - int(running.shape[0]) - 1
            cur_gamma = min(gamma, remaining, ctx_room)
            if cur_gamma <= 0:
                break
            seed = derive_seed(protocol, p_idx, round_idx)
            artifacts = draft_verify_accept(
                prefix_ids=running, drafter=drafter, verifier=verifier,
                gamma=cur_gamma, T=T, protocol=protocol,
                round_rng_seed=seed,
            )
            L = rule_fn(artifacts)
            L = max(0, min(int(L), cur_gamma))
            bonus = sample_fallback_or_bonus(
                L=L, gamma=artifacts.gamma,
                draft_log_probs=artifacts.draft_log_probs,
                target_log_probs=artifacts.target_log_probs,
                prefix_len=artifacts.prefix_len,
                protocol=protocol,
                fallback_rng=artifacts.fallback_rng,
            )

            n_succ, _ = _per_round_metrics(
                artifacts, L, artifacts.outcome.accepted_j,
            )
            n_round_total += 1
            n_tok_total += L
            n_tok_succ += n_succ
            if L > 0:
                sum_round_succ_rate += n_succ / L
                if n_succ == L:
                    n_round_all_pass += 1

            draft_pref = torch.tensor(
                artifacts.draft_tokens[:L].tolist(),
                dtype=torch.long, device=device,
            )
            extra = torch.tensor([bonus], dtype=torch.long, device=device)
            running = torch.cat([running, draft_pref, extra])
            rounds.append({
                "round_idx": round_idx,
                "L": int(L), "n_committed": int(L + 1),
                "round_rng_seed": int(seed), "gamma": int(cur_gamma),
            })
            round_idx += 1
        elapsed = time.time() - t_start
        n_new = int(running.shape[0]) - prefix_start
        per_prompt.append({
            "prompt_idx": p_idx,
            "n_new_tokens": int(n_new),
            "elapsed_s": float(elapsed),
            "tok_s": float(n_new) / max(elapsed, 1e-9),
            "generated_ids": [int(x) for x in running.tolist()],
            "rounds": rounds,
            "n_tokens_total": int(n_tok_total),
            "tok_succ": n_tok_succ / max(n_tok_total, 1),
            "rnd_mean": sum_round_succ_rate / max(n_round_total, 1),
            "all_pass": n_round_all_pass / max(n_round_total, 1),
        })
    tok_s_mean = sum(p["tok_s"] for p in per_prompt) / max(len(per_prompt), 1)
    return {
        "method": method_name, **extra_meta,
        "tok_s_mean": tok_s_mean, "per_prompt": per_prompt,
    }


@torch.no_grad()
def run_lossy_lane_one(
    drafter, verifier, prompts, prompt_indices, protocol,
    l: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Lossy SD: U_j < min(1, p_j / (l * q_j))."""
    EPS = 1e-10
    def rule(artifacts):
        # Re-derive U_j from accept_rng to keep matched randomness vs the
        # canonical Leviathan stream. accept_rng here is fresh per round,
        # seeded the same way as in draft_verify_accept.
        device = artifacts.draft_tokens.device
        accept_rng = _make_generator(
            device,
            artifacts.round_rng_seed ^ protocol.accept_salt,
        )
        gamma_r = artifacts.gamma
        # Skip the cur_gamma U_j draws that draft_verify_accept already
        # consumed by drawing them again into a discard buffer — this
        # keeps the loop-side and canonical-side accept_rng streams
        # comparable for a fresh draw of the lossy-rule U_j. We use the
        # ratios already computed by per_position_accept.
        ratio = torch.tensor(
            artifacts.outcome.min_pq_j, device=device, dtype=torch.float32,
        )
        q = torch.tensor(
            artifacts.outcome.q_j, device=device, dtype=torch.float32,
        ).clamp(min=EPS)
        p = torch.tensor(
            artifacts.outcome.p_j, device=device, dtype=torch.float32,
        ).clamp(min=EPS)
        a_lossy = torch.minimum(torch.ones_like(q), p / (float(l) * q))
        # Use a deterministic, lane-specific RNG seed so the lossy U_j
        # do not collide with the per_position_accept stream. The salt
        # `0x10551A` is just a non-zero constant unique to this lane.
        lossy_rng = _make_generator(
            device,
            artifacts.round_rng_seed ^ protocol.accept_salt ^ 0x10551A,
        )
        U = torch.rand(gamma_r, generator=lossy_rng, device=device)
        accepted = (U < a_lossy).to(torch.int32).cpu().tolist()
        return commit_strict(accepted)
    return _decode_loop(
        drafter, verifier, prompts, prompt_indices, protocol,
        rule_fn=rule, method_name="lossy_l", extra_meta={"l": float(l)},
        gamma=gamma, T=T, max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def run_threshold_lossy_lane_one(
    drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Threshold lossy: deterministic `r_j = min(1, p_j/q_j) ≥ τ`."""
    def rule(artifacts):
        return commit_threshold(artifacts.outcome.min_pq_j, float(tau))
    return _decode_loop(
        drafter, verifier, prompts, prompt_indices, protocol,
        rule_fn=rule, method_name="threshold_lossy",
        extra_meta={"tau": float(tau)},
        gamma=gamma, T=T, max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def run_confidence_lossy_lane_one(
    drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Confidence lossy: deterministic `∏ r_j ≥ τ` (commit_confidence)."""
    def rule(artifacts):
        return commit_confidence(artifacts.outcome.min_pq_j, float(tau))
    return _decode_loop(
        drafter, verifier, prompts, prompt_indices, protocol,
        rule_fn=rule, method_name="confidence_lossy",
        extra_meta={"tau": float(tau)},
        gamma=gamma, T=T, max_new_tokens=max_new_tokens,
    )
