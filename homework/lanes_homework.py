"""Homework lanes — deterministic threshold & confidence accept rules.

Two new no-predictor decoding rules used as comparison lines next to the
existing `strict` (Leviathan) and `lossy_l` (lenience-l) baselines:

  threshold_lossy(tau)
      r_j = min(1, p_j / q_j)
      accept_j iff r_j >= tau          (deterministic, left-to-right)
      L = first-zero reducer over accept_j   (commit_strict)

  confidence_lossy(tau_conf)
      r_j = min(1, p_j / q_j)
      C_k = prod_{j<k} r_j
      L = largest k with C_k >= tau_conf      (commit_confidence)

Both lanes commit `L` draft tokens + 1 bonus/fallback token per round
(strict-style, matching `run_lossy_lane_one`) so all four methods are
directly comparable: every committed token is either drafter-proposed +
verifier-accepted or a verifier-argmax bonus/fallback.

Inline tok_succ / rnd_mean / all_pass denominators follow the lossy
convention exactly (see phase26_conf_sweep.run_lossy_lane_one): every
round counts toward n_round_total even when L=0, so empty-commit rounds
correctly depress the averages. NLL is computed offline by the same
phase26.compute_nll_and_success helper used by all baselines.
"""

from __future__ import annotations

import time
from typing import Dict, List

import torch

from specdiff.commit import commit_confidence, commit_strict
from specdiff.draft_verify import _make_generator
from specdiff.protocol import derive_seed

EPS = 1e-10


@torch.no_grad()
def run_threshold_lossy_lane_one(
    drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Threshold lossy lane: accept while r_j = min(1, p_j/q_j) >= tau."""
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    for i, (prefix_ids, _text) in enumerate(prompts):
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

            draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
            draft_tokens, draft_log_probs = drafter.draft(
                prefix_ids=running, gamma=cur_gamma, T=T,
                temperature=protocol.temperature, q_mode=protocol.q_mode,
                generator=draft_rng,
            )
            candidate = torch.cat([running, draft_tokens])
            target_log_probs = verifier.score(candidate).to(torch.float32)
            prefix_len = int(running.shape[0])
            idx = torch.arange(cur_gamma, device=device)
            q_j = draft_log_probs[
                idx, draft_tokens.long()
            ].exp().clamp(min=EPS)
            p_j = target_log_probs[
                prefix_len - 1 + idx, draft_tokens.long()
            ].exp().clamp(min=EPS)

            # Deterministic accept: r_j = min(1, p_j/q_j); accept iff r_j >= tau.
            r_j = torch.minimum(torch.ones_like(q_j), p_j / q_j)
            accepted_j = (r_j >= float(tau)).to(torch.int32).cpu().tolist()
            L = commit_strict(accepted_j)
            L = max(0, min(L, cur_gamma))

            # Stream-discipline draw (matches lossy lane RNG advance).
            accept_rng = _make_generator(
                device, seed ^ protocol.accept_salt,
            )
            _ = torch.rand(cur_gamma, generator=accept_rng, device=device)

            top = target_log_probs[
                prefix_len - 1 + idx, :
            ].argmax(dim=-1)
            succ_full = (top == draft_tokens.long()).to(torch.int32)

            n_succ = int(succ_full[:L].sum().item()) if L > 0 else 0
            n_round_total += 1
            n_tok_total += L
            n_tok_succ += n_succ
            if L > 0:
                sum_round_succ_rate += n_succ / L
                if n_succ == L:
                    n_round_all_pass += 1

            draft_pref = draft_tokens[:L].to(device)
            fb_pos = prefix_len - 1 + L
            bonus_tok = int(
                target_log_probs[fb_pos, :].argmax(dim=-1).item()
            )
            extra = torch.tensor(
                [bonus_tok], dtype=torch.long, device=device,
            )
            running = torch.cat([running, draft_pref, extra])
            rounds.append({
                "round_idx": round_idx,
                "L": int(L),
                "n_committed": int(L + 1),
                "round_rng_seed": int(seed),
                "gamma": int(cur_gamma),
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
        "method": "threshold_lossy",
        "tau": float(tau),
        "tok_s_mean": tok_s_mean,
        "per_prompt": per_prompt,
    }


@torch.no_grad()
def run_confidence_lossy_lane_one(
    drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Confidence lossy lane: accept while prefix product C_k of r_j stays >= tau."""
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    for i, (prefix_ids, _text) in enumerate(prompts):
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

            draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
            draft_tokens, draft_log_probs = drafter.draft(
                prefix_ids=running, gamma=cur_gamma, T=T,
                temperature=protocol.temperature, q_mode=protocol.q_mode,
                generator=draft_rng,
            )
            candidate = torch.cat([running, draft_tokens])
            target_log_probs = verifier.score(candidate).to(torch.float32)
            prefix_len = int(running.shape[0])
            idx = torch.arange(cur_gamma, device=device)
            q_j = draft_log_probs[
                idx, draft_tokens.long()
            ].exp().clamp(min=EPS)
            p_j = target_log_probs[
                prefix_len - 1 + idx, draft_tokens.long()
            ].exp().clamp(min=EPS)

            # Deterministic accept on prefix-product confidence:
            #   r_j = min(1, p_j/q_j); C_k = prod_{j<k} r_j;
            #   L = largest k with C_k >= tau.
            r_j = torch.minimum(torch.ones_like(q_j), p_j / q_j)
            r_list = [float(x) for x in r_j.cpu().tolist()]
            L = commit_confidence(r_list, float(tau))
            L = max(0, min(L, cur_gamma))

            accept_rng = _make_generator(
                device, seed ^ protocol.accept_salt,
            )
            _ = torch.rand(cur_gamma, generator=accept_rng, device=device)

            top = target_log_probs[
                prefix_len - 1 + idx, :
            ].argmax(dim=-1)
            succ_full = (top == draft_tokens.long()).to(torch.int32)

            n_succ = int(succ_full[:L].sum().item()) if L > 0 else 0
            n_round_total += 1
            n_tok_total += L
            n_tok_succ += n_succ
            if L > 0:
                sum_round_succ_rate += n_succ / L
                if n_succ == L:
                    n_round_all_pass += 1

            draft_pref = draft_tokens[:L].to(device)
            fb_pos = prefix_len - 1 + L
            bonus_tok = int(
                target_log_probs[fb_pos, :].argmax(dim=-1).item()
            )
            extra = torch.tensor(
                [bonus_tok], dtype=torch.long, device=device,
            )
            running = torch.cat([running, draft_pref, extra])
            rounds.append({
                "round_idx": round_idx,
                "L": int(L),
                "n_committed": int(L + 1),
                "round_rng_seed": int(seed),
                "gamma": int(cur_gamma),
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
        "method": "confidence_lossy",
        "tau": float(tau),
        "tok_s_mean": tok_s_mean,
        "per_prompt": per_prompt,
    }
