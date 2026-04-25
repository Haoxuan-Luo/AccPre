"""Phase 26 — lossy SD baseline + confidence commit rule sweep.

Adds two new comparison lines on top of the Phase 25 long-horizon setup:

  Line A — original lossy SD baseline (Leviathan 2023) with lenience `l`:
      a_j = min(1, p_j / (l * q_j))
      accept iff U_j < a_j   (U from matched accept_rng)
      commit L_lossy + 1       (strict-style: L accepted + 1 bonus/fallback)

  Line B — confidence commit rule on existing predictors:
      C_k = prod_{j<k} ŷ_j
      L̂   = max{k ∈ [0, γ] : C_k ≥ τ_conf}
      commit max(1, L̂)   (no bonus — matches predictor-lane convention)

Also re-runs the pure-threshold rule at τ ∈ {0.5, 0.9} on the same four
predictors in the same run, so all Group 2 / Group 3 lanes share an
identical prompt pool and setup and can be compared row-by-row.

Setup (same as Phase 25):
  * canonical test split: indices `TRAIN_N + VAL_N .. TRAIN_N+VAL_N+n_prompts`.
  * γ=8, T=2, temperature=0 via `configs/protocol_greedy.yaml`.

Metric semantics (temp=0):
  * Per-position success indicator: `argmax(p) == draft_tok`
    (Leviathan greedy accept rule).
  * tok_succ, rnd_mean, all_pass are accumulated INLINE during decode,
    so no offline replay is needed (cf. Phase 25 oracle_exec lane).
  * NLL is computed in a separate offline pass (same helper as Phase 25).

Outputs (written under `--out_dir`):
  online_strict.json
  online_lossy_l_<l>.json           for each l ∈ lossy_ls
  online_<method>_thresh_tau_<τ>.json     for each (method, τ) in threshold sweep
  online_<method>_conf_tau_<τ>.json       for each (method, τ_conf) in conf sweep
  unified_table.{md,json}
"""

from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.commit import (
    commit_confidence, commit_strict, commit_threshold,
)
from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig, derive_seed
from accpre.data.prompts import load_owt_prompts
from accpre.data.splits import POOL_SIZE, PREFIX_LEN, PROMPT_SEED, TRAIN_N, VAL_N

# Reuse phase25 helpers for strict-lane decode + NLL scoring.
_p25_spec = _iu.spec_from_file_location(
    "p25", str(_REPO_ROOT / "scripts/phase25_longhorizon.py"),
)
p25 = _iu.module_from_spec(_p25_spec)
_p25_spec.loader.exec_module(p25)


EPS = 1e-10


# Four predictor method families to sweep under both commit rules.
PREDICTORS: List[Tuple[str, str, Optional[str]]] = [
    ("frozen_1A",     "checkpoints/acc_1a",                  None),
    ("live_jnt_10ep", "checkpoints/acc_jnt_1a_liveq2_long",
                      "checkpoints/acc_jnt_1a_liveq2_long/drafter.pt"),
    ("frozen_dep",    "checkpoints/dep_frz_1a",              None),
    ("live_jnt_dep",  "checkpoints/dep_jnt_live",
                      "checkpoints/dep_jnt_live/drafter.pt"),
]


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _fmt_l(l: float) -> str:
    return f"{int(round(l * 10))}".zfill(2).replace(" ", "")


def fmt_l_tag(l: float) -> str:
    """1.0 -> '1p0', 0.7 -> '0p7' ..."""
    whole = int(l)
    frac = int(round((l - whole) * 10))
    return f"{whole}p{frac}"


# -----------------------------------------------------------------
# Lane A — lossy baseline (strict-style loop with lossy accept rule)
# -----------------------------------------------------------------


@torch.no_grad()
def run_lossy_lane_one(
    drafter, verifier, prompts, prompt_indices, protocol,
    l: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Original lossy SD baseline with lenience factor `l`.

    Follows the strict-style loop (commit = L_lossy + 1, reserving 1 ctx
    slot for the bonus/fallback) but swaps the per-position accept rule
    to the lenient Bernoulli test
        accept_j  iff  U_j < min(1, p_j / (l * q_j))
    where U_j is drawn from the shared accept_rng (matched across lanes
    under the same round_rng_seed).

    Inline tok_succ / rnd_mean / all_pass are computed over the L_lossy
    committed-prefix positions only (bonus excluded), mirroring the
    Phase 25 oracle_exec convention.
    """
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
            # Strict-style: reserve one slot for the bonus so a full-accept
            # round can still append its bonus without overflowing MAX_CTX.
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

            # Lossy acceptance probability: min(1, p / (l * q)).
            a_lossy = torch.minimum(
                torch.ones_like(q_j),
                p_j / (float(l) * q_j),
            )
            accept_rng = _make_generator(
                device, seed ^ protocol.accept_salt,
            )
            U = torch.rand(cur_gamma, generator=accept_rng, device=device)
            accepted_j = (U < a_lossy).to(torch.int32).cpu().tolist()
            L_lossy = commit_strict(accepted_j)     # first-zero reducer
            L_lossy = max(0, min(L_lossy, cur_gamma))

            # Verifier argmax for inline success indicator (temp=0 rule).
            top = target_log_probs[
                prefix_len - 1 + idx, :
            ].argmax(dim=-1)
            succ_full = (top == draft_tokens.long()).to(torch.int32)

            # Per-round success bookkeeping over the committed prefix
            # (length L_lossy; bonus/fallback excluded because it is a
            # verifier argmax by construction).
            #
            # Denominator discipline (matches predictor-lane convention):
            # every round contributes to n_round_total. L_lossy=0 rounds
            # contribute 0 success to rnd_mean and cannot be all_pass, so
            # they still count but depress the averages. Without this,
            # lossy rnd_mean/all_pass would be biased upward vs predictor
            # lanes (which always commit ≥ 1 token so every round counts).
            n_succ = (
                int(succ_full[:L_lossy].sum().item()) if L_lossy > 0 else 0
            )
            n_round_total += 1
            n_tok_total += L_lossy            # 0 when L_lossy == 0
            n_tok_succ += n_succ              # 0 when L_lossy == 0
            if L_lossy > 0:
                sum_round_succ_rate += n_succ / L_lossy
                if n_succ == L_lossy:
                    n_round_all_pass += 1
            # L_lossy=0 rounds contribute 0 to sum_round_succ_rate and
            # never satisfy the all_pass condition.

            # Commit: L_lossy draft tokens + 1 bonus/fallback.
            # Under temp=0, bonus = argmax(target[last accepted+1]) and
            # fallback = argmax(target[L_lossy]) — both are just the
            # verifier's argmax at the appropriate position.
            draft_pref = draft_tokens[:L_lossy].to(device)
            fb_pos = prefix_len - 1 + L_lossy       # position for fallback or bonus
            if L_lossy == cur_gamma:
                # Bonus: argmax at position after the last accepted draft.
                bonus_tok = int(
                    target_log_probs[fb_pos, :].argmax(dim=-1).item()
                )
            else:
                # Fallback: argmax at the first rejected position.
                # Under temp=0, Leviathan's max(0, p - q)/norm reduces to
                # argmax(p) (sample_fallback_or_bonus in draft_verify.py).
                bonus_tok = int(
                    target_log_probs[fb_pos, :].argmax(dim=-1).item()
                )
            extra = torch.tensor(
                [bonus_tok], dtype=torch.long, device=device,
            )
            running = torch.cat([running, draft_pref, extra])
            rounds.append({
                "round_idx": round_idx,
                "L_lossy": int(L_lossy),
                "n_committed": int(L_lossy + 1),
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
        "method": "lossy_l",
        "l": float(l),
        "tok_s_mean": tok_s_mean,
        "per_prompt": per_prompt,
    }


# -----------------------------------------------------------------
# Lane B — predictor lane with pluggable commit rule
# -----------------------------------------------------------------


@torch.no_grad()
def run_predictor_lane_generic(
    predictor, drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, rule: str,
    gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Predictor lane supporting two commit rules.

    `rule` ∈ {"threshold", "confidence"}:
      - threshold:  L̂ = commit_threshold(ŷ, tau)
      - confidence: L̂ = commit_confidence(ŷ, tau)

    Semantics otherwise match `accpre.eval.online_decode._run_single_prompt`:
    commit `max(1, L̂)` draft tokens, no bonus. Inline per-position success
    (argmax(p) == draft) is accumulated per prompt.

    Calls the existing `build_online_features` / `predict_q2` interfaces,
    so no change to predictor classes is required.
    """
    if rule not in ("threshold", "confidence"):
        raise ValueError(f"rule must be 'threshold' or 'confidence', got {rule!r}")
    commit_fn = commit_threshold if rule == "threshold" else commit_confidence

    from accpre.eval.online_decode import build_online_features

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
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(gamma, remaining, ctx_room)
            if cur_gamma <= 0:
                break
            seed = derive_seed(protocol, p_idx, round_idx)

            # Mode-dependent feature extraction, mirroring _run_single_prompt.
            _pool = (
                "per_position"
                if str(predictor.family).startswith("hidden_per_pos")
                else "mean"
            )
            _layers = (
                (-2, -1) if predictor.family == "hidden_per_pos_ml" else (-1,)
            )
            draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
            draft_tokens, draft_log_probs, drafter_hidden = \
                drafter.draft_with_features(
                    prefix_ids=running, gamma=cur_gamma, T=T,
                    temperature=protocol.temperature, q_mode=protocol.q_mode,
                    generator=draft_rng, pool=_pool, layers=_layers,
                )

            verifier_prefix_hidden = None
            numeric = None
            if predictor.deploy_mode == "V-prefix":
                verifier_prefix_hidden, numeric = \
                    verifier.prefix_features(running)

            if predictor.family in (
                "hidden_per_pos_hs", "hidden_per_pos_s",
                "hidden_per_pos_v2", "hidden_per_pos_v4",
            ):
                probs_ = draft_log_probs.exp()
                entropy_t = -(probs_ * draft_log_probs).sum(dim=-1)
                top2 = draft_log_probs.topk(2, dim=-1).values
                margin_t = top2[:, 0] - top2[:, 1]
                top1_t = probs_.max(dim=-1).values
                idx_ = torch.arange(
                    int(draft_tokens.shape[0]), device=draft_tokens.device,
                )
                q_t = probs_[idx_, draft_tokens.long()].clamp(min=1e-10)
                if predictor.family in (
                    "hidden_per_pos_v2", "hidden_per_pos_v4"
                ):
                    log_q_t = torch.log(q_t)
                    scalars = torch.stack(
                        [q_t, log_q_t, entropy_t, margin_t, top1_t], dim=1,
                    ).to(torch.float32).cpu()
                else:
                    scalars = torch.stack(
                        [q_t, entropy_t, margin_t, top1_t], dim=1,
                    ).to(torch.float32).cpu()
                if predictor.family == "hidden_per_pos_s":
                    features = scalars
                elif predictor.family == "hidden_per_pos_hs":
                    h = drafter_hidden.to(torch.float32).cpu()
                    features = torch.cat([h, scalars], dim=1)
                elif predictor.family == "hidden_per_pos_v2":
                    h = drafter_hidden.to(torch.float32).cpu()
                    features = torch.cat([h, scalars], dim=1)
                else:   # v4
                    h = drafter_hidden.to(torch.float32).cpu()
                    vh = verifier_prefix_hidden.to(torch.float32).cpu()
                    g_local = int(h.shape[0])
                    vh_b = vh.unsqueeze(0).expand(
                        g_local, -1,
                    ).contiguous()
                    features = torch.cat([h, scalars, vh_b], dim=1)
            else:
                features = build_online_features(
                    family=predictor.family,
                    drafter_hidden=drafter_hidden,
                    verifier_prefix_hidden=verifier_prefix_hidden,
                    numeric=numeric,
                )

            predict_kwargs = {}
            base_pred = getattr(predictor, "base", predictor)
            if hasattr(base_pred, "token_emb"):
                predict_kwargs["token_ids"] = \
                    draft_tokens.detach().cpu().long()
            q_hat = predictor.predict_q2(features, **predict_kwargs)
            if q_hat.dim() == 2:
                q_hat = q_hat.squeeze(0)
            q_list = [float(x) for x in q_hat.detach().cpu().tolist()]
            L_hat = int(commit_fn(q_list, float(tau)))
            L_hat = max(0, min(L_hat, cur_gamma))
            n_commit = max(1, L_hat)
            committed = draft_tokens[:n_commit]

            # Draw U_j for accept_rng stream discipline (unused here at
            # temp=0; predictor lanes decide commit from ŷ, not Q2).
            accept_rng = _make_generator(
                device, seed ^ protocol.accept_salt,
            )
            _ = torch.rand(cur_gamma, generator=accept_rng, device=device)

            prefix_len_at_start = int(running.shape[0])
            rounds.append({
                "round_idx": round_idx,
                "prefix_len_at_round_start": prefix_len_at_start,
                "L_hat": int(L_hat),
                "n_committed": int(n_commit),
                "round_rng_seed": int(seed),
                "gamma": int(cur_gamma),
                "draft_tokens": [int(x) for x in draft_tokens.tolist()],
            })
            running = torch.cat([running, committed])
            round_idx += 1
        elapsed = time.time() - t_start
        n_new = int(running.shape[0]) - prefix_start
        # No inline verifier forward in the decode loop — tok_succ / rnd_mean
        # / all_pass are reconstructed offline in `compute_offline_success`
        # from stored `(prefix_len_at_round_start, draft_tokens, n_committed)`
        # against the single `verifier.score(generated_ids)` forward that
        # NLL already needs.
        per_prompt.append({
            "prompt_idx": p_idx,
            "n_new_tokens": int(n_new),
            "elapsed_s": float(elapsed),
            "tok_s": float(n_new) / max(elapsed, 1e-9),
            "generated_ids": [int(x) for x in running.tolist()],
            "rounds": rounds,
        })
    tok_s_mean = sum(p["tok_s"] for p in per_prompt) / max(len(per_prompt), 1)
    return {
        "method": "predictor",
        "rule": rule,
        "tau": float(tau),
        "tok_s_mean": tok_s_mean,
        "per_prompt": per_prompt,
    }


# -----------------------------------------------------------------
# Offline combined NLL + verifier-success pass
# -----------------------------------------------------------------


@torch.no_grad()
def compute_nll_and_success(verifier, lane: Dict, device) -> Tuple[float, Dict]:
    """Single offline pass that emits both the lane's NLL and its
    per-prompt tok_succ / rnd_mean / all_pass.

    Runs `verifier.score(generated_ids)` **once per prompt** and reuses the
    result for both metrics, avoiding the per-round verifier forward that
    inline computation would require. For predictor lanes, each round's
    committed tokens are a prefix of the drafter's draft_tokens, so
    `argmax(target_log_probs[prefix_len-1+j])` is identical to what the
    original decode-time (prefix+draft) forward would have produced at
    that position (causal attention + matching context).

    If the lane already has `tok_succ` / `rnd_mean` / `all_pass` populated
    inline on every per-prompt entry (e.g., the lossy lane or a legacy
    predictor lane JSON), those values are KEPT; only NLL is computed.

    Returns (NLL, success_summary_dict).
    """
    EPS_LOCAL = 1e-10  # unused; kept to match phase25 conventions
    total_sum = 0.0
    total_n = 0
    have_inline = all(
        ("tok_succ" in p and "rnd_mean" in p and "all_pass" in p)
        for p in lane["per_prompt"]
    )

    summary = {
        "token_success_rate":  None,
        "round_mean_success":  None,
        "round_all_pass_rate": None,
        "n_tokens_total":      0,
    }

    n_tok_total = 0; n_tok_succ = 0
    n_round_total = 0; n_round_all_pass = 0
    sum_round_succ_rate = 0.0

    for p_row in lane["per_prompt"]:
        gen = [int(x) for x in p_row["generated_ids"]]
        n_prefix = len(gen) - int(p_row["n_new_tokens"])

        # Single verifier forward on the full trajectory — used for BOTH
        # the NLL numerator and the per-round argmax success lookup.
        x = torch.tensor(gen, dtype=torch.long, device=device)
        log_probs = verifier.score(x).to(torch.float32)

        # --- NLL over the generated range ---
        if len(gen) > n_prefix:
            n_new = len(gen) - n_prefix
            idx_logits = torch.arange(
                n_prefix - 1, n_prefix + n_new - 1, device=device,
            )
            idx_tokens = torch.tensor(
                gen[n_prefix: n_prefix + n_new],
                dtype=torch.long, device=device,
            )
            selected = log_probs[idx_logits, idx_tokens]
            total_sum += float(-selected.sum().item())
            total_n += n_new

        # --- per-round success (only if not already populated inline) ---
        if have_inline:
            n_tok_total += int(p_row.get("n_tokens_total", 0))
            continue

        rounds = p_row.get("rounds", [])
        per_prompt_tok_total = 0
        per_prompt_tok_succ = 0
        per_prompt_rnd_total = 0
        per_prompt_rnd_pass = 0
        per_prompt_sum_rate = 0.0
        for r in rounds:
            if "draft_tokens" not in r or \
               "prefix_len_at_round_start" not in r:
                # Legacy rounds without stored draft_tokens: can't recover.
                continue
            prefix_len = int(r["prefix_len_at_round_start"])
            draft_tokens_r = [int(x) for x in r["draft_tokens"]]
            cur_gamma = int(r["gamma"])
            n_commit = int(r["n_committed"])
            if n_commit == 0:
                continue
            # Positions [prefix_len-1 .. prefix_len-1+cur_gamma-1] predict
            # tokens at [prefix_len .. prefix_len+cur_gamma-1]. The first
            # n_commit of those trajectory slots equal draft_tokens[:n_commit]
            # by construction (predictor lanes commit max(1, L̂) drafts).
            idx_slice = torch.arange(
                prefix_len - 1, prefix_len - 1 + cur_gamma, device=device,
            )
            # Guard: in strict/lossy lanes with a bonus or fallback, the
            # `draft_tokens` entry may still line up (bonus is at
            # prefix_len+L_lossy). Here we only use the first n_commit
            # positions, which are always draft positions.
            top = log_probs[idx_slice, :].argmax(dim=-1)
            draft_t = torch.tensor(
                draft_tokens_r, dtype=torch.long, device=device,
            )
            match = (top == draft_t).to(torch.int32)
            n_succ = int(match[:n_commit].sum().item())
            per_prompt_tok_total += n_commit
            per_prompt_tok_succ += n_succ
            per_prompt_rnd_total += 1
            if n_commit > 0:
                per_prompt_sum_rate += n_succ / n_commit
                if n_succ == n_commit:
                    per_prompt_rnd_pass += 1

        if per_prompt_tok_total > 0:
            p_row["n_tokens_total"] = int(per_prompt_tok_total)
            p_row["tok_succ"] = (
                per_prompt_tok_succ / per_prompt_tok_total
            )
            p_row["rnd_mean"] = (
                per_prompt_sum_rate / max(per_prompt_rnd_total, 1)
            )
            p_row["all_pass"] = (
                per_prompt_rnd_pass / max(per_prompt_rnd_total, 1)
            )
            n_tok_total += per_prompt_tok_total
            n_tok_succ += per_prompt_tok_succ
            n_round_total += per_prompt_rnd_total
            n_round_all_pass += per_prompt_rnd_pass
            sum_round_succ_rate += per_prompt_sum_rate
        # else: no rounds had `draft_tokens` (e.g., strict lane) — leave
        # success fields unset, which renders as "—" in the unified table.

    if not have_inline and n_tok_total > 0:
        summary["token_success_rate"] = n_tok_succ / max(n_tok_total, 1)
        summary["round_mean_success"] = (
            sum_round_succ_rate / max(n_round_total, 1)
        )
        summary["round_all_pass_rate"] = (
            n_round_all_pass / max(n_round_total, 1)
        )
        summary["n_tokens_total"] = int(n_tok_total)

    nll = total_sum / max(total_n, 1)
    return nll, summary


# -----------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------


def _load_eval_prompts(n_prompts: int) -> Tuple[List, List[int]]:
    pool = load_owt_prompts(
        n_prompts=POOL_SIZE, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )
    eval_offset = TRAIN_N + VAL_N
    prompts = pool[eval_offset: eval_offset + n_prompts]
    indices = list(range(eval_offset, eval_offset + n_prompts))
    return prompts, indices


def _rowkey(method: str, tag: str) -> str:
    return f"{method}__{tag}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--protocol_yaml", type=str,
                    default="configs/protocol_greedy.yaml")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument(
        "--lossy_ls", type=float, nargs="+",
        default=[1.0, 0.7, 0.5, 0.3],
        help="Lenience factors for the lossy SD baseline (Line A).",
    )
    ap.add_argument(
        "--taus", type=float, nargs="+", default=[0.5, 0.9],
        help="Thresholds for the threshold commit rule (Group 2).",
    )
    ap.add_argument(
        "--tau_confs", type=float, nargs="+", default=[0.5, 0.9],
        help="Thresholds for the confidence commit rule (Group 3).",
    )
    ap.add_argument(
        "--methods", type=str, nargs="+",
        default=[m[0] for m in PREDICTORS],
        help="Which predictor families to sweep under both rules.",
    )
    ap.add_argument("--skip_strict", action="store_true",
                    help="Skip the strict lane (useful for resuming).")
    ap.add_argument("--skip_lossy", action="store_true",
                    help="Skip the lossy lanes.")
    ap.add_argument("--skip_threshold", action="store_true",
                    help="Skip the predictor-threshold lanes (Group 2).")
    ap.add_argument("--skip_confidence", action="store_true",
                    help="Skip the predictor-confidence lanes (Group 3).")
    ap.add_argument(
        "--skip_existing", action="store_true",
        help=(
            "If set, load an existing online_*.json from --out_dir "
            "instead of re-decoding its lane. Useful when resuming a "
            "partially-finished sweep. NLL+success pass is still run "
            "on loaded lanes."
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(_REPO_ROOT / args.protocol_yaml)
    assert float(protocol.temperature) == 0.0, (
        f"Phase 26 expects temp=0; got {protocol.temperature}. "
        f"Check --protocol_yaml."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    from accpre.eval.online_decode import _load_predictor_from_checkpoint
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    print("[p26] loading drafter+verifier...", flush=True)
    t_setup = time.time()
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    pretrained_cpu_state = p25.stash_state(drafter)
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )
    print(f"[p26] models loaded in {time.time() - t_setup:.1f}s", flush=True)

    prompts, prompt_indices = _load_eval_prompts(int(args.n_prompts))
    print(
        f"[p26] {len(prompts)} prompts, indices "
        f"{prompt_indices[0]}..{prompt_indices[-1]}  "
        f"(max_new_tokens={args.max_new_tokens}, temp={protocol.temperature})",
        flush=True,
    )

    all_lanes: Dict[str, Dict] = {}
    timings: Dict[str, Dict] = {"decode": {}, "offline": {}}

    def _load_or_run(
        lane_key: str, filename: str, run_fn,
    ) -> Optional[Dict]:
        """Reuse an existing online_*.json if --skip_existing, else decode.

        Returns the lane dict, or None if skipped and file missing.
        Accumulates decode wallclock into timings["decode"][lane_key].
        """
        fp = out_dir / filename
        if args.skip_existing and fp.exists():
            print(
                f"[p26]   loading existing {filename} (skip_existing)",
                flush=True,
            )
            with open(fp) as fh:
                return json.load(fh)
        t0 = time.time()
        lane = run_fn()
        timings["decode"][lane_key] = time.time() - t0
        with open(fp, "w") as fh:
            json.dump(lane, fh, indent=2, default=str)
        print(
            f"[p26]   wrote {filename}  "
            f"(elapsed={timings['decode'][lane_key]:.1f}s)",
            flush=True,
        )
        return lane

    # ---------------- Lane 0: strict ----------------
    if not args.skip_strict:
        print("\n[p26] === STRICT ===", flush=True)
        lane = _load_or_run(
            "strict", "online_strict.json",
            lambda: p25.run_strict_lane(
                drafter, verifier, prompts, prompt_indices, protocol,
                gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        if lane is not None:
            print(
                f"[p26] strict: tok/s={lane['tok_s_mean']:.2f}",
                flush=True,
            )
            all_lanes["strict"] = lane

    # ---------------- Lane A: lossy baseline ----------------
    if not args.skip_lossy:
        for l_val in args.lossy_ls:
            tag = fmt_l_tag(l_val)
            print(f"\n[p26] === LOSSY l={l_val} ===", flush=True)
            lane = _load_or_run(
                f"lossy|{l_val}", f"online_lossy_l_{tag}.json",
                lambda l_v=l_val: run_lossy_lane_one(
                    drafter, verifier, prompts, prompt_indices, protocol,
                    l=float(l_v), gamma=args.gamma, T=args.T,
                    max_new_tokens=args.max_new_tokens,
                ),
            )
            if lane is not None:
                tok_succ_overall = (
                    sum(p.get("tok_succ", 0) * p.get("n_tokens_total", 0)
                        for p in lane["per_prompt"])
                    / max(sum(p.get("n_tokens_total", 0)
                              for p in lane["per_prompt"]), 1)
                )
                print(
                    f"[p26] lossy l={l_val}: "
                    f"tok/s={lane['tok_s_mean']:.2f}  "
                    f"tok_succ(prefix)={tok_succ_overall:.4f}",
                    flush=True,
                )
                all_lanes[f"lossy__{tag}"] = lane

    # ---------------- Lane B: predictors × {threshold, confidence} × τ ----------------
    for name, ckpt, dstate in PREDICTORS:
        if name not in args.methods:
            continue
        ckpt_abs = _REPO_ROOT / ckpt
        if not ckpt_abs.exists():
            print(f"[p26] SKIP {name}: missing {ckpt_abs}", flush=True)
            continue
        print(f"\n[p26] === loading predictor {name} ===", flush=True)
        predictor, _cfg, _ = _load_predictor_from_checkpoint(
            str(ckpt_abs), gamma=args.gamma,
        )
        # Restore drafter to match checkpoint's decode-time drafter.
        if dstate is not None:
            print(
                f"[p26] overriding drafter state <- {dstate}",
                flush=True,
            )
            state = torch.load(
                str(_REPO_ROOT / dstate), map_location=device,
            )
            drafter.model.load_state_dict(state)
        else:
            p25.restore_state(drafter, pretrained_cpu_state, device)
        drafter.model.eval()

        if not args.skip_threshold:
            for tau in args.taus:
                tag = p25.fmt_tau_tag(float(tau))
                print(
                    f"[p26]   {name} threshold τ={tau}",
                    flush=True,
                )
                fn = f"online_{name}_thresh_tau_{tag}.json"
                lane = _load_or_run(
                    f"{name}|thresh|{tau}", fn,
                    lambda t=tau: run_predictor_lane_generic(
                        predictor, drafter, verifier, prompts,
                        prompt_indices, protocol,
                        tau=float(t), rule="threshold",
                        gamma=args.gamma, T=args.T,
                        max_new_tokens=args.max_new_tokens,
                    ),
                )
                if lane is not None:
                    print(
                        f"[p26]     tok/s={lane['tok_s_mean']:.2f}",
                        flush=True,
                    )
                    all_lanes[f"{name}__thresh__{tag}"] = lane

        if not args.skip_confidence:
            for tau in args.tau_confs:
                tag = p25.fmt_tau_tag(float(tau))
                print(
                    f"[p26]   {name} confidence τ_conf={tau}",
                    flush=True,
                )
                fn = f"online_{name}_conf_tau_{tag}.json"
                lane = _load_or_run(
                    f"{name}|conf|{tau}", fn,
                    lambda t=tau: run_predictor_lane_generic(
                        predictor, drafter, verifier, prompts,
                        prompt_indices, protocol,
                        tau=float(t), rule="confidence",
                        gamma=args.gamma, T=args.T,
                        max_new_tokens=args.max_new_tokens,
                    ),
                )
                if lane is not None:
                    print(
                        f"[p26]     tok/s={lane['tok_s_mean']:.2f}",
                        flush=True,
                    )
                    all_lanes[f"{name}__conf__{tag}"] = lane

    # ---------------- Offline combined NLL + success pass ----------------
    print("\n[p26] === offline NLL + success pass ===", flush=True)
    method_nll: Dict[str, float] = {}
    for key, lane in all_lanes.items():
        t0 = time.time()
        nll, summary = compute_nll_and_success(verifier, lane, device)
        method_nll[key] = nll
        timings["offline"][key] = time.time() - t0
        # Persist the offline-filled success metrics back to disk so
        # future re-aggregations don't re-run the forward.
        fname_map = {
            "strict": "online_strict.json",
        }
        # Infer filename from key for non-strict lanes.
        if key == "strict":
            fname = "online_strict.json"
        elif key.startswith("lossy__"):
            tag = key.split("__", 1)[1]
            fname = f"online_lossy_l_{tag}.json"
        else:
            # Predictor key pattern: <method>__<rule>__<tag>
            parts = key.split("__")
            fname = f"online_{parts[0]}_{parts[1]}_tau_{parts[2]}.json"
        with open(out_dir / fname, "w") as fh:
            json.dump(lane, fh, indent=2, default=str)
        tok_succ_str = (
            f"{summary['token_success_rate']:.4f}"
            if summary['token_success_rate'] is not None else "(inline)"
        )
        print(
            f"[p26]   {key}: NLL={nll:.4f}  "
            f"tok_succ={tok_succ_str}  "
            f"elapsed={timings['offline'][key]:.1f}s",
            flush=True,
        )

    # ---------------- unified table ----------------
    strict_nll = method_nll.get("strict")
    rows: List[Dict] = []

    def _row(key: str, family: str, rule: str, val_label: str, val) -> Dict:
        lane = all_lanes[key]
        pp = lane["per_prompt"]
        tot_n = sum(int(p.get("n_tokens_total", 0)) for p in pp)
        tok_succ = (
            sum(
                float(p.get("tok_succ", 0.0))
                * int(p.get("n_tokens_total", 0))
                for p in pp
            ) / max(tot_n, 1)
        ) if tot_n > 0 else None
        pp_with_rnd = [p for p in pp if "rnd_mean" in p]
        rnd_mean = (
            sum(p["rnd_mean"] for p in pp_with_rnd) / len(pp_with_rnd)
        ) if pp_with_rnd else None
        pp_with_ap = [p for p in pp if "all_pass" in p]
        all_pass = (
            sum(p["all_pass"] for p in pp_with_ap) / len(pp_with_ap)
        ) if pp_with_ap else None
        # Per-prompt averages for optional columns
        n_rounds_per_prompt = [len(p.get("rounds", [])) for p in pp]
        n_committed_per_round = [
            (int(p.get("n_tokens_total", 0)) / max(len(p.get("rounds", [])), 1))
            for p in pp
        ]
        nll = method_nll[key]
        return {
            "family": family,
            "rule": rule,
            "threshold_label": val_label,
            "threshold_value": val,
            "tok_s_mean": float(lane["tok_s_mean"]),
            "NLL": float(nll),
            "delta_NLL": (
                float(nll - strict_nll)
                if strict_nll is not None else None
            ),
            "tok_succ": None if tot_n == 0 else float(tok_succ),
            "rnd_mean": float(rnd_mean) if rnd_mean is not None else None,
            "all_pass": float(all_pass) if all_pass is not None else None,
            "avg_rounds":
                sum(n_rounds_per_prompt) / max(len(n_rounds_per_prompt), 1),
            "avg_commit_per_round":
                sum(n_committed_per_round) / max(len(n_committed_per_round), 1),
            "lane_key": key,
        }

    # Group 1 — baselines
    if "strict" in all_lanes:
        lane = all_lanes["strict"]
        rows.append({
            "family": "strict",
            "rule": "strict",
            "threshold_label": "—",
            "threshold_value": None,
            "tok_s_mean": float(lane["tok_s_mean"]),
            "NLL": float(method_nll["strict"]),
            "delta_NLL": 0.0,
            "tok_succ": None,
            "rnd_mean": None,
            "all_pass": None,
            "avg_rounds":
                sum(len(p["rounds"]) for p in lane["per_prompt"])
                / max(len(lane["per_prompt"]), 1),
            "avg_commit_per_round": None,
            "lane_key": "strict",
        })
    for l_val in args.lossy_ls:
        tag = fmt_l_tag(l_val)
        key = f"lossy__{tag}"
        if key in all_lanes:
            rows.append(_row(key, "lossy", "lossy-l", f"l={l_val}", float(l_val)))

    # Group 2 — predictor threshold
    for name, _c, _d in PREDICTORS:
        if name not in args.methods:
            continue
        for tau in args.taus:
            tag = p25.fmt_tau_tag(float(tau))
            key = f"{name}__thresh__{tag}"
            if key in all_lanes:
                rows.append(
                    _row(key, name, "threshold", f"τ={tau}", float(tau))
                )

    # Group 3 — predictor confidence
    for name, _c, _d in PREDICTORS:
        if name not in args.methods:
            continue
        for tau in args.tau_confs:
            tag = p25.fmt_tau_tag(float(tau))
            key = f"{name}__conf__{tag}"
            if key in all_lanes:
                rows.append(
                    _row(key, name, "confidence", f"τ_conf={tau}", float(tau))
                )

    # ---------------- markdown render ----------------
    def _fmt_opt(v, w=10, p=4):
        if v is None:
            return f"{'—':>{w}}"
        return f"{float(v):>{w}.{p}f}"

    def _fmt_signed(v, w=10, p=4):
        if v is None:
            return f"{'—':>{w}}"
        return f"{float(v):>+{w}.{p}f}"

    md: List[str] = []
    md.append(
        f"# Phase 26 — lossy SD baseline + confidence commit rule "
        f"({len(prompts)} prompts, max_new_tokens={args.max_new_tokens}, "
        f"γ={args.gamma}, T={args.T}, temperature=0)\n\n"
    )
    md.append(
        "tok/s measured inside the decode loop. NLL, ΔNLL, tok_succ, "
        "rnd_mean, all_pass are computed on the stored decode JSONs — "
        "tok_succ/rnd_mean/all_pass inline during decode (temp=0: "
        "argmax(p)==draft), NLL in an offline pass. Lossy-l commits "
        "L_lossy + 1 per round (bonus/fallback included, matches the "
        "original SD paper). Predictor threshold and confidence lanes "
        "commit max(1, L̂) per round (no bonus).\n\n"
    )
    md.append("```\n")
    header = (
        f"  {'family':<14}{'rule':<12}{'thresh':>10}"
        f"{'tok/s':>8}{'NLL':>9}{'ΔNLL':>10}"
        f"{'tok_succ':>10}{'rnd_mean':>10}{'all_pass':>10}\n"
    )
    md.append(header)
    for r in rows:
        md.append(
            f"  {r['family']:<14}{r['rule']:<12}"
            f"{r['threshold_label']:>10}"
            f"{float(r['tok_s_mean']):>8.2f}"
            f"{_fmt_opt(r['NLL'], 9, 4)}"
            f"{_fmt_signed(r['delta_NLL'], 10, 4)}"
            f"{_fmt_opt(r['tok_succ'], 10, 4)}"
            f"{_fmt_opt(r['rnd_mean'], 10, 4)}"
            f"{_fmt_opt(r['all_pass'], 10, 4)}\n"
        )
    md.append("```\n\n")
    md.append(
        f"Prompt indices: {prompt_indices[0]}..{prompt_indices[-1]} "
        f"({len(prompts)} prompts)\n"
    )
    md.append(
        f"Decode wallclock (sum): "
        f"{sum(timings['decode'].values()):.1f}s\n"
    )
    md.append(
        f"Offline NLL+success wallclock (sum): "
        f"{sum(timings['offline'].values()):.1f}s\n"
    )

    md_text = "".join(md)
    print("\n" + md_text, flush=True)

    with open(out_dir / "unified_table.md", "w") as f:
        f.write(md_text)
    with open(out_dir / "unified_table.json", "w") as f:
        json.dump({
            "rows": rows,
            "n_prompts": len(prompts),
            "prompt_indices": [int(i) for i in prompt_indices],
            "max_new_tokens": int(args.max_new_tokens),
            "gamma": int(args.gamma),
            "T": int(args.T),
            "temperature": float(protocol.temperature),
            "timings_s": timings,
            "strict_nll": (
                float(strict_nll) if strict_nll is not None else None
            ),
        }, f, indent=2, default=str)

    print(f"[p26] wrote {out_dir}/unified_table.md", flush=True)
    print(f"[p26] wrote {out_dir}/unified_table.json", flush=True)
    print("[p26] DONE.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
