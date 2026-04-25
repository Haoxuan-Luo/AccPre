"""Phase 24 — clean 50-prompt method-selection comparison.

Goal: run ONE unified pass that gives us enough signal to drop methods
and decide which 2-4 non-baseline methods survive for the paper / mainline.

Prompt set (deterministic, reproducible):
    load_owt_prompts(n_prompts=110, prefix_len=32, seed=PROMPT_SEED=42)[60:110]
  → 50 prompts. Indices 60..79 are the canonical test split (overlap with
    Phase 22 / Phase 23 results — gives continuity); indices 80..109 are
    30 fresh prompts pulled from OWT under the same deterministic seed.
    No prompt in the 50 was seen during training (training used 0..39).

Online setting (matches Phase 22/23):
    max_new_tokens = 64,  γ = 8,  T = 2,
    drafter = MDLM-OWT, verifier = GPT-2-XL,
    same protocol.yaml, same matched-randomness discipline.

Methods:
    --- Baselines ---
    strict
    oracle_exec       τ ∈ {0.5, 0.7, 0.9}

    --- Main predictors (for selection) ---
    frozen_1A         τ ∈ {0.5, 0.7, 0.9}
    live_jnt_5ep      τ ∈ {0.5, 0.7, 0.9}
    live_jnt_10ep     τ ∈ {0.5, 0.7, 0.9}
    frozen_dep        τ = 0.9   (Phase 23 best operating point)

    --- Reference / appendix (off the main table) ---
    lazy_jnt          τ ∈ {0.5, 0.7, 0.9}
    live_jnt_dep      τ ∈ {0.5, 0.7, 0.9}

Predictor MAE (frozen heads only — depends on preds_test.pt, which is
written on the canonical 20-prompt test split, not on the 50-prompt eval
set):
    MAE_overall  = mean |Q̂ − target| weighted by `survived`
    MAE_acc      = mean |Q̂ − target| restricted to positions where strict
                   ACCEPTED (j < record.L)
    MAE_wgt      = survived-masked MAE with prefix weight w_j = 1/(1+j)
For dep-target methods, "target" is s_j; for Q2-target methods, "target"
is Q2_j. The MAE numbers are head-quality metrics on the head's own
training target; do NOT compare across different targets numerically.

Output (under `--out_dir`):
    online_strict.json
    online_<method>_tau_<*>.json
    online_oracle_tau_<*>.json
    unified_table.{md,json}     - Table 1 (main, no appendix)
    appendix_table.{md,json}    - lazy_jnt + live_jnt_dep (reference only)
    summary_table.{md,json}     - Table 2 (one row per method, best op)
    pareto_delta_nll_tok_s.png
    pareto_tok_succ_tok_s.png
    pareto_mae_acc_tok_s.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.commit import commit_threshold
from accpre.core.draft_verify import (
    _make_generator, draft_verify_round,
)
from accpre.core.protocol import ProtocolConfig, derive_seed
from accpre.data.prompts import load_owt_prompts
from accpre.data.splits import POOL_SIZE, PREFIX_LEN, PROMPT_SEED, TRAIN_N, VAL_N


EPS = 1e-10


# ---------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------


# Each entry: (name, ckpt_dir, drafter_state_path | None, τ tuple, tier)
#   tier="main"     -> appears in Table 1 + summary + clean Pareto
#   tier="appendix" -> separate appendix table only
METHODS: List[Tuple[str, str, Optional[str], Tuple[float, ...], str]] = [
    ("frozen_1A",     "checkpoints/acc_1a",                 None,
                      (0.5, 0.7, 0.9), "main"),
    ("live_jnt_5ep",  "checkpoints/acc_jnt_1a_liveq2",
                      "checkpoints/acc_jnt_1a_liveq2/drafter.pt",
                      (0.5, 0.7, 0.9), "main"),
    ("live_jnt_10ep", "checkpoints/acc_jnt_1a_liveq2_long",
                      "checkpoints/acc_jnt_1a_liveq2_long/drafter.pt",
                      (0.5, 0.7, 0.9), "main"),
    ("frozen_dep",    "checkpoints/dep_frz_1a",             None,
                      (0.9,), "main"),
    # --- appendix / reference ---
    ("lazy_jnt",      "checkpoints/acc_jnt_1a",
                      "checkpoints/acc_jnt_1a/drafter.pt",
                      (0.5, 0.7, 0.9), "appendix"),
    ("live_jnt_dep",  "checkpoints/dep_jnt_live",
                      "checkpoints/dep_jnt_live/drafter.pt",
                      (0.5, 0.7, 0.9), "appendix"),
]

ORACLE_TAUS: Tuple[float, ...] = (0.5, 0.7, 0.9)

COLORS: Dict[str, str] = {
    "strict":        "#000000",
    "oracle_exec":   "#f58518",
    "frozen_1A":     "#3b76b3",
    "live_jnt_5ep":  "#54a24b",
    "live_jnt_10ep": "#d22b2b",
    "frozen_dep":    "#9467bd",
    "lazy_jnt":      "#a9a9a9",
    "live_jnt_dep":  "#e377c2",
}


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def stash_state(drafter):
    return {k: v.detach().cpu().clone()
            for k, v in drafter.model.state_dict().items()}


def restore_state(drafter, cpu_state, device):
    drafter.model.load_state_dict(
        {k: v.to(device) for k, v in cpu_state.items()}
    )
    drafter.model.eval()


def fmt_tau_tag(tau: float) -> str:
    return f"0p{int(round(tau * 10))}"


# ---------------------------------------------------------------
# Online lanes — copied from phase23 driver, identical semantics.
# Timing scope: drafter forward + (optional) verifier prefix +
# predictor forward + commit + cat. NO offline NLL or replay inside.
# ---------------------------------------------------------------


@torch.no_grad()
def run_strict_lane(
    drafter, verifier, prompts, prompt_indices, protocol,
    gamma: int = 8, T: int = 2, max_new_tokens: int = 64,
) -> Dict:
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    for i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[i])
        running = prefix_ids.to(device).clone()
        prefix_start = int(running.shape[0])
        rounds: List[Dict] = []
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
            rec = draft_verify_round(
                prefix_ids=running, drafter=drafter, verifier=verifier,
                gamma=cur_gamma, T=T, protocol=protocol,
                prompt_idx=p_idx, round_idx=round_idx, round_rng_seed=seed,
            )
            L = int(rec.L)
            draft_pref = torch.tensor(
                rec.draft_tokens[:L], dtype=torch.long, device=device,
            )
            extra = torch.tensor(
                [rec.bonus_or_fallback_token], dtype=torch.long, device=device,
            )
            running = torch.cat([running, draft_pref, extra])
            rounds.append({
                "round_idx": round_idx, "L": L, "n_committed": L + 1,
                "round_rng_seed": int(seed), "gamma": int(cur_gamma),
            })
            round_idx += 1
        elapsed = time.time() - t_start
        n_new = int(running.shape[0]) - prefix_start
        per_prompt.append({
            "prompt_idx": p_idx, "n_new_tokens": int(n_new),
            "elapsed_s": float(elapsed),
            "tok_s": float(n_new) / max(elapsed, 1e-9),
            "generated_ids": [int(x) for x in running.tolist()],
            "rounds": rounds,
        })
    tok_s_mean = sum(p["tok_s"] for p in per_prompt) / max(len(per_prompt), 1)
    return {"method": "strict", "tok_s_mean": tok_s_mean,
            "per_prompt": per_prompt}


@torch.no_grad()
def run_predictor_lane_one(
    predictor, drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 64,
) -> Dict:
    from accpre.eval.online_decode import run_predictor_lane
    lane = run_predictor_lane(
        predictor=predictor, drafter=drafter, verifier=verifier,
        prompts=prompts, protocol=protocol, gamma=gamma, T=T,
        max_new_tokens=max_new_tokens, tau=tau,
        prompt_indices=prompt_indices,
    )
    return lane.to_dict()


@torch.no_grad()
def run_oracle_exec_lane(
    drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 64,
) -> Dict:
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    for i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[i])
        running = prefix_ids.to(device).clone()
        prefix_start = int(running.shape[0])
        rounds: List[Dict] = []
        n_tok_total = 0
        n_tok_succ = 0
        n_round_total = 0
        n_round_all_pass = 0
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
            Q2 = torch.minimum(torch.ones_like(q_j), p_j / q_j)
            q2_list = [float(x) for x in Q2.cpu().tolist()]
            L_hat = commit_threshold(q2_list, float(tau))
            L_hat = max(0, min(L_hat, cur_gamma))
            n_commit = max(1, L_hat)

            accept_rng = _make_generator(
                device, seed ^ protocol.accept_salt,
            )
            U = torch.rand(cur_gamma, generator=accept_rng, device=device)
            successes = (U[:n_commit] < Q2[:n_commit]).cpu().int().tolist()
            n_succ = sum(successes)

            n_round_total += 1
            n_tok_total += n_commit
            n_tok_succ += n_succ
            if n_commit > 0:
                sum_round_succ_rate += n_succ / n_commit
                if n_succ == n_commit:
                    n_round_all_pass += 1

            running = torch.cat([running, draft_tokens[:n_commit]])
            rounds.append({
                "round_idx": round_idx,
                "L_hat": int(L_hat), "n_commit": int(n_commit),
                "round_rng_seed": int(seed), "gamma": int(cur_gamma),
            })
            round_idx += 1
        elapsed = time.time() - t_start
        n_new = int(running.shape[0]) - prefix_start
        per_prompt.append({
            "prompt_idx": p_idx, "n_new_tokens": int(n_new),
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
        "method": "oracle_exec", "tau": float(tau),
        "tok_s_mean": tok_s_mean, "per_prompt": per_prompt,
    }


@torch.no_grad()
def score_nll_on_new(
    verifier, generated_ids: List[int], n_prefix: int, device,
) -> Tuple[float, int]:
    if len(generated_ids) <= n_prefix:
        return 0.0, 0
    x = torch.tensor(generated_ids, dtype=torch.long, device=device)
    log_probs = verifier.score(x).to(torch.float32)
    n_new = len(generated_ids) - n_prefix
    idx_logits = torch.arange(
        n_prefix - 1, n_prefix + n_new - 1, device=device,
    )
    idx_tokens = torch.tensor(
        generated_ids[n_prefix: n_prefix + n_new],
        dtype=torch.long, device=device,
    )
    selected = log_probs[idx_logits, idx_tokens]
    return float(-selected.sum().item()), int(n_new)


def lane_nll(verifier, lane_per_prompt, device) -> float:
    total_sum = 0.0
    total_n = 0
    for p in lane_per_prompt:
        gen = [int(x) for x in p["generated_ids"]]
        n_prefix = len(gen) - int(p["n_new_tokens"])
        s, n = score_nll_on_new(verifier, gen, n_prefix, device)
        total_sum += s
        total_n += n
    return total_sum / max(total_n, 1)


@torch.no_grad()
def replay_verifier_success(
    drafter, verifier, lane_per_prompt, protocol,
    pool_by_idx: Dict[int, torch.Tensor], device,
) -> Dict:
    n_tok_total = 0; n_tok_succ = 0
    n_round_total = 0; n_round_all_pass = 0
    sum_round_succ_rate = 0.0
    for prompt in lane_per_prompt:
        p_idx = int(prompt["prompt_idx"])
        running = [int(x) for x in pool_by_idx[p_idx].tolist()]
        for r in prompt["rounds"]:
            if r.get("prefix_len_at_round_start") is None:
                break
            if int(r["prefix_len_at_round_start"]) != len(running):
                raise RuntimeError(
                    f"prefix_len mismatch at prompt {p_idx} "
                    f"round {r['round_idx']}"
                )
            prefix_ids = torch.tensor(
                running, dtype=torch.long, device=device,
            )
            seed = int(r["round_rng_seed"])
            cur_gamma = int(r["gamma"])
            T = int(r["T"])
            draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
            draft_tokens, draft_log_probs = drafter.draft(
                prefix_ids=prefix_ids, gamma=cur_gamma, T=T,
                temperature=protocol.temperature, q_mode=protocol.q_mode,
                generator=draft_rng,
            )
            candidate = torch.cat([prefix_ids, draft_tokens])
            target_log_probs = verifier.score(candidate).to(torch.float32)
            prefix_len = len(running)
            idx = torch.arange(cur_gamma, device=device)
            q_j = draft_log_probs[
                idx, draft_tokens.long()
            ].exp().clamp(min=EPS)
            p_j = target_log_probs[
                prefix_len - 1 + idx, draft_tokens.long()
            ].exp().clamp(min=EPS)
            Q2 = torch.minimum(torch.ones_like(q_j), p_j / q_j)
            accept_rng = _make_generator(
                device, seed ^ protocol.accept_salt,
            )
            U = torch.rand(cur_gamma, generator=accept_rng, device=device)
            committed = [int(x) for x in r["committed_tokens"]]
            n_commit = len(committed)
            if n_commit == 0:
                continue
            successes = (U[:n_commit] < Q2[:n_commit]).cpu().int().tolist()
            n_succ = sum(successes)
            n_round_total += 1
            n_tok_total += n_commit
            n_tok_succ += n_succ
            sum_round_succ_rate += n_succ / n_commit
            if n_succ == n_commit:
                n_round_all_pass += 1
            running.extend(committed)
    return {
        "token_success_rate":  n_tok_succ / max(n_tok_total, 1),
        "round_mean_success":  sum_round_succ_rate / max(n_round_total, 1),
        "round_all_pass_rate": n_round_all_pass / max(n_round_total, 1),
        "n_tokens_total":      n_tok_total,
        "n_rounds_total":      n_round_total,
    }


# ---------------------------------------------------------------
# Predictor MAE (offline, from preds_test.pt; covers the canonical
# 20-prompt test split, not the 50-prompt online set — MAE is a
# head-quality metric independent of the online prompt set).
# ---------------------------------------------------------------


def compute_predictor_mae(ckpt_dir: Path) -> Dict[str, float]:
    """Return {MAE_overall, MAE_acc, MAE_wgt} computed from preds_test.pt.

      MAE_overall = Σ |Q̂ − tgt| · survived  /  Σ survived
      MAE_acc     = Σ |Q̂ − tgt| · accepted  /  Σ accepted
      MAE_wgt     = Σ |Q̂ − tgt| · survived · w_j  /  Σ survived · w_j,
                    w_j = 1 / (1 + j)

    `tgt` is the value stored in preds_test.pt's `q2_target` field —
    Q2_j for acceptance-target heads, s_j for dependence-target heads.
    Cross-target comparison of these MAE numbers is NOT meaningful;
    they are head-quality metrics on the head's own training target.
    """
    pp = ckpt_dir / "preds_test.pt"
    if not pp.exists():
        return {"MAE_overall": float("nan"), "MAE_acc": float("nan"),
                "MAE_wgt": float("nan")}
    rows = torch.load(str(pp), weights_only=False)
    sse_overall = 0.0; n_overall = 0.0
    sse_acc     = 0.0; n_acc     = 0.0
    sse_wgt     = 0.0; n_wgt     = 0.0
    for r in rows:
        q_hat = torch.tensor(r["q2_hat"], dtype=torch.float32)
        q_tgt = torch.tensor(r["q2_target"], dtype=torch.float32)
        survived = torch.tensor(r["survived"], dtype=torch.float32)
        accepted = torch.tensor(r["accepted"], dtype=torch.float32)
        gamma = q_hat.shape[0]
        ae = (q_hat - q_tgt).abs()
        sse_overall += float((ae * survived).sum())
        n_overall   += float(survived.sum())
        sse_acc     += float((ae * accepted).sum())
        n_acc       += float(accepted.sum())
        w = 1.0 / (1.0 + torch.arange(gamma, dtype=torch.float32))
        sse_wgt     += float((ae * survived * w).sum())
        n_wgt       += float((survived * w).sum())
    return {
        "MAE_overall": sse_overall / max(n_overall, 1.0),
        "MAE_acc":     sse_acc     / max(n_acc, 1.0),
        "MAE_wgt":     sse_wgt     / max(n_wgt, 1.0),
    }


# ---------------------------------------------------------------
# Tables and plots
# ---------------------------------------------------------------


def _f(v, fmt="{:>10.4f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'   —':>10}"
    return fmt.format(float(v))


def fmt_main_table(rows: List[Dict], n_prompts: int) -> str:
    out = []
    out.append(
        f"# Phase 24 — main comparison ({n_prompts} prompts, "
        f"max_new_tokens=64, γ=8, T=2)\n\n"
    )
    out.append(
        "tok/s measured during the decode loop only; NLL and "
        "verifier-success from separate offline passes.  "
        "MAE columns from each method's `preds_test.pt` (head quality "
        "on its own training target, computed over the 20-prompt "
        "canonical test split — independent of the 50-prompt online "
        "set).  N/A means the metric is undefined for this method.\n\n"
    )
    out.append("```\n")
    out.append(
        f"  {'method':<14}{'τ':>6}{'tok/s':>8}{'NLL':>9}{'ΔNLL':>9}"
        f"{'tok_succ':>10}{'rnd_mean':>10}{'all_pass':>10}"
        f"{'MAE_ovr':>10}{'MAE_acc':>10}{'MAE_wgt':>10}\n"
    )
    for row in rows:
        tau_s = "—" if row["tau"] is None else f"{row['tau']:.1f}"
        out.append(
            f"  {row['method']:<14}{tau_s:>6}"
            f"{row['tok_s']:>8.2f}"
            f"{row['NLL']:>9.4f}{row['delta_NLL']:>+9.4f}"
            f"{_f(row.get('tok_succ'))}{_f(row.get('rnd_mean'))}{_f(row.get('all_pass'))}"
            f"{_f(row.get('MAE_overall'))}{_f(row.get('MAE_acc'))}{_f(row.get('MAE_wgt'))}\n"
        )
    out.append("```\n")
    return "".join(out)


def best_op_per_method(rows: List[Dict]) -> List[Dict]:
    """Pick one (method) row using a fixed scoring rule.

    Score per row (higher is better):
      score = z(tok_s) − 0.5 · z(ΔNLL) + 0.5 · z(tok_succ)

    z scores normalise by mean / std across the full main table so the
    three signals are commensurate. Strict / oracle skipped here — they
    appear in their own dedicated rows below.
    """
    pred_rows = [r for r in rows if r["method"] not in ("strict", "oracle_exec")]
    if not pred_rows:
        return []
    arr_speed = np.array([r["tok_s"] for r in pred_rows])
    arr_dnll  = np.array([r["delta_NLL"] for r in pred_rows])
    arr_ts    = np.array([
        r["tok_succ"] if r.get("tok_succ") is not None else 0.0
        for r in pred_rows
    ])
    def _z(a):
        sd = a.std()
        return (a - a.mean()) / (sd if sd > 1e-9 else 1.0)
    score = _z(arr_speed) - 0.5 * _z(arr_dnll) + 0.5 * _z(arr_ts)
    by_method: Dict[str, Tuple[float, Dict]] = {}
    for s, r in zip(score, pred_rows):
        prev = by_method.get(r["method"])
        if prev is None or s > prev[0]:
            by_method[r["method"]] = (float(s), r)
    return [v[1] for v in by_method.values()]


def fmt_summary_table(
    main_rows: List[Dict], strict_row: Dict, n_prompts: int,
) -> str:
    out = []
    out.append(
        f"# Phase 24 — Table 2: best operating point per method "
        f"({n_prompts} prompts)\n\n"
    )
    out.append(
        "Each non-baseline method's single best row by combined "
        "score = z(tok_s) − 0.5·z(ΔNLL) + 0.5·z(tok_succ).  Note column "
        "is a short editorial verdict.\n\n"
    )
    out.append("```\n")
    out.append(
        f"  {'method':<14}{'τ':>6}{'tok/s':>8}{'ΔNLL':>9}{'tok_succ':>10}"
        f"{'MAE_acc':>10}  note\n"
    )
    out.append(
        f"  {'strict':<14}{'—':>6}{strict_row['tok_s']:>8.2f}"
        f"{0.0:>+9.4f}{'   —':>10}{'   —':>10}  reference\n"
    )
    best = best_op_per_method(main_rows)
    # Sort for readability: by tok/s descending.
    best.sort(key=lambda r: -r["tok_s"])
    for row in best:
        note = _summary_note(row)
        out.append(
            f"  {row['method']:<14}{row['tau']:>6.1f}"
            f"{row['tok_s']:>8.2f}{row['delta_NLL']:>+9.4f}"
            f"{_f(row.get('tok_succ'))}{_f(row.get('MAE_acc'))}  {note}\n"
        )
    out.append("```\n")
    return "".join(out)


def _summary_note(row: Dict) -> str:
    name = row["method"]
    dnll = row["delta_NLL"]
    if name == "frozen_dep":
        if dnll > 0.10:
            return "fast but big quality regression"
        return "fastest near-strict (V-free, no fine-tune)"
    if name == "frozen_1A":
        return "stable quality baseline (V-free, no fine-tune)"
    if name == "lazy_jnt":
        return "best ΔNLL on Q2 line"
    if name == "live_jnt_5ep":
        return "Q2 collapse — drop"
    if name == "live_jnt_10ep":
        return "moderate (similar to frozen_1A, joint)"
    if name == "live_jnt_dep":
        return "joint-dep collapse — drop"
    return ""


def _maybe_pareto(
    main_rows: List[Dict], strict_row: Dict,
    out_dnll: Path, out_succ: Path, out_mae: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"[phase24] matplotlib unavailable ({e})")
        return
    markers_by_tau = {0.5: "o", 0.7: "s", 0.9: "^", None: "D"}

    def _plot(rows_to_plot, x_key, y_key, xlabel, ylabel, title, out_path,
              x_strict=None, y_strict=None):
        fig, ax = plt.subplots(figsize=(8.0, 5.6))
        for row in rows_to_plot:
            color = COLORS.get(row["method"], "#888")
            tau = row["tau"]
            marker = markers_by_tau.get(tau, "x")
            x = row.get(x_key); y = row.get(y_key)
            if x is None or y is None or (isinstance(x, float) and not np.isfinite(x)):
                continue
            ax.scatter(
                x, y, s=110, color=color, marker=marker,
                edgecolor="black" if row["method"] in ("strict", "oracle_exec") else None,
                linewidth=(1.0 if row["method"] in ("strict", "oracle_exec") else 0.0),
            )
            ax.annotate(
                f"{row['method']}\nτ={tau}", (x, y),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7.0, color=color, alpha=0.95,
            )
        if x_strict is not None and y_strict is not None:
            ax.scatter(
                x_strict, y_strict, s=140, color="black", marker="D",
                edgecolor="black", linewidth=1.4,
            )
            ax.annotate(
                "strict", (x_strict, y_strict),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8.0, color="black",
            )
            if x_key == "delta_NLL":
                ax.axvline(0.0, linestyle="--", color="#bbbbbb", linewidth=1)
            ax.axhline(y_strict, linestyle=":",
                       color="#bbbbbb", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Legend by method.
        leg_h = []
        seen_methods = set()
        for r in rows_to_plot + ([strict_row] if x_strict is not None else []):
            n = r["method"]
            if n in seen_methods:
                continue
            seen_methods.add(n)
            leg_h.append(Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=COLORS.get(n, "#888"),
                label=n, markersize=8,
                markeredgecolor=("black" if n in ("strict", "oracle_exec") else COLORS.get(n, "#888")),
            ))
        ax.legend(handles=leg_h, loc="best", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=130)
        plt.close(fig)
        print(f"[phase24] wrote {out_path}")

    # Build the rows used by the clean Pareto: main + oracle baselines.
    plot_rows = [r for r in main_rows if r["method"] != "strict"]

    _plot(
        plot_rows, "delta_NLL", "tok_s",
        "Δ NLL vs strict  (↓ better, strict at 0)",
        "tok/s  (↑ better)",
        "Phase 24 — Pareto: speed vs Δ NLL",
        out_dnll, x_strict=0.0, y_strict=strict_row["tok_s"],
    )
    _plot(
        plot_rows, "tok_succ", "tok_s",
        "token verifier-success rate  (↑ better)",
        "tok/s  (↑ better)",
        "Phase 24 — Pareto: speed vs verifier success",
        out_succ,
    )
    _plot(
        plot_rows, "MAE_acc", "tok_s",
        "MAE_acc on head's own target  (↓ better)",
        "tok/s  (↑ better)",
        "Phase 24 — Pareto: speed vs predictor MAE on accepted positions",
        out_mae,
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def _load_eval_prompts(n_prompts: int) -> Tuple[List, List[int]]:
    """Deterministic 50-prompt eval set: load_owt_prompts(n_prompts=N+60)
    then slice [60:60+n_prompts]. Indices 60..79 are the canonical test
    split (overlap with prior phase results); 80..(60+n_prompts-1) are
    fresh prompts pulled under the same seed.

    Training never used any prompt with index >= TRAIN_N (=40), so this
    50-prompt set is fully held out from training.
    """
    needed = TRAIN_N + VAL_N + n_prompts   # = 60 + 50 = 110 by default
    pool = load_owt_prompts(
        n_prompts=needed, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )
    eval_offset = TRAIN_N + VAL_N
    prompts = pool[eval_offset: eval_offset + n_prompts]
    indices = list(range(eval_offset, eval_offset + n_prompts))
    return prompts, indices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_prompts", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument(
        "--skip_missing", action="store_true",
        help="If set, methods whose ckpt_dir does not exist are skipped.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(_REPO_ROOT / "configs/protocol.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier
    from accpre.eval.online_decode import _load_predictor_from_checkpoint

    print("[phase24] loading models...")
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    pretrained_cpu_state = stash_state(drafter)
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompts, prompt_indices = _load_eval_prompts(int(args.n_prompts))
    pool_by_idx = {p_idx: prompts[i][0] for i, p_idx in enumerate(prompt_indices)}
    print(
        f"[phase24] {len(prompts)} eval prompts, indices "
        f"{prompt_indices[0]}..{prompt_indices[-1]}  "
        f"(max_new_tokens={args.max_new_tokens})"
    )

    all_lanes: Dict = {}

    # ---- 1. strict ----
    print("\n[phase24] === STRICT ===")
    t0 = time.time()
    strict_lane = run_strict_lane(
        drafter, verifier, prompts, prompt_indices, protocol,
        gamma=args.gamma, T=args.T, max_new_tokens=args.max_new_tokens,
    )
    print(f"[phase24] strict: tok/s={strict_lane['tok_s_mean']:.2f}  "
          f"elapsed={time.time() - t0:.1f}s")
    with open(out_dir / "online_strict.json", "w") as f:
        json.dump(strict_lane, f, indent=2, default=str)
    all_lanes["strict"] = strict_lane

    # ---- 2. predictor lanes ----
    skipped: List[str] = []
    for name, ckpt_dir, dstate, taus, tier in METHODS:
        ckpt_abs = _REPO_ROOT / ckpt_dir
        if not ckpt_abs.exists():
            msg = f"[phase24] missing ckpt {ckpt_abs} for {name!r}"
            if args.skip_missing:
                print(msg); skipped.append(name); continue
            raise FileNotFoundError(msg)
        print(f"\n[phase24] === {name} ({tier}) ===")
        predictor, _cfg, _ = _load_predictor_from_checkpoint(
            str(ckpt_abs), gamma=args.gamma,
        )
        if dstate is not None:
            state = torch.load(
                str(_REPO_ROOT / dstate), map_location=device,
            )
            drafter.model.load_state_dict(state)
        else:
            restore_state(drafter, pretrained_cpu_state, device)
        drafter.model.eval()
        for tau in taus:
            t0 = time.time()
            lane = run_predictor_lane_one(
                predictor, drafter, verifier, prompts, prompt_indices, protocol,
                tau=tau, gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            )
            print(
                f"[phase24]   {name} τ={tau}: tok/s={lane['tok_s_mean']:.2f}  "
                f"elapsed={time.time() - t0:.1f}s"
            )
            with open(out_dir / f"online_{name}_tau_{fmt_tau_tag(tau)}.json", "w") as f:
                json.dump(lane, f, indent=2, default=str)
            all_lanes[(name, tau)] = lane

    # ---- 3. oracle_exec ----
    restore_state(drafter, pretrained_cpu_state, device)
    for tau in ORACLE_TAUS:
        print(f"\n[phase24] === oracle_exec τ={tau} ===")
        t0 = time.time()
        lane = run_oracle_exec_lane(
            drafter, verifier, prompts, prompt_indices, protocol,
            tau=tau, gamma=args.gamma, T=args.T,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[phase24]   oracle_exec τ={tau}: tok/s={lane['tok_s_mean']:.2f}  "
              f"elapsed={time.time() - t0:.1f}s")
        with open(out_dir / f"online_oracle_tau_{fmt_tau_tag(tau)}.json", "w") as f:
            json.dump(lane, f, indent=2, default=str)
        all_lanes[("oracle_exec", tau)] = lane

    # ---- 4. NLL ----
    print("\n[phase24] === NLL scoring (offline) ===")
    method_nll: Dict = {}
    nll_strict = lane_nll(verifier, strict_lane["per_prompt"], device)
    method_nll["strict"] = nll_strict
    print(f"[phase24]   strict NLL={nll_strict:.4f}")
    for name, _c, _d, taus, _tier in METHODS:
        if name in skipped:
            continue
        for tau in taus:
            nll = lane_nll(verifier, all_lanes[(name, tau)]["per_prompt"], device)
            method_nll[(name, tau)] = nll
            print(f"[phase24]   {name} τ={tau}: NLL={nll:.4f}")
    for tau in ORACLE_TAUS:
        nll = lane_nll(verifier, all_lanes[("oracle_exec", tau)]["per_prompt"], device)
        method_nll[("oracle_exec", tau)] = nll
        print(f"[phase24]   oracle_exec τ={tau}: NLL={nll:.4f}")

    # ---- 5. verifier-success replay (V-free predictor methods) ----
    print("\n[phase24] === verifier-success replay ===")
    vsucc: Dict = {}
    for name, _ckpt, dstate, taus, _tier in METHODS:
        if name in skipped:
            continue
        if dstate is not None:
            state = torch.load(
                str(_REPO_ROOT / dstate), map_location=device,
            )
            drafter.model.load_state_dict(state)
        else:
            restore_state(drafter, pretrained_cpu_state, device)
        drafter.model.eval()
        for tau in taus:
            t0 = time.time()
            r = replay_verifier_success(
                drafter, verifier,
                all_lanes[(name, tau)]["per_prompt"],
                protocol, pool_by_idx, device,
            )
            vsucc[(name, tau)] = r
            print(
                f"[phase24]   {name} τ={tau}: "
                f"tok_succ={r['token_success_rate']:.4f}  "
                f"rnd_mean={r['round_mean_success']:.4f}  "
                f"all_pass={r['round_all_pass_rate']:.4f}  "
                f"({time.time() - t0:.1f}s)"
            )
    for tau in ORACLE_TAUS:
        lane = all_lanes[("oracle_exec", tau)]
        tot_n = sum(p["n_tokens_total"] for p in lane["per_prompt"])
        tok_succ_num = sum(
            p["tok_succ"] * p["n_tokens_total"] for p in lane["per_prompt"]
        )
        rnd_mean = sum(p["rnd_mean"] for p in lane["per_prompt"]) / \
            max(len(lane["per_prompt"]), 1)
        all_pass = sum(p["all_pass"] for p in lane["per_prompt"]) / \
            max(len(lane["per_prompt"]), 1)
        vsucc[("oracle_exec", tau)] = {
            "token_success_rate":  tok_succ_num / max(tot_n, 1),
            "round_mean_success":  rnd_mean,
            "round_all_pass_rate": all_pass,
            "n_tokens_total":      tot_n,
        }

    # ---- 6. predictor MAE ----
    print("\n[phase24] === predictor MAE on canonical test split ===")
    pred_mae: Dict[str, Dict[str, float]] = {}
    for name, ckpt_dir, _d, _taus, _tier in METHODS:
        if name in skipped:
            continue
        m = compute_predictor_mae(_REPO_ROOT / ckpt_dir)
        pred_mae[name] = m
        print(
            f"[phase24]   {name}: MAE_overall={m['MAE_overall']:.4f}  "
            f"MAE_acc={m['MAE_acc']:.4f}  MAE_wgt={m['MAE_wgt']:.4f}"
        )

    # ---- 7. assemble tables ----
    def _row(method, tau):
        if method == "strict":
            return {
                "method": "strict", "tau": None,
                "tok_s": strict_lane["tok_s_mean"],
                "NLL": nll_strict, "delta_NLL": 0.0,
                "tok_succ": None, "rnd_mean": None, "all_pass": None,
                "MAE_overall": None, "MAE_acc": None, "MAE_wgt": None,
            }
        if method == "oracle_exec":
            lane = all_lanes[("oracle_exec", tau)]
            nll = method_nll[("oracle_exec", tau)]
            vs = vsucc[("oracle_exec", tau)]
            return {
                "method": method, "tau": float(tau),
                "tok_s": lane["tok_s_mean"],
                "NLL": nll, "delta_NLL": nll - nll_strict,
                "tok_succ": vs["token_success_rate"],
                "rnd_mean": vs["round_mean_success"],
                "all_pass": vs["round_all_pass_rate"],
                "MAE_overall": None, "MAE_acc": None, "MAE_wgt": None,
            }
        lane = all_lanes[(method, tau)]
        nll = method_nll[(method, tau)]
        vs = vsucc[(method, tau)]
        m = pred_mae.get(method, {})
        return {
            "method": method, "tau": float(tau),
            "tok_s": lane["tok_s_mean"],
            "NLL": nll, "delta_NLL": nll - nll_strict,
            "tok_succ": vs["token_success_rate"],
            "rnd_mean": vs["round_mean_success"],
            "all_pass": vs["round_all_pass_rate"],
            "MAE_overall": m.get("MAE_overall"),
            "MAE_acc":     m.get("MAE_acc"),
            "MAE_wgt":     m.get("MAE_wgt"),
        }

    main_rows: List[Dict] = [_row("strict", None)]
    for tau in ORACLE_TAUS:
        main_rows.append(_row("oracle_exec", tau))
    appendix_rows: List[Dict] = []
    for name, _c, _d, taus, tier in METHODS:
        if name in skipped:
            continue
        target_list = main_rows if tier == "main" else appendix_rows
        for tau in taus:
            target_list.append(_row(name, tau))

    main_md = fmt_main_table(main_rows, int(args.n_prompts))
    with open(out_dir / "unified_table.md", "w") as f:
        f.write(main_md)
    with open(out_dir / "unified_table.json", "w") as f:
        json.dump({
            "rows": main_rows, "strict_nll": nll_strict,
            "prompt_indices": prompt_indices,
            "n_prompts": int(args.n_prompts),
            "max_new_tokens": int(args.max_new_tokens),
            "skipped": skipped,
        }, f, indent=2, default=str)
    print(main_md)

    if appendix_rows:
        appx_md = fmt_main_table(
            appendix_rows, int(args.n_prompts),
        ).replace("Phase 24 — main comparison",
                  "Phase 24 — APPENDIX (reference / dominated)")
        with open(out_dir / "appendix_table.md", "w") as f:
            f.write(appx_md)
        with open(out_dir / "appendix_table.json", "w") as f:
            json.dump({"rows": appendix_rows}, f, indent=2, default=str)
        print("\n" + appx_md)

    summary_md = fmt_summary_table(
        main_rows, _row("strict", None), int(args.n_prompts),
    )
    with open(out_dir / "summary_table.md", "w") as f:
        f.write(summary_md)
    with open(out_dir / "summary_table.json", "w") as f:
        json.dump({
            "best_per_method": best_op_per_method(main_rows),
            "strict": _row("strict", None),
        }, f, indent=2, default=str)
    print(summary_md)

    _maybe_pareto(
        main_rows, _row("strict", None),
        out_dir / "pareto_delta_nll_tok_s.png",
        out_dir / "pareto_tok_succ_tok_s.png",
        out_dir / "pareto_mae_acc_tok_s.png",
    )

    print("\n[phase24] DONE.")
    if skipped:
        print(f"[phase24] SKIPPED: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
