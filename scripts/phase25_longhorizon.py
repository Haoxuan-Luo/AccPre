"""Phase 25 — SpecDiff-style longer-horizon greedy sweep.

Setting (reproducible):
  * Prompts: canonical test split = indices 60..79 from
    `load_owt_prompts(n_prompts=80, prefix_len=32, seed=42)`.
  * 20 prompts, max_new_tokens = 1024 (caps at MAX_CTX=1024 → ≤ 992
    new tokens per prompt in practice).
  * γ=8, T=2.
  * `configs/protocol_greedy.yaml` — temperature = 0.0, greedy decode.

Methods:
  Baselines
    strict
    oracle_exec       τ ∈ {0.5, 0.7, 0.9}
  Predictors
    frozen_1A         τ ∈ {0.5, 0.7, 0.9}
    live_jnt_10ep     τ ∈ {0.5, 0.7, 0.9}
    frozen_dep        τ ∈ {0.5, 0.7, 0.9}
    live_jnt_dep      τ ∈ {0.5, 0.7, 0.9}

tok/s timed INSIDE the decode loop only. NLL scoring and
verifier-success replay are separate offline passes.

Metric semantics under temperature = 0:
  * Accept rule: `argmax(p_target) == draft_tok`. (NOT `U < Q2`.)
  * `tok_succ`: fraction of committed tokens whose draft token equals
    the verifier's argmax at that position. `rnd_mean`, `all_pass`
    aggregated with the same per-position success indicator.
  * Oracle-exec commit rule is still `commit_threshold(Q2, τ)` with
    `Q2 = min(1, p_j / q_j)` — same math as before — but the inline
    per-position success flag uses the argmax test, to match the
    accept rule under temp=0.

Outputs (`--out_dir`):
  online_strict.json, online_<method>_tau_<τ>.json, online_oracle_tau_<τ>.json
  unified_table.{md,json}
  pareto_delta_nll_tok_s.png
  pareto_tok_succ_tok_s.png
  timing_breakdown.json  (decode vs NLL vs replay, per method)
"""

from __future__ import annotations

import argparse
import json
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


METHODS: List[Tuple[str, str, Optional[str], Tuple[float, ...]]] = [
    ("frozen_1A",     "checkpoints/acc_1a",                  None,
                      (0.5, 0.7, 0.9)),
    ("live_jnt_10ep", "checkpoints/acc_jnt_1a_liveq2_long",
                      "checkpoints/acc_jnt_1a_liveq2_long/drafter.pt",
                      (0.5, 0.7, 0.9)),
    ("frozen_dep",    "checkpoints/dep_frz_1a",              None,
                      (0.5, 0.7, 0.9)),
    ("live_jnt_dep",  "checkpoints/dep_jnt_live",
                      "checkpoints/dep_jnt_live/drafter.pt",
                      (0.5, 0.7, 0.9)),
]

ORACLE_TAUS: Tuple[float, ...] = (0.5, 0.7, 0.9)

COLORS: Dict[str, str] = {
    "strict":        "#000000",
    "oracle_exec":   "#f58518",
    "frozen_1A":     "#3b76b3",
    "live_jnt_10ep": "#d22b2b",
    "frozen_dep":    "#9467bd",
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


# -----------------------------------------------------------------
# Lanes
# -----------------------------------------------------------------


@torch.no_grad()
def run_strict_lane(
    drafter, verifier, prompts, prompt_indices, protocol,
    gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
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
            # Strict commits up to `cur_gamma + 1` tokens per round
            # (draft prefix of length L ≤ cur_gamma + one bonus/fallback
            # token). Reserve one slot for the bonus, else a round with
            # L = cur_gamma pushes `running.shape[0]` past MAX_CTX and the
            # downstream verifier / GPT-2-XL forward hits its seq-len
            # limit. (Phase 25 caught this: at max_new_tokens=1024 strict
            # generated 1025 tokens and NLL scoring crashed.)
            ctx_room = MAX_CTX - int(running.shape[0]) - 1
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
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    from accpre.eval.online_decode import run_predictor_lane
    lane = run_predictor_lane(
        predictor=predictor, drafter=drafter, verifier=verifier,
        prompts=prompts, protocol=protocol, gamma=gamma, T=T,
        max_new_tokens=max_new_tokens, tau=tau,
        prompt_indices=prompt_indices,
    )
    return lane.to_dict()


def _position_accepted(
    target_log_probs: torch.Tensor,    # (L+γ, V)
    draft_tokens: torch.Tensor,        # (γ,) long, on device
    prefix_len: int,
    temperature: float,
    accept_rng,
    cur_gamma: int,
) -> torch.Tensor:
    """Per-position success indicator in [0, 1]^cur_gamma (on device).

    Under temp=0 (greedy): `argmax(target[pos]) == draft_tok`.
    Under temp>0          : `U < Q2`, where U drawn from accept_rng and
                             Q2 = min(1, p/q). Matches the Leviathan
                             accept rule in `per_position_accept`.

    For the Phase-25 greedy sweep we use temp=0, so the Bernoulli path
    is effectively dead code here, but kept so the script also runs
    under temp=1 if someone points --protocol_yaml at the mainline.
    """
    device = draft_tokens.device
    idx = torch.arange(cur_gamma, device=device)
    tgt_row = target_log_probs[prefix_len - 1 + idx]       # (γ, V)
    if float(temperature) == 0.0:
        top = tgt_row.argmax(dim=-1)                       # (γ,)
        return (top == draft_tokens.long()).to(torch.int32)
    # temperature > 0 → need q from the drafter; caller must provide it
    # another way. This branch is not reached in the Phase-25 sweep.
    raise NotImplementedError(
        "_position_accepted: non-greedy path needs q_j too; call the "
        "explicit U<Q2 block inline."
    )


@torch.no_grad()
def run_oracle_exec_lane(
    drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
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

            # Per-position success indicator consistent with temp=0 accept.
            accept_rng = _make_generator(
                device, seed ^ protocol.accept_salt,
            )
            # Draw U_j for stream discipline (unused at temp=0).
            U = torch.rand(cur_gamma, generator=accept_rng, device=device)
            if float(protocol.temperature) == 0.0:
                top = target_log_probs[
                    prefix_len - 1 + idx, :
                ].argmax(dim=-1)
                succ_full = (top == draft_tokens.long()).to(torch.int32)
            else:
                succ_full = (U < Q2).to(torch.int32)
            successes = succ_full[:n_commit].cpu().int().tolist()
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
    total_sum = 0.0; total_n = 0
    for p in lane_per_prompt:
        gen = [int(x) for x in p["generated_ids"]]
        n_prefix = len(gen) - int(p["n_new_tokens"])
        s, n = score_nll_on_new(verifier, gen, n_prefix, device)
        total_sum += s; total_n += n
    return total_sum / max(total_n, 1)


@torch.no_grad()
def replay_verifier_success(
    drafter, verifier, lane_per_prompt, protocol,
    pool_by_idx: Dict[int, torch.Tensor], device,
) -> Dict:
    """Temp-aware verifier-success replay.

    At temp=0: per-position success = argmax(p_target) == draft_tok.
    At temp>0: per-position success = U < Q2.
    """
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
            if float(protocol.temperature) == 0.0:
                top = target_log_probs[
                    prefix_len - 1 + idx, :
                ].argmax(dim=-1)
                succ_full = (top == draft_tokens.long()).to(torch.int32)
            else:
                succ_full = (U < Q2).to(torch.int32)
            committed = [int(x) for x in r["committed_tokens"]]
            n_commit = len(committed)
            if n_commit == 0:
                continue
            successes = succ_full[:n_commit].cpu().int().tolist()
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


# -----------------------------------------------------------------
# Predictor MAE — same as Phase 24; head-on-own-target, 20-prompt test.
# -----------------------------------------------------------------


def compute_predictor_mae(ckpt_dir: Path) -> Dict[str, float]:
    pp = ckpt_dir / "preds_test.pt"
    if not pp.exists():
        return {k: float("nan") for k in ("MAE_overall", "MAE_acc", "MAE_wgt")}
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
        sse_wgt += float((ae * survived * w).sum())
        n_wgt   += float((survived * w).sum())
    return {
        "MAE_overall": sse_overall / max(n_overall, 1.0),
        "MAE_acc":     sse_acc     / max(n_acc, 1.0),
        "MAE_wgt":     sse_wgt     / max(n_wgt, 1.0),
    }


# -----------------------------------------------------------------
# Tables & plots
# -----------------------------------------------------------------


def _f(v, fmt="{:>10.4f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'   —':>10}"
    return fmt.format(float(v))


def fmt_main_table(rows: List[Dict], n_prompts: int, max_new_tokens: int) -> str:
    out = [
        f"# Phase 25 — long-horizon greedy comparison "
        f"({n_prompts} prompts, max_new_tokens={max_new_tokens}, "
        f"γ=8, T=2, temperature=0)\n\n",
        "tok/s measured strictly during the decode loop. NLL and "
        "verifier-success are separate offline passes.  Under temp=0, "
        "`tok_succ` is the fraction of committed tokens where the "
        "drafter-sampled token matches the verifier's argmax at that "
        "position (Leviathan greedy accept rule).  MAE columns from "
        "`preds_test.pt` — head quality on its own training target, "
        "computed under the temp=1 training regime.  N/A where "
        "undefined.\n\n",
        "```\n",
        f"  {'method':<14}{'τ':>6}{'tok/s':>8}{'NLL':>9}{'ΔNLL':>9}"
        f"{'tok_succ':>10}{'rnd_mean':>10}{'all_pass':>10}"
        f"{'MAE_ovr':>10}{'MAE_acc':>10}{'MAE_wgt':>10}\n",
    ]
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


def _maybe_pareto(rows: List[Dict], out_dnll: Path, out_succ: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"[phase25] matplotlib unavailable ({e})")
        return
    markers_by_tau = {0.5: "o", 0.7: "s", 0.9: "^", None: "D"}

    def _plot(x_key, xlabel, out_path, title, x_strict=None):
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        strict_tok_s = next((r["tok_s"] for r in rows if r["method"] == "strict"), None)
        for row in rows:
            color = COLORS.get(row["method"], "#888")
            tau = row["tau"]
            marker = markers_by_tau.get(tau, "x")
            x = row.get(x_key); y = row["tok_s"]
            if x is None or y is None or (isinstance(x, float) and not np.isfinite(x)):
                continue
            if row["method"] == "strict" and x_key == "tok_succ":
                continue
            ax.scatter(
                x, y, s=(150 if row["method"] == "strict" else 110),
                color=color, marker=marker,
                edgecolor="black" if row["method"] in ("strict", "oracle_exec") else None,
                linewidth=(1.2 if row["method"] in ("strict", "oracle_exec") else 0.0),
            )
            lbl = "strict" if row["method"] == "strict" else f"{row['method']}\nτ={tau}"
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7.0, color=color, alpha=0.95)
        if x_strict is not None:
            ax.axvline(x_strict, linestyle="--", color="#bbbbbb", linewidth=1)
        if strict_tok_s is not None:
            ax.axhline(strict_tok_s, linestyle=":", color="#bbbbbb", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("tok/s  (↑ better)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        leg_h = []
        seen = set()
        for r in rows:
            n = r["method"]
            if n in seen:
                continue
            seen.add(n)
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
        print(f"[phase25] wrote {out_path}")

    _plot(
        "delta_NLL", "Δ NLL vs strict  (↓ better, strict at 0)",
        out_dnll,
        "Phase 25 (greedy, 1024 tok): speed vs Δ NLL",
        x_strict=0.0,
    )
    _plot(
        "tok_succ", "token verifier-success rate  (↑ better)",
        out_succ,
        "Phase 25 (greedy, 1024 tok): speed vs greedy-accept success",
    )


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


def _load_eval_prompts(n_prompts: int) -> Tuple[List, List[int]]:
    # Canonical test split: indices 60..79 in the POOL_SIZE=80 pool.
    # Phase 25 asks for exactly 20 prompts -> reuse this slice unchanged.
    pool = load_owt_prompts(
        n_prompts=POOL_SIZE, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )
    eval_offset = TRAIN_N + VAL_N
    prompts = pool[eval_offset: eval_offset + n_prompts]
    indices = list(range(eval_offset, eval_offset + n_prompts))
    return prompts, indices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--protocol_yaml", type=str,
                    default="configs/protocol_greedy.yaml")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--skip_missing", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(_REPO_ROOT / args.protocol_yaml)
    assert float(protocol.temperature) == 0.0, (
        f"Phase 25 expects temp=0; got {protocol.temperature}. "
        f"Check --protocol_yaml."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier
    from accpre.eval.online_decode import _load_predictor_from_checkpoint

    print("[phase25] loading models...")
    t_setup = time.time()
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    pretrained_cpu_state = stash_state(drafter)
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )
    print(f"[phase25] setup done in {time.time() - t_setup:.1f}s")

    prompts, prompt_indices = _load_eval_prompts(int(args.n_prompts))
    pool_by_idx = {p_idx: prompts[i][0] for i, p_idx in enumerate(prompt_indices)}
    print(
        f"[phase25] {len(prompts)} prompts, indices "
        f"{prompt_indices[0]}..{prompt_indices[-1]}  "
        f"(max_new_tokens={args.max_new_tokens}, temp={protocol.temperature})"
    )

    all_lanes: Dict = {}
    timings: Dict[str, Dict] = {
        "decode": {},
        "nll":    {},
        "replay": {},
    }

    # ---- strict ----
    print("\n[phase25] === STRICT ===")
    t0 = time.time()
    strict_lane = run_strict_lane(
        drafter, verifier, prompts, prompt_indices, protocol,
        gamma=args.gamma, T=args.T, max_new_tokens=args.max_new_tokens,
    )
    timings["decode"]["strict"] = time.time() - t0
    print(f"[phase25] strict: tok/s={strict_lane['tok_s_mean']:.2f}  "
          f"elapsed={timings['decode']['strict']:.1f}s")
    with open(out_dir / "online_strict.json", "w") as f:
        json.dump(strict_lane, f, indent=2, default=str)
    all_lanes["strict"] = strict_lane

    # ---- predictors ----
    skipped: List[str] = []
    for name, ckpt_dir, dstate, taus in METHODS:
        ckpt_abs = _REPO_ROOT / ckpt_dir
        if not ckpt_abs.exists():
            msg = f"[phase25] missing ckpt {ckpt_abs} for {name!r}"
            if args.skip_missing:
                print(msg); skipped.append(name); continue
            raise FileNotFoundError(msg)
        print(f"\n[phase25] === {name} ===")
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
            el = time.time() - t0
            timings["decode"][f"{name}|{tau}"] = el
            print(f"[phase25]   {name} τ={tau}: tok/s={lane['tok_s_mean']:.2f}  "
                  f"elapsed={el:.1f}s")
            with open(out_dir / f"online_{name}_tau_{fmt_tau_tag(tau)}.json", "w") as f:
                json.dump(lane, f, indent=2, default=str)
            all_lanes[(name, tau)] = lane

    # ---- oracle_exec ----
    restore_state(drafter, pretrained_cpu_state, device)
    for tau in ORACLE_TAUS:
        print(f"\n[phase25] === oracle_exec τ={tau} ===")
        t0 = time.time()
        lane = run_oracle_exec_lane(
            drafter, verifier, prompts, prompt_indices, protocol,
            tau=tau, gamma=args.gamma, T=args.T,
            max_new_tokens=args.max_new_tokens,
        )
        el = time.time() - t0
        timings["decode"][f"oracle_exec|{tau}"] = el
        print(f"[phase25]   oracle_exec τ={tau}: tok/s={lane['tok_s_mean']:.2f}  "
              f"elapsed={el:.1f}s")
        with open(out_dir / f"online_oracle_tau_{fmt_tau_tag(tau)}.json", "w") as f:
            json.dump(lane, f, indent=2, default=str)
        all_lanes[("oracle_exec", tau)] = lane

    # ---- NLL ----
    print("\n[phase25] === NLL scoring (offline) ===")
    method_nll: Dict = {}
    t0 = time.time()
    nll_strict = lane_nll(verifier, strict_lane["per_prompt"], device)
    timings["nll"]["strict"] = time.time() - t0
    method_nll["strict"] = nll_strict
    print(f"[phase25]   strict NLL={nll_strict:.4f}")
    for name, _c, _d, taus in METHODS:
        if name in skipped:
            continue
        for tau in taus:
            t0 = time.time()
            nll = lane_nll(verifier, all_lanes[(name, tau)]["per_prompt"], device)
            timings["nll"][f"{name}|{tau}"] = time.time() - t0
            method_nll[(name, tau)] = nll
            print(f"[phase25]   {name} τ={tau}: NLL={nll:.4f}")
    for tau in ORACLE_TAUS:
        t0 = time.time()
        nll = lane_nll(verifier, all_lanes[("oracle_exec", tau)]["per_prompt"], device)
        timings["nll"][f"oracle_exec|{tau}"] = time.time() - t0
        method_nll[("oracle_exec", tau)] = nll
        print(f"[phase25]   oracle_exec τ={tau}: NLL={nll:.4f}")

    # ---- verifier-success replay ----
    print("\n[phase25] === verifier-success replay ===")
    vsucc: Dict = {}
    for name, _ckpt, dstate, taus in METHODS:
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
            timings["replay"][f"{name}|{tau}"] = time.time() - t0
            vsucc[(name, tau)] = r
            print(
                f"[phase25]   {name} τ={tau}: "
                f"tok_succ={r['token_success_rate']:.4f}  "
                f"rnd_mean={r['round_mean_success']:.4f}  "
                f"all_pass={r['round_all_pass_rate']:.4f}  "
                f"({timings['replay'][f'{name}|{tau}']:.1f}s)"
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

    # ---- predictor MAE ----
    print("\n[phase25] === predictor MAE ===")
    pred_mae: Dict[str, Dict[str, float]] = {}
    for name, ckpt_dir, _d, _taus in METHODS:
        if name in skipped:
            continue
        m = compute_predictor_mae(_REPO_ROOT / ckpt_dir)
        pred_mae[name] = m
        print(
            f"[phase25]   {name}: MAE_overall={m['MAE_overall']:.4f}  "
            f"MAE_acc={m['MAE_acc']:.4f}  MAE_wgt={m['MAE_wgt']:.4f}"
        )

    # ---- assemble rows ----
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

    rows: List[Dict] = [_row("strict", None)]
    for tau in ORACLE_TAUS:
        rows.append(_row("oracle_exec", tau))
    for name, _c, _d, taus in METHODS:
        if name in skipped:
            continue
        for tau in taus:
            rows.append(_row(name, tau))

    main_md = fmt_main_table(rows, int(args.n_prompts), int(args.max_new_tokens))
    with open(out_dir / "unified_table.md", "w") as f:
        f.write(main_md)
    with open(out_dir / "unified_table.json", "w") as f:
        json.dump({
            "rows": rows, "strict_nll": nll_strict,
            "prompt_indices": prompt_indices,
            "n_prompts": int(args.n_prompts),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(protocol.temperature),
            "skipped": skipped,
        }, f, indent=2, default=str)
    print(main_md)

    _maybe_pareto(
        rows,
        out_dir / "pareto_delta_nll_tok_s.png",
        out_dir / "pareto_tok_succ_tok_s.png",
    )

    # ---- timing breakdown ----
    sums = {
        k: sum(v for v in stage.values())
        for k, stage in timings.items()
    }
    sums["total"] = sums["decode"] + sums["nll"] + sums["replay"]
    timing_blob = {
        "per_method_s": timings,
        "totals_s": sums,
    }
    with open(out_dir / "timing_breakdown.json", "w") as f:
        json.dump(timing_blob, f, indent=2, default=str)
    print(
        f"\n[phase25] timing totals (s):  "
        f"decode={sums['decode']:.1f}  "
        f"nll={sums['nll']:.1f}  "
        f"replay={sums['replay']:.1f}  "
        f"TOTAL={sums['total']:.1f}  "
        f"({sums['total'] / 60:.1f} min)"
    )

    print("\n[phase25] DONE.")
    if skipped:
        print(f"[phase25] SKIPPED: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
