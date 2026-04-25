"""Phase 22 — unified 20-prompt online evaluation.

One driver that, under a single process, evaluates:
  - strict SpecDiff
  - frozen_1A / lazy_jnt / live_jnt_5ep / live_jnt_10ep at τ ∈ {0.5,0.7,0.9}
  - oracle_exec at τ ∈ {0.5,0.7,0.9}

on the canonical 20-prompt test split (global indices 60..79, max 64
new tokens).

**tok/s is measured ONLY during the decoding loop** — the subsequent
verifier-NLL scoring and verifier-success replay passes run AFTER the
timed region and are NOT charged against tok/s.

Outputs:
  <out_dir>/
    online_strict.json
    online_<method>_tau_<0p5|0p7|0p9>.json     (per predictor method)
    online_oracle_tau_<0p5|0p7|0p9>.json
    unified_table.{md,json}
    pareto_delta_nll_tok_s.png
    pareto_tok_succ_tok_s.png                   (second figure)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N, test_split


TAUS = (0.5, 0.7, 0.9)
EPS = 1e-10


# name, ckpt_dir, drafter_state_path (None = pretrained)
METHODS_PREDICTOR = [
    ("frozen_1A",     "checkpoints/acc_1a",
     None),
    ("lazy_jnt",      "checkpoints/acc_jnt_1a",
     "checkpoints/acc_jnt_1a/drafter.pt"),
    ("live_jnt_5ep",  "checkpoints/acc_jnt_1a_liveq2",
     "checkpoints/acc_jnt_1a_liveq2/drafter.pt"),
    ("live_jnt_10ep", "checkpoints/acc_jnt_1a_liveq2_long",
     "checkpoints/acc_jnt_1a_liveq2_long/drafter.pt"),
]


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
# 1. Strict lane (timed, no verifier scoring inside the timer)
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
                "round_idx": round_idx, "L": L,
                "n_committed": L + 1,
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


# ---------------------------------------------------------------
# 2. Predictor lane (delegates to the repo's run_predictor_lane)
# ---------------------------------------------------------------


@torch.no_grad()
def run_one_predictor_lane(
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


# ---------------------------------------------------------------
# 3. Oracle-exec lane — verifier is intrinsic; track everything inline.
# ---------------------------------------------------------------


@torch.no_grad()
def run_oracle_exec_lane(
    drafter, verifier, prompts, prompt_indices, protocol,
    tau: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 64,
) -> Dict:
    """Runs the oracle-Q2 rollout and collects tok/s AS WELL AS
    verifier-success stats inline. For consistency with the
    timed-region rule, tok/s here does include the verifier forwards
    (which the method itself needs). NLL is scored separately below
    over the generated_ids under the same verifier, so that the
    `NLL` column is computed the same way across methods (a separate
    offline pass)."""
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
            # inline verifier-success aggregates:
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


# ---------------------------------------------------------------
# 4. Offline NLL scoring (separate pass, NOT in tok/s)
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# 5. Verifier-success replay (V-free predictor methods only)
# ---------------------------------------------------------------


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
                # Not a predictor-lane round (e.g. strict/oracle format);
                # skip this prompt's rounds as unsupported.
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
# Assembling table + figures
# ---------------------------------------------------------------


def fmt_table(rows: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# Phase 22 — unified 20-prompt evaluation\n\n")
    lines.append(
        "tok/s measured during decoding only; NLL and verifier-success "
        "from separate offline passes.  Strict is the reference "
        "(ΔNLL = 0 by construction; verifier-success not applicable — "
        "strict's commits are defined by Leviathan accept).\n\n"
    )
    lines.append("```\n")
    lines.append(
        f"  {'method':<14}{'τ':>6}{'tok/s':>8}{'NLL':>9}"
        f"{'ΔNLL':>9}{'tok_succ':>10}{'rnd_mean':>10}{'all_pass':>10}\n"
    )
    for row in rows:
        tau_s = "—" if row["tau"] is None else f"{row['tau']:.1f}"
        nll = row.get("NLL", float("nan"))
        dnll = row.get("delta_NLL", float("nan"))
        ts = row.get("tok_succ")
        rm = row.get("rnd_mean")
        ap = row.get("all_pass")
        def _f(v, fmt="{:>10.4f}"):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return f"{'   —':>10}"
            return fmt.format(float(v))
        lines.append(
            f"  {row['method']:<14}{tau_s:>6}"
            f"{row['tok_s']:>8.2f}"
            f"{nll:>9.4f}{dnll:>+9.4f}"
            f"{_f(ts)}{_f(rm)}{_f(ap)}\n"
        )
    lines.append("```\n")
    return "".join(lines)


def _maybe_pareto(
    rows: List[Dict], out_path_nll: Path, out_path_succ: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[phase22] matplotlib unavailable ({e})")
        return
    colors = {
        "strict":         "#000000",
        "oracle_exec":    "#f58518",
        "frozen_1A":      "#4c78a8",
        "lazy_jnt":       "#a0a0a0",
        "live_jnt_5ep":   "#54a24b",
        "live_jnt_10ep":  "#e45756",
    }
    markers_by_tau = {0.5: "o", 0.7: "s", 0.9: "^", None: "D"}

    # Figure 1: ΔNLL vs tok/s
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for row in rows:
        color = colors.get(row["method"], "#888")
        tau = row["tau"]
        marker = markers_by_tau.get(tau, "x")
        x = row["delta_NLL"]
        y = row["tok_s"]
        if x is None or y is None:
            continue
        ax.scatter(
            x, y, s=(130 if row["method"] == "strict" else 95),
            color=color, marker=marker,
            edgecolor="black" if row["method"] in ("strict", "oracle_exec") else None,
            linewidth=(1.2 if row["method"] in ("strict", "oracle_exec") else 0.0),
        )
        lbl = "strict" if row["method"] == "strict" else f"{row['method']}\nτ={tau}"
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.2, color=color, alpha=0.95)
    ax.axvline(0.0, linestyle="--", color="#bbbbbb", linewidth=1)
    # Strict y-line
    strict_rows = [r for r in rows if r["method"] == "strict"]
    if strict_rows:
        ax.axhline(strict_rows[0]["tok_s"], linestyle=":",
                   color="#bbbbbb", linewidth=1)
    ax.set_xlabel("Δ NLL vs strict  (↓ better, strict at 0)")
    ax.set_ylabel("tok/s  (↑ better)")
    ax.set_title("Phase 22 Pareto (20 prompts): speed vs quality gap to strict")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    leg_h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               label="τ = 0.5", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               label="τ = 0.7", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
               label="τ = 0.9", markersize=8),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
               markeredgecolor="black", label="strict", markersize=8),
    ]
    for name, color in colors.items():
        if name == "strict":
            continue
        leg_h.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color, label=name, markersize=8,
            markeredgecolor=("black" if name == "oracle_exec" else color),
        ))
    ax.legend(handles=leg_h, loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path_nll), dpi=130)
    plt.close(fig)
    print(f"[phase22] wrote {out_path_nll}")

    # Figure 2: tok_succ vs tok/s
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for row in rows:
        color = colors.get(row["method"], "#888")
        tau = row["tau"]
        marker = markers_by_tau.get(tau, "x")
        x = row.get("tok_succ")
        y = row["tok_s"]
        if x is None or y is None:
            continue
        ax.scatter(
            x, y, s=95, color=color, marker=marker,
            edgecolor="black" if row["method"] == "oracle_exec" else None,
            linewidth=1.2 if row["method"] == "oracle_exec" else 0.0,
        )
        ax.annotate(f"{row['method']}\nτ={tau}", (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7.2, color=color, alpha=0.95)
    ax.set_xlabel("token verifier-success rate  (↑ better)")
    ax.set_ylabel("tok/s  (↑ better)")
    ax.set_title("Phase 22: token-level verifier success vs throughput")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=leg_h, loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path_succ), dpi=130)
    plt.close(fig)
    print(f"[phase22] wrote {out_path_succ}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
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

    print("[phase22] loading models...")
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    pretrained_cpu_state = stash_state(drafter)
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompts = test_split()[: int(args.n_prompts)]
    prompt_indices = list(range(
        TRAIN_N + VAL_N, TRAIN_N + VAL_N + int(args.n_prompts),
    ))
    pool_by_idx: Dict[int, torch.Tensor] = {
        p_idx: prompts[i][0] for i, p_idx in enumerate(prompt_indices)
    }
    print(
        f"[phase22] 20-prompt canonical test range: global indices "
        f"{prompt_indices[0]}..{prompt_indices[-1]}  "
        f"(n={len(prompts)}, max_new_tokens={args.max_new_tokens})"
    )

    all_lanes: Dict = {}

    # ----- 1. strict -----
    print("\n[phase22] === STRICT ===")
    t0 = time.time()
    strict_lane = run_strict_lane(
        drafter, verifier, prompts, prompt_indices, protocol,
        gamma=args.gamma, T=args.T, max_new_tokens=args.max_new_tokens,
    )
    print(
        f"[phase22] strict: tok/s={strict_lane['tok_s_mean']:.2f}  "
        f"elapsed={time.time() - t0:.1f}s"
    )
    with open(out_dir / "online_strict.json", "w") as f:
        json.dump(strict_lane, f, indent=2, default=str)
    all_lanes["strict"] = strict_lane

    # ----- 2. predictor methods -----
    for name, ckpt_dir, drafter_state_path in METHODS_PREDICTOR:
        print(f"\n[phase22] === {name} ===")
        predictor, _cfg, _ = _load_predictor_from_checkpoint(
            str(_REPO_ROOT / ckpt_dir), gamma=args.gamma,
        )
        if drafter_state_path is not None:
            state = torch.load(
                str(_REPO_ROOT / drafter_state_path), map_location=device,
            )
            drafter.model.load_state_dict(state)
        else:
            restore_state(drafter, pretrained_cpu_state, device)
        drafter.model.eval()
        for tau in TAUS:
            t0 = time.time()
            lane = run_one_predictor_lane(
                predictor, drafter, verifier, prompts, prompt_indices, protocol,
                tau=tau, gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            )
            print(
                f"[phase22]   {name} τ={tau}: tok/s={lane['tok_s_mean']:.2f}  "
                f"elapsed={time.time() - t0:.1f}s"
            )
            with open(out_dir / f"online_{name}_tau_{fmt_tau_tag(tau)}.json", "w") as f:
                json.dump(lane, f, indent=2, default=str)
            all_lanes[(name, tau)] = lane

    # ----- 3. oracle_exec (reset drafter to pretrained) -----
    restore_state(drafter, pretrained_cpu_state, device)
    for tau in TAUS:
        print(f"\n[phase22] === oracle_exec τ={tau} ===")
        t0 = time.time()
        lane = run_oracle_exec_lane(
            drafter, verifier, prompts, prompt_indices, protocol,
            tau=tau, gamma=args.gamma, T=args.T,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"[phase22]   oracle_exec τ={tau}: tok/s={lane['tok_s_mean']:.2f}  "
            f"elapsed={time.time() - t0:.1f}s"
        )
        with open(out_dir / f"online_oracle_tau_{fmt_tau_tag(tau)}.json", "w") as f:
            json.dump(lane, f, indent=2, default=str)
        all_lanes[("oracle_exec", tau)] = lane

    # ----- 4. offline NLL scoring -----
    print("\n[phase22] === NLL scoring (offline) ===")
    method_nll: Dict = {}
    nll_strict = lane_nll(verifier, strict_lane["per_prompt"], device)
    method_nll["strict"] = nll_strict
    print(f"[phase22]   strict NLL={nll_strict:.4f}")
    for name, _c, _d in METHODS_PREDICTOR:
        for tau in TAUS:
            nll = lane_nll(verifier, all_lanes[(name, tau)]["per_prompt"], device)
            method_nll[(name, tau)] = nll
            print(f"[phase22]   {name} τ={tau}: NLL={nll:.4f}")
    for tau in TAUS:
        nll = lane_nll(verifier, all_lanes[("oracle_exec", tau)]["per_prompt"], device)
        method_nll[("oracle_exec", tau)] = nll
        print(f"[phase22]   oracle_exec τ={tau}: NLL={nll:.4f}")

    # ----- 5. verifier-success replay (V-free methods) -----
    print("\n[phase22] === verifier-success replay (V-free predictors) ===")
    vsucc: Dict = {}
    for name, _ckpt, drafter_state_path in METHODS_PREDICTOR:
        if drafter_state_path is not None:
            state = torch.load(
                str(_REPO_ROOT / drafter_state_path), map_location=device,
            )
            drafter.model.load_state_dict(state)
        else:
            restore_state(drafter, pretrained_cpu_state, device)
        drafter.model.eval()
        for tau in TAUS:
            t0 = time.time()
            r = replay_verifier_success(
                drafter, verifier,
                all_lanes[(name, tau)]["per_prompt"],
                protocol, pool_by_idx, device,
            )
            vsucc[(name, tau)] = r
            print(
                f"[phase22]   {name} τ={tau}: tok_succ="
                f"{r['token_success_rate']:.4f}  "
                f"rnd_mean={r['round_mean_success']:.4f}  "
                f"all_pass={r['round_all_pass_rate']:.4f}  "
                f"({time.time() - t0:.1f}s)"
            )

    # Oracle vsucc aggregated from per_prompt inline stats.
    for tau in TAUS:
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

    # ----- 6. assemble unified table -----
    print("\n[phase22] === assembling table + figures ===")
    rows: List[Dict] = []
    rows.append({
        "method": "strict", "tau": None,
        "tok_s": strict_lane["tok_s_mean"],
        "NLL": nll_strict, "delta_NLL": 0.0,
        "tok_succ": None, "rnd_mean": None, "all_pass": None,
    })
    for name, _c, _d in METHODS_PREDICTOR:
        for tau in TAUS:
            lane = all_lanes[(name, tau)]
            nll = method_nll[(name, tau)]
            vs = vsucc[(name, tau)]
            rows.append({
                "method": name, "tau": float(tau),
                "tok_s": lane["tok_s_mean"],
                "NLL": nll,
                "delta_NLL": nll - nll_strict,
                "tok_succ": vs["token_success_rate"],
                "rnd_mean": vs["round_mean_success"],
                "all_pass": vs["round_all_pass_rate"],
            })
    for tau in TAUS:
        lane = all_lanes[("oracle_exec", tau)]
        nll = method_nll[("oracle_exec", tau)]
        vs = vsucc[("oracle_exec", tau)]
        rows.append({
            "method": "oracle_exec", "tau": float(tau),
            "tok_s": lane["tok_s_mean"],
            "NLL": nll,
            "delta_NLL": nll - nll_strict,
            "tok_succ": vs["token_success_rate"],
            "rnd_mean": vs["round_mean_success"],
            "all_pass": vs["round_all_pass_rate"],
        })

    rep = fmt_table(rows)
    with open(out_dir / "unified_table.md", "w") as f:
        f.write(rep)
    with open(out_dir / "unified_table.json", "w") as f:
        json.dump({"rows": rows, "strict_nll": nll_strict,
                   "prompt_indices": prompt_indices,
                   "n_prompts": int(args.n_prompts),
                   "max_new_tokens": int(args.max_new_tokens)},
                  f, indent=2, default=str)
    print(rep)
    print(f"[phase22] wrote {out_dir}/unified_table.md")

    _maybe_pareto(
        rows,
        out_dir / "pareto_delta_nll_tok_s.png",
        out_dir / "pareto_tok_succ_tok_s.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
