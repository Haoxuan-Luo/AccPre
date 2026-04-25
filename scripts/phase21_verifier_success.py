"""Phase 21 — verifier-success metrics (evaluation only).

For each method × τ, we replay the rollout recorded in its online JSON
and, at each round, re-run drafter + verifier on the round's prefix +
draft_tokens to recover (q_j, p_j), compute Q2_j = min(1, p/q), and
draw U_j from the matched accept_rng. We then test U_j < Q2_j on the
positions the method COMMITTED (not the full γ draft positions).

Metrics reported per (method, τ):
    token_success_rate   = fraction of committed tokens with U < Q2.
                           This is the "token-level" verifier-success.
    round_mean_success   = mean over rounds of (within-round success
                           fraction on committed positions). Differs
                           from token_success_rate when rounds commit
                           different token counts.
    round_all_pass_rate  = fraction of rounds where every committed
                           token passes (U < Q2 for all j < n_commit).

Oracle_exec is handled by re-running the lane in-process (rather than
reading a stored JSON), because our earlier oracle_exec log didn't save
per-round draft_tokens. Matched RNG + pretrained drafter in eval mode
means the second rollout is bit-identical to the first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.commit import commit_threshold
from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig, derive_seed
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N, test_split


EPS = 1e-10


METHODS_ONLINE = {
    "frozen_1A": {
        "online_dir": "outputs/phase12_1a1b_11830736/acc_1a",
        "drafter_state": None,
    },
    "lazy_jnt": {
        "online_dir": "outputs/phase15_jnt1a_11859744/online",
        "drafter_state": "checkpoints/acc_jnt_1a/drafter.pt",
    },
    "live_jnt_5ep": {
        "online_dir": "outputs/phase16_liveq2_11860043/online",
        "drafter_state": "checkpoints/acc_jnt_1a_liveq2/drafter.pt",
    },
    "live_jnt_10ep": {
        "online_dir": "outputs/phase17_liveq2_long_11860808/online",
        "drafter_state": "checkpoints/acc_jnt_1a_liveq2_long/drafter.pt",
    },
}

TAUS = (0.5, 0.7, 0.9)


def _load_protocol(path: Path) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _load_drafter(protocol, dtype, device, state_path=None):
    from accpre.models.drafter_mdlm import MDLMDrafter
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    if state_path is not None:
        state = torch.load(state_path, map_location=device)
        drafter.model.load_state_dict(state)
        print(f"  loaded drafter state from {state_path}")
    drafter.model.eval()
    return drafter


@torch.no_grad()
def _success_from_round(
    drafter, verifier, running: torch.Tensor, seed: int,
    gamma: int, T: int, protocol: ProtocolConfig, device,
    committed_tokens: List[int],
) -> Tuple[int, int, bool]:
    """Returns (n_commit, n_success, all_pass)."""
    prefix_len = int(running.shape[0])
    draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
    draft_tokens, draft_log_probs = drafter.draft(
        prefix_ids=running, gamma=gamma, T=T,
        temperature=protocol.temperature, q_mode=protocol.q_mode,
        generator=draft_rng,
    )
    n_commit = len(committed_tokens)
    if n_commit == 0:
        return 0, 0, True
    # Under matched RNG + same drafter state, replay should yield the
    # same draft_tokens the online run logged. We don't assert-hard,
    # but print a warning if they diverge.
    replayed = [int(x) for x in draft_tokens[:n_commit].cpu().tolist()]
    if replayed != [int(x) for x in committed_tokens]:
        # Note: method's committed_tokens come FROM draft_tokens (cheap
        # lane), so this is a strong replay check — if it fails,
        # something is off.
        print(
            f"  WARN: replay mismatch n_commit={n_commit}  "
            f"stored={committed_tokens[:5]}  replay={replayed[:5]}"
        )
    candidate = torch.cat([running, draft_tokens])
    target_log_probs = verifier.score(candidate).to(torch.float32)
    idx = torch.arange(gamma, device=device)
    q_j = draft_log_probs[idx, draft_tokens.long()].exp().clamp(min=EPS)
    p_j = target_log_probs[
        prefix_len - 1 + idx, draft_tokens.long()
    ].exp().clamp(min=EPS)
    Q2 = torch.minimum(torch.ones_like(q_j), p_j / q_j)

    accept_rng = _make_generator(device, seed ^ protocol.accept_salt)
    U = torch.rand(gamma, generator=accept_rng, device=device)

    Q2_c = Q2[:n_commit]
    U_c = U[:n_commit]
    successes = (U_c < Q2_c).to(torch.int64)
    n_success = int(successes.sum().item())
    all_pass = bool(n_success == n_commit)
    return n_commit, n_success, all_pass


def score_v_free_method(
    drafter, verifier, online_dir: Path, protocol: ProtocolConfig,
    device, tau: float,
) -> Dict:
    """Replay one V-free method × τ from the stored online JSON."""
    tau_tag = f"0p{int(tau * 10)}"
    json_path = online_dir / f"online_acceptance_tau_{tau_tag}.json"
    with open(json_path) as f:
        d = json.load(f)
    pool = test_split()

    agg = {
        "n_tokens_total": 0, "n_tokens_success": 0,
        "n_rounds_total": 0, "n_rounds_all_pass": 0,
        "sum_round_success_rate": 0.0,
    }
    for prompt in d["online_lane"]["per_prompt"]:
        p_idx = int(prompt["prompt_idx"])
        local_idx = p_idx - (TRAIN_N + VAL_N)
        initial_prefix = [int(x) for x in pool[local_idx][0].tolist()]
        running_list = list(initial_prefix)
        for round_log in prompt["rounds"]:
            running = torch.tensor(
                running_list, dtype=torch.long, device=device,
            )
            seed = int(round_log["round_rng_seed"])
            gamma = int(round_log["gamma"])
            T = int(round_log["T"])
            committed = [int(x) for x in round_log["committed_tokens"]]
            n_commit, n_success, all_pass = _success_from_round(
                drafter, verifier, running, seed, gamma, T,
                protocol, device, committed,
            )
            agg["n_rounds_total"] += 1
            agg["n_tokens_total"] += n_commit
            agg["n_tokens_success"] += n_success
            if n_commit > 0:
                agg["sum_round_success_rate"] += n_success / n_commit
                if all_pass:
                    agg["n_rounds_all_pass"] += 1
            running_list.extend(committed)

    n_rounds = max(agg["n_rounds_total"], 1)
    return {
        "token_success_rate":  agg["n_tokens_success"] / max(agg["n_tokens_total"], 1),
        "round_mean_success":  agg["sum_round_success_rate"] / n_rounds,
        "round_all_pass_rate": agg["n_rounds_all_pass"] / n_rounds,
        "n_tokens_total":      agg["n_tokens_total"],
        "n_tokens_success":    agg["n_tokens_success"],
        "n_rounds_total":      agg["n_rounds_total"],
    }


def score_oracle_exec(
    drafter, verifier, protocol, device, tau: float,
    n_prompts: int, max_new_tokens: int, gamma: int = 8, T: int = 2,
) -> Dict:
    """Re-run oracle_exec lane at this τ with verifier-success tracking."""
    pool = test_split()[:n_prompts]
    prompt_indices = list(
        range(TRAIN_N + VAL_N, TRAIN_N + VAL_N + n_prompts)
    )
    MAX_CTX = protocol.max_verifier_ctx
    agg = {
        "n_tokens_total": 0, "n_tokens_success": 0,
        "n_rounds_total": 0, "n_rounds_all_pass": 0,
        "sum_round_success_rate": 0.0,
    }
    for i, (prefix_ids, _text) in enumerate(pool):
        p_idx = prompt_indices[i]
        running = prefix_ids.to(device).clone()
        prefix_start = int(running.shape[0])
        round_idx = 0
        while (running.shape[0] - prefix_start) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_start)
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(int(gamma), int(remaining), int(ctx_room))
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
            Q2_c = Q2[:n_commit]
            U_c = U[:n_commit]
            successes = (U_c < Q2_c).to(torch.int64)
            n_success = int(successes.sum().item())
            all_pass = bool(n_success == n_commit)

            agg["n_rounds_total"] += 1
            agg["n_tokens_total"] += n_commit
            agg["n_tokens_success"] += n_success
            if n_commit > 0:
                agg["sum_round_success_rate"] += n_success / n_commit
                if all_pass:
                    agg["n_rounds_all_pass"] += 1

            committed = draft_tokens[:n_commit]
            running = torch.cat([running, committed])
            round_idx += 1

    n_rounds = max(agg["n_rounds_total"], 1)
    return {
        "token_success_rate":  agg["n_tokens_success"] / max(agg["n_tokens_total"], 1),
        "round_mean_success":  agg["sum_round_success_rate"] / n_rounds,
        "round_all_pass_rate": agg["n_rounds_all_pass"] / n_rounds,
        "n_tokens_total":      agg["n_tokens_total"],
        "n_tokens_success":    agg["n_tokens_success"],
        "n_rounds_total":      agg["n_rounds_total"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_prompts_oracle", type=int, default=10)
    ap.add_argument("--max_new_tokens_oracle", type=int, default=64)
    ap.add_argument("--phase20_dir", type=str,
                    default="outputs/phase20_eval_11869681")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = _load_protocol(_REPO_ROOT / "configs/protocol.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    from accpre.models.verifier_gpt2 import GPT2Verifier
    print("[phase21] loading verifier...")
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    print(f"[phase21] will evaluate {len(METHODS_ONLINE)} V-free methods + oracle_exec, τ={TAUS}")

    # Predictor-level quick reference from phase20 (offline CF@1, tok/s, ΔNLL).
    with open(_REPO_ROOT / args.phase20_dir / "system_table.json") as f:
        nll_data = json.load(f)
    with open(_REPO_ROOT / args.phase20_dir / "predictor_table.json") as f:
        pred_rows = json.load(f)
    off_cf = {row["method"]: row["offline_per_tau"] for row in pred_rows}

    results: Dict = {"methods": {}}

    for name, cfg in METHODS_ONLINE.items():
        print(f"\n[phase21] === {name} ===")
        state_path = (
            None if cfg["drafter_state"] is None
            else str(_REPO_ROOT / cfg["drafter_state"])
        )
        drafter = _load_drafter(protocol, dtype, device, state_path)
        results["methods"][name] = {}
        for tau in TAUS:
            t0 = time.time()
            r = score_v_free_method(
                drafter, verifier,
                _REPO_ROOT / cfg["online_dir"], protocol, device, tau,
            )
            dt = time.time() - t0
            r["elapsed_s"] = dt
            results["methods"][name][str(tau)] = r
            print(
                f"  τ={tau}  token_succ={r['token_success_rate']:.4f}  "
                f"round_mean={r['round_mean_success']:.4f}  "
                f"all_pass={r['round_all_pass_rate']:.4f}  "
                f"(n_tok={r['n_tokens_total']}, n_rounds={r['n_rounds_total']}, {dt:.1f}s)"
            )
        # Free the drafter; allocator will be reused on next load.
        del drafter

    # Oracle_exec — uses pretrained drafter, recomputed online.
    print("\n[phase21] === oracle_exec ===")
    drafter = _load_drafter(protocol, dtype, device, state_path=None)
    results["methods"]["oracle_exec"] = {}
    for tau in TAUS:
        t0 = time.time()
        r = score_oracle_exec(
            drafter, verifier, protocol, device, tau,
            n_prompts=int(args.n_prompts_oracle),
            max_new_tokens=int(args.max_new_tokens_oracle),
        )
        dt = time.time() - t0
        r["elapsed_s"] = dt
        results["methods"]["oracle_exec"][str(tau)] = r
        print(
            f"  τ={tau}  token_succ={r['token_success_rate']:.4f}  "
            f"round_mean={r['round_mean_success']:.4f}  "
            f"all_pass={r['round_all_pass_rate']:.4f}  "
            f"(n_tok={r['n_tokens_total']}, n_rounds={r['n_rounds_total']}, {dt:.1f}s)"
        )

    # Emit combined table.
    lines: List[str] = []
    lines.append("# Phase 21 — verifier-success metrics\n\n")
    lines.append(
        "`token_succ` = fraction of committed tokens with `U < Q2` "
        "(strict Leviathan accept under matched RNG). "
        "`round_mean` = mean over rounds of within-round success rate. "
        "`all_pass` = fraction of rounds where every committed token passes. "
        "`offCF1`, `tok/s`, `ΔNLL` are from Phase 20 (same runs).\n\n"
    )
    lines.append("```\n")
    lines.append(
        f"  {'method':<14}{'τ':>6}{'offCF1':>8}{'tok/s':>8}{'ΔNLL':>8}"
        f"{'tok_succ':>10}{'rnd_mean':>10}{'all_pass':>10}\n"
    )
    order = ["frozen_1A", "lazy_jnt", "live_jnt_5ep", "live_jnt_10ep",
             "oracle_exec"]
    for name in order:
        if name not in results["methods"]:
            continue
        for tau in TAUS:
            r = results["methods"][name].get(str(tau))
            if r is None:
                continue
            if name == "oracle_exec":
                # Use Phase 14 offline CF@1 ceiling + Phase 20 oracle exec NLL/tok_s.
                off_cfv = {
                    0.5: 0.8360, 0.7: 0.8151, 0.9: 0.7781,
                }[tau]
                tok_s = {
                    0.5: 22.34, 0.7: 22.39, 0.9: 17.47,
                }[tau]
                dnll = {
                    0.5: +0.1512, 0.7: -0.2921, 0.9: -0.0500,
                }[tau]
            else:
                off = off_cf.get(name, {}).get(str(float(tau)), {})
                off_cfv = off.get("offline_cf", float("nan"))
                m20 = nll_data["methods"].get(name, {}).get(str(tau), {})
                tok_s = m20.get("tok_s_mean", float("nan"))
                dnll = m20.get("delta_nll_vs_strict", float("nan"))
            lines.append(
                f"  {name:<14}{float(tau):>6.1f}"
                f"{off_cfv:>8.4f}{tok_s:>8.2f}{dnll:>+8.3f}"
                f"{r['token_success_rate']:>10.4f}"
                f"{r['round_mean_success']:>10.4f}"
                f"{r['round_all_pass_rate']:>10.4f}\n"
            )
    lines.append("```\n")
    report = "".join(lines)

    with open(out_dir / "verifier_success.md", "w") as f:
        f.write(report)
    with open(out_dir / "verifier_success.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n" + report)
    print(f"[phase21] wrote {out_dir}/verifier_success.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
