"""Phase 25 finalize — ONE lane at a time.

Recovers NLL (and, for predictor lanes, verifier-success replay) from
an existing decode JSON in `outputs/phase25_lh_<job>/online_*.json`.
Each invocation processes exactly one lane so a single call fits inside
a short wallclock (≤ 2h on the 1g.10gb MIG for the heaviest predictor
lane; ≤ 5 min for strict/oracle NLL-only).

Usage:
  python3 -u scripts/phase25_finalize_lane.py \\
      --decode_dir outputs/phase25_lh_11979860 \\
      --out_dir    outputs/phase25_lh_11979860/lanes \\
      --method     strict
  python3 -u scripts/phase25_finalize_lane.py \\
      --decode_dir outputs/phase25_lh_11979860 \\
      --out_dir    outputs/phase25_lh_11979860/lanes \\
      --method     frozen_1A --tau 0.5

Per-call output: `<out_dir>/lane_<method>[_tau_<τ>].json` with:
  method, tau, tok_s_mean, NLL, n_tokens_evaluated,
  tok_succ, rnd_mean, all_pass, n_tokens_total, timings_s.

All progress `print`s use `flush=True`; invoke Python with `-u` in the
slurm job so partial output survives SIGTERM.

Strict-trim fix: every loaded JSON passes through
`_trim_trajectories_inplace` (caps each trajectory at GPT-2-XL's 1024
token limit). This is the same patch as `scripts/phase25_finalize.py`;
re-applying is idempotent.
"""

from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the phase25 longhorizon helpers (fmt_tau_tag, METHODS,
# score_nll_on_new, stash_state, restore_state, etc.).
_p25_spec = _iu.spec_from_file_location(
    "p25", str(_REPO_ROOT / "scripts/phase25_longhorizon.py"),
)
p25 = _iu.module_from_spec(_p25_spec)
_p25_spec.loader.exec_module(p25)


GPT2_XL_MAX = 1024


def _trim_trajectories_inplace(lane: Dict, cap: int = GPT2_XL_MAX) -> int:
    """Cap every generated trajectory at `cap` tokens. Idempotent."""
    fixed = 0
    for p in lane.get("per_prompt", []):
        gen = p["generated_ids"]
        if len(gen) > cap:
            over = len(gen) - cap
            p["generated_ids"] = gen[:cap]
            p["n_new_tokens"] = int(p["n_new_tokens"]) - over
            fixed += 1
    return fixed


def _method_drafter_override(method: str) -> Optional[str]:
    """Return the drafter state path (if any) for a predictor method."""
    for name, _ckpt, dstate, _taus in p25.METHODS:
        if name == method:
            return dstate
    return None


def _fmt_tau_tag(tau: Optional[float]) -> str:
    if tau is None:
        return ""
    return f"_tau_{p25.fmt_tau_tag(float(tau))}"


def _replay_with_progress(
    drafter, verifier, lane_per_prompt, protocol,
    pool_by_idx: Dict[int, torch.Tensor], device,
) -> Dict:
    """Same math as p25.replay_verifier_success, with per-prompt flushes."""
    from accpre.core.draft_verify import _make_generator
    EPS = p25.EPS
    n_tok_total = 0
    n_tok_succ = 0
    n_round_total = 0
    n_round_all_pass = 0
    sum_round_succ_rate = 0.0
    n_prompts = len(lane_per_prompt)
    for pi, prompt in enumerate(lane_per_prompt):
        p_idx = int(prompt["prompt_idx"])
        running = [int(x) for x in pool_by_idx[p_idx].tolist()]
        n_rounds = len(prompt["rounds"])
        t0 = time.time()
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
        print(
            f"[lane]   replay prompt {pi + 1}/{n_prompts} "
            f"(idx={p_idx}, rounds={n_rounds}, {time.time() - t0:.1f}s)  "
            f"running tok_succ={n_tok_succ / max(n_tok_total, 1):.4f}",
            flush=True,
        )
    return {
        "token_success_rate":  n_tok_succ / max(n_tok_total, 1),
        "round_mean_success":  sum_round_succ_rate / max(n_round_total, 1),
        "round_all_pass_rate": n_round_all_pass / max(n_round_total, 1),
        "n_tokens_total":      int(n_tok_total),
        "n_rounds_total":      int(n_round_total),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol_yaml", default="configs/protocol_greedy.yaml")
    ap.add_argument(
        "--method", required=True,
        choices=["strict", "oracle_exec",
                 "frozen_1A", "live_jnt_10ep",
                 "frozen_dep", "live_jnt_dep"],
    )
    ap.add_argument("--tau", type=float, default=None,
                    help="Required unless --method strict.")
    args = ap.parse_args()

    decode_dir = Path(args.decode_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "strict":
        if args.tau is not None:
            print("[lane] WARN: --tau ignored for strict", flush=True)
        args.tau = None
        lane_path = decode_dir / "online_strict.json"
    elif args.method == "oracle_exec":
        if args.tau is None:
            print("[lane] FATAL: oracle_exec requires --tau", flush=True)
            return 2
        lane_path = (
            decode_dir / f"online_oracle_tau_{p25.fmt_tau_tag(args.tau)}.json"
        )
    else:
        if args.tau is None:
            print(f"[lane] FATAL: {args.method} requires --tau", flush=True)
            return 2
        lane_path = (
            decode_dir
            / f"online_{args.method}_tau_{p25.fmt_tau_tag(args.tau)}.json"
        )

    if not lane_path.exists():
        print(f"[lane] FATAL: decode JSON not found: {lane_path}", flush=True)
        return 1

    print(f"[lane] method={args.method} tau={args.tau}", flush=True)
    print(f"[lane] decode JSON: {lane_path}", flush=True)

    protocol = p25.load_protocol(_REPO_ROOT / args.protocol_yaml)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    print(f"[lane] device={device} dtype={protocol.dtype}", flush=True)

    with open(lane_path) as f:
        lane = json.load(f)

    n_fix = _trim_trajectories_inplace(lane)
    if n_fix:
        print(f"[lane] trimmed {n_fix} trajectories to ≤{GPT2_XL_MAX}",
              flush=True)
        with open(lane_path, "w") as f:
            json.dump(lane, f, indent=2, default=str)

    n_prompts = len(lane["per_prompt"])
    print(f"[lane] n_prompts={n_prompts}  "
          f"tok_s_mean={lane['tok_s_mean']:.4f}", flush=True)

    from accpre.models.verifier_gpt2 import GPT2Verifier
    t0 = time.time()
    print("[lane] loading verifier (GPT-2-XL)...", flush=True)
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )
    print(f"[lane] verifier loaded in {time.time() - t0:.1f}s", flush=True)

    is_predictor = args.method in {m[0] for m in p25.METHODS}
    drafter = None
    if is_predictor:
        from accpre.models.drafter_mdlm import MDLMDrafter
        t0 = time.time()
        print("[lane] loading drafter (MDLM)...", flush=True)
        drafter = MDLMDrafter(
            model_name=protocol.drafter_model, device=device, dtype=dtype,
        )
        drafter.model.eval()
        dstate = _method_drafter_override(args.method)
        if dstate is not None:
            print(f"[lane] overriding drafter state <- {dstate}", flush=True)
            state = torch.load(
                str(_REPO_ROOT / dstate), map_location=device,
            )
            drafter.model.load_state_dict(state)
        drafter.model.eval()
        print(f"[lane] drafter ready in {time.time() - t0:.1f}s", flush=True)

    # ---------------- NLL pass ----------------
    print("[lane] === NLL pass ===", flush=True)
    t0 = time.time()
    total_sum = 0.0
    total_n = 0
    for i, prompt in enumerate(lane["per_prompt"]):
        gen = [int(x) for x in prompt["generated_ids"]]
        n_prefix = len(gen) - int(prompt["n_new_tokens"])
        s, n = p25.score_nll_on_new(verifier, gen, n_prefix, device)
        total_sum += s
        total_n += n
        print(
            f"[lane]   nll prompt {i + 1}/{n_prompts} "
            f"(idx={int(prompt['prompt_idx'])}, n_new={n}): "
            f"nll_sum={s:.2f}  running_NLL="
            f"{total_sum / max(total_n, 1):.4f}",
            flush=True,
        )
    nll = total_sum / max(total_n, 1)
    nll_elapsed = time.time() - t0
    print(
        f"[lane] NLL={nll:.4f}  n_tok={total_n}  "
        f"elapsed={nll_elapsed:.1f}s",
        flush=True,
    )

    # ---------------- replay / success ----------------
    replay_elapsed = 0.0
    vsucc: Optional[Dict] = None
    if is_predictor:
        print("[lane] === verifier-success replay ===", flush=True)
        from accpre.data.prompts import load_owt_prompts
        from accpre.data.splits import POOL_SIZE, PREFIX_LEN, PROMPT_SEED
        pool = load_owt_prompts(
            n_prompts=POOL_SIZE, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
        )
        pool_by_idx: Dict[int, torch.Tensor] = {}
        for p_row in lane["per_prompt"]:
            p_idx = int(p_row["prompt_idx"])
            pool_by_idx[p_idx] = pool[p_idx][0]

        t0 = time.time()
        vsucc = _replay_with_progress(
            drafter, verifier, lane["per_prompt"], protocol,
            pool_by_idx, device,
        )
        replay_elapsed = time.time() - t0
        print(
            f"[lane] replay: tok_succ={vsucc['token_success_rate']:.4f}  "
            f"rnd_mean={vsucc['round_mean_success']:.4f}  "
            f"all_pass={vsucc['round_all_pass_rate']:.4f}  "
            f"(n_tok_committed={vsucc['n_tokens_total']}  "
            f"elapsed={replay_elapsed:.1f}s)",
            flush=True,
        )
    elif args.method == "oracle_exec":
        pp = lane["per_prompt"]
        tot_n = sum(int(p["n_tokens_total"]) for p in pp)
        tok_succ_num = sum(
            p["tok_succ"] * int(p["n_tokens_total"]) for p in pp
        )
        rnd_mean = (
            sum(p["rnd_mean"] for p in pp) / max(len(pp), 1)
        )
        all_pass = (
            sum(p["all_pass"] for p in pp) / max(len(pp), 1)
        )
        vsucc = {
            "token_success_rate":  tok_succ_num / max(tot_n, 1),
            "round_mean_success":  rnd_mean,
            "round_all_pass_rate": all_pass,
            "n_tokens_total":      int(tot_n),
        }
        print(
            f"[lane] oracle stored-metrics: "
            f"tok_succ={vsucc['token_success_rate']:.4f}  "
            f"rnd_mean={vsucc['round_mean_success']:.4f}  "
            f"all_pass={vsucc['round_all_pass_rate']:.4f}  "
            f"(n_tok_committed={vsucc['n_tokens_total']})",
            flush=True,
        )

    # ---------------- write lane row ----------------
    row = {
        "method":             args.method,
        "tau":                None if args.tau is None else float(args.tau),
        "tok_s_mean":         float(lane["tok_s_mean"]),
        "NLL":                float(nll),
        "n_tokens_evaluated": int(total_n),
        "timings_s": {
            "nll":    float(nll_elapsed),
            "replay": float(replay_elapsed),
        },
        "decode_json": str(lane_path),
    }
    if vsucc is not None:
        row["tok_succ"]       = float(vsucc["token_success_rate"])
        row["rnd_mean"]       = float(vsucc["round_mean_success"])
        row["all_pass"]       = float(vsucc["round_all_pass_rate"])
        row["n_tokens_total"] = int(vsucc["n_tokens_total"])

    out_path = out_dir / f"lane_{args.method}{_fmt_tau_tag(args.tau)}.json"
    with open(out_path, "w") as f:
        json.dump(row, f, indent=2, default=str)
    print(f"[lane] wrote {out_path}", flush=True)
    print("[lane] DONE.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
