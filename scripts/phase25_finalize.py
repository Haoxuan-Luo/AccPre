"""Phase 25 finalize — run only the offline NLL + verifier-success passes
on existing decode JSONs from `phase25_longhorizon.py`.

The original Phase-25 job (11979860) completed all 16 decode lanes
successfully, then crashed during the first NLL forward because the
strict lane produced a 1025-token trajectory (20 prompts × 64 tok the
lane always stayed under 1024; at 1024 new tokens strict's
`L + 1`-per-round commit can overflow the context by one).

This script:
  1. Loads every `online_*.json` in the decode dir.
  2. Truncates any trajectory whose length exceeds GPT-2-XL's 1024
     context (drops the last token, decrements `n_new_tokens` by 1).
     Only the strict lane should need this.
  3. Runs the NLL pass + verifier-success replay pass (same code as
     phase25_longhorizon), temp-aware (argmax rule under temp=0).
  4. Emits unified_table.{md,json}, pareto PNGs, timing_breakdown.json.

Decode timings are preserved from the original decode pass (parsed from
logs). The finalize script only measures the offline passes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Reuse pieces from phase25.
import importlib.util as _iu
_p25_spec = _iu.spec_from_file_location(
    "p25", str(_REPO_ROOT / "scripts/phase25_longhorizon.py"),
)
p25 = _iu.module_from_spec(_p25_spec)
_p25_spec.loader.exec_module(p25)


GPT2_XL_MAX = 1024


def _trim_trajectories_inplace(lane: Dict, cap: int = GPT2_XL_MAX) -> int:
    """If any prompt in `lane` has len(generated_ids) > cap, drop tail."""
    fixed = 0
    for p in lane.get("per_prompt", []):
        gen = p["generated_ids"]
        if len(gen) > cap:
            over = len(gen) - cap
            p["generated_ids"] = gen[:cap]
            p["n_new_tokens"] = int(p["n_new_tokens"]) - over
            fixed += 1
    return fixed


def _parse_decode_timings(log_path: Path) -> Dict[str, float]:
    """Extract per-lane decode elapsed seconds from the phase25 log.
    Falls back to {} on any parse error.
    """
    if not log_path.exists():
        return {}
    timings: Dict[str, float] = {}
    # Known patterns:
    #   [phase25] strict: tok/s=X  elapsed=Ys
    #   [phase25]   <method> τ=T: tok/s=X  elapsed=Ys
    re_strict = re.compile(
        r"\[phase25\] strict: tok/s=([\d.]+)\s+elapsed=([\d.]+)s"
    )
    re_pred = re.compile(
        r"\[phase25\]\s+(\S+)\s+τ=([\d.]+):\s+tok/s=([\d.]+)\s+"
        r"elapsed=([\d.]+)s"
    )
    with open(log_path) as f:
        for line in f:
            m = re_strict.search(line)
            if m:
                timings["strict"] = float(m.group(2))
                continue
            m = re_pred.search(line)
            if m:
                name = m.group(1); tau = float(m.group(2))
                timings[f"{name}|{tau}"] = float(m.group(4))
    return timings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decode_dir", type=str, required=True,
                    help="Existing phase25 output dir with online_*.json")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="If set, write tables/plots here. Defaults to decode_dir.")
    ap.add_argument("--protocol_yaml", type=str,
                    default="configs/protocol_greedy.yaml")
    ap.add_argument("--log_file", type=str, default=None,
                    help="Optional phase25 .out file to parse decode timings "
                         "from, so the timing breakdown carries the decode "
                         "numbers from the original run.")
    args = ap.parse_args()

    decode_dir = Path(args.decode_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else decode_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = p25.load_protocol(_REPO_ROOT / args.protocol_yaml)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    print("[p25_fin] loading models...")
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    pretrained_cpu_state = p25.stash_state(drafter)
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    # --- load decode JSONs & reconstruct prompt_indices / pool_by_idx ---
    strict_path = decode_dir / "online_strict.json"
    if not strict_path.exists():
        print(f"[p25_fin] FATAL: {strict_path} missing"); return 1
    with open(strict_path) as f:
        strict_lane = json.load(f)

    prompt_indices = [int(p["prompt_idx"]) for p in strict_lane["per_prompt"]]
    n_prompts = len(prompt_indices)
    print(f"[p25_fin] {n_prompts} prompts: {prompt_indices}")

    # Trim strict (1025 → 1024).
    n_fixed = _trim_trajectories_inplace(strict_lane)
    if n_fixed:
        print(f"[p25_fin] trimmed {n_fixed} strict trajectories to ≤{GPT2_XL_MAX}")
        with open(strict_path, "w") as f:
            json.dump(strict_lane, f, indent=2, default=str)

    # Reconstruct pool_by_idx from the original OWT prompts (deterministic).
    from accpre.data.prompts import load_owt_prompts
    from accpre.data.splits import POOL_SIZE, PREFIX_LEN, PROMPT_SEED
    pool = load_owt_prompts(
        n_prompts=POOL_SIZE, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )
    pool_by_idx: Dict[int, torch.Tensor] = {}
    for p_idx in prompt_indices:
        pool_by_idx[p_idx] = pool[p_idx][0]

    # Load predictor + oracle lanes from disk.
    all_lanes: Dict = {"strict": strict_lane}
    for name, _ckpt, _ds, taus in p25.METHODS:
        for tau in taus:
            tag = p25.fmt_tau_tag(tau)
            path = decode_dir / f"online_{name}_tau_{tag}.json"
            if not path.exists():
                print(f"[p25_fin] missing {path}, skipping"); continue
            with open(path) as f:
                d = json.load(f)
            nfix = _trim_trajectories_inplace(d)
            if nfix:
                print(f"[p25_fin] trimmed {nfix} trajectories in {path.name}")
                with open(path, "w") as f2:
                    json.dump(d, f2, indent=2, default=str)
            all_lanes[(name, tau)] = d
    for tau in p25.ORACLE_TAUS:
        tag = p25.fmt_tau_tag(tau)
        path = decode_dir / f"online_oracle_tau_{tag}.json"
        if not path.exists():
            print(f"[p25_fin] missing {path}, skipping"); continue
        with open(path) as f:
            d = json.load(f)
        _trim_trajectories_inplace(d)
        all_lanes[("oracle_exec", tau)] = d

    timings: Dict[str, Dict] = {
        "decode": {}, "nll": {}, "replay": {},
    }
    if args.log_file:
        timings["decode"] = _parse_decode_timings(Path(args.log_file))
        print(f"[p25_fin] parsed decode timings from {args.log_file}: "
              f"{len(timings['decode'])} entries")

    # --- NLL pass ---
    print("\n[p25_fin] === NLL scoring ===")
    method_nll: Dict = {}
    t0 = time.time()
    nll_strict = p25.lane_nll(verifier, strict_lane["per_prompt"], device)
    timings["nll"]["strict"] = time.time() - t0
    method_nll["strict"] = nll_strict
    print(f"[p25_fin]   strict NLL={nll_strict:.4f}  "
          f"({timings['nll']['strict']:.1f}s)")
    for name, _c, _d, taus in p25.METHODS:
        for tau in taus:
            key = (name, tau)
            if key not in all_lanes:
                continue
            t0 = time.time()
            nll = p25.lane_nll(verifier, all_lanes[key]["per_prompt"], device)
            timings["nll"][f"{name}|{tau}"] = time.time() - t0
            method_nll[key] = nll
            print(f"[p25_fin]   {name} τ={tau}: NLL={nll:.4f}  "
                  f"({timings['nll'][f'{name}|{tau}']:.1f}s)")
    for tau in p25.ORACLE_TAUS:
        key = ("oracle_exec", tau)
        if key not in all_lanes:
            continue
        t0 = time.time()
        nll = p25.lane_nll(verifier, all_lanes[key]["per_prompt"], device)
        timings["nll"][f"oracle_exec|{tau}"] = time.time() - t0
        method_nll[key] = nll
        print(f"[p25_fin]   oracle_exec τ={tau}: NLL={nll:.4f}  "
              f"({timings['nll'][f'oracle_exec|{tau}']:.1f}s)")

    # --- verifier-success replay ---
    print("\n[p25_fin] === verifier-success replay ===")
    vsucc: Dict = {}
    for name, _ckpt, dstate, taus in p25.METHODS:
        if dstate is not None:
            state = torch.load(
                str(_REPO_ROOT / dstate), map_location=device,
            )
            drafter.model.load_state_dict(state)
        else:
            p25.restore_state(drafter, pretrained_cpu_state, device)
        drafter.model.eval()
        for tau in taus:
            key = (name, tau)
            if key not in all_lanes:
                continue
            t0 = time.time()
            r = p25.replay_verifier_success(
                drafter, verifier,
                all_lanes[key]["per_prompt"],
                protocol, pool_by_idx, device,
            )
            timings["replay"][f"{name}|{tau}"] = time.time() - t0
            vsucc[key] = r
            print(
                f"[p25_fin]   {name} τ={tau}: "
                f"tok_succ={r['token_success_rate']:.4f}  "
                f"rnd_mean={r['round_mean_success']:.4f}  "
                f"all_pass={r['round_all_pass_rate']:.4f}  "
                f"({timings['replay'][f'{name}|{tau}']:.1f}s)"
            )
    for tau in p25.ORACLE_TAUS:
        key = ("oracle_exec", tau)
        if key not in all_lanes:
            continue
        lane = all_lanes[key]
        tot_n = sum(p["n_tokens_total"] for p in lane["per_prompt"])
        tok_succ_num = sum(
            p["tok_succ"] * p["n_tokens_total"] for p in lane["per_prompt"]
        )
        rnd_mean = sum(p["rnd_mean"] for p in lane["per_prompt"]) / \
            max(len(lane["per_prompt"]), 1)
        all_pass = sum(p["all_pass"] for p in lane["per_prompt"]) / \
            max(len(lane["per_prompt"]), 1)
        vsucc[key] = {
            "token_success_rate":  tok_succ_num / max(tot_n, 1),
            "round_mean_success":  rnd_mean,
            "round_all_pass_rate": all_pass,
            "n_tokens_total":      tot_n,
        }

    # --- MAE ---
    pred_mae: Dict = {}
    for name, ckpt_dir, _d, _taus in p25.METHODS:
        m = p25.compute_predictor_mae(_REPO_ROOT / ckpt_dir)
        pred_mae[name] = m
        print(
            f"[p25_fin] MAE {name}: overall={m['MAE_overall']:.4f}  "
            f"acc={m['MAE_acc']:.4f}  wgt={m['MAE_wgt']:.4f}"
        )

    # --- rows ---
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
    for tau in p25.ORACLE_TAUS:
        rows.append(_row("oracle_exec", tau))
    for name, _c, _d, taus in p25.METHODS:
        for tau in taus:
            if (name, tau) not in all_lanes:
                continue
            rows.append(_row(name, tau))

    # Read the max_new_tokens from the strict lane (inferred from n_new).
    mnt_guess = max(
        int(p["n_new_tokens"]) for p in strict_lane["per_prompt"]
    )
    main_md = p25.fmt_main_table(rows, int(n_prompts), int(mnt_guess))
    with open(out_dir / "unified_table.md", "w") as f:
        f.write(main_md)
    with open(out_dir / "unified_table.json", "w") as f:
        json.dump({
            "rows": rows, "strict_nll": nll_strict,
            "prompt_indices": prompt_indices,
            "n_prompts": n_prompts,
            "max_new_tokens_inferred": int(mnt_guess),
            "temperature": float(protocol.temperature),
        }, f, indent=2, default=str)
    print(main_md)

    p25._maybe_pareto(
        rows,
        out_dir / "pareto_delta_nll_tok_s.png",
        out_dir / "pareto_tok_succ_tok_s.png",
    )

    sums = {k: sum(v for v in stage.values()) for k, stage in timings.items()}
    sums["total"] = sums["decode"] + sums["nll"] + sums["replay"]
    with open(out_dir / "timing_breakdown.json", "w") as f:
        json.dump({
            "per_method_s": timings,
            "totals_s": sums,
            "note": "decode timings parsed from original job log; nll/"
                    "replay measured by this finalize run.",
        }, f, indent=2, default=str)
    print(
        f"\n[p25_fin] timing totals (s):  "
        f"decode={sums['decode']:.1f}  "
        f"nll={sums['nll']:.1f}  "
        f"replay={sums['replay']:.1f}  "
        f"TOTAL={sums['total']:.1f}  ({sums['total']/60:.1f} min)"
    )
    print("\n[p25_fin] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
