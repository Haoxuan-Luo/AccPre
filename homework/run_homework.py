"""Homework experiment — 4-method throughput-quality comparison on OWT.

Runs four no-predictor decoding rules on the same OpenWebText test prompts
(prompt indices 60..79 by default; 32-token prefix), at gamma=15, T=2,
max_new_tokens=1024, and the protocol's temperature (greedy by default):

  strict             — Leviathan strict speculative decoding (1 lane)
  lossy_l            — lenience-l Bernoulli accept (5 lanes by default)
  threshold_lossy    — deterministic accept while r_j >= tau (9 lanes)
  confidence_lossy   — deterministic accept while prod r_j >= tau (9 lanes)

For every lane:
  * online decode produces a JSON with per-prompt rounds, generated tokens,
    inline tok_succ / rnd_mean / all_pass, and tok/s.
  * an offline NLL pass scores each generated continuation under the
    verifier (GPT-2 XL).
  * delta_nll is reported relative to the strict baseline.

Default outputs (under --out_dir, default `homework/results`):
  online_<method>[_param].json     one file per lane
  results.csv                      flat per-lane rows
  unified.{md,json}                pretty summary table

The exact prompts used are saved to --prompts_cache (default
`homework/data/prompts_owt_20.json`); on subsequent runs the script reads
the prompts from that cache file (no OpenWebText download needed).

Smoke usage (1 prompt, 64 tokens, single tau):
    python homework/run_homework.py \
        --n_prompts 1 --max_new_tokens 64 \
        --lossy_ls 1.0 --thresh_taus 0.5 --conf_taus 0.5 \
        --out_dir homework/results_smoke

Full usage (matches scripts/run_homework.slurm):
    python homework/run_homework.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Resolve the homework folder so the local `specdiff/` and helper modules
# are importable regardless of the working directory the script is
# launched from.
_HW_DIR = Path(__file__).resolve().parent
if str(_HW_DIR) not in sys.path:
    sys.path.insert(0, str(_HW_DIR))

from homework_utils import (
    compute_nll_and_success,
    load_owt_test_prompts,
    load_protocol,
    run_lossy_lane_one,
    run_strict_lane,
    write_prompts_dump,
)
import lanes_homework as hw


def _fmt_param(v: float) -> str:
    return f"{v:g}".replace(".", "p")


def _row_metrics(lane: Dict, nll: float) -> Dict:
    """Round/token-weighted aggregation across per-prompt rows."""
    pp = lane.get("per_prompt", [])
    have_inline = all(
        ("tok_succ" in p and "rnd_mean" in p and "all_pass" in p)
        for p in pp
    )
    if have_inline:
        n_tok = 0
        n_rnd = 0
        sum_succ = 0.0
        sum_rnd = 0.0
        sum_all = 0.0
        for p in pp:
            n = int(p.get("n_tokens_total", 0) or 0)
            r = len(p.get("rounds", []) or [])
            n_tok += n
            n_rnd += r
            sum_succ += float(p["tok_succ"]) * n
            sum_rnd += float(p["rnd_mean"]) * r
            sum_all += float(p["all_pass"]) * r
        tok_succ = (sum_succ / n_tok) if n_tok > 0 else None
        rnd_mean = (sum_rnd / n_rnd) if n_rnd > 0 else None
        all_pass = (sum_all / n_rnd) if n_rnd > 0 else None
    else:
        tok_succ = rnd_mean = all_pass = None
    L_vals: List[int] = []
    gamma_vals: List[int] = []
    for p in pp:
        for rd in p.get("rounds", []) or []:
            if "L" in rd:
                L_vals.append(int(rd["L"]))
            elif "L_lossy" in rd:
                L_vals.append(int(rd["L_lossy"]))
            elif "L_hat" in rd:
                L_vals.append(int(rd["L_hat"]))
            else:
                L_vals.append(int(rd.get("n_committed", 1)) - 1)
            gamma_vals.append(int(rd.get("gamma", 0)))
    if L_vals:
        n = len(L_vals)
        mean_L = sum(L_vals) / n
        frac_L0 = sum(1 for x in L_vals if x == 0) / n
        frac_Lg = (
            sum(1 for L, g in zip(L_vals, gamma_vals) if g > 0 and L == g) / n
        )
    else:
        mean_L = frac_L0 = frac_Lg = None
    return {
        "tok_s_mean": float(lane["tok_s_mean"]),
        "nll":        float(nll),
        "tok_succ":   tok_succ,
        "rnd_mean":   rnd_mean,
        "all_pass":   all_pass,
        "mean_L":     mean_L,
        "frac_L0":    frac_L0,
        "frac_Lgamma": frac_Lg,
    }


def _fmt(v, prec=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    with open(out_dir / "unified.json", "w") as f:
        json.dump(rows, f, indent=2)
    keys = [
        "method", "param", "tok_s_mean", "nll", "delta_nll",
        "tok_succ", "rnd_mean", "all_pass",
        "mean_L", "frac_L0", "frac_Lgamma",
    ]
    csv_lines = [",".join(keys)]
    for r in rows:
        csv_lines.append(",".join(
            "" if r.get(k) is None else (
                f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
            )
            for k in keys
        ))
    (out_dir / "results.csv").write_text("\n".join(csv_lines) + "\n")
    md = [
        "| method | param | tok/s | NLL | ΔNLL | tok_succ | rnd_mean | "
        "all_pass | mean_L | frac_L=0 | frac_L=γ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["method"], r["param"],
                _fmt(r["tok_s_mean"], 1),
                _fmt(r["nll"]),
                _fmt(r["delta_nll"]),
                _fmt(r["tok_succ"]),
                _fmt(r["rnd_mean"]),
                _fmt(r["all_pass"]),
                _fmt(r.get("mean_L"), 2),
                _fmt(r.get("frac_L0"), 3),
                _fmt(r.get("frac_Lgamma"), 3),
            )
        )
    (out_dir / "unified.md").write_text("\n".join(md) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str,
                    default=str(_HW_DIR / "results"),
                    help="Where to write online_*.json + tables.")
    ap.add_argument("--protocol_yaml", type=str,
                    default=str(_HW_DIR / "configs/homework_greedy.yaml"))
    ap.add_argument("--prompts_cache", type=str,
                    default=str(_HW_DIR / "data/prompts_owt_20.json"),
                    help="Prompts JSON cache (loaded if exists, else written).")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--gamma", type=int, default=15)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--lossy_ls", type=float, nargs="+",
                    default=[1.0, 0.7, 0.5, 0.3, 0.1])
    ap.add_argument("--thresh_taus", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--conf_taus", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--skip_existing", action="store_true",
                    help="Reuse pre-existing online_*.json files.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(Path(args.protocol_yaml).resolve())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    from specdiff.drafter_mdlm import MDLMDrafter
    from specdiff.verifier_gpt2 import GPT2Verifier

    print(
        f"[hw] device={device} dtype={protocol.dtype} "
        f"temp={protocol.temperature} gamma={args.gamma} T={args.T} "
        f"max_new={args.max_new_tokens}",
        flush=True,
    )
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompts, prompt_indices = load_owt_test_prompts(
        n_prompts=args.n_prompts,
        cache_path=Path(args.prompts_cache).resolve(),
    )
    print(
        f"[hw] {len(prompts)} OWT test prompts, indices "
        f"{prompt_indices[0]}..{prompt_indices[-1]}",
        flush=True,
    )
    write_prompts_dump(prompts, prompt_indices, out_dir / "prompts.json")

    def _load_or_run(filename: str, run_fn):
        fp = out_dir / filename
        if args.skip_existing and fp.exists():
            print(f"[hw]   reusing {filename}", flush=True)
            with open(fp) as fh:
                return json.load(fh)
        t0 = time.time()
        lane = run_fn()
        with open(fp, "w") as fh:
            json.dump(lane, fh, indent=2, default=str)
        print(
            f"[hw]   wrote {filename} (decode {time.time()-t0:.1f}s)",
            flush=True,
        )
        return lane

    rows: List[Dict] = []

    print("\n[hw] === STRICT ===", flush=True)
    strict_lane = _load_or_run(
        "online_strict.json",
        lambda: run_strict_lane(
            drafter, verifier, prompts, prompt_indices, protocol,
            gamma=args.gamma, T=args.T, max_new_tokens=args.max_new_tokens,
        ),
    )
    t0 = time.time()
    strict_nll, _ = compute_nll_and_success(verifier, strict_lane, device)
    print(
        f"[hw]   strict NLL={strict_nll:.4f} "
        f"(offline {time.time()-t0:.1f}s)",
        flush=True,
    )
    rows.append({
        "method": "strict", "param": "—",
        **_row_metrics(strict_lane, strict_nll),
        "delta_nll": 0.0,
    })

    for l_val in args.lossy_ls:
        tag = _fmt_param(float(l_val))
        print(f"\n[hw] === LOSSY l={l_val} ===", flush=True)
        lane = _load_or_run(
            f"online_lossy_l_{tag}.json",
            lambda lv=l_val: run_lossy_lane_one(
                drafter, verifier, prompts, prompt_indices, protocol,
                l=float(lv), gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        t0 = time.time()
        nll, _ = compute_nll_and_success(verifier, lane, device)
        print(
            f"[hw]   lossy l={l_val} NLL={nll:.4f} "
            f"(offline {time.time()-t0:.1f}s)",
            flush=True,
        )
        rows.append({
            "method": "lossy_l",
            "param": f"l={l_val}",
            **_row_metrics(lane, nll),
            "delta_nll": float(nll) - float(strict_nll),
        })

    for tau in args.thresh_taus:
        tag = _fmt_param(float(tau))
        print(f"\n[hw] === THRESHOLD_LOSSY tau={tau} ===", flush=True)
        lane = _load_or_run(
            f"online_threshold_lossy_tau_{tag}.json",
            lambda t=tau: hw.run_threshold_lossy_lane_one(
                drafter, verifier, prompts, prompt_indices, protocol,
                tau=float(t), gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        t0 = time.time()
        nll, _ = compute_nll_and_success(verifier, lane, device)
        print(
            f"[hw]   threshold tau={tau} NLL={nll:.4f} "
            f"(offline {time.time()-t0:.1f}s)",
            flush=True,
        )
        rows.append({
            "method": "threshold_lossy",
            "param": f"tau={tau}",
            **_row_metrics(lane, nll),
            "delta_nll": float(nll) - float(strict_nll),
        })

    for tau in args.conf_taus:
        tag = _fmt_param(float(tau))
        print(f"\n[hw] === CONFIDENCE_LOSSY tau={tau} ===", flush=True)
        lane = _load_or_run(
            f"online_confidence_lossy_tau_{tag}.json",
            lambda t=tau: hw.run_confidence_lossy_lane_one(
                drafter, verifier, prompts, prompt_indices, protocol,
                tau=float(t), gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        t0 = time.time()
        nll, _ = compute_nll_and_success(verifier, lane, device)
        print(
            f"[hw]   confidence tau={tau} NLL={nll:.4f} "
            f"(offline {time.time()-t0:.1f}s)",
            flush=True,
        )
        rows.append({
            "method": "confidence_lossy",
            "param": f"tau={tau}",
            **_row_metrics(lane, nll),
            "delta_nll": float(nll) - float(strict_nll),
        })

    write_outputs(rows, out_dir)
    print(f"\n[hw] wrote unified.{{json,md}} + results.csv -> {out_dir}",
          flush=True)
    print(f"[hw] strict NLL reference: {strict_nll:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
