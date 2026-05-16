"""Run OWT_Frozen_0429 baselines on a parametrized split & horizon.

Lanes:
  - strict_temp0          (reuse outputs/.../greedy_lossy_baseline/lanes.py)
  - strict_soft_T1        (reuse scripts/phase26_conf_sweep.py:run_lossy_lane_one with l=1.0)
  - lossy_soft_T1_l       (reuse scripts/phase26_conf_sweep.py:run_lossy_lane_one
                           with formula `accept iff U_j < min(1, p_j / (l · q_j))`)

CLI flags allow shrinking for the smoke run (--n_prompts, --max_new_tokens).
Outputs land at <out_dir>/online_<name>.json with the same per-prompt schema
the audit's add_baseline_nll.py expects.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "outputs/analysis/greedy_lossy_baseline"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from accpre.core.protocol import ProtocolConfig
from accpre.data.prompts import load_prompts
from accpre.data.splits import get_split_config

from lanes import run_strict_temp0_lane                  # noqa: E402
from phase26_conf_sweep import run_lossy_lane_one         # noqa: E402


def _load_protocol(path: str) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _l_tag(l: float) -> str:
    """0.7 -> '0p7', 1.0 -> '1', 0.95 -> '0p95'."""
    s = f"{float(l):.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol_temp0",
                    default="configs/protocol_greedy.yaml")
    ap.add_argument("--protocol_temp1",
                    default="configs/protocol.yaml")
    ap.add_argument("--dataset", default="owt_300")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--prompt_offset", type=int, default=200,
                    help="OWT_Frozen_0429 test split starts at index 200 (owt_300)")
    ap.add_argument("--gamma", type=int, default=15)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--lossy_ls", nargs="+", type=float,
                    default=[1.0, 0.7, 0.5, 0.3])
    ap.add_argument("--include_strict_T1", action="store_true",
                    help="Also run strict_soft_T1 (== lossy_soft_T1_l=1.0).")
    ap.add_argument("--lanes", default="all",
                    help="Comma-separated subset of {strict_temp0, strict_soft_T1, lossy_soft_T1}; "
                         "default 'all' runs every lane.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lanes_filter = (set(s.strip() for s in args.lanes.split(","))
                    if args.lanes != "all"
                    else {"strict_temp0", "strict_soft_T1", "lossy_soft_T1"})

    proto0 = _load_protocol(args.protocol_temp0)
    proto1 = _load_protocol(args.protocol_temp1)
    print(f"[baselines] dataset={args.dataset} n={args.n_prompts} "
          f"max_new={args.max_new_tokens} gamma={args.gamma}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[proto0.dtype]

    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier
    drafter = MDLMDrafter(model_name=proto0.drafter_model,
                          device=device, dtype=dtype)
    verifier = GPT2Verifier(model_name=proto0.verifier_model,
                            device=device, dtype=dtype)

    cfg = get_split_config(args.dataset)
    pool = load_prompts(dataset=cfg.name, n_prompts=cfg.pool_size,
                        prefix_len=cfg.prefix_len, seed=cfg.seed)
    chosen = []
    indices = []
    for i in range(args.prompt_offset,
                   args.prompt_offset + args.n_prompts):
        ids, text = pool[i]
        chosen.append((ids, text))
        indices.append(i)
    print(f"[baselines] prompts {indices[0]}..{indices[-1]}  ({len(indices)} total)")

    # ---- temp=0 strict ----
    if "strict_temp0" in lanes_filter:
        out = out_dir / "online_strict_temp0.json"
        if out.exists():
            print(f"\n=== strict_temp0 (temp=0) ===  [SKIP: {out} exists]")
        else:
            print("\n=== strict_temp0 (temp=0) ===")
            t0 = time.time()
            res = run_strict_temp0_lane(
                drafter, verifier, chosen, indices, proto0,
                gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            )
            with open(out, "w") as f:
                json.dump(res, f)
            print(f"  wrote {out}  ({time.time() - t0:.1f}s)")

    # ---- temp=1 strict (== lossy_soft_T1_l=1.0) ----
    if args.include_strict_T1 and "strict_soft_T1" in lanes_filter:
        out = out_dir / "online_strict_soft_T1.json"
        if out.exists():
            print(f"\n=== strict_soft_T1 (== lossy l=1.0, temp=1) ===  [SKIP: {out} exists]")
        else:
            print("\n=== strict_soft_T1 (== lossy l=1.0, temp=1) ===")
            t0 = time.time()
            res = run_lossy_lane_one(
                drafter, verifier, chosen, indices, proto1,
                l=1.0, gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            )
            with open(out, "w") as f:
                json.dump(res, f)
            print(f"  wrote {out}  ({time.time() - t0:.1f}s)")

    # ---- temp=1 lossy sweep ----
    if "lossy_soft_T1" in lanes_filter:
        for l in args.lossy_ls:
            out = out_dir / f"online_lossy_soft_T1_l{_l_tag(l)}.json"
            if out.exists():
                print(f"\n=== lossy_soft_T1 l={l} (temp=1) ===  [SKIP: {out} exists]")
                continue
            print(f"\n=== lossy_soft_T1 l={l} (temp=1) ===")
            t0 = time.time()
            res = run_lossy_lane_one(
                drafter, verifier, chosen, indices, proto1,
                l=float(l), gamma=args.gamma, T=args.T,
                max_new_tokens=args.max_new_tokens,
            )
            with open(out, "w") as f:
                json.dump(res, f)
            print(f"  wrote {out}  ({time.time() - t0:.1f}s)")

    print("\n[baselines] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
