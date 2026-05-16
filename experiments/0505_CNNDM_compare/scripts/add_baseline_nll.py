"""Post-hoc NLL + tok_succ filler for baseline JSONs.

Mirrors outputs/analysis/retrain_temp0_gamma15_owt/scripts/add_baseline_nll.py
exactly, with no behavior changes — written here so the smoke pipeline
runs end-to-end against scripts that all live under experiments/OWT_Frozen_0429/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.protocol import ProtocolConfig


def _load_protocol(path: str) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines_dir", required=True)
    ap.add_argument("--protocol", default="configs/protocol_greedy.yaml")
    args = ap.parse_args()

    protocol = _load_protocol(args.protocol)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from accpre.models.verifier_gpt2 import GPT2Verifier
    verifier = GPT2Verifier(model_name=protocol.verifier_model,
                            device=device, dtype=torch.bfloat16)
    verifier.model.eval()

    base_dir = Path(args.baselines_dir)
    files = sorted(base_dir.glob("online_*.json"))
    print(f"[baseline_nll] found {len(files)} JSON(s)")

    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        per_prompt = d.get("per_prompt", [])
        nlls = []; succs = []
        for pp in per_prompt:
            gen_ids = pp.get("generated_ids")
            n_new = int(pp.get("n_new_tokens", 0))
            if not gen_ids or n_new <= 0:
                pp["nll"] = float("nan")
                pp["tok_succ"] = float("nan")
                continue
            gen = torch.tensor(gen_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                log_p = verifier.score(gen)
            prefix_len = int(gen.shape[0] - n_new)
            target_pos = torch.arange(prefix_len, gen.shape[0], device=device)
            target_tok = gen[target_pos]
            log_p_at_tok = log_p[target_pos - 1, target_tok]
            nll = float(-log_p_at_tok.mean().item())
            argmax_tok = log_p[target_pos - 1].argmax(dim=-1)
            tok_succ = float((argmax_tok == target_tok).float().mean().item())
            pp["nll"] = nll
            pp["tok_succ"] = tok_succ
            nlls.append(nll); succs.append(tok_succ)
        d["nll_mean"] = float(sum(nlls) / len(nlls)) if nlls else float("nan")
        d["tok_succ_mean"] = float(sum(succs) / len(succs)) if succs else float("nan")
        with open(fp, "w") as f:
            json.dump(d, f)
        print(f"  {fp.name}  nll={d['nll_mean']:.4f}  tok_succ={d['tok_succ_mean']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
