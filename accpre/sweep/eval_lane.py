"""Online-eval wrapper for predictor sweeps.

Wraps the trusted Phase 26 helpers so a sweep run can evaluate ONE
trained predictor without invoking the full Phase 26 multi-lane driver
(which also runs strict + lossy + confidence). The helpers themselves
are imported from `scripts/phase26_conf_sweep.py` via `importlib.util`
so we don't duplicate decode logic.

Public API:
  evaluate_predictor(...) -> dict          # one threshold lane
  strict_nll(...) -> float                 # strict-lane NLL reference

The strict NLL is cached in a file keyed by (dataset, n_prompts,
max_new_tokens, gamma, T, drafter_signature). Sweep runs then compute
ΔNLL against the cached value instead of re-running strict per grid
point.
"""

from __future__ import annotations

import hashlib
import importlib.util as _iu
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Import phase26_conf_sweep via importlib so we can reuse
# run_predictor_lane_generic and compute_nll_and_success without having
# to re-import the full script (which has its own argparse main).
_p26_spec = _iu.spec_from_file_location(
    "p26_sweep_impl",
    str(_REPO_ROOT / "scripts/phase26_conf_sweep.py"),
)
_p26 = _iu.module_from_spec(_p26_spec)
sys.modules["p26_sweep_impl"] = _p26
_p26_spec.loader.exec_module(_p26)

# Same trick for phase25 (for run_strict_lane).
_p25_spec = _iu.spec_from_file_location(
    "p25_sweep_impl",
    str(_REPO_ROOT / "scripts/phase25_longhorizon.py"),
)
_p25 = _iu.module_from_spec(_p25_spec)
sys.modules["p25_sweep_impl"] = _p25
_p25_spec.loader.exec_module(_p25)


def load_greedy_protocol(path: Optional[Path] = None):
    """Load the temp=0 protocol used by the sweep eval."""
    from accpre.core.protocol import ProtocolConfig
    p = path or (_REPO_ROOT / "configs/protocol_greedy.yaml")
    with open(p) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _cache_key(
    dataset: str, n_prompts: int, max_new_tokens: int, gamma: int, T: int,
    drafter_sig: str,
) -> str:
    s = f"{dataset}|n={n_prompts}|h={max_new_tokens}|g={gamma}|T={T}|d={drafter_sig}"
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _drafter_signature(drafter) -> str:
    """Coarse fingerprint of current drafter weights.

    Used so strict NLL cached under a pretrained drafter doesn't leak
    into a joint-drafter cache key.
    """
    with torch.no_grad():
        h = hashlib.sha1()
        # Use a tiny slice of the first couple of parameters; robust to
        # float-precision noise and order-stable within this repo.
        for i, (_name, p) in enumerate(drafter.model.state_dict().items()):
            if i >= 4:
                break
            flat = p.detach().float().flatten()
            n = min(64, int(flat.numel()))
            if n > 0:
                h.update(flat[:n].cpu().numpy().tobytes())
        return h.hexdigest()[:12]


def strict_nll(
    drafter,
    verifier,
    prompts,
    prompt_indices: List[int],
    protocol,
    *,
    gamma: int = 8,
    T: int = 2,
    max_new_tokens: int = 1024,
    dataset_name: str = "owt",
    cache_dir: Optional[Path] = None,
) -> float:
    """Return strict-lane NLL for the given eval config.

    Cached at `<cache_dir>/strict_<key>.json` so subsequent sweep runs
    reuse the value. Pass `cache_dir=None` to disable caching.
    """
    drafter_sig = _drafter_signature(drafter)
    key = _cache_key(
        dataset_name, len(prompts), max_new_tokens, gamma, T, drafter_sig,
    )
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"strict_{key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            print(
                f"[eval] strict NLL cache hit: {cached['nll']:.4f} "
                f"(key={key}, from {cache_path})"
            )
            return float(cached["nll"])

    print("[eval] running strict lane for NLL reference ...")
    t0 = time.time()
    lane = _p25.run_strict_lane(
        drafter, verifier, prompts, prompt_indices, protocol,
        gamma=gamma, T=T, max_new_tokens=max_new_tokens,
    )
    device = drafter.device
    nll, _ = _p26.compute_nll_and_success(verifier, lane, device)
    print(f"[eval] strict NLL = {nll:.4f}  (took {time.time() - t0:.1f}s)")

    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump({
                "nll": float(nll),
                "tok_s_mean": float(lane["tok_s_mean"]),
                "dataset": dataset_name,
                "n_prompts": len(prompts),
                "max_new_tokens": int(max_new_tokens),
                "gamma": int(gamma), "T": int(T),
                "drafter_sig": drafter_sig,
                "key": key,
            }, f, indent=2)
    return float(nll)


def evaluate_predictor(
    predictor,
    drafter,
    verifier,
    prompts,
    prompt_indices: List[int],
    protocol,
    *,
    tau: float = 0.5,
    rule: str = "threshold",
    gamma: int = 8,
    T: int = 2,
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run ONE predictor lane and return online metrics.

    Returns dict with keys: tok_s_mean, nll, tok_succ, rnd_mean,
    all_pass, decode_s, offline_s, per_prompt (lane dump), rule, tau.
    """
    device = drafter.device
    t_decode0 = time.time()
    lane = _p26.run_predictor_lane_generic(
        predictor, drafter, verifier, prompts, prompt_indices, protocol,
        tau=float(tau), rule=rule, gamma=gamma, T=T,
        max_new_tokens=max_new_tokens,
    )
    decode_s = time.time() - t_decode0
    t_off0 = time.time()
    nll, _summary = _p26.compute_nll_and_success(verifier, lane, device)
    offline_s = time.time() - t_off0

    pp = lane["per_prompt"]
    tot_n = sum(int(p.get("n_tokens_total", 0)) for p in pp)
    tok_succ = (
        sum(
            float(p.get("tok_succ", 0.0)) * int(p.get("n_tokens_total", 0))
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

    return {
        "rule": rule,
        "tau": float(tau),
        "gamma": int(gamma), "T": int(T),
        "max_new_tokens": int(max_new_tokens),
        "n_prompts": len(prompts),
        "tok_s_mean": float(lane["tok_s_mean"]),
        "nll": float(nll),
        "tok_succ": float(tok_succ) if tok_succ is not None else None,
        "rnd_mean": float(rnd_mean) if rnd_mean is not None else None,
        "all_pass": float(all_pass) if all_pass is not None else None,
        "decode_s": float(decode_s),
        "offline_s": float(offline_s),
        "lane": lane,
    }
