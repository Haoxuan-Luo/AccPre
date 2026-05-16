"""Stage 5 — V-free predictor eval driver, one (run_dir, rule, param) cell.

Mirrors `experiments/OWT_Frozen_0429/scripts/eval_with_fallback.py` but:
  - Builds heads via `scripts.heads.build_head` (the new architecture).
  - Restricts commit rules to {hard_threshold, prefix_product}; relative_max
    is forbidden (sanity_no_relative_max.py also gates this).
  - Per-target fallback policy is fixed by `--fallback` (or by the run's
    target if `--fallback auto` is requested):
        relmax   -> confidence_gated_sampled_first  (eta=0.5)
        alpha_q2 -> confidence_gated_sampled_first  (eta=0.5)
        dep      -> drafter_argmax_first
    These match the OWT_Frozen_0429 plan (T=1 plan).

V-free invariant:
  No `verifier.score`, `verifier.prefix_features`, or any verifier-derived
  feature appears inside the per-round predictor decode loop. The verifier
  is loaded only for the OFFLINE NLL / tok_succ pass at the end. The
  function `_offline_compute_nll_tok_succ` is the only place the verifier
  is invoked. `sanity_vfree_inference.py` static-greps this file.

Output JSON schema (extends OWT_Frozen_0429's `online_predictor_full.json`):
  method:                "predictor"
  rule:                  "hard_threshold" | "prefix_product"
  param:                 float
  param_tag:             stable string (e.g. "tau_0p7", "tauc_0p9")
  fallback_policy:       string
  fallback_eta:          float | null
  gamma, T:              ints
  n_prompts:             int
  n_new_tokens_total:    int
  tok_s_mean:            float (drafter+head time only — verifier is offline)
  nll_mean:              float
  tok_succ_mean:         float
  per_prompt:            list of dicts (see code)
  triple:                {regime, target, arch}  — for downstream aggregation

The per-round records carry only drafter-side fields (draft_tokens, score_pred,
L_hat, n_committed, fallback_*). No verifier output is logged per round.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXP_ROOT))

from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig, derive_seed
from accpre.data.prompts import load_prompts
from accpre.data.splits import get_split_config

from scripts.heads import build_head


# Allowed commit rules. relative_max is FORBIDDEN.
_ALLOWED_RULES = ("hard_threshold", "prefix_product")
_ALLOWED_FALLBACKS = (
    "current_sampled_first",
    "drafter_argmax_first",
    "confidence_gated_sampled_first",
)
_FALLBACK_BY_TARGET = {
    "relmax":   ("confidence_gated_sampled_first", 0.5),
    "alpha_q2": ("confidence_gated_sampled_first", 0.5),
    "dep":      ("drafter_argmax_first",           0.5),  # eta unused
}

EPS = 1e-10


# ----------------------------------------------------------------------
# Commit rules — hard-inlined (also implemented in
# outputs/analysis/retrain_temp0_gamma15_owt/scripts/commit_rules.py).
# We re-implement here so this file is self-contained and the V-free scan
# does not have to chase imports across the repo.
# ----------------------------------------------------------------------


def _commit_hard_threshold(scores: List[float], tau: float) -> int:
    for j, a in enumerate(scores):
        if a < tau:
            return j
    return len(scores)


def _commit_prefix_product(scores: List[float], tau_conf: float) -> int:
    prod = 1.0
    for j, a in enumerate(scores):
        prod = prod * float(a)
        if prod < tau_conf:
            return j
    return len(scores)


def _apply_commit(rule: str, scores: List[float], param: float) -> int:
    if rule == "hard_threshold":
        return _commit_hard_threshold(scores, float(param))
    if rule == "prefix_product":
        return _commit_prefix_product(scores, float(param))
    raise ValueError(f"unknown / forbidden commit rule {rule!r}")


def _param_tag(rule: str, param: float) -> str:
    prefix = "tau" if rule == "hard_threshold" else "tauc"
    s = f"{float(param):.3f}".rstrip("0").rstrip(".")
    return f"{prefix}_{s.replace('.', 'p')}"


# ----------------------------------------------------------------------
# Fallbacks. Verifier is FORBIDDEN here (V-free); we drop the
# `oracle_verifier_argmax_first` policy from the predecessor.
# ----------------------------------------------------------------------


def _pick_first_token(
    fallback: str,
    eta: float,
    draft_tokens: torch.Tensor,
    draft_log_probs: torch.Tensor,
    score_pred: List[float],
) -> int:
    if fallback == "current_sampled_first":
        return int(draft_tokens[0].item())
    if fallback == "drafter_argmax_first":
        return int(draft_log_probs[0].argmax().item())
    if fallback == "confidence_gated_sampled_first":
        if score_pred and score_pred[0] >= eta:
            return int(draft_tokens[0].item())
        return int(draft_log_probs[0].argmax().item())
    raise ValueError(f"unknown / forbidden fallback {fallback!r}")


# ----------------------------------------------------------------------
# Protocol + head loaders.
# ----------------------------------------------------------------------


def _load_protocol(path: str) -> ProtocolConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _load_head(run_dir: Path, device: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load the trained head and its run config."""
    ckpt_dir = run_dir / "ckpt"
    cfg_path = ckpt_dir / "config.yaml" if (ckpt_dir / "config.yaml").exists() else run_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    head = build_head(
        cfg["arch"],
        gamma=int(cfg.get("gamma", 15)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        d_model=int(cfg.get("d_model", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        token_emb_dim=int(cfg.get("token_emb_dim", 64)),
    )
    model_path = ckpt_dir / "model.pt" if (ckpt_dir / "model.pt").exists() else run_dir / "model.pt"
    head.load_state_dict(torch.load(str(model_path), map_location=device))
    head.to(device)
    head.eval()
    return head, cfg


# ----------------------------------------------------------------------
# V-free per-round forward.
#
# Critical invariant: `verifier` is NOT touched anywhere in this function
# or anywhere it calls. Inputs to `head(...)` are derived ONLY from the
# drafter forward, plus the head's own learned token embedding and
# pos_scalar buffer.
# ----------------------------------------------------------------------


@torch.no_grad()
def _vfree_round(
    head: torch.nn.Module,
    drafter,
    protocol: ProtocolConfig,
    running: torch.Tensor,
    cur_gamma: int,
    seed: int,
    T: int,
    rule: str,
    param: float,
    fallback: str,
    eta: float,
    device: str,
) -> Dict[str, Any]:
    """Run one round of V-free decode. Returns a per-round record."""
    draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
    t_d = time.time()
    draft_tokens, draft_log_probs, drafter_hidden = drafter.draft_with_features(
        prefix_ids=running, gamma=cur_gamma, T=T,
        temperature=protocol.temperature, q_mode=protocol.q_mode,
        generator=draft_rng,
        pool="per_position", layers=(-1,),
    )
    # Drafter may return tensors on its own device; coerce to head's device.
    # Mirrors train_joint.py's defensive .to(device) pattern.
    draft_tokens = draft_tokens.to(device)
    draft_log_probs = draft_log_probs.to(device)
    drafter_hidden = drafter_hidden.to(device)
    draft_time_s = time.time() - t_d

    t_p = time.time()
    # Five drafter scalars per position.
    probs = draft_log_probs.exp()
    entropy_t = -(probs * draft_log_probs).sum(dim=-1)
    top2 = draft_log_probs.topk(2, dim=-1).values
    margin_t = top2[:, 0] - top2[:, 1]
    top1_t = probs.max(dim=-1).values
    idx = torch.arange(int(draft_tokens.shape[0]),
                       device=draft_tokens.device)
    q_t = probs[idx, draft_tokens.long()].clamp(min=EPS)
    log_q_t = torch.log(q_t)
    scalars = torch.stack([q_t, log_q_t, entropy_t, margin_t, top1_t],
                          dim=1).to(torch.float32)
    h = drafter_hidden.to(torch.float32)
    features = torch.cat([h, scalars], dim=1)        # (gamma, 773)

    # Head expects (B, gamma, 773); add batch axis.
    features_b = features.unsqueeze(0)
    tokens_b = draft_tokens.unsqueeze(0).long()
    q_hat = head(features_b, token_ids=tokens_b)     # (1, gamma)
    if q_hat.dim() == 2:
        q_hat = q_hat.squeeze(0)
    score_pred = [float(x) for x in q_hat.detach().cpu().tolist()]

    L_hat = _apply_commit(rule, score_pred, param)
    L_hat = max(0, min(int(L_hat), int(cur_gamma)))
    predictor_time_s = time.time() - t_p

    fallback_used = 0
    fallback_first_token: Optional[int] = None
    if L_hat == 0:
        fallback_used = 1
        first_tok = _pick_first_token(
            fallback, eta, draft_tokens, draft_log_probs, score_pred,
        )
        fallback_first_token = int(first_tok)
        committed = torch.tensor(
            [first_tok], device=device, dtype=draft_tokens.dtype,
        )
        n_commit = 1
    else:
        n_commit = int(L_hat)
        committed = draft_tokens[:n_commit]

    return dict(
        L_hat=int(L_hat),
        n_commit=int(n_commit),
        committed=committed,
        score_pred=score_pred,
        draft_tokens=[int(x) for x in draft_tokens.tolist()],
        draft_time_s=float(draft_time_s),
        predictor_time_s=float(predictor_time_s),
        fallback_used=int(fallback_used),
        fallback_first_token=fallback_first_token,
        seed=int(seed),
        gamma=int(cur_gamma),
    )


# ----------------------------------------------------------------------
# Online (V-free) decode loop.
#
# Per protocol, the drafter is the only model in the per-round loop.
# Verifier loads happen ONLY in `_offline_compute_nll_tok_succ` below.
# ----------------------------------------------------------------------


@torch.no_grad()
def run_online_vfree(
    head: torch.nn.Module,
    drafter,
    protocol: ProtocolConfig,
    prompts: List[Tuple[int, torch.Tensor, str]],
    gamma: int,
    T: int,
    max_new_tokens: int,
    rule: str,
    param: float,
    fallback: str,
    eta: float,
    device: str,
) -> List[Dict[str, Any]]:
    """Run V-free decode for every prompt; return per-prompt records.

    NB: this function never imports or calls a verifier model. The
    sanity_vfree_inference.py scan asserts this statically.
    """
    out: List[Dict[str, Any]] = []
    MAX_CTX = protocol.max_verifier_ctx

    for prompt_idx, prefix_ids, _text in prompts:
        prefix_ids = prefix_ids.to(device)
        running = prefix_ids.clone()
        prefix_start_len = int(prefix_ids.shape[0])
        rounds: List[Dict[str, Any]] = []
        round_idx = 0
        t_start = time.time()
        while (running.shape[0] - prefix_start_len) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_start_len)
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(int(gamma), int(remaining), int(ctx_room))
            if cur_gamma <= 0:
                break
            seed = derive_seed(protocol, prompt_idx, round_idx)
            rec = _vfree_round(
                head, drafter, protocol, running, cur_gamma, seed,
                T, rule, param, fallback, eta, device,
            )
            committed = rec.pop("committed")
            running = torch.cat([running, committed])
            rec["round_idx"] = int(round_idx)
            rec["committed_tokens"] = [int(x) for x in committed.tolist()]
            rounds.append(rec)
            round_idx += 1
        elapsed_s = time.time() - t_start
        n_new_tokens = int(running.shape[0] - prefix_start_len)
        tok_s = float(n_new_tokens) / elapsed_s if elapsed_s > 0 else 0.0
        out.append(dict(
            prompt_idx=int(prompt_idx),
            n_new_tokens=int(n_new_tokens),
            elapsed_s=float(elapsed_s),
            tok_s=float(tok_s),
            rounds=rounds,
            generated_ids=[int(x) for x in running.tolist()],
        ))
    return out


# ----------------------------------------------------------------------
# Offline NLL / tok_succ pass.
#
# This is the ONLY place the verifier is touched. It happens AFTER
# `run_online_vfree` returns and operates on already-committed sequences.
# Tok/s is reported from the online loop ONLY (verifier walltime is not
# included in tok_s — see eval_plan.md §2).
# ----------------------------------------------------------------------


@torch.no_grad()
def _offline_compute_nll_tok_succ(
    per_prompt: List[Dict[str, Any]],
    verifier,
    device: str,
) -> None:
    """Mutates `per_prompt` in place: adds `nll`, `tok_succ`, `all_pass`."""
    for pp in per_prompt:
        gen = torch.tensor(pp["generated_ids"], dtype=torch.long, device=device)
        if gen.shape[0] <= 1 or pp["n_new_tokens"] <= 0:
            pp["nll"] = float("nan")
            pp["tok_succ"] = float("nan")
            pp["all_pass"] = False
            continue
        log_p = verifier.score(gen)
        prefix_len = int(gen.shape[0] - pp["n_new_tokens"])
        target_pos = torch.arange(prefix_len, gen.shape[0], device=device)
        target_tok = gen[target_pos]
        log_p_at_tok = log_p[target_pos - 1, target_tok]
        nll = float(-log_p_at_tok.mean().item())
        argmax_tok = log_p[target_pos - 1].argmax(dim=-1)
        tok_succ = float((argmax_tok == target_tok).float().mean().item())
        all_pass = bool((argmax_tok == target_tok).all().item())
        pp["nll"] = nll
        pp["tok_succ"] = tok_succ
        pp["all_pass"] = all_pass


# ----------------------------------------------------------------------
# Entry.
# ----------------------------------------------------------------------


def _run_metadata(run_dir: Path) -> Dict[str, str]:
    """Try to recover (regime, target, arch) from a run_dir like
    runs/<regime>/<target>/<arch>/<cell>/.

    Skips any underscore-prefixed wrapper dirs between `runs/` and the
    regime (e.g. `runs/_dryrun/<regime>/...`, `runs/_smoke/<regime>/...`).
    If the path is shorter or differently-shaped, returns "?" for
    unknown components — `main()` then falls back to the cfg.
    """
    parts = run_dir.parts
    try:
        i = parts.index("runs")
    except ValueError:
        return {"regime": "?", "target": "?", "arch": "?"}
    # Skip past any underscore-prefixed wrapper dir(s).
    j = i + 1
    while j < len(parts) and parts[j].startswith("_"):
        j += 1
    try:
        return {"regime": parts[j], "target": parts[j + 1], "arch": parts[j + 2]}
    except IndexError:
        return {"regime": "?", "target": "?", "arch": "?"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="trained predictor run directory (must contain ckpt/model.pt)")
    ap.add_argument("--rule", required=True, choices=_ALLOWED_RULES)
    ap.add_argument("--param", required=True, type=float)
    ap.add_argument("--fallback", default="auto",
                    help=f"one of {_ALLOWED_FALLBACKS}, or 'auto' to pick by target")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--gamma", type=int, default=15)
    ap.add_argument("--T", type=int, default=1,
                    help="MDLM denoising steps; T=1 matches the stage-1 records")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--dataset", default="owt_300")
    ap.add_argument("--device", default=None,
                    help="override device; defaults to cuda if available")
    args = ap.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[eval] SKIP {out_path} (exists)")
        return 0

    run_dir = Path(args.run_dir).resolve()
    rule = args.rule
    if rule not in _ALLOWED_RULES:
        raise ValueError(f"forbidden rule {rule!r}; must be in {_ALLOWED_RULES}")
    param = float(args.param)

    protocol = _load_protocol(args.protocol)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[protocol.dtype]

    # Load drafter (for online decode) and verifier (for offline NLL only).
    from accpre.models.drafter_mdlm import MDLMDrafter
    drafter = MDLMDrafter(model_name=protocol.drafter_model,
                          device=device, dtype=dtype)
    drafter.model.eval()

    head, cfg = _load_head(run_dir, device)
    triple = _run_metadata(run_dir)
    if triple["target"] == "?" and "target" in cfg:
        triple["target"] = str(cfg["target"])
    if triple["arch"] == "?" and "arch" in cfg:
        triple["arch"] = str(cfg["arch"])
    if triple["regime"] == "?" and "mode" in cfg:
        triple["regime"] = str(cfg["mode"])

    # Resolve fallback policy.
    fb = args.fallback
    if fb == "auto":
        target = triple["target"]
        if target not in _FALLBACK_BY_TARGET:
            raise ValueError(
                f"cannot auto-pick fallback for unknown target {target!r}; "
                f"pass --fallback explicitly"
            )
        fb, eta = _FALLBACK_BY_TARGET[target]
    else:
        if fb not in _ALLOWED_FALLBACKS:
            raise ValueError(f"forbidden fallback {fb!r}; must be in {_ALLOWED_FALLBACKS}")
        eta = float(args.eta)

    # Load prompts. owt_300 test split is indices [200, 300).
    split_cfg = get_split_config(args.dataset)
    pool = load_prompts(dataset=split_cfg.name,
                        n_prompts=split_cfg.pool_size,
                        prefix_len=split_cfg.prefix_len,
                        seed=split_cfg.seed)
    test_lo = split_cfg.train_n + split_cfg.val_n
    n = min(int(args.n_prompts), split_cfg.pool_size - test_lo)
    chosen: List[Tuple[int, torch.Tensor, str]] = []
    for i in range(test_lo, test_lo + n):
        ids, text = pool[i]
        chosen.append((i, ids, text))
    print(f"[eval] run={run_dir.name} triple={triple} rule={rule} param={param} "
          f"fallback={fb}{' eta='+str(eta) if fb=='confidence_gated_sampled_first' else ''} "
          f"prompts={len(chosen)}")

    # Online (V-free) decode.
    per_prompt = run_online_vfree(
        head, drafter, protocol, chosen,
        args.gamma, args.T, args.max_new_tokens,
        rule, param, fb, eta, device,
    )

    # Offline NLL / tok_succ — verifier loaded ONLY here.
    from accpre.models.verifier_gpt2 import GPT2Verifier
    verifier = GPT2Verifier(model_name=protocol.verifier_model,
                            device=device, dtype=torch.bfloat16)
    verifier.model.eval()
    _offline_compute_nll_tok_succ(per_prompt, verifier, device)

    # Aggregate.
    nlls = [pp["nll"] for pp in per_prompt
            if isinstance(pp.get("nll"), float) and pp["nll"] == pp["nll"]]
    succs = [pp["tok_succ"] for pp in per_prompt
             if isinstance(pp.get("tok_succ"), float) and pp["tok_succ"] == pp["tok_succ"]]
    tokss = [pp["tok_s"] for pp in per_prompt]
    n_total = sum(pp["n_new_tokens"] for pp in per_prompt)
    res: Dict[str, Any] = dict(
        method="predictor",
        rule=rule,
        param=float(param),
        param_tag=_param_tag(rule, param),
        fallback_policy=fb,
        fallback_eta=float(eta) if fb == "confidence_gated_sampled_first" else None,
        gamma=int(args.gamma),
        T=int(args.T),
        n_prompts=len(per_prompt),
        n_new_tokens_total=int(n_total),
        tok_s_mean=float(sum(tokss) / len(tokss)) if tokss else 0.0,
        nll_mean=float(sum(nlls) / len(nlls)) if nlls else float("nan"),
        tok_succ_mean=float(sum(succs) / len(succs)) if succs else float("nan"),
        per_prompt=per_prompt,
        triple=triple,
        run_dir=str(run_dir),
        protocol_fingerprint=protocol.fingerprint(),
    )
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"[eval] tok_s={res['tok_s_mean']:.2f} nll={res['nll_mean']:.4f} "
          f"tok_succ={res['tok_succ_mean']:.4f}")
    print(f"[eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
