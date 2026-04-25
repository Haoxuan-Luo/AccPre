"""Online predictor decode — the real cheap-lane inference path.

*** SEMANTIC WARNING: this loop is NOT strict SpecDiff. ***

Differences from strict SpecDiff (DESIGN_PHASE2.md §C.5):
  1. NO full verifier-on-draft pass. Strict runs
     `verifier.score(concat(prefix, draft_tokens))` every round; this
     loop does not. That is where the speedup comes from.
  2. NO strict fallback / NO bonus token. Strict samples one extra
     token from either `target_log_probs[-1]` (full-accept bonus) or
     the `(p − q)_+` adjusted distribution (partial-accept fallback).
     Cheap decode has neither — there is no target distribution
     available without paying the verifier cost we are trying to save.
  3. Commit convention: `max(1, L̂)` tokens per round, from the
     already-sampled `draft_tokens`. Progress is guaranteed — a round
     always advances the context by at least one token — by emitting
     `draft_tokens[0]` when `L̂ == 0`.

Under this contract, each round's per-token count is `max(1, L̂)`,
compared to strict's `L_strict + 1`. Both are honest accountings; tok/s
is measured by the real elapsed time.

Deploy modes (§C.3):
  - V-free:   NO verifier call of any kind per round. Only the
              `hidden` family qualifies.
  - V-prefix: exactly one `verifier.prefix_features(running)` call per
              round (prefix only; never on draft positions). Families
              `numeric`, `hidden_numeric`, `upper_bound` use this.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig, derive_seed


# ------------------------------------------------------------------
# Per-round / per-prompt / per-lane bookkeeping dataclasses
# ------------------------------------------------------------------


@dataclass
class PredictorRoundLog:
    """One round of predictor-driven cheap online decode."""
    prompt_idx: int
    round_idx: int
    round_rng_seed: int
    gamma: int
    T: int
    prefix_len_at_round_start: int
    L_hat: int
    n_committed: int                # == max(1, L_hat); NO extra bonus/fallback
    draft_tokens: List[int]         # (gamma,)
    committed_tokens: List[int]     # (n_committed,); subset of draft_tokens
    draft_time_s: float
    verifier_prefix_time_s: float   # == 0.0 for V-free
    predictor_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictorPromptResult:
    prompt_idx: int
    n_new_tokens: int
    elapsed_s: float
    tok_s: float
    generated_ids: List[int]
    rounds: List[PredictorRoundLog] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_idx": self.prompt_idx,
            "n_new_tokens": self.n_new_tokens,
            "elapsed_s": self.elapsed_s,
            "tok_s": self.tok_s,
            "generated_ids": self.generated_ids,
            "rounds": [r.to_dict() for r in self.rounds],
        }


@dataclass
class PredictorLaneResult:
    predictor_name: str
    kind: str
    family: str
    deploy_mode: str
    tau: float
    gamma: int
    T: int
    per_prompt: List[PredictorPromptResult] = field(default_factory=list)

    @property
    def tok_s_per_prompt(self) -> List[float]:
        return [p.tok_s for p in self.per_prompt]

    @property
    def tok_s_mean(self) -> float:
        xs = self.tok_s_per_prompt
        return sum(xs) / max(len(xs), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictor_name": self.predictor_name,
            "kind": self.kind,
            "family": self.family,
            "deploy_mode": self.deploy_mode,
            "tau": self.tau,
            "gamma": self.gamma,
            "T": self.T,
            "tok_s_mean": self.tok_s_mean,
            "tok_s_per_prompt": self.tok_s_per_prompt,
            "per_prompt": [p.to_dict() for p in self.per_prompt],
        }


# ------------------------------------------------------------------
# Feature assembly from raw drafter / verifier-prefix outputs
# ------------------------------------------------------------------


def _numeric_tensor(numeric: Dict[str, float]) -> torch.Tensor:
    return torch.tensor(
        [
            float(numeric["entropy"]),
            float(numeric["margin"]),
            float(numeric["top1_prob"]),
        ],
        dtype=torch.float32,
    )


def build_online_features(
    family: str,
    drafter_hidden: Optional[torch.Tensor],
    verifier_prefix_hidden: Optional[torch.Tensor],
    numeric: Optional[Dict[str, float]],
) -> torch.Tensor:
    """Construct the predictor input tensor from raw forward outputs.

    Shape-equivalent to `accpre.collect.features.extract_features` on a
    RoundRecord with matching fields populated. Lives here (not in
    `collect/features.py`) because the inputs are raw tensors and
    numeric dicts — not a RoundRecord.
    """
    if family == "numeric":
        if numeric is None:
            raise ValueError("family='numeric' requires verifier prefix features")
        return _numeric_tensor(numeric)
    if family == "hidden":
        if drafter_hidden is None:
            raise ValueError("family='hidden' requires drafter hidden")
        return drafter_hidden.to(torch.float32).flatten().cpu()
    if family == "hidden_numeric":
        if drafter_hidden is None or numeric is None:
            raise ValueError("family='hidden_numeric' requires drafter hidden + numeric")
        return torch.cat([
            drafter_hidden.to(torch.float32).flatten().cpu(),
            _numeric_tensor(numeric),
        ])
    if family == "upper_bound":
        if drafter_hidden is None or numeric is None or verifier_prefix_hidden is None:
            raise ValueError("family='upper_bound' requires all feature sources")
        return torch.cat([
            verifier_prefix_hidden.to(torch.float32).flatten().cpu(),
            drafter_hidden.to(torch.float32).flatten().cpu(),
            _numeric_tensor(numeric),
        ])
    if family in ("hidden_per_pos", "hidden_per_pos_ml"):
        if drafter_hidden is None:
            raise ValueError(
                f"family={family!r} requires drafter hidden (per-position)"
            )
        # Expect `(γ, k*H)` where k is the number of layers the caller
        # asked the drafter to return (pp_base → (γ, H); pp_ml → (γ, 2H)).
        # Do NOT flatten — the per-position head consumes a 2-D tensor.
        return drafter_hidden.to(torch.float32).cpu()
    if family == "hidden_per_pos_numeric":
        # Phase 7 V1 (pp_vpn): per-position drafter hidden `(γ, H)`
        # concatenated with the 3 verifier-prefix numeric scalars
        # broadcast to every draft position. Produces `(γ, H + 3)`.
        # V-prefix deploy mode — `numeric` comes from
        # `verifier.prefix_features(running)` called upstream.
        if drafter_hidden is None or numeric is None:
            raise ValueError(
                "family='hidden_per_pos_numeric' requires drafter hidden "
                "(per-position) + verifier prefix numeric"
            )
        h = drafter_hidden.to(torch.float32).cpu()              # (γ, H)
        gamma = int(h.shape[0])
        num_t = _numeric_tensor(numeric)                         # (3,)
        num_b = num_t.unsqueeze(0).expand(gamma, -1).contiguous()  # (γ, 3)
        return torch.cat([h, num_b], dim=1)                      # (γ, H+3)
    if family in ("hidden_per_pos_hs", "hidden_per_pos_s"):
        # Phase 5: drafter-only per-position scalars (± hidden).
        # `drafter_hidden` here is (γ, H) because the online loop passes
        # pool="per_position" for every hidden_per_pos* family. We also
        # need the 4 per-position scalars, which online_decode must
        # compute from the drafter's log-probs row at each position.
        # These come through build_online_features's `extra` arg (we
        # pack them into the `numeric` slot with per-position vectors
        # in the online loop; see _run_single_prompt).
        raise ValueError(
            "hidden_per_pos_hs/s features are assembled inside "
            "_run_single_prompt; build_online_features should not be "
            "called for them directly."
        )
    if family == "hidden_per_pos_v1":
        # Phase 9 A1/L1: drafter hidden only.
        if drafter_hidden is None:
            raise ValueError(
                f"family={family!r} requires drafter hidden (per-position)"
            )
        return drafter_hidden.to(torch.float32).cpu()
    if family == "hidden_per_pos_v3":
        # Phase 9 A3/L3: drafter hidden + 3 prefix-numeric broadcast.
        if drafter_hidden is None or numeric is None:
            raise ValueError(
                f"family={family!r} requires drafter hidden (per-position) "
                "+ verifier prefix numeric"
            )
        h = drafter_hidden.to(torch.float32).cpu()                # (γ, H)
        gamma = int(h.shape[0])
        num_t = _numeric_tensor(numeric)                           # (3,)
        num_b = num_t.unsqueeze(0).expand(gamma, -1).contiguous()  # (γ, 3)
        return torch.cat([h, num_b], dim=1)                        # (γ, H+3)
    if family in ("hidden_per_pos_v2", "hidden_per_pos_v4"):
        # Phase 9 A2/A4/L2/L4: need drafter-side scalars computed
        # per-round from the actual draft_log_probs (not from the
        # offline record). Assembled inside `_run_single_prompt`.
        raise ValueError(
            f"family={family!r} features are assembled inside "
            "_run_single_prompt; build_online_features should not be "
            "called for them directly."
        )
    raise ValueError(f"unknown family {family!r}")


# ------------------------------------------------------------------
# Per-prompt online decode loop
# ------------------------------------------------------------------


@torch.no_grad()
def _run_single_prompt(
    predictor,
    drafter,
    verifier,
    prefix_ids: torch.Tensor,
    protocol: ProtocolConfig,
    gamma: int,
    T: int,
    max_new_tokens: int,
    tau: float,
    prompt_idx: int,
) -> Tuple[List[PredictorRoundLog], torch.Tensor, float]:
    device = drafter.device
    running = prefix_ids.to(device).clone()
    rounds: List[PredictorRoundLog] = []
    round_idx = 0
    MAX_CTX = protocol.max_verifier_ctx
    prefix_start_len = int(prefix_ids.shape[0])

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
        draft_rng = _make_generator(device, seed ^ protocol.draft_salt)

        # --- 1. drafter forward (always with hidden features) ---
        # Per-position families (hidden_per_pos / _hs / _s / _ml) need the
        # unpooled `(γ, k*H)` tensor; legacy families use the mean-pooled `(H,)`.
        # `layers` picks which transformer blocks to extract at the final
        # DDPM step: pp_ml asks for (penultimate, final); every other family
        # gets only the final block.
        _pool = (
            "per_position"
            if str(predictor.family).startswith("hidden_per_pos")
            else "mean"
        )
        _layers = (
            (-2, -1) if predictor.family == "hidden_per_pos_ml" else (-1,)
        )
        t0 = time.time()
        draft_tokens, draft_log_probs, drafter_hidden = drafter.draft_with_features(
            prefix_ids=running,
            gamma=cur_gamma,
            T=T,
            temperature=protocol.temperature,
            q_mode=protocol.q_mode,
            generator=draft_rng,
            pool=_pool,
            layers=_layers,
        )
        draft_time_s = time.time() - t0

        # --- 2. mode-specific feature build ---
        verifier_prefix_time_s = 0.0
        verifier_prefix_hidden: Optional[torch.Tensor] = None
        numeric: Optional[Dict[str, float]] = None
        if predictor.deploy_mode == "V-prefix":
            t0 = time.time()
            verifier_prefix_hidden, numeric = verifier.prefix_features(running)
            verifier_prefix_time_s = time.time() - t0
        # V-free: NO verifier call at all; both stay None.

        t0 = time.time()
        if predictor.family in (
            "hidden_per_pos_hs", "hidden_per_pos_s",
            "hidden_per_pos_v2", "hidden_per_pos_v4",
        ):
            # Phase 5 / Phase 9: compute drafter-side per-position
            # scalars from the already-available `draft_log_probs` — no
            # extra forward pass. EPS clamp on q_t matches the offline
            # convention in accpre/core/accept.py.
            probs = draft_log_probs.exp()
            entropy_t = -(probs * draft_log_probs).sum(dim=-1)     # (γ,)
            top2 = draft_log_probs.topk(2, dim=-1).values          # (γ, 2)
            margin_t = top2[:, 0] - top2[:, 1]                     # (γ,)
            top1_t = probs.max(dim=-1).values                      # (γ,)
            idx = torch.arange(int(draft_tokens.shape[0]),
                               device=draft_tokens.device)
            q_t = probs[idx, draft_tokens.long()].clamp(min=1e-10)  # (γ,)

            if predictor.family in ("hidden_per_pos_v2", "hidden_per_pos_v4"):
                # Phase 9: 5 scalars [q, log q, entropy, margin, top1].
                log_q_t = torch.log(q_t)                            # (γ,)
                scalars = torch.stack(
                    [q_t, log_q_t, entropy_t, margin_t, top1_t],
                    dim=1,
                ).to(torch.float32).cpu()                           # (γ, 5)
            else:
                # Phase 5: 4 scalars [q, entropy, margin, top1].
                scalars = torch.stack(
                    [q_t, entropy_t, margin_t, top1_t], dim=1,
                ).to(torch.float32).cpu()                           # (γ, 4)

            if predictor.family == "hidden_per_pos_s":
                features = scalars                                  # (γ, 4)
            elif predictor.family == "hidden_per_pos_hs":
                h = drafter_hidden.to(torch.float32).cpu()          # (γ, H)
                features = torch.cat([h, scalars], dim=1)           # (γ, H+4)
            elif predictor.family == "hidden_per_pos_v2":
                h = drafter_hidden.to(torch.float32).cpu()          # (γ, H)
                features = torch.cat([h, scalars], dim=1)           # (γ, H+5)
            else:  # hidden_per_pos_v4
                if verifier_prefix_hidden is None:
                    raise ValueError(
                        "hidden_per_pos_v4 requires verifier prefix hidden; "
                        "deploy_mode must be V-prefix"
                    )
                h = drafter_hidden.to(torch.float32).cpu()          # (γ, H)
                vh = verifier_prefix_hidden.to(torch.float32).cpu()  # (H_v,)
                gamma_local = int(h.shape[0])
                vh_b = vh.unsqueeze(0).expand(
                    gamma_local, -1,
                ).contiguous()                                       # (γ, H_v)
                features = torch.cat([h, scalars, vh_b], dim=1)     # (γ, H+5+H_v)
        else:
            features = build_online_features(
                family=predictor.family,
                drafter_hidden=drafter_hidden,
                verifier_prefix_hidden=verifier_prefix_hidden,
                numeric=numeric,
            )
        # --- 3. predictor inference ---
        # Pass per-position drafted-token IDs only to predictors that
        # actually consume them (Phase 12 tokemb heads). Other heads
        # accept **_kwargs and ignore this. predict_L forwards kwargs
        # through to predict_q2 via PredictorBase.
        predict_kwargs = {}
        base_pred = getattr(predictor, "base", predictor)
        if hasattr(base_pred, "token_emb"):
            predict_kwargs["token_ids"] = draft_tokens.detach().cpu().long()
        L_hat = int(predictor.predict_L(features, float(tau), **predict_kwargs))
        L_hat = max(0, min(L_hat, cur_gamma))
        predictor_time_s = time.time() - t0

        # --- 4. commit: max(1, L_hat) draft tokens; NO extra token ---
        n_commit = max(1, L_hat)
        committed = draft_tokens[:n_commit]

        rounds.append(PredictorRoundLog(
            prompt_idx=prompt_idx,
            round_idx=round_idx,
            round_rng_seed=int(seed),
            gamma=int(cur_gamma),
            T=int(T),
            prefix_len_at_round_start=int(running.shape[0]),
            L_hat=int(L_hat),
            n_committed=int(n_commit),
            draft_tokens=[int(x) for x in draft_tokens.tolist()],
            committed_tokens=[int(x) for x in committed.tolist()],
            draft_time_s=float(draft_time_s),
            verifier_prefix_time_s=float(verifier_prefix_time_s),
            predictor_time_s=float(predictor_time_s),
        ))

        running = torch.cat([running, committed])
        round_idx += 1

    elapsed = time.time() - t_start
    return rounds, running, float(elapsed)


def run_predictor_lane(
    predictor,
    drafter,
    verifier,
    prompts: List[Tuple[torch.Tensor, str]],
    protocol: ProtocolConfig,
    gamma: int,
    T: int,
    max_new_tokens: int,
    tau: float,
    prompt_indices: Optional[List[int]] = None,
) -> PredictorLaneResult:
    """Execute the cheap online decode loop for every prompt.

    Identical loop body for acceptance and length predictors — they
    differ only in how `predict_L(features, tau)` is computed.

    `prompt_indices` optionally overrides the default local `enumerate`
    indices. Supplying the **global** prompt indices (matching the
    Stage 2A.0 collection convention) is required for matched-randomness
    CF@1 replay against stored strict records.

    Returns a full log structure for metric computation + plotting.
    """
    if prompt_indices is not None and len(prompt_indices) != len(prompts):
        raise ValueError(
            f"prompt_indices length {len(prompt_indices)} != prompts length {len(prompts)}"
        )
    per_prompt: List[PredictorPromptResult] = []
    for i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[i]) if prompt_indices is not None else i
        rounds, generated, elapsed = _run_single_prompt(
            predictor=predictor,
            drafter=drafter,
            verifier=verifier,
            prefix_ids=prefix_ids,
            protocol=protocol,
            gamma=gamma,
            T=T,
            max_new_tokens=max_new_tokens,
            tau=float(tau),
            prompt_idx=p_idx,
        )
        n_new = int(generated.shape[0]) - int(prefix_ids.shape[0])
        per_prompt.append(PredictorPromptResult(
            prompt_idx=p_idx,
            n_new_tokens=int(n_new),
            elapsed_s=float(elapsed),
            tok_s=float(n_new) / max(elapsed, 1e-9),
            generated_ids=[int(x) for x in generated.tolist()],
            rounds=rounds,
        ))

    return PredictorLaneResult(
        predictor_name=type(predictor).__name__,
        kind=predictor.kind,
        family=predictor.family,
        deploy_mode=predictor.deploy_mode,
        tau=float(tau),
        gamma=int(gamma),
        T=int(T),
        per_prompt=per_prompt,
    )


# ------------------------------------------------------------------
# Joint-mode strict re-collection (measurement-gap fix)
# ------------------------------------------------------------------


@torch.no_grad()
def recollect_strict_with_drafter(
    prompts: List[Tuple[torch.Tensor, str]],
    prompt_indices: List[int],
    protocol: ProtocolConfig,
    drafter,
    verifier,
    gamma: int,
    T: int,
    max_new_tokens: int,
):
    """Re-run strict SpecDiff records using the drafter currently loaded
    (for joint checkpoints, this is the fine-tuned MDLM).

    Purpose: align the CF@1 offline-replay reference with the drafter
    state used by the online tok/s measurement. Without this, the two
    Pareto axes are measured against different drafter states — see
    DESIGN_PHASE2 joint measurement-gap fix.

    Returns a list of `RoundRecord`s with optional feature fields
    populated. These records are equivalent to what
    `accpre.collect.cli` would write, but scoped to the evaluation
    prompts and produced in-memory.
    """
    from accpre.core.draft_verify import draft_verify_round_with_features

    MAX_CTX = protocol.max_verifier_ctx
    out: List = []
    for prompt_i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[prompt_i])
        prefix_ids = prefix_ids.to(drafter.device)
        running = prefix_ids.clone()
        round_idx = 0
        while (running.shape[0] - prefix_ids.shape[0]) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_ids.shape[0])
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(int(gamma), int(remaining), int(ctx_room))
            if cur_gamma <= 0:
                break
            seed = derive_seed(protocol, p_idx, round_idx)
            rec = draft_verify_round_with_features(
                prefix_ids=running,
                drafter=drafter, verifier=verifier,
                gamma=cur_gamma, T=T, protocol=protocol,
                prompt_idx=p_idx, round_idx=round_idx,
                round_rng_seed=seed,
            )
            out.append(rec)
            draft_pref = torch.tensor(
                rec.draft_tokens[: rec.L], dtype=torch.long, device=drafter.device,
            )
            extra = torch.tensor(
                [rec.bonus_or_fallback_token], dtype=torch.long, device=drafter.device,
            )
            running = torch.cat([running, draft_pref, extra])
            round_idx += 1
    return out


# ------------------------------------------------------------------
# CF@1 bridge — uses the offline faithfulness function
# ------------------------------------------------------------------


def offline_cf_at_1(
    predictor,
    strict_records,  # List[RoundRecord] from Phase 1 strict
    tau: float,
    bootstrap_samples: int = 1000,
) -> Dict[str, Any]:
    """Compute offline CF@1 for this predictor at this τ.

    Uses the Phase-1 canonical `controlled_faithfulness_at_1`. For each
    strict record, builds features from the record's optional fields
    and derives `L̂(τ) = predictor.predict_L(features, τ)`. Compares to
    `record.L`. Under the v1 shared-fallback coupling this reduces to
    `mean 1[L̂ == L_strict]`.
    """
    from accpre.collect.features import extract_features
    from accpre.eval.faithfulness import controlled_faithfulness_at_1

    # Tokemb heads require per-position drafted-token IDs at inference;
    # pass them through predict_L → predict_q2 via **kwargs. Other heads
    # accept **_kwargs and ignore.
    base_pred = getattr(predictor, "base", predictor)
    _needs_tok = hasattr(base_pred, "token_emb")

    def _commit_fn(record):
        feats = extract_features(record, predictor.family)
        kw = {}
        if _needs_tok:
            kw["token_ids"] = torch.tensor(record.draft_tokens, dtype=torch.long)
        return int(predictor.predict_L(feats, float(tau), **kw))

    return controlled_faithfulness_at_1(
        strict_records, method_commit_fn=_commit_fn,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=0,
    )


# ------------------------------------------------------------------
# CLI — run one predictor checkpoint through online decode + offline CF@1
# ------------------------------------------------------------------


def _load_protocol(path: str) -> ProtocolConfig:
    import yaml
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


class _TemperatureCalibrated(torch.nn.Module):
    """Wraps an acceptance predictor with post-hoc temperature scaling.

    Given the base predictor's Q̂ = sigmoid(z), this wrapper returns
    Q̂_cal = sigmoid(z / T) by inverting the sigmoid, dividing the
    logit by a fixed scalar `T > 0`, and re-applying sigmoid. Used only
    for acceptance predictors; length predictors don't go through this.

    The wrapper is transparent to `commit_threshold` — it keeps the
    same kind / family / deploy_mode / gamma and overrides only
    `predict_q2` and `predict_L`. No weights of the base predictor are
    changed; T comes from `<ckpt>/temperature.json`.
    """

    def __init__(self, base, T: float) -> None:
        super().__init__()
        if base.kind != "acceptance":
            raise ValueError(
                f"Temperature calibration wraps acceptance predictors only; "
                f"got kind={base.kind!r}"
            )
        self.base = base
        self.T = float(T)
        # Mirror metadata that callers (online decode, compare) read.
        self.kind = base.kind
        self.family = base.family
        self.deploy_mode = base.deploy_mode
        self.gamma = base.gamma

    def _calibrate(self, q: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        q = q.clamp(eps, 1.0 - eps)
        z = torch.log(q / (1.0 - q))
        return torch.sigmoid(z / self.T)

    def predict_q2(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        q = self.base.predict_q2(features, **kwargs)
        return self._calibrate(q)

    def predict_L(self, features: torch.Tensor, tau: float, **kwargs) -> int:
        from accpre.core.commit import commit_threshold
        q = self.predict_q2(features, **kwargs)
        if q.dim() == 2:
            q = q.squeeze(0)
        q_list = [float(x) for x in q.detach().cpu().tolist()]
        return commit_threshold(q_list, float(tau))

    def describe(self):
        d = dict(self.base.describe())
        d["class"] = f'{d.get("class", type(self.base).__name__)}+T={self.T:.3f}'
        return d


def _load_predictor_from_checkpoint(checkpoint_dir: str, gamma: int):
    """Rebuild the predictor from its config + state_dict.

    Returns `(predictor, cfg, drafter_state_path)`. The third element is
    `None` for frozen-mode checkpoints; for Stage 2B.1 joint checkpoints
    it points to `<ckpt>/drafter.pt` so the caller can override the
    global drafter's weights before running the online loop.

    If `<checkpoint_dir>/temperature.json` exists and the predictor is
    an acceptance predictor, the loaded predictor is wrapped with
    `_TemperatureCalibrated` at the fitted T (post-hoc calibration,
    no architecture change).
    """
    import yaml
    with open(os.path.join(checkpoint_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    # Dependence target (Phase 23) shares the acceptance-predictor
    # heads and inference path — only the training target differs.
    # Dispatch into the acceptance branch so online_decode can load
    # dep_frz / dep_jnt checkpoints without any other change.
    if cfg["target"] in ("acceptance", "dependence"):
        if str(cfg["family"]).startswith("hidden_per_pos"):
            head_arch = str(cfg.get("head_arch", "base"))
            if head_arch == "base":
                from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPos
                p = AcceptanceMLPPerPos(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    hidden_dim=int(cfg.get("hidden_dim", 128)),
                    dropout=float(cfg.get("dropout", 0.1)),
                )
            elif head_arch == "wide":
                from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPosWide
                p = AcceptanceMLPPerPosWide(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    hidden_dim=int(cfg.get("hidden_dim", 256)),
                    dropout=float(cfg.get("dropout", 0.1)),
                )
            elif head_arch == "attn":
                from accpre.predictors.acceptance_mlp import AcceptanceMLPPerPosAttn
                p = AcceptanceMLPPerPosAttn(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    hidden_dim=int(cfg.get("hidden_dim", 128)),
                    dropout=float(cfg.get("dropout", 0.1)),
                )
            elif head_arch == "tx":
                # Phase 9: per-position transformer family (A1..A4).
                from accpre.predictors.pp_transformer import AcceptanceTx
                p = AcceptanceTx(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    d_model=int(cfg.get("d_model", 128)),
                    num_layers=int(cfg.get("num_layers", 2)),
                    num_heads=int(cfg.get("num_heads", 4)),
                    dropout=float(cfg.get("dropout", 0.1)),
                )
            elif head_arch == "tokemb_mlp":
                # Phase 12 / 1A.
                from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmb
                p = AcceptanceMLPPerPosTokEmb(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    hidden_dim=int(cfg.get("hidden_dim", 128)),
                    dropout=float(cfg.get("dropout", 0.1)),
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            elif head_arch == "tokemb_mlp_sepln":
                # Phase 13 / 1A fusion-C ablation.
                from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmbSepLN
                p = AcceptanceMLPPerPosTokEmbSepLN(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    hidden_dim=int(cfg.get("hidden_dim", 128)),
                    dropout=float(cfg.get("dropout", 0.1)),
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            elif head_arch == "tokemb_tx":
                # Phase 12 / 1B.
                from accpre.predictors.pp_tokemb import AcceptanceTxTokEmb
                p = AcceptanceTxTokEmb(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    d_model=int(cfg.get("d_model", 128)),
                    num_layers=int(cfg.get("num_layers", 2)),
                    num_heads=int(cfg.get("num_heads", 4)),
                    dropout=float(cfg.get("dropout", 0.1)),
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            elif head_arch == "thresh_multihead":
                # Phase 14: direct threshold classifier.
                from accpre.predictors.pp_threshold import AcceptanceThresholdMultiHead
                p = AcceptanceThresholdMultiHead(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    hidden_dim=int(cfg.get("hidden_dim", 128)),
                    dropout=float(cfg.get("dropout", 0.1)),
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                )
            elif head_arch == "tokemb_mlp_vhproj":
                # Phase 19 Vlite-H: 1A + projected verifier-hidden prefix.
                from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmbVHProj
                p = AcceptanceMLPPerPosTokEmbVHProj(
                    family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                    hidden_dim=int(cfg.get("hidden_dim", 128)),
                    dropout=float(cfg.get("dropout", 0.1)),
                    token_emb_dim=int(cfg.get("token_emb_dim", 64)),
                    vh_proj_dim=int(cfg.get("vh_proj_dim", 64)),
                )
            else:
                raise ValueError(f"unknown head_arch {head_arch!r}")
        else:
            from accpre.predictors.acceptance_mlp import AcceptanceMLP
            p = AcceptanceMLP(
                family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                hidden_dim=int(cfg.get("hidden_dim", 128)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
    elif cfg["target"] == "committed_length":
        head_arch = str(cfg.get("head_arch", "base"))
        if head_arch == "tx":
            # Phase 9: per-position transformer length head (L1..L4).
            from accpre.predictors.pp_transformer import LengthTx
            p = LengthTx(
                family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                d_model=int(cfg.get("d_model", 128)),
                num_layers=int(cfg.get("num_layers", 2)),
                num_heads=int(cfg.get("num_heads", 4)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
        else:
            from accpre.predictors.length_mlp import LengthMLP
            p = LengthMLP(
                family=cfg["family"], gamma=int(cfg.get("gamma", gamma)),
                hidden_dim=int(cfg.get("hidden_dim", 128)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
    else:
        raise ValueError(f"unknown target {cfg['target']!r}")
    p.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt"),
                                 map_location="cpu"))
    p.eval()

    # Optional post-hoc temperature calibration (acceptance only).
    temp_path = os.path.join(checkpoint_dir, "temperature.json")
    if os.path.isfile(temp_path) and cfg["target"] == "acceptance":
        with open(temp_path, "r") as f:
            T = float(json.load(f)["T"])
        p = _TemperatureCalibrated(p, T=T)
        print(f"[online] applied post-hoc temperature calibration T={T:.4f}")

    drafter_state_path = None
    if str(cfg.get("mode", "frozen")) in ("joint", "joint_multitask"):
        cand = os.path.join(checkpoint_dir, "drafter.pt")
        if os.path.isfile(cand):
            drafter_state_path = cand
        else:
            raise FileNotFoundError(
                f"joint checkpoint {checkpoint_dir!r} missing drafter.pt — "
                f"online decode cannot reproduce the fine-tuned drafter."
            )
    return p, cfg, drafter_state_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2A.3 — online predictor decode (cheap lane)."
    )
    ap.add_argument("--config", type=str, required=True, help="protocol.yaml")
    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="Directory holding model.pt + config.yaml from trainer.")
    ap.add_argument("--strict_records", type=str, required=True,
                    help="Path to Phase-1 strict records (for CF@1 offline replay).")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--taus", type=str, default="0.3,0.5,0.7,0.9")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--drafter_state", type=str, default=None,
        help="Optional drafter .pt state_dict to load, overriding the "
             "default pretrained MDLM and any joint-checkpoint drafter.pt. "
             "Used for Phase 18 (aligned drafter + frozen head).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    protocol = _load_protocol(args.config)
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    from accpre.core.schema import load_records
    from accpre.data.splits import test_split, val_split, train_split
    from accpre.eval.quality import xl_audit
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
    }[protocol.dtype]

    # Predictor (CPU is fine for frozen MLPs).
    predictor, cfg, drafter_state_path = _load_predictor_from_checkpoint(
        args.checkpoint_dir, gamma=args.gamma,
    )
    print(f"[online] predictor: {predictor.describe()}")

    # Models.
    drafter = MDLMDrafter(model_name=protocol.drafter_model,
                          device=device, dtype=dtype)
    verifier = GPT2Verifier(model_name=protocol.verifier_model,
                            device=device, dtype=dtype)
    # Drafter-state override precedence:
    #   1) CLI --drafter_state (Phase 18: aligned drafter + frozen head).
    #   2) Joint-checkpoint drafter.pt (Phase 15/16 joint mode).
    #   3) Pretrained MDLM (default).
    state_path = args.drafter_state or drafter_state_path
    if state_path is not None:
        src = ("CLI override" if args.drafter_state else "joint checkpoint")
        print(f"[online] loading drafter state ({src}): {state_path}")
        drafter.model.load_state_dict(
            torch.load(state_path, map_location=device)
        )
        drafter.model.eval()

    # Prompts — and their global pool indices so seed derivation matches
    # Stage 2A.0 collection (which uses global indices). This is
    # load-bearing for matched-randomness CF@1 replay.
    from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N
    split_fn = {"train": train_split, "val": val_split, "test": test_split}[args.split]
    split_offset = {
        "train": 0, "val": TRAIN_N, "test": TRAIN_N + VAL_N,
    }[args.split]
    prompts = split_fn()[:args.n_prompts]
    prompt_indices = list(range(split_offset, split_offset + len(prompts)))
    print(
        f"[online] {len(prompts)} prompts from split={args.split} "
        f"(global indices {prompt_indices[0]}..{prompt_indices[-1]})"
    )

    # Offline CF@1 needs strict records for the SAME prompt indices and
    # SAME protocol. For FROZEN checkpoints the canonical source is
    # `data_collected/stage1.pt` (Stage 2A.0). For JOINT checkpoints
    # the fine-tuned drafter means Stage-1 records are on the wrong
    # drafter state — we re-collect strict records in-memory using the
    # currently-loaded fine-tuned drafter so CF@1 and tok/s share one
    # drafter state (DESIGN_PHASE2 joint measurement-gap fix).
    if drafter_state_path is not None:
        print(
            "[online] JOINT mode: re-collecting strict records with the "
            "fine-tuned drafter (CF@1 / tok_s drafter-state alignment)"
        )
        from accpre.core.schema import save_records as _save_records
        strict = recollect_strict_with_drafter(
            prompts=prompts, prompt_indices=prompt_indices,
            protocol=protocol, drafter=drafter, verifier=verifier,
            gamma=args.gamma, T=args.T, max_new_tokens=args.max_new_tokens,
        )
        recol_path = os.path.join(args.out_dir, "strict_records_recollected.pt")
        _save_records(strict, recol_path)
        print(
            f"[online] re-collected {len(strict)} strict records "
            f"→ {recol_path}"
        )
    else:
        strict = load_records(args.strict_records, expected_protocol=protocol)
        allowed = set(prompt_indices)
        strict = [r for r in strict if int(r.prompt_idx) in allowed]
        print(
            f"[online] filtered strict records to {len(strict)} "
            f"(matching global prompt indices)"
        )

    for tau in taus:
        print(f"[online] running tau={tau} ...")
        lane = run_predictor_lane(
            predictor=predictor, drafter=drafter, verifier=verifier,
            prompts=prompts, protocol=protocol, gamma=args.gamma, T=args.T,
            max_new_tokens=args.max_new_tokens, tau=tau,
            prompt_indices=prompt_indices,
        )

        # XL-audit on the online-generated sequences (auxiliary).
        xl_per_prompt = []
        for (prefix, _), prompt_result in zip(prompts, lane.per_prompt):
            gen = torch.tensor(prompt_result.generated_ids, dtype=torch.long,
                               device=device)
            xl_per_prompt.append(
                float(xl_audit(gen, int(prefix.shape[0]), verifier))
            )
        xl_mean = sum(xl_per_prompt) / max(len(xl_per_prompt), 1)

        # CF@1 offline via replay on strict records.
        cf = offline_cf_at_1(predictor, strict, tau=tau)

        summary = {
            "protocol_fingerprint": protocol.fingerprint(),
            "predictor": predictor.describe(),
            "predictor_config": cfg,
            "tau": float(tau),
            "n_prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "gamma": args.gamma,
            "T": args.T,
            "l3": {
                "tok_s_per_prompt": lane.tok_s_per_prompt,
                "tok_s_mean": lane.tok_s_mean,
                "cf_at_1_aggregate": cf["aggregate"],
                "cf_at_1_bootstrap_ci": cf["bootstrap_ci"],
                "cf_at_1_per_prompt": cf["per_prompt"],
                "xl_audit_aux_per_prompt": xl_per_prompt,
                "xl_audit_aux_mean": xl_mean,
            },
            "online_lane": lane.to_dict(),
        }

        tag = f"{predictor.kind}_tau_{tau}".replace(".", "p")
        path = os.path.join(args.out_dir, f"online_{tag}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[online]   tok/s={lane.tok_s_mean:.2f}  "
            f"CF@1={cf['aggregate']:.3f}  "
            f"XL-aux={xl_mean:.3f}  -> {path}"
        )


if __name__ == "__main__":
    main()
