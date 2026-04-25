"""Canonical draft -> verify -> accept -> commit composer.

This module is the single place where a SpecDiff round is composed end
to end. Everyone else (baselines, eval lanes, future predictors) imports
from here or reuses the exposed lower-level helpers.

Two public entry points:

  draft_verify_accept(...)  -> RoundArtifacts
      Shared scaffold: runs the drafter with draft_rng, the verifier,
      and the per-position accept test with accept_rng. Does NOT commit;
      the caller applies a commit rule to the returned AcceptOutcome.

  draft_verify_round(...)   -> RoundRecord
      The canonical strict-rule round. Composes draft_verify_accept +
      commit_strict + fallback/bonus sampling, and packages the result
      into a fully populated RoundRecord.

Matched-randomness discipline (DESIGN.md §D.5.2):
  - `round_rng_seed` is the only randomness input.
  - Three independent sub-streams are derived via protocol salts:
        draft_rng    = Generator(seed XOR protocol.draft_salt)
        accept_rng   = Generator(seed XOR protocol.accept_salt)
        fallback_rng = Generator(seed XOR protocol.fallback_salt)
  - Under identical round_rng_seed, two lanes consuming the same
    (prefix_ids, drafter, verifier, gamma, T, protocol) observe
    identical draft_tokens, draft_log_probs, target_log_probs, U_j, and
    fallback_rng state at the moment of the extra-token sample. Only
    their commit rules and therefore their L values differ.

The Phase-4 anti-drift test will additionally assert that a training-
time joint predictor uses this same draft-verify-accept path (no inline
DDPM re-implementation). Phase 1 does not exercise that test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Union

import torch

from accpre.core.accept import AcceptOutcome, per_position_accept
from accpre.core.commit import commit_strict
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import RoundRecord


def _make_generator(
    device: Union[str, torch.device], seed: int
) -> torch.Generator:
    """Build a `torch.Generator` on the right device with a masked seed.

    `torch.Generator.manual_seed` expects an int64; we mask to 63 bits to
    avoid overflow surprises across platforms.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


@dataclass
class RoundArtifacts:
    """Raw outputs of draft_verify_accept, shared across lanes.

    A caller picks a commit rule, runs it on `outcome`, and then feeds
    the resulting L into `sample_fallback_or_bonus` using the returned
    `fallback_rng`.
    """
    draft_tokens: torch.Tensor          # (gamma,)
    draft_log_probs: torch.Tensor       # (gamma, vocab_drafter)
    target_log_probs: torch.Tensor      # (prefix_len+gamma, vocab_verifier)
    outcome: AcceptOutcome              # per-position arrays incl. U_j, min_pq_j
    prefix_len: int
    gamma: int
    T: int
    round_rng_seed: int
    draft_time: float
    verify_time: float
    fallback_rng: torch.Generator       # untouched by accept; pass to fallback


@torch.no_grad()
def draft_verify_accept(
    prefix_ids: torch.Tensor,
    drafter,
    verifier,
    gamma: int,
    T: int,
    protocol: ProtocolConfig,
    round_rng_seed: int,
) -> RoundArtifacts:
    """Shared draft + verify + accept scaffold.

    Does NOT apply a commit rule. The caller is responsible for applying
    `commit_strict`, `commit_threshold`, or (future) a predictor-derived
    rule to the returned `outcome`, and then calling
    `sample_fallback_or_bonus` with the returned `fallback_rng`.
    """
    device = prefix_ids.device
    prefix_len = int(prefix_ids.shape[0])

    draft_rng = _make_generator(device, round_rng_seed ^ protocol.draft_salt)
    accept_rng = _make_generator(device, round_rng_seed ^ protocol.accept_salt)
    fallback_rng = _make_generator(device, round_rng_seed ^ protocol.fallback_salt)

    t0 = time.time()
    draft_tokens, draft_log_probs = drafter.draft(
        prefix_ids=prefix_ids,
        gamma=gamma,
        T=T,
        temperature=protocol.temperature,
        q_mode=protocol.q_mode,
        generator=draft_rng,
    )
    draft_time = time.time() - t0

    t0 = time.time()
    candidate = torch.cat([prefix_ids, draft_tokens])
    target_log_probs = verifier.score(candidate)
    verify_time = time.time() - t0

    outcome = per_position_accept(
        draft_tokens=draft_tokens,
        draft_log_probs=draft_log_probs,
        target_log_probs=target_log_probs,
        prefix_len=prefix_len,
        protocol=protocol,
        accept_rng=accept_rng,
    )

    return RoundArtifacts(
        draft_tokens=draft_tokens,
        draft_log_probs=draft_log_probs,
        target_log_probs=target_log_probs,
        outcome=outcome,
        prefix_len=prefix_len,
        gamma=int(gamma),
        T=int(T),
        round_rng_seed=int(round_rng_seed),
        draft_time=float(draft_time),
        verify_time=float(verify_time),
        fallback_rng=fallback_rng,
    )


def sample_fallback_or_bonus(
    L: int,
    gamma: int,
    draft_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    prefix_len: int,
    protocol: ProtocolConfig,
    fallback_rng: torch.Generator,
) -> int:
    """Sample the single extra token appended after the accepted prefix.

    Shared-fallback protocol used by strict and every v1 lossy/predictor
    lane:
      - L == gamma (full accept): sample BONUS from target_log_probs at
        the position following the last accepted draft token.
      - L <  gamma (partial accept): sample FALLBACK from the adjusted
        distribution norm(max(0, p - q)) at position L.

    `fallback_rng` is the per-round fallback generator. Under matched
    randomness, this function produces the same token across lanes that
    share `round_rng_seed` and agree on `L`.
    """
    if L == gamma:
        # Bonus token: position right after the last accepted draft token.
        pos = prefix_len - 1 + gamma
        logits_row = target_log_probs[pos]
        if protocol.temperature == 0.0:
            return int(logits_row.argmax().item())
        probs = (logits_row / protocol.temperature).softmax(dim=-1)
        return int(
            torch.multinomial(probs, num_samples=1, generator=fallback_rng).item()
        )

    # Partial-accept fallback: adjusted distribution at position L.
    pos = prefix_len - 1 + L
    target_row = target_log_probs[pos]
    draft_row = draft_log_probs[L]

    if protocol.temperature == 0.0:
        return int(target_row.argmax().item())

    # Vocabulary-size mismatch guard (MDLM 50258 vs GPT-2 50257).
    p = target_row.exp()
    q = draft_row.exp()
    min_size = min(p.shape[0], q.shape[0])
    p_trim = p[:min_size]
    q_trim = q[:min_size]

    adjusted = torch.clamp(p_trim - q_trim, min=0.0)
    total = adjusted.sum()
    if float(total.item()) < 1e-10:
        return int(p_trim.argmax().item())
    adjusted = adjusted / total
    return int(
        torch.multinomial(adjusted, num_samples=1, generator=fallback_rng).item()
    )


def build_record(
    artifacts: RoundArtifacts,
    L: int,
    bonus_or_fallback_token: int,
    protocol: ProtocolConfig,
    prompt_idx: int,
    round_idx: int,
) -> RoundRecord:
    """Package RoundArtifacts + commit outputs into a RoundRecord."""
    out = artifacts.outcome
    return RoundRecord(
        schema_version=protocol.schema_version,
        protocol=protocol,
        prompt_idx=int(prompt_idx),
        round_idx=int(round_idx),
        round_rng_seed=int(artifacts.round_rng_seed),
        gamma=int(artifacts.gamma),
        T=int(artifacts.T),
        prefix_len=int(artifacts.prefix_len),
        draft_tokens=[int(t) for t in artifacts.draft_tokens.tolist()],
        q_j=list(out.q_j),
        p_j=list(out.p_j),
        min_pq_j=list(out.min_pq_j),
        U_j=list(out.U_j),
        accepted_j=list(out.accepted_j),
        survived_j=list(out.survived_j),
        L=int(L),
        bonus_or_fallback_token=int(bonus_or_fallback_token),
        draft_time=float(artifacts.draft_time),
        verify_time=float(artifacts.verify_time),
    )


def draft_verify_round(
    prefix_ids: torch.Tensor,
    drafter,
    verifier,
    gamma: int,
    T: int,
    protocol: ProtocolConfig,
    prompt_idx: int,
    round_idx: int,
    round_rng_seed: int,
) -> RoundRecord:
    """Canonical strict-rule draft-verify-accept-commit round.

    Composes `draft_verify_accept` + `commit_strict` + the shared-
    fallback extra-token sample. Returns a fully populated RoundRecord.
    """
    artifacts = draft_verify_accept(
        prefix_ids=prefix_ids,
        drafter=drafter,
        verifier=verifier,
        gamma=gamma,
        T=T,
        protocol=protocol,
        round_rng_seed=round_rng_seed,
    )
    L = commit_strict(artifacts.outcome.accepted_j)
    bonus = sample_fallback_or_bonus(
        L=L,
        gamma=artifacts.gamma,
        draft_log_probs=artifacts.draft_log_probs,
        target_log_probs=artifacts.target_log_probs,
        prefix_len=artifacts.prefix_len,
        protocol=protocol,
        fallback_rng=artifacts.fallback_rng,
    )
    return build_record(
        artifacts=artifacts,
        L=L,
        bonus_or_fallback_token=bonus,
        protocol=protocol,
        prompt_idx=prompt_idx,
        round_idx=round_idx,
    )


@torch.no_grad()
def draft_verify_round_with_features(
    prefix_ids: torch.Tensor,
    drafter,
    verifier,
    gamma: int,
    T: int,
    protocol: ProtocolConfig,
    prompt_idx: int,
    round_idx: int,
    round_rng_seed: int,
) -> RoundRecord:
    """Strict round + populated optional feature fields for predictor training.

    Identical semantics to `draft_verify_round` (same matched-randomness
    discipline, same accept/commit/fallback path) PLUS:
      - uses `drafter.draft_with_features` → drafter_hidden is extracted
        in the same final-step forward (no extra drafter forward);
      - calls `verifier.prefix_features(prefix_ids)` once per round for
        verifier prefix hidden + numerics.

    The returned RoundRecord has the optional feature fields populated
    (`verifier_hidden`, `drafter_hidden`, `prefix_tail`,
    `verifier_entropy`, `verifier_margin`, `verifier_top1_prob`). The
    Phase-1 schema version is preserved because those fields were
    declared optional in `RoundRecord` from day one.

    Cost overhead vs `draft_verify_round`:
      - one verifier prefix forward (prefix_len tokens) per round.
      - one extra tensor slice + mean-pool on the drafter forward
        (negligible).
    """
    device = prefix_ids.device
    prefix_len = int(prefix_ids.shape[0])

    draft_rng = _make_generator(device, round_rng_seed ^ protocol.draft_salt)
    accept_rng = _make_generator(device, round_rng_seed ^ protocol.accept_salt)
    fallback_rng = _make_generator(device, round_rng_seed ^ protocol.fallback_salt)

    # Verifier prefix features. This is a PREFIX-ONLY forward; it is
    # independent of verifier.score() on (prefix + draft) below.
    verifier_hidden, numeric = verifier.prefix_features(prefix_ids)

    # Drafter with hidden features (single-forward on final step).
    # Use pool="per_position" + layers=(-2, -1) so the record can store
    # every drafter-hidden feature any current v1 predictor family needs:
    #   - drafter_hidden         (mean-pooled final layer) — legacy
    #   - drafter_hidden_per_pos (γ, H) final layer        — pp_base / pp_hs / pp_s
    #   - drafter_hidden_per_pos_ml (γ, 2H) concat(penultimate, final)  — pp_ml
    # The multi-layer tensor is computed once; the final-layer (γ, H) view
    # is a slice of its last H columns by construction.
    t0 = time.time()
    draft_tokens, draft_log_probs, drafter_hidden_per_pos_ml = drafter.draft_with_features(
        prefix_ids=prefix_ids,
        gamma=gamma,
        T=T,
        temperature=protocol.temperature,
        q_mode=protocol.q_mode,
        generator=draft_rng,
        pool="per_position",
        layers=(-2, -1),
    )
    draft_time = time.time() - t0
    if drafter_hidden_per_pos_ml is not None:
        H_final = int(drafter_hidden_per_pos_ml.shape[-1]) // 2
        drafter_hidden_per_pos = drafter_hidden_per_pos_ml[:, H_final:].contiguous()
        drafter_hidden = drafter_hidden_per_pos.mean(dim=0)
    else:
        drafter_hidden_per_pos = None
        drafter_hidden = None

    t0 = time.time()
    candidate = torch.cat([prefix_ids, draft_tokens])
    target_log_probs = verifier.score(candidate)
    verify_time = time.time() - t0

    outcome = per_position_accept(
        draft_tokens=draft_tokens,
        draft_log_probs=draft_log_probs,
        target_log_probs=target_log_probs,
        prefix_len=prefix_len,
        protocol=protocol,
        accept_rng=accept_rng,
    )
    L = commit_strict(outcome.accepted_j)

    bonus_token = sample_fallback_or_bonus(
        L=L,
        gamma=int(gamma),
        draft_log_probs=draft_log_probs,
        target_log_probs=target_log_probs,
        prefix_len=prefix_len,
        protocol=protocol,
        fallback_rng=fallback_rng,
    )

    prefix_tail = prefix_ids[-32:].detach().cpu() if prefix_len >= 32 else prefix_ids.detach().cpu()

    # Phase 5 per-position drafter scalars, derived from `draft_log_probs`:
    #   entropy   = -Σ_v p_θ[v] log p_θ[v]
    #   margin    = log p_θ[top1] - log p_θ[top2]
    #   top1_prob = max_v p_θ[v]
    # Computed once per round from the already-computed SUBS log_p_x0 rows.
    log_probs = draft_log_probs                             # (γ, vocab)
    probs = log_probs.exp()
    entropy_t = -(probs * log_probs).sum(dim=-1)            # (γ,)
    top2 = log_probs.topk(2, dim=-1).values                 # (γ, 2)
    margin_t = top2[:, 0] - top2[:, 1]                      # (γ,)
    top1_prob_t = probs.max(dim=-1).values                  # (γ,)
    entropies = entropy_t.detach().cpu().tolist()
    margins = margin_t.detach().cpu().tolist()
    top1_probs = top1_prob_t.detach().cpu().tolist()

    return RoundRecord(
        schema_version=protocol.schema_version,
        protocol=protocol,
        prompt_idx=int(prompt_idx),
        round_idx=int(round_idx),
        round_rng_seed=int(round_rng_seed),
        gamma=int(gamma),
        T=int(T),
        prefix_len=prefix_len,
        draft_tokens=[int(t) for t in draft_tokens.tolist()],
        q_j=list(outcome.q_j),
        p_j=list(outcome.p_j),
        min_pq_j=list(outcome.min_pq_j),
        U_j=list(outcome.U_j),
        accepted_j=list(outcome.accepted_j),
        survived_j=list(outcome.survived_j),
        L=int(L),
        bonus_or_fallback_token=int(bonus_token),
        draft_time=float(draft_time),
        verify_time=float(verify_time),
        # Optional feature fields (populated here, None in Phase-1 strict).
        verifier_hidden=verifier_hidden,
        drafter_hidden=drafter_hidden,
        drafter_hidden_per_pos=drafter_hidden_per_pos,
        drafter_hidden_per_pos_ml=drafter_hidden_per_pos_ml,
        prefix_tail=prefix_tail,
        verifier_entropy=numeric["entropy"],
        verifier_margin=numeric["margin"],
        verifier_top1_prob=numeric["top1_prob"],
        drafter_entropy_j=[float(x) for x in entropies],
        drafter_margin_j=[float(x) for x in margins],
        drafter_top1_prob_j=[float(x) for x in top1_probs],
    )
