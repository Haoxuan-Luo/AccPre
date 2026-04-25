"""MDLM diffusion drafter — thin wrapper with explicit RNG discipline.

Public API:

  draft(prefix_ids, gamma, T, temperature, q_mode, generator=None)
    -> (draft_tokens, draft_log_probs)

  draft_with_features(prefix_ids, gamma, T, temperature, q_mode, generator=None,
                      pool="mean", layers=(-1,))
    -> (draft_tokens, draft_log_probs, drafter_hidden)
    # drafter_hidden's shape depends on `pool` and `layers`:
    #   pool="mean", layers=(-1,)        → (H,)           mean over γ positions
    #   pool="per_position", layers=(-1,) → (γ, H)         per-position, last layer
    #   pool="per_position", layers=(-2,-1) → (γ, 2H)     per-position, last 2 layers
    #                                                       concatenated along H
    # `layers` lists which entries of the MDLM forward's
    # `output_hidden_states` tuple to extract at the FINAL DDPM step;
    # -1 is the last transformer block's output (current v1 default),
    # -2 is the penultimate block (Phase 6 "pp_ml" enrichment).

Both methods share the SAME underlying DDPM loop (`_run_ddpm_loop`) to
guarantee that hidden-state extraction cannot silently drift from the
inference-only draft path. `draft_with_features` differs only in that
the final-step forward also requests `output_hidden_states=True`.

Matched-randomness discipline (DESIGN.md §D.5.2):
  - `generator` is a `torch.Generator`. When provided, every stochastic
    step inside the DDPM loop — intermediate categorical sampling and
    final-step multinomial sampling — uses it instead of torch's default
    generator. Two calls with the same generator state and same inputs
    produce identical `draft_tokens` and `draft_log_probs`.
  - Intermediate DDPM steps are *always* stochastic, even at
    `temperature == 0`; greedy at intermediate steps would collapse the
    diffusion to a trivial one-shot unmask.
  - Commit-time q extraction: for each draft position j,
    `draft_log_probs[j]` is the SUBS-parameterized `log p_theta(x_0|x_t)`
    row at the DDPM step where j was first unmasked. This is q_mode "A"
    (canonical v1). q_mode "B" applies a per-step log-factor shift;
    both select the same drafted token.
"""

from __future__ import annotations

import contextlib
import math
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MASK_INDEX: int = 50257  # MDLM mask-token id (appended to GPT-2 vocab of 50257)


class MDLMDrafter:
    """Pretrained MDLM drafter with a clean `draft` / `draft_with_features` API."""

    def __init__(
        self,
        model_name: str = "kuleshov-group/mdlm-owt",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.mask_index = MASK_INDEX

        print(f"[MDLMDrafter] Loading model: {model_name}")
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size: int = 50258       # GPT-2 (50257) + mask token
        self.max_seq_len: int = 1024
        # MDLM hidden size can be read from config if present. Fall back to 768
        # (RoBERTa-base-style) which matches the kuleshov-group checkpoint.
        self.hidden_size: int = int(getattr(self.model.config, "hidden_size", 768))
        # Vocab-layout invariant (audit issue 6): v1 assumes MDLM vocab =
        # GPT-2 vocab + one MASK token at index 50257. Everything in
        # accept/fallback/vocab-trim logic depends on this. If swapping the
        # drafter, revise this assertion and the trim logic in draft_verify.
        assert self.vocab_size == 50258, (
            f"MDLM drafter vocab_size must be 50258 (GPT-2 50257 + MASK), "
            f"got {self.vocab_size}. See audit issue 6."
        )
        print(f"[MDLMDrafter] Loaded. vocab_size={self.vocab_size} hidden_size={self.hidden_size}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def draft(
        self,
        prefix_ids: torch.Tensor,
        gamma: int,
        T: int = 2,
        temperature: float = 1.0,
        eps: float = 1e-5,
        q_mode: str = "A",
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft `gamma` tokens. Returns (draft_tokens, draft_log_probs)."""
        tokens, log_probs, _hidden = self._run_ddpm_loop(
            prefix_ids=prefix_ids,
            gamma=gamma,
            T=T,
            temperature=temperature,
            eps=eps,
            q_mode=q_mode,
            generator=generator,
            extract_hidden=False,
        )
        return tokens, log_probs

    @torch.no_grad()
    def draft_with_features(
        self,
        prefix_ids: torch.Tensor,
        gamma: int,
        T: int = 2,
        temperature: float = 1.0,
        eps: float = 1e-5,
        q_mode: str = "A",
        generator: Optional[torch.Generator] = None,
        pool: str = "mean",
        layers: Tuple[int, ...] = (-1,),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draft `gamma` tokens AND extract drafter hidden features.

        Final-step forward is done ONCE with `output_hidden_states=True`,
        so `draft_tokens` / `draft_log_probs` are bit-identical to what
        `draft(...)` would produce with the same generator state.

        `pool`:
          - "mean" (default, BACKWARD-COMPAT): returned `drafter_hidden`
            has shape `(hidden_size * len(layers),)`, mean over γ draft
            positions of the per-position hidden rows (already concatenated
            across requested layers), placed on CPU.
          - "per_position": returned `drafter_hidden` has shape
            `(γ, hidden_size * len(layers))`, one row per draft position,
            placed on CPU. Used by the per-position acceptance predictor.

        `layers`: tuple of indices into the MDLM forward's
        `output_hidden_states` tuple, extracted at the FINAL DDPM step
        and concatenated along the hidden axis in the order given.
        Default `(-1,)` = final transformer block only. Phase 6 `pp_ml`
        uses `(-2, -1)` = penultimate + final, in that order, so the last
        H columns correspond to the final layer (matches legacy slicing).
        """
        if pool not in ("mean", "per_position"):
            raise ValueError(f"pool must be 'mean' or 'per_position', got {pool!r}")
        tokens, log_probs, hidden_per_pos = self._run_ddpm_loop(
            prefix_ids=prefix_ids,
            gamma=gamma,
            T=T,
            temperature=temperature,
            eps=eps,
            q_mode=q_mode,
            generator=generator,
            extract_hidden=True,
            grad_at_final_only=False,
            layers=tuple(int(x) for x in layers),
        )
        if hidden_per_pos is None:
            return tokens, log_probs, None
        if pool == "mean":
            hidden = hidden_per_pos.mean(dim=0).detach().cpu()
        else:
            hidden = hidden_per_pos.detach().cpu()
        return tokens, log_probs, hidden

    def draft_with_features_grad(
        self,
        prefix_ids: torch.Tensor,
        gamma: int,
        T: int = 2,
        temperature: float = 1.0,
        eps: float = 1e-5,
        q_mode: str = "A",
        generator: Optional[torch.Generator] = None,
        pool: str = "mean",
        layers: Tuple[int, ...] = (-1,),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage-2B grad variant; see `draft_with_features` for kwargs.

        Returns `drafter_hidden` on-device with the autograd graph
        intact (NOT moved to CPU).
        """
        if pool not in ("mean", "per_position"):
            raise ValueError(f"pool must be 'mean' or 'per_position', got {pool!r}")
        tokens, log_probs, hidden_per_pos = self._run_ddpm_loop(
            prefix_ids=prefix_ids,
            gamma=gamma,
            T=T,
            temperature=temperature,
            eps=eps,
            q_mode=q_mode,
            generator=generator,
            extract_hidden=True,
            grad_at_final_only=True,
            layers=tuple(int(x) for x in layers),
        )
        if hidden_per_pos is None:
            return tokens, log_probs, None
        hidden = hidden_per_pos.mean(dim=0) if pool == "mean" else hidden_per_pos
        return tokens, log_probs, hidden

    # ------------------------------------------------------------------
    # Shared DDPM loop (the single source of truth for drafting logic)
    # ------------------------------------------------------------------

    def _run_ddpm_loop(
        self,
        prefix_ids: torch.Tensor,
        gamma: int,
        T: int,
        temperature: float,
        eps: float,
        q_mode: str,
        generator: Optional[torch.Generator],
        extract_hidden: bool,
        grad_at_final_only: bool = False,
        layers: Tuple[int, ...] = (-1,),
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """The canonical DDPM loop. Used by `draft`, `draft_with_features`,
        AND `draft_with_features_grad` so hidden-state extraction, drafted
        tokens, and q-rows cannot drift between inference and training.

        Returns `(draft_tokens, draft_log_probs, drafter_hidden)`.
        `drafter_hidden` is `None` iff `extract_hidden=False`.

        When `grad_at_final_only=True`, the FINAL DDPM step runs under
        the ambient grad mode (so gradients flow through the final MDLM
        forward into `drafter_hidden`), while intermediate steps are
        forced to `torch.no_grad()` (their outputs only propagate to
        the next step via non-differentiable multinomial sampling). For
        `grad_at_final_only=False`, all steps run under the ambient
        grad mode — the no_grad decorators on `draft` /
        `draft_with_features` mean they default to no_grad end-to-end.
        """
        if q_mode not in ("A", "B"):
            raise ValueError(f"q_mode must be 'A' or 'B', got {q_mode!r}")

        prefix_len = int(prefix_ids.shape[0])
        total_len = prefix_len + int(gamma)
        if total_len > self.max_seq_len:
            trim = total_len - self.max_seq_len
            prefix_ids = prefix_ids[trim:]
            prefix_len = int(prefix_ids.shape[0])
            total_len = self.max_seq_len

        x = torch.full(
            (1, total_len), self.mask_index, dtype=torch.long, device=self.device
        )
        x[0, :prefix_len] = prefix_ids

        timesteps = torch.linspace(1.0, eps, T + 1, device=self.device)

        # q_mode B: per-step log-factor shifts on the stored q row.
        # For q_mode A these are all zero; committed distributions are the
        # raw SUBS rows. See DESIGN.md: q_mode = "A" is canonical for v1.
        intermediate_log_factors = [0.0] * max(T - 1, 0)
        final_log_factor = 0.0
        if q_mode == "B":
            for k in range(T - 1):
                f = float(((timesteps[k] - timesteps[k + 1]) / timesteps[k]).item())
                intermediate_log_factors[k] = math.log(max(f, 1e-30))
            ff = 1.0
            for k in range(T - 1):
                ff *= float((timesteps[k + 1] / timesteps[k]).item())
            final_log_factor = math.log(max(ff, 1e-30))

        # Per-position commitment tracking. For each draft position j we
        # store, AT THE STEP WHERE j IS FIRST UNMASKED, the SUBS-processed
        # log p(x0|xt) row at position j and the committed token. This
        # avoids the SUBS carry-over collapse that would otherwise make
        # q == 1 at any position committed at an intermediate step.
        committed_rows: List[Optional[torch.Tensor]] = [None] * gamma
        committed_tokens: List[Optional[torch.Tensor]] = [None] * gamma

        drafter_hidden: Optional[torch.Tensor] = None

        for step_idx in range(T):
            t_now = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            sigma = t_now * torch.ones(1, device=self.device)

            is_final = (step_idx == T - 1)

            # Grad context for this step's forward. When
            # grad_at_final_only=True: intermediate steps forced
            # no_grad, final step inherits ambient grad mode.
            # Otherwise: every step inherits ambient grad mode (the
            # public @torch.no_grad()-decorated methods default to
            # no_grad for the whole loop).
            if grad_at_final_only and not is_final:
                fwd_ctx = torch.no_grad()
            else:
                fwd_ctx = contextlib.nullcontext()

            with fwd_ctx:
                # Final step + hidden extraction: one forward with
                # output_hidden_states=True. Otherwise: plain forward.
                if is_final and extract_hidden:
                    out = self.model(
                        input_ids=x,
                        timesteps=sigma,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    logits_all = out.logits
                    hidden_states = out.hidden_states  # tuple or None
                    if hidden_states is not None and len(hidden_states) > 0:
                        # Extract requested layer indices at the γ draft
                        # positions, concatenate along the hidden axis.
                        # Order preserves `layers`, so `layers=(-2,-1)`
                        # puts the final block's rows in the last H cols.
                        slices = []
                        for li in layers:
                            h = hidden_states[li]  # (1, total_len, hidden)
                            slices.append(
                                h[0, prefix_len:prefix_len + gamma]
                            )
                        drafter_hidden = (
                            slices[0] if len(slices) == 1
                            else torch.cat(slices, dim=-1)
                        )
                    else:
                        # Defensive fallback — should not happen with MDLM.
                        drafter_hidden = torch.zeros(
                            gamma, self.hidden_size * len(layers),
                            device=self.device,
                        )
                else:
                    out = self.model(input_ids=x, timesteps=sigma)
                    if hasattr(out, "logits"):
                        logits_all = out.logits
                    else:
                        logits_all = out  # raw tensor fallback

                log_p_x0 = self._subs_parameterization(logits_all[0], x[0])

            if not is_final:
                # Intermediate DDPM step: sample which MASK positions unmask.
                p_x0 = log_p_x0.exp()
                draft_p = p_x0[prefix_len:]

                move_chance_t = t_now
                move_chance_s = t_next

                # For masked positions: prob of unmasking to token v;
                # prob of staying MASK is move_chance_s.
                q_xs = draft_p * (move_chance_t - move_chance_s)
                q_xs[:, self.mask_index] = move_chance_s

                # Intermediate steps must be stochastic even at temp=0;
                # otherwise greedy would unmask everything in one shot.
                new_tokens = torch.multinomial(
                    q_xs, num_samples=1, generator=generator
                ).squeeze(-1)

                # Snapshot newly-committed positions BEFORE writing back.
                draft_slice = x[0, prefix_len:]
                is_masked = (draft_slice == self.mask_index)
                newly_committed = is_masked & (new_tokens != self.mask_index)
                if newly_committed.any():
                    nc_idx = newly_committed.nonzero(as_tuple=True)[0]
                    shift = intermediate_log_factors[step_idx]
                    for jj in nc_idx.tolist():
                        row = log_p_x0[prefix_len + jj].clone()
                        if shift != 0.0:
                            row = row + shift
                        committed_rows[jj] = row
                        committed_tokens[jj] = new_tokens[jj].detach().clone()

                draft_slice[is_masked] = new_tokens[is_masked]
                x[0, prefix_len:] = draft_slice
            else:
                # Final step: commit any still-uncommitted positions.
                for jj in range(gamma):
                    if committed_rows[jj] is None:
                        row_raw = log_p_x0[prefix_len + jj]
                        # Token selection uses the UNSHIFTED row so q_mode
                        # A and B agree on the drafted token; shift only
                        # affects the stored q row (for q_mode B).
                        if temperature == 0.0:
                            tok = row_raw.argmax().detach().clone()
                        else:
                            probs = (row_raw / temperature).softmax(dim=-1)
                            tok = (
                                torch.multinomial(
                                    probs, num_samples=1, generator=generator
                                )
                                .squeeze()
                                .detach()
                                .clone()
                            )
                        row = row_raw.clone()
                        if final_log_factor != 0.0:
                            row = row + final_log_factor
                        committed_rows[jj] = row
                        committed_tokens[jj] = tok

        draft_log_probs = torch.stack(committed_rows, dim=0)  # type: ignore[arg-type]
        draft_tokens = torch.stack(
            [t.long() for t in committed_tokens]  # type: ignore[union-attr]
        , dim=0)
        return draft_tokens, draft_log_probs, drafter_hidden

    # ------------------------------------------------------------------
    # SUBS parameterization
    # ------------------------------------------------------------------

    def _subs_parameterization(
        self, logits: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        """Apply SUBS parameterization: raw logits -> log p(x_0 | x_t).

        Zero-mask the mask-token logit (can't predict MASK as output),
        normalize, then pin unmasked positions to their current token
        with probability 1.
        """
        NEG_INF = -1e9
        logits = logits.clone()
        logits[:, self.mask_index] = NEG_INF
        log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = (xt != self.mask_index)
        if unmasked.any():
            log_p[unmasked] = NEG_INF
            log_p[unmasked, xt[unmasked].long()] = 0.0
        return log_p
