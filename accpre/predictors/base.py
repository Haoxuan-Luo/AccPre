"""Predictor base class and shared interface.

Two KINDs of predictors in v1:
  - "acceptance" predictors emit `Q̂_j ∈ [0,1]^γ`. A commit length
    `L̂` is derived downstream via `commit_threshold(Q̂, τ)`.
  - "committed_length" predictors emit `P(L | features, τ)` over
    `{0, ..., γ}` directly; `L̂ = argmax P(L|·)`.

Both kinds share:
  - `forward(features, **kwargs)` — torch.nn.Module forward for training.
  - `predict_q2(features)` — only acceptance predictors; raises otherwise.
  - `predict_L(features, tau)` — both; acceptance predictors default to
    thresholding `predict_q2`; length predictors override with native
    `argmax`.

Metadata attributes `family` and `deploy_mode` are set as INSTANCE
attributes via PredictorBase.__init__ kwargs. `kind` is class-static per
subclass (always `"acceptance"` or `"committed_length"`) and declared as
a `ClassVar` on each subclass. This avoids the class-attribute mutation
that earlier revisions used (`type(self).family = ...`), which caused
cross-contamination if two predictors of the same class coexisted in
one process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict

import torch
from torch import nn

from accpre.core.commit import commit_threshold


_VALID_FAMILIES = (
    "numeric", "hidden", "hidden_numeric", "upper_bound",
    "hidden_per_pos",
    "hidden_per_pos_hs", "hidden_per_pos_s",
    "hidden_per_pos_ml",
    "hidden_per_pos_numeric",
    # Phase 9 — per-position transformer family (v1..v4).
    "hidden_per_pos_v1",
    "hidden_per_pos_v2",
    "hidden_per_pos_v3",
    "hidden_per_pos_v4",
)

# V-free ⇔ drafter-only feature families (no verifier call at inference).
_V_FREE_FAMILIES = (
    "hidden",
    "hidden_per_pos",
    "hidden_per_pos_hs", "hidden_per_pos_s",
    "hidden_per_pos_ml",
    # Phase 9: v1 and v2 are drafter-only; v3 and v4 consume verifier
    # prefix features and are therefore V-prefix.
    "hidden_per_pos_v1",
    "hidden_per_pos_v2",
)


class PredictorBase(nn.Module, ABC):
    """Abstract base class for all Phase 2 predictors.

    Subclasses MUST declare `kind` as a ClassVar ("acceptance" or
    "committed_length"). They pass `family` and `deploy_mode` to this
    `__init__` which validates them and stores them as instance
    attributes on self.
    """

    kind: ClassVar[str] = ""   # subclass overrides with "acceptance" / "committed_length"

    def __init__(
        self,
        *,
        family: str,
        deploy_mode: str,
        gamma: int,
    ) -> None:
        super().__init__()
        if self.kind not in ("acceptance", "committed_length"):
            raise TypeError(
                f"Predictor subclass {type(self).__name__} must set "
                f"`kind` ClassVar to 'acceptance' or 'committed_length'."
            )
        if family not in _VALID_FAMILIES:
            raise TypeError(
                f"Predictor subclass {type(self).__name__} has invalid "
                f"family={family!r}."
            )
        if deploy_mode not in ("V-free", "V-prefix"):
            raise TypeError(
                f"Predictor subclass {type(self).__name__} must declare "
                f"`deploy_mode` 'V-free' or 'V-prefix'; got {deploy_mode!r}."
            )
        if deploy_mode == "V-free" and family not in _V_FREE_FAMILIES:
            raise TypeError(
                f"Predictor subclass {type(self).__name__} declared "
                f"deploy_mode='V-free' but family={family!r}. "
                f"V-free families: {_V_FREE_FAMILIES}."
            )
        # Store as INSTANCE attributes — safe across coexisting instances
        # of the same class.
        self.family = family
        self.deploy_mode = deploy_mode
        self.gamma = int(gamma)

    # ------------------------------------------------------------------
    # Forward — abstract; subclasses define their own signatures.
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Training-time forward.

        Acceptance predictors: return Q̂ ∈ [0,1]^(batch, γ).
        Length predictors: return logits ∈ R^(batch, γ+1).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Inference-time predictions.
    # ------------------------------------------------------------------

    def predict_q2(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return `Q̂ ∈ [0,1]^γ`. Acceptance predictors only."""
        raise NotImplementedError(
            f"Predictor {type(self).__name__} is not an acceptance predictor."
        )

    def predict_L(
        self, features: torch.Tensor, tau: float, **kwargs
    ) -> int:
        """Return `L̂ ∈ [0, γ]`.

        Default implementation for `kind == "acceptance"`:
            Q̂ = self.predict_q2(features)
            L̂ = commit_threshold(Q̂, τ)

        Length predictors must override this method.
        """
        if self.kind != "acceptance":
            raise NotImplementedError(
                f"Predictor {type(self).__name__} must override predict_L."
            )
        q = self.predict_q2(features, **kwargs)
        if q.dim() == 2:
            q = q.squeeze(0)  # (γ,)
        q_list = [float(x) for x in q.detach().cpu().tolist()]
        return commit_threshold(q_list, float(tau))

    # ------------------------------------------------------------------
    # Runtime metadata.
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, str]:
        return {
            "class": type(self).__name__,
            "kind": self.kind,
            "family": self.family,
            "deploy_mode": self.deploy_mode,
            "gamma": str(self.gamma),
        }
