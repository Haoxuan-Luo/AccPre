"""GPT-2 autoregressive verifier — thin wrapper.

Provides two methods:
  - score(token_ids) -> log_probs of shape (seq_len, vocab): the per-
    position next-token log-probabilities (log_probs[i] is
    log P(x_{i+1} | x_{0:i})).
  - sample_next(token_ids, temperature): greedy or temperature-scaled
    sample of the next token after the given prefix.

Copied from the trusted HPC_SpecDiff baseline with only class rename
(GPT2XLVerifier -> GPT2Verifier) and a dtype/device pass-through. No
logic changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPT2Verifier:
    def __init__(
        self,
        model_name: str = "gpt2-xl",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype

        print(f"[GPT2Verifier] Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size: int = int(self.model.config.vocab_size)  # 50257 for gpt2-xl
        # Vocab-layout invariant (audit issue 6): v1 paired with MDLM, which
        # adds ONE mask token at index 50257 beyond GPT-2's 50257 vocab.
        # Trim logic in draft_verify.sample_fallback_or_bonus depends on
        # this pairing.
        assert self.vocab_size == 50257, (
            f"GPT-2 verifier vocab_size must be 50257 (matches MDLM minus MASK), "
            f"got {self.vocab_size}. See audit issue 6."
        )
        print(f"[GPT2Verifier] Loaded. vocab_size={self.vocab_size}")

    @torch.no_grad()
    def score(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return per-position log-probabilities for a token sequence.

        Args:
            token_ids: (seq_len,) long tensor.

        Returns:
            log_probs: (seq_len, vocab) — log_probs[i] is the next-token
                log-distribution given prefix token_ids[:i+1].
        """
        input_ids = token_ids.unsqueeze(0)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab)
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def prefix_features(
        self, prefix_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """One forward over the PREFIX ONLY; return hidden + numeric features.

        Used by Stage 2A.0 collection and by the online V-prefix decode
        mode. Cost is one verifier forward over `prefix_len` tokens per
        call — NOT the full `prefix_len + gamma` forward that strict
        SpecDiff runs on (prefix + draft).

        Returns:
            hidden: (hidden_size,) last-layer, last-token hidden state on CPU.
            numeric: dict with keys:
                - "entropy":  predictive entropy at the last prefix token
                - "margin":   log p(top1) - log p(top2)
                - "top1_prob": p(argmax)
        """
        input_ids = prefix_ids.unsqueeze(0)
        outputs = self.model(
            input_ids=input_ids, output_hidden_states=True, return_dict=True
        )
        last_hidden = outputs.hidden_states[-1][0, -1]  # (hidden_size,)
        logits = outputs.logits[0, -1]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum().item())
        top2 = log_probs.topk(2)
        margin = float((top2.values[0] - top2.values[1]).item())
        top1_prob = float(probs.max().item())
        return last_hidden.detach().cpu(), {
            "entropy": entropy,
            "margin": margin,
            "top1_prob": top1_prob,
        }

    @torch.no_grad()
    def sample_next(
        self,
        token_ids: torch.Tensor,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample the next token after the given prefix.

        Greedy argmax when `temperature == 0.0`, else temperature-scaled
        multinomial. If `generator` is provided, use it for the
        multinomial draw; otherwise use torch's default generator.
        """
        log_probs = self.score(token_ids)
        last_log_probs = log_probs[-1]
        if temperature == 0.0:
            return int(last_log_probs.argmax().item())
        probs = (last_log_probs / temperature).softmax(dim=-1)
        return int(
            torch.multinomial(probs, num_samples=1, generator=generator).item()
        )
