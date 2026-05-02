"""GPT-2 autoregressive verifier — thin wrapper.

Vendored from `accpre/models/verifier_gpt2.py`. Loads the verifier model
through the HuggingFace AutoModelForCausalLM interface; no model weights
are stored in this repo.

API:
  score(token_ids) -> log_probs of shape (seq_len, vocab)
  sample_next(token_ids, temperature, generator=None) -> int
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
        self.vocab_size: int = int(self.model.config.vocab_size)
        # MDLM (50258) - 1 mask token = GPT-2 (50257). Trim logic in
        # draft_verify.sample_fallback_or_bonus depends on this.
        assert self.vocab_size == 50257, (
            f"GPT-2 verifier vocab_size must be 50257, got {self.vocab_size}."
        )
        print(f"[GPT2Verifier] Loaded. vocab_size={self.vocab_size}")

    @torch.no_grad()
    def score(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Per-position log-probabilities for a token sequence."""
        input_ids = token_ids.unsqueeze(0)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0]
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def sample_next(
        self,
        token_ids: torch.Tensor,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> int:
        """Greedy argmax (temp=0) or temperature-scaled multinomial."""
        log_probs = self.score(token_ids)
        last_log_probs = log_probs[-1]
        if temperature == 0.0:
            return int(last_log_probs.argmax().item())
        probs = (last_log_probs / temperature).softmax(dim=-1)
        return int(
            torch.multinomial(probs, num_samples=1, generator=generator).item()
        )
