"""Load prompts from OpenWebText as fixed-length prefix tensors.

Copied verbatim from the trusted HPC_SpecDiff baseline.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer


def load_owt_prompts(
    n_prompts: int = 10,
    prefix_len: int = 32,
    seed: int = 42,
    tokenizer_name: str = "gpt2",
) -> List[Tuple[torch.Tensor, str]]:
    """Return `n_prompts` (prefix_ids, prefix_text) tuples from OWT.

    Args:
        n_prompts: number of prompts to load.
        prefix_len: number of tokens per prefix.
        seed: RNG seed for streaming shuffle (reproducible).
        tokenizer_name: HF tokenizer to use.

    Returns:
        list of (prefix_ids: LongTensor(prefix_len,), prefix_text: str).
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("[Data] Loading OpenWebText (streaming)...")
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    prompts: List[Tuple[torch.Tensor, str]] = []
    for sample in dataset:
        text = sample["text"]
        tokens = tokenizer.encode(text)
        # Require a bit of headroom after the prefix so sequences aren't truncated.
        if len(tokens) >= prefix_len + 100:
            prefix_ids = torch.tensor(tokens[:prefix_len], dtype=torch.long)
            prefix_text = tokenizer.decode(prefix_ids)
            prompts.append((prefix_ids, prefix_text))
            if len(prompts) >= n_prompts:
                break

    print(f"[Data] Loaded {len(prompts)} prompts (prefix_len={prefix_len}).")
    return prompts
