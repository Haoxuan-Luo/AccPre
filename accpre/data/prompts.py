"""Prompt loaders for predictor sweeps.

Three datasets are wired:
  owt      OpenWebText, streamed via HF datasets, GPT-2 tokenizer,
           min-length filter `len(tokens) >= prefix_len + 100`.
  cnn_dm   CNN/DailyMail v3.0.0 article bodies, same tokenizer + filter.
           Uses the deterministic `train` split shuffled by `seed`.
  mt_bench MT-Bench user prompts (80 official questions). Each question
           is tokenized; we take the first `prefix_len` tokens. Because
           MT-Bench questions are short, the headroom filter is dropped
           and a question shorter than `prefix_len` is skipped. The
           pool may end up smaller than the requested `n_prompts`, in
           which case the loader raises so the split sizes can be
           adjusted.

Public API:
  load_prompts(dataset, n_prompts, prefix_len, seed)  -> list[(ids, text)]
  load_owt_prompts(...)        — preserved for back-compat
  load_cnn_dm_prompts(...)
  load_mt_bench_prompts(...)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

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


def load_cnn_dm_prompts(
    n_prompts: int = 10,
    prefix_len: int = 32,
    seed: int = 42,
    tokenizer_name: str = "gpt2",
) -> List[Tuple[torch.Tensor, str]]:
    """Return `n_prompts` prompts drawn from CNN/DailyMail article bodies.

    Same shape as `load_owt_prompts`. Uses the official v3.0.0 release
    on HuggingFace datasets, streamed train split, deterministically
    shuffled by `seed`.
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("[Data] Loading CNN/DailyMail v3.0.0 (streaming)...")
    dataset = load_dataset(
        "cnn_dailymail", "3.0.0", split="train", streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    prompts: List[Tuple[torch.Tensor, str]] = []
    for sample in dataset:
        text = sample.get("article") or sample.get("text") or ""
        if not text:
            continue
        tokens = tokenizer.encode(text)
        if len(tokens) >= prefix_len + 100:
            prefix_ids = torch.tensor(tokens[:prefix_len], dtype=torch.long)
            prefix_text = tokenizer.decode(prefix_ids)
            prompts.append((prefix_ids, prefix_text))
            if len(prompts) >= n_prompts:
                break

    print(f"[Data] Loaded {len(prompts)} prompts (prefix_len={prefix_len}).")
    return prompts


def load_mt_bench_prompts(
    n_prompts: int = 10,
    prefix_len: int = 32,
    seed: int = 42,
    tokenizer_name: str = "gpt2",
) -> List[Tuple[torch.Tensor, str]]:
    """Return `n_prompts` prompts from the official MT-Bench question set.

    MT-Bench consists of 80 short user prompts across 8 categories.
    Each item has a `turns` list (length 2 — first turn + follow-up).
    We use the first turn as the seed prefix; questions whose tokenised
    length is below `prefix_len` are skipped. Pool order is the
    canonical category order from the lmsys/mt_bench_human_judgments
    release; `seed` is used only to break ties in a stable shuffle so
    the same `n_prompts` request is reproducible across calls.

    If the requested `n_prompts` exceeds the number of usable
    questions a ValueError is raised so split sizes can be reduced.
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("[Data] Loading MT-Bench question set...")
    # `lmsys/mt_bench_human_judgments` exposes the original 80 questions
    # as the unique `question_id` set. Each row has a `turns` list whose
    # element 0 is the user's first turn — exactly the prompt we want.
    ds = load_dataset(
        "lmsys/mt_bench_human_judgments", split="human", streaming=False,
    )
    seen: Dict[int, str] = {}
    for row in ds:
        qid = int(row["question_id"])
        if qid in seen:
            continue
        # `lmsys/mt_bench_human_judgments` row schema: each row is a
        # judgment over (model_a, model_b) and exposes `conversation_a`
        # and `conversation_b` — each a list of {"role","content"}
        # dicts. Index 0 is always the first user turn (identical
        # across `_a` / `_b` for the same question), which is the
        # prompt we want.
        conv = row.get("conversation_a") or row.get("conversation_b")
        if not conv or not isinstance(conv, list):
            continue
        first = conv[0]
        if not isinstance(first, dict):
            continue
        content = first.get("content", "")
        if content:
            seen[qid] = str(content)
    # Stable order: by question_id, optionally re-shuffled by `seed`.
    # Audit fix 2026-04-26: explicitly check for None so seed=0 still
    # triggers a (deterministic) shuffle instead of being silently
    # treated as "no seed".
    ordered_qids = sorted(seen)
    if seed is not None:
        import random
        rng = random.Random(int(seed))
        rng.shuffle(ordered_qids)

    prompts: List[Tuple[torch.Tensor, str]] = []
    too_short: int = 0
    for qid in ordered_qids:
        text = seen[qid]
        tokens = tokenizer.encode(text)
        if len(tokens) < prefix_len:
            too_short += 1
            continue
        prefix_ids = torch.tensor(tokens[:prefix_len], dtype=torch.long)
        prefix_text = tokenizer.decode(prefix_ids)
        prompts.append((prefix_ids, prefix_text))
        if len(prompts) >= n_prompts:
            break

    if len(prompts) < n_prompts:
        raise ValueError(
            f"MT-Bench has only {len(prompts)} usable prompts at "
            f"prefix_len={prefix_len} (skipped {too_short} short ones); "
            f"requested {n_prompts}. Reduce the split sizes or lower "
            f"prefix_len."
        )
    print(f"[Data] Loaded {len(prompts)} prompts (prefix_len={prefix_len}).")
    return prompts


# Dispatcher used by everything outside the legacy phaseN scripts.
_LOADERS: Dict[str, Callable[..., List[Tuple[torch.Tensor, str]]]] = {
    "owt":         load_owt_prompts,
    # OWT_Frozen_0429 experiment splits (alias to the same OWT corpus).
    "owt_300":     load_owt_prompts,
    "owt_smoke":   load_owt_prompts,
    "cnn_dm":      load_cnn_dm_prompts,
    # 0505_CNNDM_compare experiment split (alias to the same CNN/DM corpus).
    "cnn_dm_300":  load_cnn_dm_prompts,
    "mt_bench":    load_mt_bench_prompts,
}


def load_prompts(
    dataset: str,
    n_prompts: int,
    prefix_len: int,
    seed: int,
    tokenizer_name: str = "gpt2",
) -> List[Tuple[torch.Tensor, str]]:
    """Dispatch to the right loader by name.

    Raises KeyError on unknown dataset names so callers fail loudly.
    """
    if dataset not in _LOADERS:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(_LOADERS)}"
        )
    return _LOADERS[dataset](
        n_prompts=n_prompts, prefix_len=prefix_len, seed=seed,
        tokenizer_name=tokenizer_name,
    )
