"""Helpers for the coursework experiment.

Three things are pulled together here so the rest of the homework code
stays small:

  load_protocol(path)              YAML -> ProtocolConfig
  load_owt_test_prompts(...)       deterministic 20-prompt OWT test split,
                                   optionally backed by a cached JSON dump
  run_strict_lane(...)             strict (Leviathan) lane decode
  run_lossy_lane_one(...)          lossy-l (Leviathan + lenience) lane decode
  compute_nll_and_success(...)     offline NLL pass + success summary

The two strict/lossy lane functions are vendored from
`scripts/phase25_longhorizon.py` and `scripts/phase26_conf_sweep.py` of
the parent project. The new `run_threshold_lossy_lane_one` and
`run_confidence_lossy_lane_one` lanes live in `lanes_homework.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoTokenizer

from specdiff.commit import commit_strict
from specdiff.draft_verify import (
    _make_generator, draft_verify_round_strict,
)
from specdiff.protocol import ProtocolConfig, derive_seed


EPS = 1e-10


# ---------------------------------------------------------------------------
# Config + prompt loading
# ---------------------------------------------------------------------------


def load_protocol(path: Path) -> ProtocolConfig:
    """Load a YAML protocol file into a `ProtocolConfig`."""
    with open(path) as f:
        d = yaml.safe_load(f)
    for k in (
        "draft_salt", "accept_salt", "fallback_salt",
        "schema_version", "max_verifier_ctx",
    ):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _stream_owt_pool(
    pool_size: int = 80,
    prefix_len: int = 32,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, str]]:
    """Stream OpenWebText from HuggingFace and tokenise the first `pool_size`
    documents that have at least `prefix_len + 100` GPT-2 tokens.

    This is the deterministic 80-prompt pool used throughout the project.
    Indices 0..39 are train, 40..59 are val, 60..79 are test.
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("[Data] Loading OpenWebText (streaming)...", flush=True)
    ds = load_dataset("openwebtext", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    out: List[Tuple[torch.Tensor, str]] = []
    for sample in ds:
        text = sample["text"]
        tokens = tokenizer.encode(text)
        if len(tokens) >= prefix_len + 100:
            prefix_ids = torch.tensor(tokens[:prefix_len], dtype=torch.long)
            prefix_text = tokenizer.decode(prefix_ids)
            out.append((prefix_ids, prefix_text))
            if len(out) >= pool_size:
                break
    print(f"[Data] Loaded {len(out)} prompts (prefix_len={prefix_len}).", flush=True)
    return out


def load_owt_test_prompts(
    n_prompts: int = 20,
    cache_path: Optional[Path] = None,
) -> Tuple[List[Tuple[torch.Tensor, str]], List[int]]:
    """Return the first `n_prompts` prompts from the OWT test split.

    The canonical project split is 80 OWT documents, prefix_len=32, seed=42:
    indices 0..39 = train, 40..59 = val, 60..79 = test.

    If `cache_path` exists, prompts are read from it (each item must store a
    `prefix_ids` int list); otherwise we stream OWT from HuggingFace and write
    the result to `cache_path` (if provided).
    """
    test_offset = 60
    indices = list(range(test_offset, test_offset + n_prompts))

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            d = json.load(f)
        items = d.get("items", [])
        if items and all("prefix_ids" in it for it in items[:n_prompts]):
            prompts: List[Tuple[torch.Tensor, str]] = []
            for it in items[:n_prompts]:
                ids = torch.tensor(it["prefix_ids"], dtype=torch.long)
                prompts.append((ids, it.get("text", "")))
            if len(prompts) < n_prompts:
                raise ValueError(
                    f"cached {cache_path} has {len(prompts)} prompts; need {n_prompts}."
                )
            print(
                f"[Data] Loaded {len(prompts)} prompts from cache {cache_path}.",
                flush=True,
            )
            return prompts, indices
        # Cache exists but is the legacy text-only format — fall through and
        # rebuild the cache from OpenWebText (writes back with prefix_ids).
        print(
            f"[Data] Cache {cache_path} lacks prefix_ids; rebuilding from OWT.",
            flush=True,
        )

    pool = _stream_owt_pool(pool_size=80, prefix_len=32, seed=42)
    test_pool = pool[test_offset: 80]
    if n_prompts > len(test_pool):
        raise ValueError(
            f"OWT test split has {len(test_pool)} prompts; requested {n_prompts}."
        )
    prompts = test_pool[:n_prompts]

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        items = [
            {
                "prompt_idx": int(idx),
                "prefix_len": int(prefix_ids.shape[0]),
                "prefix_ids": [int(x) for x in prefix_ids.tolist()],
                "text": text,
            }
            for (prefix_ids, text), idx in zip(prompts, indices)
        ]
        with open(cache_path, "w") as f:
            json.dump({
                "dataset": "owt",
                "split": "test",
                "n_prompts": len(items),
                "indices": indices,
                "items": items,
            }, f, indent=2)
        print(f"[Data] Cached {len(items)} prompts to {cache_path}.", flush=True)

    return prompts, indices


def write_prompts_dump(
    prompts, indices: List[int], out_path: Path,
) -> None:
    """Save the exact prompts used (text + token IDs) for full reproducibility."""
    items = []
    for (prefix_ids, text), idx in zip(prompts, indices):
        items.append({
            "prompt_idx": int(idx),
            "prefix_len": int(prefix_ids.shape[0]),
            "prefix_ids": [int(x) for x in prefix_ids.tolist()],
            "text": text,
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "dataset": "owt",
            "split": "test",
            "n_prompts": len(items),
            "indices": indices,
            "items": items,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Lane: strict (Leviathan)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_strict_lane(
    drafter, verifier, prompts, prompt_indices, protocol,
    gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Strict speculative decoding lane.

    Vendored from `scripts/phase25_longhorizon.py` (`run_strict_lane`).
    Reserves one context slot per round so a full-accept (L=gamma) round can
    still append its bonus token without overflowing `protocol.max_verifier_ctx`.
    """
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    for i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[i])
        running = prefix_ids.to(device).clone()
        prefix_start = int(running.shape[0])
        rounds: List[Dict] = []
        round_idx = 0
        t_start = time.time()
        while (running.shape[0] - prefix_start) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_start)
            ctx_room = MAX_CTX - int(running.shape[0]) - 1
            cur_gamma = min(gamma, remaining, ctx_room)
            if cur_gamma <= 0:
                break
            seed = derive_seed(protocol, p_idx, round_idx)
            L, draft_tokens, bonus_tok, _ = draft_verify_round_strict(
                prefix_ids=running, drafter=drafter, verifier=verifier,
                gamma=cur_gamma, T=T, protocol=protocol, round_rng_seed=seed,
            )
            L = int(L)
            draft_pref = draft_tokens[:L].to(device).long()
            extra = torch.tensor([int(bonus_tok)], dtype=torch.long, device=device)
            running = torch.cat([running, draft_pref, extra])
            rounds.append({
                "round_idx": round_idx, "L": L, "n_committed": L + 1,
                "round_rng_seed": int(seed), "gamma": int(cur_gamma),
            })
            round_idx += 1
        elapsed = time.time() - t_start
        n_new = int(running.shape[0]) - prefix_start
        per_prompt.append({
            "prompt_idx": p_idx, "n_new_tokens": int(n_new),
            "elapsed_s": float(elapsed),
            "tok_s": float(n_new) / max(elapsed, 1e-9),
            "generated_ids": [int(x) for x in running.tolist()],
            "rounds": rounds,
        })
    tok_s_mean = sum(p["tok_s"] for p in per_prompt) / max(len(per_prompt), 1)
    return {
        "method": "strict",
        "tok_s_mean": tok_s_mean,
        "per_prompt": per_prompt,
    }


# ---------------------------------------------------------------------------
# Lane: lossy_l (Leviathan + lenience l)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_lossy_lane_one(
    drafter, verifier, prompts, prompt_indices, protocol,
    l: float, gamma: int = 8, T: int = 2, max_new_tokens: int = 1024,
) -> Dict:
    """Lossy SD baseline with lenience factor `l`.

    Vendored from `scripts/phase26_conf_sweep.py` (`run_lossy_lane_one`).
    Per-position accept rule:
        accept_j  iff  U_j < min(1, p_j / (l * q_j))
    """
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    for i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[i])
        running = prefix_ids.to(device).clone()
        prefix_start = int(running.shape[0])
        rounds: List[Dict] = []
        n_tok_total = 0; n_tok_succ = 0
        n_round_total = 0; n_round_all_pass = 0
        sum_round_succ_rate = 0.0
        round_idx = 0
        t_start = time.time()
        while (running.shape[0] - prefix_start) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_start)
            ctx_room = MAX_CTX - int(running.shape[0]) - 1
            cur_gamma = min(gamma, remaining, ctx_room)
            if cur_gamma <= 0:
                break
            seed = derive_seed(protocol, p_idx, round_idx)

            draft_rng = _make_generator(device, seed ^ protocol.draft_salt)
            draft_tokens, draft_log_probs = drafter.draft(
                prefix_ids=running, gamma=cur_gamma, T=T,
                temperature=protocol.temperature, q_mode=protocol.q_mode,
                generator=draft_rng,
            )
            candidate = torch.cat([running, draft_tokens])
            target_log_probs = verifier.score(candidate).to(torch.float32)
            prefix_len = int(running.shape[0])
            idx = torch.arange(cur_gamma, device=device)
            q_j = draft_log_probs[
                idx, draft_tokens.long()
            ].exp().clamp(min=EPS)
            p_j = target_log_probs[
                prefix_len - 1 + idx, draft_tokens.long()
            ].exp().clamp(min=EPS)

            a_lossy = torch.minimum(
                torch.ones_like(q_j),
                p_j / (float(l) * q_j),
            )
            accept_rng = _make_generator(device, seed ^ protocol.accept_salt)
            U = torch.rand(cur_gamma, generator=accept_rng, device=device)
            accepted_j = (U < a_lossy).to(torch.int32).cpu().tolist()
            L_lossy = commit_strict(accepted_j)
            L_lossy = max(0, min(L_lossy, cur_gamma))

            top = target_log_probs[prefix_len - 1 + idx, :].argmax(dim=-1)
            succ_full = (top == draft_tokens.long()).to(torch.int32)

            n_succ = int(succ_full[:L_lossy].sum().item()) if L_lossy > 0 else 0
            n_round_total += 1
            n_tok_total += L_lossy
            n_tok_succ += n_succ
            if L_lossy > 0:
                sum_round_succ_rate += n_succ / L_lossy
                if n_succ == L_lossy:
                    n_round_all_pass += 1

            draft_pref = draft_tokens[:L_lossy].to(device)
            fb_pos = prefix_len - 1 + L_lossy
            bonus_tok = int(target_log_probs[fb_pos, :].argmax(dim=-1).item())
            extra = torch.tensor([bonus_tok], dtype=torch.long, device=device)
            running = torch.cat([running, draft_pref, extra])
            rounds.append({
                "round_idx": round_idx,
                "L_lossy": int(L_lossy),
                "n_committed": int(L_lossy + 1),
                "round_rng_seed": int(seed),
                "gamma": int(cur_gamma),
            })
            round_idx += 1
        elapsed = time.time() - t_start
        n_new = int(running.shape[0]) - prefix_start
        per_prompt.append({
            "prompt_idx": p_idx,
            "n_new_tokens": int(n_new),
            "elapsed_s": float(elapsed),
            "tok_s": float(n_new) / max(elapsed, 1e-9),
            "generated_ids": [int(x) for x in running.tolist()],
            "rounds": rounds,
            "n_tokens_total": int(n_tok_total),
            "tok_succ": n_tok_succ / max(n_tok_total, 1),
            "rnd_mean": sum_round_succ_rate / max(n_round_total, 1),
            "all_pass": n_round_all_pass / max(n_round_total, 1),
        })
    tok_s_mean = sum(p["tok_s"] for p in per_prompt) / max(len(per_prompt), 1)
    return {
        "method": "lossy_l",
        "l": float(l),
        "tok_s_mean": tok_s_mean,
        "per_prompt": per_prompt,
    }


# ---------------------------------------------------------------------------
# Offline NLL + success summary
# ---------------------------------------------------------------------------


def compute_nll_and_success(verifier, lane: Dict, device) -> Tuple[float, Dict]:
    """Single offline pass: per-prompt verifier forward yielding NLL + success.

    Vendored from `scripts/phase26_conf_sweep.py` (`compute_nll_and_success`).
    If `tok_succ` / `rnd_mean` / `all_pass` are already populated inline on
    every per-prompt entry (e.g., the lossy lanes), they are kept and only
    NLL is computed.
    """
    total_sum = 0.0
    total_n = 0
    have_inline = all(
        ("tok_succ" in p and "rnd_mean" in p and "all_pass" in p)
        for p in lane["per_prompt"]
    )

    summary = {
        "token_success_rate":  None,
        "round_mean_success":  None,
        "round_all_pass_rate": None,
        "n_tokens_total":      0,
    }

    n_tok_total = 0; n_tok_succ = 0
    n_round_total = 0; n_round_all_pass = 0
    sum_round_succ_rate = 0.0

    for p_row in lane["per_prompt"]:
        gen = [int(x) for x in p_row["generated_ids"]]
        n_prefix = len(gen) - int(p_row["n_new_tokens"])

        x = torch.tensor(gen, dtype=torch.long, device=device)
        log_probs = verifier.score(x).to(torch.float32)

        if len(gen) > n_prefix:
            n_new = len(gen) - n_prefix
            idx_logits = torch.arange(
                n_prefix - 1, n_prefix + n_new - 1, device=device,
            )
            idx_tokens = torch.tensor(
                gen[n_prefix: n_prefix + n_new],
                dtype=torch.long, device=device,
            )
            selected = log_probs[idx_logits, idx_tokens]
            total_sum += float(-selected.sum().item())
            total_n += n_new

        if have_inline:
            n_tok_total += int(p_row.get("n_tokens_total", 0))
            continue

        rounds = p_row.get("rounds", [])
        per_prompt_tok_total = 0
        per_prompt_tok_succ = 0
        per_prompt_rnd_total = 0
        per_prompt_rnd_pass = 0
        per_prompt_sum_rate = 0.0
        for r in rounds:
            if "draft_tokens" not in r or \
               "prefix_len_at_round_start" not in r:
                continue
            prefix_len = int(r["prefix_len_at_round_start"])
            draft_tokens_r = [int(x) for x in r["draft_tokens"]]
            cur_gamma = int(r["gamma"])
            n_commit = int(r["n_committed"])
            if n_commit == 0:
                continue
            idx_slice = torch.arange(
                prefix_len - 1, prefix_len - 1 + cur_gamma, device=device,
            )
            top = log_probs[idx_slice, :].argmax(dim=-1)
            draft_t = torch.tensor(
                draft_tokens_r, dtype=torch.long, device=device,
            )
            match = (top == draft_t).to(torch.int32)
            n_succ = int(match[:n_commit].sum().item())
            per_prompt_tok_total += n_commit
            per_prompt_tok_succ += n_succ
            per_prompt_rnd_total += 1
            if n_commit > 0:
                per_prompt_sum_rate += n_succ / n_commit
                if n_succ == n_commit:
                    per_prompt_rnd_pass += 1

        if per_prompt_tok_total > 0:
            p_row["n_tokens_total"] = int(per_prompt_tok_total)
            p_row["tok_succ"] = (
                per_prompt_tok_succ / per_prompt_tok_total
            )
            p_row["rnd_mean"] = (
                per_prompt_sum_rate / max(per_prompt_rnd_total, 1)
            )
            p_row["all_pass"] = (
                per_prompt_rnd_pass / max(per_prompt_rnd_total, 1)
            )
            n_tok_total += per_prompt_tok_total
            n_tok_succ += per_prompt_tok_succ
            n_round_total += per_prompt_rnd_total
            n_round_all_pass += per_prompt_rnd_pass
            sum_round_succ_rate += per_prompt_sum_rate

    if not have_inline and n_tok_total > 0:
        summary["token_success_rate"] = n_tok_succ / max(n_tok_total, 1)
        summary["round_mean_success"] = (
            sum_round_succ_rate / max(n_round_total, 1)
        )
        summary["round_all_pass_rate"] = (
            n_round_all_pass / max(n_round_total, 1)
        )
    summary["n_tokens_total"] = int(n_tok_total)
    nll = total_sum / max(total_n, 1)
    return nll, summary
