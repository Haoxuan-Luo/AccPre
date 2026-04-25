"""XL-audit quality metric. AUXILIARY only in v1.

Definition (DESIGN.md §D.5.3): for a generated sequence `gen[0:N]` whose
first `prefix_len` tokens are the prompt prefix, the XL-audit quality is
the fraction of positions `k in [prefix_len, N)` where

    gen[k] == argmax(verifier.score(gen[:k])[-1])

i.e., the fraction of generated tokens that match the verifier's greedy
next-token prediction given the running prefix. Implemented as a single
forward pass over the full sequence: `verifier.score(gen)` returns
row-per-position log-probs, and argmax at row `k-1` is the greedy
next-token prediction for position `k`.

**Auxiliary only.** Not on the Pareto axes. Reported in a side table
alongside the primary (wallclock_tok_s, CF@1) figure.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def xl_audit(
    generated_ids: torch.Tensor, prefix_len: int, verifier
) -> float:
    """Post-hoc greedy-match fraction. Returns 0.0 if no generated tokens."""
    total_len = int(generated_ids.shape[0])
    if total_len <= prefix_len:
        return 0.0

    log_probs = verifier.score(generated_ids)  # (total_len, vocab)
    greedy = log_probs.argmax(dim=-1)          # (total_len,)

    matches = 0
    for k in range(prefix_len, total_len):
        if int(generated_ids[k].item()) == int(greedy[k - 1].item()):
            matches += 1
    return matches / float(total_len - prefix_len)
