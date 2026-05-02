"""Quantify how often strict's argmax-match rule and the ratio-based
rule (used by lossy/threshold/confidence) diverge.

Runs a short decode (256 new tokens) on 5 OWT test prompts, recording
for every drafted position:  q_j, p_j, ratio = min(1, p_j/q_j),
argmax_match (= strict accept at temp=0), and whether that position lay
inside strict's accepted prefix.

Outputs a single-pass summary to stdout.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

_HW_DIR = Path(__file__).resolve().parent
if str(_HW_DIR) not in sys.path:
    sys.path.insert(0, str(_HW_DIR))

from homework_utils import _stream_owt_pool, load_protocol
from specdiff.draft_verify import _make_generator
from specdiff.drafter_mdlm import MDLMDrafter
from specdiff.protocol import derive_seed
from specdiff.verifier_gpt2 import GPT2Verifier


def main() -> int:
    proto = load_protocol(_HW_DIR / "configs/homework_greedy.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"[diag] device={device} temp={proto.temperature} q_mode={proto.q_mode}",
          flush=True)

    drafter = MDLMDrafter(
        model_name=proto.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    verifier = GPT2Verifier(
        model_name=proto.verifier_model, device=device, dtype=dtype,
    )

    pool = _stream_owt_pool(pool_size=80, prefix_len=32, seed=42)
    prompts = pool[60:65]
    indices = list(range(60, 65))
    print(f"[diag] {len(prompts)} OWT test prompts (indices {indices})",
          flush=True)

    GAMMA = 15
    T = 2
    EPS = 1e-10
    records = []  # (q, p, ratio, match, in_strict_prefix)
    for i, (prefix_ids, _txt) in enumerate(prompts):
        p_idx = indices[i]
        running = prefix_ids.to(device).clone()
        prefix_start = int(running.shape[0])
        round_idx = 0
        while running.shape[0] - prefix_start < 256:
            if running.shape[0] >= proto.max_verifier_ctx:
                break
            ctx_room = proto.max_verifier_ctx - int(running.shape[0]) - 1
            cur_gamma = min(GAMMA, ctx_room)
            if cur_gamma <= 0:
                break
            seed = derive_seed(proto, p_idx, round_idx)
            draft_rng = _make_generator(device, seed ^ proto.draft_salt)
            with torch.no_grad():
                draft_tokens, draft_log_probs = drafter.draft(
                    prefix_ids=running, gamma=cur_gamma, T=T,
                    temperature=proto.temperature, q_mode=proto.q_mode,
                    generator=draft_rng,
                )
                candidate = torch.cat([running, draft_tokens])
                target_log_probs = verifier.score(candidate).to(torch.float32)
            prefix_len = int(running.shape[0])
            idx = torch.arange(cur_gamma, device=device)
            q_j = draft_log_probs[idx, draft_tokens.long()].exp().clamp(min=EPS)
            p_j = target_log_probs[
                prefix_len - 1 + idx, draft_tokens.long()
            ].exp().clamp(min=EPS)
            ratio = torch.minimum(torch.ones_like(q_j), p_j / q_j)
            top = target_log_probs[prefix_len - 1 + idx, :].argmax(dim=-1)
            match = (top == draft_tokens.long())
            strict_acc = match.int().cpu().tolist()
            L_strict = next(
                (k for k, a in enumerate(strict_acc) if a == 0),
                len(strict_acc),
            )
            for j in range(cur_gamma):
                records.append((
                    float(q_j[j].item()),
                    float(p_j[j].item()),
                    float(ratio[j].item()),
                    int(match[j].item()),
                    j < L_strict,
                ))
            draft_pref = draft_tokens[:L_strict].to(device)
            bonus_pos = prefix_len - 1 + L_strict
            bonus_tok = int(
                target_log_probs[bonus_pos, :].argmax(dim=-1).item()
            )
            running = torch.cat([
                running, draft_pref,
                torch.tensor([bonus_tok], dtype=torch.long, device=device),
            ])
            round_idx += 1

    print(f"\n[diag] {len(records)} drafted positions across "
          f"{len(prompts)} prompts", flush=True)

    matches = [r for r in records if r[3] == 1]
    mismatches = [r for r in records if r[3] == 0]
    n = len(records)
    print(f"[diag] argmax-match positions:    {len(matches):5d} "
          f"({len(matches) / n:.2%})")
    print(f"[diag] argmax-mismatch positions: {len(mismatches):5d} "
          f"({len(mismatches) / n:.2%})")

    print(f"\n[diag] q_j (drafter prob of drafted token):")
    print(f"   mean={statistics.mean(r[0] for r in records):.3f}  "
          f"median={statistics.median(r[0] for r in records):.3f}")
    print(f"[diag] p_j (verifier prob of drafted token):")
    print(f"   mean={statistics.mean(r[1] for r in records):.3f}  "
          f"median={statistics.median(r[1] for r in records):.3f}")
    print(f"[diag] ratio = min(1, p/q) on argmax-MATCH positions:")
    print(f"   mean={statistics.mean(r[2] for r in matches):.3f}  "
          f"median={statistics.median(r[2] for r in matches):.3f}")
    print(f"[diag] ratio on argmax-MISMATCH positions:")
    if mismatches:
        print(f"   mean={statistics.mean(r[2] for r in mismatches):.3f}  "
              f"median={statistics.median(r[2] for r in mismatches):.3f}")

    print("\n[diag] Of the argmax-MATCH positions (which strict ALWAYS "
          "accepts at temp=0), how many would the threshold rule REJECT?")
    for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
        below = sum(1 for r in matches if r[2] < tau)
        pct = below / max(len(matches), 1)
        print(f"   tau={tau}:  {below:5d}/{len(matches):5d} = {pct:6.2%}  "
              f"→ this is the gap that makes ratio-rules slower than strict")

    print("\n[diag] Equivalently, of all positions inside strict's accepted "
          "prefix, how many would threshold reject (truncating earlier)?")
    inside = [r for r in records if r[4]]
    for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
        diverge = sum(1 for r in inside if r[2] < tau)
        pct = diverge / max(len(inside), 1)
        print(f"   tau={tau}:  {diverge:5d}/{len(inside):5d} = {pct:6.2%}")

    print("\n[diag] Lossy-l Bernoulli accept rate on match positions "
          "(should equal mean(min(1, p/(l*q)))):")
    for l in (1.0, 0.7, 0.5, 0.3, 0.1):
        rates = [min(1.0, r[1] / max(l * r[0], EPS)) for r in matches]
        print(f"   l={l}: mean accept = {statistics.mean(rates):.3f}  "
              f"(strict at temp=0 always accepts these = 1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
