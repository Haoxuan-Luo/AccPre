# Diagnosis — why "lossy" is slower than "strict" in the homework run

## TL;DR
The lanes are coded correctly. The experiment ran with `temperature=0`, and at
that setting **strict uses a different acceptance rule from every other lane**:

  - Strict at temp=0: `accept iff argmax(verifier[pos]) == draft_token`
    (argmax-match — fully deterministic, very permissive)
  - Lossy / threshold / confidence (any temp): `accept` based on the ratio
    `r_j = min(1, p_j / q_j)` (Bernoulli, threshold, or prefix-product)

These are not the same algorithm. At temp=0 the strict rule is **the most
permissive deterministic rule possible** — every position the verifier would
have generated is accepted, regardless of how diffuse or sharp `p_j` happens
to be. The ratio rules drop positions whenever the verifier is "less sharp"
than the drafter, even when the argmax matches.

## Code path (proof)

`accpre/core/accept.py:103–107` is the single per-position accept test:

```python
if protocol.temperature == 0.0:
    top = int(target_log_probs[target_pos].argmax().item())
    accepted = (top == tok)             # ← strict at temp=0
else:
    accepted = (u < ratio)              # ← strict at temp>0 (matches lossy l=1.0)
```

`run_strict_lane` calls this via `draft_verify_round`, so strict at temp=0
takes the **argmax-match** branch. Every other lane (`run_lossy_lane_one`, my
`run_threshold_lossy_lane_one`, `run_confidence_lossy_lane_one`) computes
`r_j = min(1, p_j/q_j)` directly and applies its own rule on top.

`configs/protocol_greedy.yaml` (which I used) has `temperature: 0.0`. The
mainline `configs/protocol.yaml` has `temperature: 1.0` precisely so this
mismatch goes away.

## Quantitative evidence (5 OWT prompts, 256 new tokens — `diagnose_rules.log`)

5 325 drafted positions:

| measurement | value |
|---|---:|
| argmax-match positions | 38.0 % |
| q_j (drafter prob)     mean / median | 0.397 / 0.252 |
| p_j (verifier prob)    mean / median | 0.309 / 0.032 |
| ratio on **match** positions: mean / median | **0.910 / 1.000** |
| ratio on mismatch positions: mean / median  | 0.216 / 0.025 |

Of the **match** positions (which strict always accepts), the threshold
rule rejects:

| τ | rejected match positions |
|---:|---:|
| 0.1 |  0.00 % |
| 0.3 |  2.92 % |
| 0.5 |  6.57 % |
| 0.7 | 12.75 % |
| 0.9 | 21.15 % |

And the lossy-l Bernoulli accept rate on match positions:

| l | mean accept rate (vs strict 1.0) |
|---:|---:|
| 1.0 | 0.910 |
| 0.7 | 0.957 |
| 0.5 | 0.978 |
| 0.3 | 0.993 |
| 0.1 | 1.000 |

So **lossy l=1.0 already concedes 9 % of the match positions** strict would
have committed. With γ=15 and `mean_L_strict ≈ 4.83`, that 9 % gap propagates
through the trajectory: `mean_L_lossy_l=1.0 ≈ 3.11`, which is the entire
throughput gap (27.2 → 20.7 tok/s).

## Why the throughput numbers come out the way they do

Throughput is bounded by `(mean_L + 1) / per_round_cost`. Per-round cost is
the same for all lanes (one drafter pass + one verifier pass). So the tok/s
ratio between any two lanes ≈ ratio of `mean_L + 1`:

| lane | mean_L | predicted tok/s relative to strict | observed tok/s |
|---|---:|---:|---:|
| strict       | 4.83 | 1.000 | 27.2 |
| lossy l=1.0  | 3.11 | 0.705 | 20.7 (76 %) |
| lossy l=0.3  | 4.26 | 0.901 | 25.1 (92 %) |
| threshold τ=0.6 | 3.74 | 0.812 | 23.6 (87 %) |
| confidence τ=0.9 | 3.07 | 0.696 | 20.9 (77 %) |

The match is good — the throughput ranking is fully explained by `mean_L`,
which is in turn fully explained by the rule mismatch above.

## Why NLL also worsens

In addition to committing fewer tokens, lossy/threshold/confidence at temp=0
**also commit some mismatched tokens** (positions where verifier_argmax ≠
draft, but `r_j ≥ τ` or `U < r_j`). Strict never commits those. Hence
`NLL_lossy > NLL_strict` even though `mean_L_lossy < mean_L_strict`.

## What this means

**The homework experiment, as currently configured, asks an unanswerable
question.** It compares strict's argmax-match rule against ratio-based rules
that are not designed for greedy decoding. Strict will dominate every other
method on tok/s by construction, and on NLL by being deterministic.

Speculative decoding's classical speed-up story only emerges at
**temperature > 0** (sampling), where:
  - strict and lossy l=1.0 use the **same** Bernoulli rule (`U < min(1, p/q)`),
    so lossy l=1.0 reproduces strict exactly (matched-randomness invariance),
  - lossy l<1.0 strictly dominates strict on accept rate (`min(1, p/(l·q)) ≥
    min(1, p/q)`), giving the expected speed-up,
  - threshold and confidence rules trade quality for throughput along a clean
    curve.

## Proposed fix (recommend re-running)

Switch `--protocol_yaml` from `configs/protocol_greedy.yaml` to
`configs/protocol.yaml` (temperature=1, the v1 mainline). This requires three
small code changes:

1. Drop the `assert temp == 0.0` in `homework/run_homework.py`.
2. In `homework/lanes_homework.py`, replace the hardcoded
   `argmax(target_log_probs[fb_pos])` bonus/fallback with a call to
   `accpre.core.draft_verify.sample_fallback_or_bonus` (which already handles
   both temperatures correctly).
3. Note that `phase26.run_lossy_lane_one` *also* hardcodes argmax for its
   bonus — at temp=1 this is wrong. Easiest fix: copy its body into a
   homework-local lossy lane that uses `sample_fallback_or_bonus`.

Re-running 24 lanes at temp=1 takes ~7 h on the 1-MIG GPU. A cheaper sweep
(3 lossy + 5 threshold + 5 confidence + 1 strict = 14 lanes) would take ~4 h
and still produce a clean Pareto curve.
