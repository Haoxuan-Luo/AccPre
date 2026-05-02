# Homework — Throughput–Quality Trade-off on OpenWebText

## Setup
- 20 OpenWebText test prompts (indices 60–79; 32-token prefix).
- MDLM drafter, GPT-2 XL verifier, γ=15, T=2, max_new_tokens=1024, temperature=0.
- All four methods run on **the same prompts** so tok/s and NLL are directly comparable.
- NLL is GPT-2 XL's mean negative log-likelihood on the generated continuation.
- ΔNLL is the gap to the strict baseline (column NLL − strict NLL).

## Methods (no learned predictor)
| Method | Accept rule | Rule type |
|---|---|---|
| `strict` | Leviathan: `U_j < min(1, p_j/q_j)` | stochastic |
| `lossy_l` | `U_j < min(1, p_j/(l·q_j))` | stochastic, lenient |
| `threshold_lossy` (new) | accept while `r_j = min(1, p_j/q_j) ≥ τ` | **deterministic** |
| `confidence_lossy` (new) | accept while `∏_{j<k} r_j ≥ τ` | **deterministic** |

All methods commit `L` accepted draft tokens + 1 verifier-argmax bonus/fallback per round.

## Headline numbers (`unified.md`)

| method | param | tok/s | NLL | ΔNLL |
|---|---|---:|---:|---:|
| strict | — | **27.2** | **0.107** | 0.000 |
| lossy_l | l=1.0 | 20.7 | 0.377 | 0.270 |
| lossy_l | l=0.3 | 25.1 | 0.562 | 0.455 |
| threshold_lossy | τ=0.6 | 23.6 | 0.210 | 0.103 |
| threshold_lossy | τ=0.9 | 21.4 | **0.182** | 0.075 |
| confidence_lossy | τ=0.9 | 20.9 | **0.180** | 0.073 |

## Interpretation

1. **Strict still wins on both axes.** It is both the fastest (27.2 tok/s) and
   highest-quality (NLL 0.107). No no-predictor lossy variant beats it in this
   setup — the speedup that lossy/threshold/confidence rules promise must come
   from accepting more drafts than strict does, but at γ=15 strict already
   commits a healthy `mean_L=4.83` per round, so the lossy methods buy nothing
   on speed and pay a real cost on quality.

2. **Both new deterministic rules strictly dominate `lossy_l`.** At similar
   throughput (≈21–24 tok/s), threshold and confidence reach NLL ≈ 0.18–0.27,
   while `lossy_l` lives at NLL ≈ 0.38–1.10. Visually, the orange/green curves
   sit well to the left of the blue curve in `pareto_nll.png`.

3. **`threshold_lossy` and `confidence_lossy` are essentially equivalent on
   this corpus.** At τ=0.9 they tie within noise (NLL 0.182 vs 0.180; tok/s
   21.4 vs 20.9). The prefix-product rule is not adding measurable value over
   the per-position threshold here — `r_j` is usually either ≈1 (clearly
   accept) or quite small, so the cumulative product crosses τ at the same
   position the per-position rule rejects.

4. **The threshold rule has a sweet spot near τ=0.5–0.6.** Throughput peaks at
   23.6 tok/s with NLL 0.21 — the best non-strict speed/quality point we
   measured. Lower τ accepts too many bad positions (NLL spikes); higher τ
   commits fewer tokens per round (slower). Confidence behaves similarly but
   peaks slightly later (τ=0.6, 22.5 tok/s, NLL 0.25).

5. **Why `lossy_l` does so badly here.** The lenience-l rule is stochastic
   over `r_j/l`, so it accepts "almost-good" positions probabilistically. With
   γ=15 and a strong drafter, this introduces enough random low-probability
   commits to push NLL way past the deterministic rules. The deterministic
   threshold and confidence rules cleanly cut off as soon as the local ratio
   drops, which preserves the verifier's distribution much better.

## Practical takeaway for the talk

> If you ever want a no-predictor lossy speculative decoder, prefer the
> deterministic threshold rule (`accept while r_j ≥ τ`) over Leviathan's
> lossy-l. Same throughput envelope, **2–4× lower ΔNLL**. But none of these
> rules beat plain strict speculative decoding when the drafter is already
> well-aligned with the verifier — strict is a hard ceiling at this γ.

## Reproducibility

```bash
sbatch jobs/run_homework.slurm        # ~7h on 1×MIG GPU
# Outputs land under homework/:
#   prompts.json, run.log,
#   online_<method>[_param].json (24 lanes),
#   results.csv, unified.{md,json},
#   pareto_nll.{png,pdf}, pareto_delta_nll.{png,pdf}
```
