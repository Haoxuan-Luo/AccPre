# Verifier-committed tokens per round vs. accept-rule parameter

`mean_L` is the average number of accepted *drafted* tokens per round.
`tokens/round = mean_L + 1` because every round also commits a single
verifier-argmax bonus/fallback token regardless of L.
`frac L=0` is the fraction of rounds where the rule rejected at the
very first position.

Tables sorted from most-conservative (top) to most-permissive (bottom).

## Lossy-l

| l | mean_L | tokens/round | tok/s | NLL | frac L=0 |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.83 | 1.83 | 6.9 | 0.191 | 0.555 |
| 1.0 | 3.11 | 4.11 | 20.7 | 0.377 | 0.304 |
| 0.7 | 3.33 | 4.33 | 21.6 | 0.470 | 0.253 |
| 0.5 | 3.25 | 4.25 | 20.9 | 0.548 | 0.260 |
| 0.3 | 4.26 | 5.26 | 25.1 | 0.562 | 0.169 |
| 0.1 | 3.94 | 4.94 | 20.9 | 1.098 | 0.123 |

## Threshold

| τ | mean_L | tokens/round | tok/s | NLL | frac L=0 |
|---:|---:|---:|---:|---:|---:|
| 0.95 | 2.79 | 3.79 | 19.6 | 0.182 | 0.414 |
| 0.9 | 3.12 | 4.12 | 21.4 | 0.182 | 0.397 |
| 0.8 | 3.22 | 4.22 | 21.6 | 0.223 | 0.374 |
| 0.7 | 3.28 | 4.28 | 21.9 | 0.258 | 0.347 |
| 0.6 | 3.74 | 4.74 | 23.6 | 0.210 | 0.286 |
| 0.5 | 4.03 | 5.03 | 23.5 | 0.218 | 0.259 |
| 0.4 | 3.43 | 4.43 | 21.9 | 0.346 | 0.269 |
| 0.3 | 3.42 | 4.42 | 21.1 | 0.427 | 0.239 |
| 0.2 | 3.06 | 4.06 | 19.6 | 0.564 | 0.250 |
| 0.1 | 4.17 | 5.17 | 23.5 | 0.638 | 0.134 |

## Confidence

| τ | mean_L | tokens/round | tok/s | NLL | frac L=0 |
|---:|---:|---:|---:|---:|---:|
| 0.95 | 2.51 | 3.51 | 18.2 | 0.199 | 0.437 |
| 0.9 | 3.07 | 4.07 | 20.9 | 0.180 | 0.397 |
| 0.8 | 2.87 | 3.87 | 20.1 | 0.247 | 0.398 |
| 0.7 | 3.24 | 4.24 | 21.7 | 0.255 | 0.355 |
| 0.6 | 3.39 | 4.39 | 22.5 | 0.253 | 0.316 |
| 0.5 | 3.37 | 4.37 | 21.1 | 0.269 | 0.293 |
| 0.4 | 3.61 | 4.61 | 21.8 | 0.265 | 0.250 |
| 0.3 | 3.68 | 4.68 | 21.7 | 0.306 | 0.211 |
| 0.2 | 3.40 | 4.40 | 22.1 | 0.458 | 0.210 |
| 0.1 | 3.55 | 4.55 | 20.8 | 0.560 | 0.146 |

## Notes

- `frac_L=0` decreases monotonically with permissiveness across all three
  methods — that is the "naive expected" within-round effect: looser rule
  ⇒ first position gets accepted more often.
- `mean_L` (and therefore tokens/round) is *not* monotone in the same
  direction. It rises with permissiveness only in the conservative half
  of each sweep (τ ≥ 0.5 for thresh/conf, l ≥ 0.3 for lossy) and then
  oscillates in the permissive tail.
- The breakdown of monotonicity is the trajectory-divergence cascade:
  off-policy commits drift the prefix away from the verifier's natural
  distribution; the drafter then produces drafts the verifier rejects
  earlier; mean_L drops despite the per-position rule being looser.
