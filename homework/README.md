# Speculative Diffusion Decoding with Lossy Acceptance Rules — Coursework

Throughput–quality comparison of four no-predictor speculative-decoding
acceptance rules on OpenWebText, with an MDLM discrete-diffusion drafter and
a GPT-2 XL autoregressive verifier. **This folder is self-contained**: every
module the experiment touches is vendored under `homework/specdiff/` and
`homework/homework_utils.py`. No imports outside `homework/` are required.

## Methods compared

| Method               | Per-position accept rule                                       | Commit-length rule                                       | Control |
|----------------------|----------------------------------------------------------------|-----------------------------------------------------------|---------|
| `strict`             | Leviathan strict speculative decoding                          | first reject                                              | —       |
| `lossy_l`            | `U_j < min(1, p_j / (l · q_j))` (Bernoulli, lenient)           | first reject                                              | `l`     |
| `threshold_lossy`    | `r_j = min(1, p_j / q_j); accept while r_j ≥ τ` (deterministic) | first position with `r_j < τ`                             | `τ`     |
| `confidence_lossy`   | same `r_j`                                                      | largest prefix length `m` with `∏_{j=1..m} r_j ≥ τ`        | `τ`     |

All four lanes commit `L` accepted draft tokens plus one verifier-argmax
bonus / fallback token per round, so they are directly comparable on tok/s
and NLL.

## Experimental setup

| Knob                 | Value                                                              |
|----------------------|--------------------------------------------------------------------|
| Dataset              | OpenWebText (HuggingFace `openwebtext`, streaming)                |
| Prompts              | 20 prompts, OWT canonical test split (indices 60..79; pool of 80) |
| Prefix length        | 32 tokens (GPT-2 BPE)                                              |
| Verifier (target)    | `gpt2-xl` (loaded from HuggingFace at runtime)                    |
| Drafter              | `kuleshov-group/mdlm-owt` (loaded from HuggingFace at runtime)    |
| `gamma` (draft len)  | 15                                                                 |
| `T` (DDPM steps)     | 2                                                                  |
| `max_new_tokens`     | 1024                                                               |
| Temperature          | **0** (greedy; `configs/homework_greedy.yaml`)                    |
| dtype                | float32                                                            |
| `max_verifier_ctx`   | 1024                                                               |

> **Models are NOT included.** `gpt2-xl` (~6.4 GB float32 state) and the MDLM
> drafter are downloaded from HuggingFace on first run via `transformers`.
> Set `HF_HOME` if you want to control where they cache.

## Directory layout

```
homework/
├── README.md                          ← this file
├── requirements.txt                   ← pip install -r
├── run_homework.py                    ← entry script
├── lanes_homework.py                  ← threshold + confidence lossy lanes
├── lanes_homework_v2.py               ← temp-aware alternative (not in temp=0 run)
├── homework_utils.py                  ← protocol load, prompt cache, strict/lossy lanes,
│                                        offline NLL pass
├── _diagnose_rules.py                 ← diagnostic: rule divergence at temp=0
├── _make_plot.py                      ← Pareto NLL + ΔNLL plots
├── _make_plot_final.py                ← FINAL presentation figure
├── _make_plot_curated.py              ← curated 3-points-per-method
├── _make_plot_anchored.py             ← anchored at lossy l=1.0
├── specdiff/                          ← vendored MDLM drafter, GPT-2 verifier,
│   ├── __init__.py                       protocol, commit, accept, draft_verify
│   ├── protocol.py
│   ├── commit.py
│   ├── accept.py
│   ├── draft_verify.py
│   ├── drafter_mdlm.py
│   └── verifier_gpt2.py
├── configs/
│   └── homework_greedy.yaml           ← protocol (temp=0; greedy)
├── scripts/
│   ├── run_homework.slurm             ← full SLURM submission (≈6–7 h on 1×MIG GPU)
│   ├── run_smoke.sh                   ← 1-prompt × 64-token smoke test (~5 min)
│   └── diagnose_rules.slurm           ← runs _diagnose_rules.py
├── data/
│   └── prompts_owt_20.json            ← canonical 20-prompt cache (text + token IDs;
│                                        regenerated from OWT on first run)
├── results/                           ← raw decode JSONs + summary tables
│   ├── online_strict.json
│   ├── online_lossy_l_*.json          (6 files)
│   ├── online_threshold_lossy_tau_*.json (10 files)
│   ├── online_confidence_lossy_tau_*.json (10 files)
│   ├── results.csv                    ← flat per-lane rows
│   ├── unified.json                   ← same data, JSON
│   ├── unified.md                     ← pretty Markdown table
│   └── run.log                        ← stdout of the SLURM run
├── figures/                           ← all 5 Pareto figures, png + pdf
│   ├── pareto_final.{png,pdf}         ← FINAL FIGURE
│   ├── pareto_nll.{png,pdf}
│   ├── pareto_delta_nll.{png,pdf}
│   ├── pareto_curated.{png,pdf}
│   └── pareto_anchored.{png,pdf}
└── diagnostics/                       ← analysis written for the talk
    ├── diagnose_rules.log
    ├── diagnosis.md
    ├── interpretation.md
    └── tokens_per_round.md
```

`homework/logs/` is created by SLURM at runtime and is gitignored.

## Install

```bash
pip install -r homework/requirements.txt
```

## Smoke test (recommended first run)

Verifies the entire code path end to end on 1 prompt × 64 new tokens × one
parameter per lossy family. Runs in ~5 minutes on a single GPU; on CPU it
will work but be very slow. Outputs land in `homework/results_smoke/`.

```bash
bash homework/scripts/run_smoke.sh
```

Equivalent direct invocation:

```bash
python -u homework/run_homework.py \
    --n_prompts 1 --max_new_tokens 64 \
    --lossy_ls 1.0 --thresh_taus 0.5 --conf_taus 0.5 \
    --out_dir homework/results_smoke
```

## Full experiment

```bash
sbatch homework/scripts/run_homework.slurm                              # ~6–7 h on 1×MIG GPU
sbatch --export=ALL,SKIP_EXISTING=1 homework/scripts/run_homework.slurm # reuse existing JSONs
```

Direct invocation (no SLURM):

```bash
python -u homework/run_homework.py
```

CLI knobs (defaults match the SLURM script):

```bash
python -u homework/run_homework.py \
    --out_dir homework/results \
    --protocol_yaml homework/configs/homework_greedy.yaml \
    --prompts_cache homework/data/prompts_owt_20.json \
    --n_prompts 20 \
    --max_new_tokens 1024 \
    --gamma 15 --T 2 \
    --lossy_ls 1.0 0.7 0.5 0.3 0.1 \
    --thresh_taus 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --conf_taus  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

Pass `--skip_existing` to reuse any `online_<method>[_param].json` already
on disk and only run the missing lanes.

## Regenerate the figures and tables

The aggregated tables (`results.csv`, `unified.{md,json}`) are produced at
the end of `run_homework.py`. The figures are produced by separate plotting
scripts that read `homework/results/unified.json` and write to
`homework/figures/` by default:

```bash
python -u homework/_make_plot_final.py     # final presentation figure
python -u homework/_make_plot.py           # NLL + ΔNLL Pareto
python -u homework/_make_plot_curated.py   # curated 3-points-per-method
python -u homework/_make_plot_anchored.py  # anchored at lossy l=1.0
```

## Headline numbers (current run)

| method            | param   | tok/s | NLL    | ΔNLL   |
|-------------------|---------|------:|-------:|-------:|
| strict            | —       | 27.2  | 0.107  | 0.000  |
| lossy_l           | l=1.0   | 20.7  | 0.377  | 0.270  |
| lossy_l           | l=0.3   | 25.1  | 0.562  | 0.455  |
| threshold_lossy   | τ=0.6   | 23.6  | 0.210  | 0.103  |
| threshold_lossy   | τ=0.9   | 21.4  | 0.182  | 0.075  |
| confidence_lossy  | τ=0.9   | 20.9  | 0.180  | 0.073  |

Full 27-row table is in `results/unified.md`. Interpretation, including why
`lossy_l` lags the deterministic rules and why no method beats `strict` at
this γ under temp=0, is in `diagnostics/interpretation.md`.

## Limitations

- **Coursework scale.** Only 20 OWT test prompts are used to keep wallclock
  ≤ ~7 h on a single MIG GPU partition. Larger prompt counts would tighten
  the throughput error bars but should not change the qualitative ordering.
- **Greedy regime.** Strict at temp=0 reduces to argmax-match (not
  Leviathan's Bernoulli ratio rule), so the strict point is "free" relative
  to the ratio rules. `diagnostics/diagnosis.md` quantifies the gap; if you
  need an apples-to-apples Leviathan comparison, use `lossy_l` at `l=1.0`
  as the strict reference (this is what `_make_plot_final.py` does).
- **Deterministic bonus token.** All four lanes commit one verifier-argmax
  bonus / fallback per round. At temp=0 this is correct; at temp>0 you
  should use `specdiff.draft_verify.sample_fallback_or_bonus` (see
  `lanes_homework_v2.py` for the temp-aware drop-in).
- **Models loaded at runtime.** `gpt2-xl` and `kuleshov-group/mdlm-owt`
  download from HuggingFace on first run. Internet access (or a primed HF
  cache) is required.
