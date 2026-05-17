# 0505_CNNDM_compare — CNN/DailyMail predictor sweep

A faithful CNN/DailyMail clone of the `experiments/0505_OWT_compare/` pipeline, designed to be runnable end-to-end by a collaborator from a fresh checkout.

## Quick-start (collaborator)

```bash
# 1. Clone the repo (or pull this branch into an existing checkout).
git clone <repo-url>     # or: git fetch && git checkout <branch>
cd AccPre

# 2. (One-time, on a login node) Install the Python deps that are NOT in
#    the shared apptainer SIF — transformers, datasets, dill, multiprocess
#    — into a PROJECT-LOCAL directory:
#        experiments/0505_CNNDM_compare/.pydeps/
#    NOT into ~/.local. ~/.local is unreliably visible inside apptainer
#    batch jobs (a previous run of this pipeline failed in 6 seconds with
#    "ModuleNotFoundError: No module named 'transformers'" because the
#    batch node could not see the login-node user-site). The project-local
#    path is always visible because it sits inside the repo itself.
#    Idempotent — re-running is safe; pip skips packages already at the
#    pinned version.
bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh

# 3. (Optional) Point HuggingFace caches at a project / scratch dir so that
#    the GPT-2 XL (~6 GB) and MDLM-OWT (~880 MB) downloads land where you
#    want. Defaults to ~/.cache/huggingface.
export HF_HOME=/path/to/your/hf_cache

# 4. (Recommended) Run the LOGIN-NODE preflight. It checks the registration
#    in accpre/, the slurm files, container readability, dataset config,
#    cell counts (144 frozen + 96 joint + 180 eval), AND that .pydeps is
#    populated and resolves correctly. It does NOT submit any job.
bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --preflight-only

# 5. (Strongly recommended after the user-site -> batch issue) Submit ONLY
#    the small batch-environment preflight. This is a CPU-only Slurm job
#    (<2 min walltime budget; CPU partition `standard`) that re-runs the
#    same preflight inside the BATCH apptainer context, where ~/.local may
#    not be visible. If this passes, every other cnndm_300 slurm job will
#    import transformers/datasets correctly too.
SLURM_ACCOUNT=<your-account> bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --batch-preflight

#    Wait for it to finish (squeue -j <id>), then inspect:
#        cat experiments/0505_CNNDM_compare/logs/preflight_300_<id>.out
#    It must end with "[batch-preflight] PASS". If it does NOT, do not
#    proceed to step 6 — fix the underlying issue first.

# 6. Submit the full 8-stage pipeline with YOUR Slurm account.
#    Either:
SLURM_ACCOUNT=<your-account> bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh
#    Or with a positional argument:
bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh <your-account>
```

The submit script:
- prints the git branch and HEAD commit at the start (so you can report exactly what you ran)
- runs the same preflight as step 4 (it is always executed before any sbatch); the preflight runs with `PYTHONNOUSERSITE=1` and `PYTHONPATH=<repo>:<.pydeps>` — the exact same env every batch job uses
- if the preflight fails (including a missing `.pydeps/` from skipping step 2), exits with no jobs submitted
- if the preflight passes, prints the 8 job IDs and the dependency graph, plus copy-pasteable `squeue` / `scancel` commands

Every cnndm_300 slurm job carries the same env block:
```bash
CNNDM_PYDEPS="$SLURM_SUBMIT_DIR/experiments/0505_CNNDM_compare/.pydeps"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="$SLURM_SUBMIT_DIR:$CNNDM_PYDEPS"
```
so login-node preflight and every batch job use IDENTICAL Python paths.

### Monitor

The script's final stanza shows the exact `squeue` line. Equivalent:

```bash
squeue -u "$USER" -o "%.12i %.20j %.12T %.10M %.20R"
sacct -j <JC>,<JT>,<JB>,<JF>,<JJ>,<JP>,<JE>,<JA> \
    --format=JobID,State,ExitCode,Elapsed,Start -X
```

### Where outputs appear

| Path | Producer |
|---|---|
| `experiments/0505_CNNDM_compare/runs/{frozen,joint}/<target>/<arch>/<lr>__ep<E>__s<s>/{model.pt,train_history.json,preds_*.pt,config.yaml}` | train_frozen / train_joint |
| `experiments/0505_CNNDM_compare/results/best_per_cell.json` and `best_per_cell_report.md` | pick_best |
| `experiments/0505_CNNDM_compare/results/baselines/online_*.json` and `baselines_metrics.csv` | baselines |
| `experiments/0505_CNNDM_compare/results/eval/<regime>/<target>/<arch>/<rule>/<param_tag>/online_predictor_full.json` | eval (180 cells) |
| `experiments/0505_CNNDM_compare/results/{all_metrics.csv, final_table.{csv,md}, plots/{nll,bytes,latency}_vs_speedup.png}` | aggregate |
| `experiments/0505_CNNDM_compare/logs/*_{<job_id>,<array_id>_<task>}.{out,err}` | every stage |

None of these paths is committed; they're listed in `.gitignore`.

## Design (matches OWT_300 exactly)

- **Dataset:** CNN/DailyMail v3.0.0, **300-prompt pool**, **160 train / 40 val / 100 test**, prefix_len=32, seed=42 (registered as `cnn_dm_300` in `accpre/data/splits.py`)
- **Test prompts:** indices `[200, 300)` — analogous to OWT_300's `[200, 300)`
- **Protocol:** T=1, gamma=15, max_new_tokens=512, temperature=1.0, q_mode=A, float32
- **Drafter:** `kuleshov-group/mdlm-owt` (frozen throughout)
- **Verifier:** `gpt2-xl` (used live in joint training; offline in evaluation only)
- **Targets:** `relmax`, `alpha_q2`, `dep`
- **Frozen heads:** `mlp_pos`, `causal_transformer_pos`, `bidirectional_transformer_pos`
- **Joint heads:** `causal_transformer_pos`, `bidirectional_transformer_pos`
- **Hyperparameter grid:** lr ∈ {1e-3, 3e-4, 1e-4, 1e-5} × epochs ∈ {1, 3, 5, 10} × seed=0
  - Frozen sweep: **144 cells** (3 targets × 3 archs × 4 lrs × 4 epochs)
  - Joint sweep: **96 cells** (3 targets × 2 archs × 4 lrs × 4 epochs)
- **Commit rules:** `hard_threshold` (τ ∈ {0.3, 0.5, 0.6, 0.7, 0.8, 0.9}), `prefix_product` (τc ∈ {0.30, 0.45, 0.60, 0.75, 0.90, 0.95}). `relative_max` is forbidden.
- **Eval sweep:** 15 best checkpoints × 2 rules × 6 thresholds = **180 cells**
- **Fallback policies (all V-free):** relmax/alpha_q2 → `confidence_gated_sampled_first` (η=0.5); dep → `drafter_argmax_first`
- **Baselines (own, NOT reused from OWT):** 5 lanes on CNN/DM 100 test prompts at T=1: `strict_soft_T1`, `lossy_soft_T1` at λ ∈ {1.0, 0.7, 0.5, 0.3}
- **Loss:** `expected_survival_weighted_mse` with survival α = min(1, p_v/q). Frozen training uses fixed target files; joint training uses live targets. Only predictor head parameters are trained.

## Data files

These four files must exist at `data_collected/` before training starts:

| File | Source | Approx size |
|---|---|---|
| `stage1_pp_cnn_dm_300_g15_T1.pt` | Stage 1 (collect) | ~12 GB |
| `stage1_pp_cnn_dm_300_g15_T1_relmax.pt` | Stage 2 (build_targets) | ~13 MB |
| `stage1_pp_cnn_dm_300_g15_T1_q2.pt` | Stage 2 | ~13 MB |
| `stage1_pp_cnn_dm_300_g15_T1_dep.pt` | Stage 2 | ~13 MB |

**Two options:**

- **(A) Run them yourself.** The submit script automatically schedules `job_collect_cnndm_300.slurm` and `job_build_targets_cnndm_300.slurm`. Cost: ~8 h on A40 for collect + ~6 h for targets = **~14 h GPU wall-time** before training starts.
- **(B) Use pre-copied data.** If we transfer these 4 files to your `data_collected/` directory before you run the submit script, both `collect` and `build_targets` will skip themselves (each slurm script checks `[ -s $OUT ]` and exits 0 if present), and the pipeline starts directly at frozen + joint training.

These `.pt` files are not committed (`data_collected/` is `.gitignore`d).

## Per-stage cost estimates (A40)

| Stage | Job var | Resource | Walltime budget | Notes |
|---|---|---|---|---|
| 1. collect | JC | 1 × A40, 80 G mem | 8 h | 300 prompts × 512 new_tokens |
| 2. build_targets | JT | 1 × A40, 80 G mem | 6 h | relmax verifier replay dominates |
| 3a. baselines | JB | 1 × A40, 80 G mem | 8 h | 5 lanes × 100 prompts × 512 tokens |
| 3b. train_frozen | JF | 16 × MIG 1g.10gb (array) | 2 h each | 144 cells / 16 shards |
| 3c. train_joint | JJ | 8 × A40 (array) | 48 h each | 96 cells / 8 shards |
| 4. pick_best | JP | CPU only | 30 min | reduces 240 → 15 |
| 5. eval | JE | 16 × A40 (array) | 4 h each | 180 cells / 16 shards |
| 6. aggregate | JA | CPU only | 30 min | CSVs + Pareto plots |

Total project GPU-hours: ≈ **600–800 GPU-hours** including the high end of joint training.

## Dependency graph

```
collect ──┬─→ build_targets ──→ frozen ──┐
          │                              ├─→ pick_best ──→ eval ──┐
          ├──────────────────→ joint ────┘                         ├─→ aggregate
          └─→ baselines ───────────────────────────────────────────┘
```

## Metrics produced

- `tok_s_mean` — tokens per second (drafter + head only; verifier is offline)
- `nll_mean` — primary quality metric (offline NLL under the canonical verifier)
- `tok_succ_mean` — diagnostic under T=1 (verifier-agreement rate)
- `tokens/round`, `mean_L_hat`, `frac_Lhat_eq_0`, `frac_Lhat_eq_gamma` — supporting fields

## Slurm account

The 8 cnndm_300 slurm files **do not carry a hardcoded `#SBATCH -A` directive**. The account is supplied at submit time by `submit_full_cnndm_300.sh` via `sbatch -A "$SLURM_ACCOUNT"`. Set it via env or positional arg — see Quick-start above.

If you want to submit a single slurm file directly (e.g. for a one-off rerun), you must pass `--account=<your-account>` to sbatch yourself, e.g.

```bash
sbatch --account=<your-account> experiments/0505_CNNDM_compare/jobs/job_train_frozen_cnndm_300.slurm
```

## What this workspace touches outside its own directory

These four `accpre/` files were modified **additively** to register the new `cnn_dm_300` split (no existing entries changed):

- **`accpre/data/splits.py`** — added the `cnn_dm_300` `DatasetSplitConfig`
- **`accpre/data/prompts.py`** — added `"cnn_dm_300": load_cnn_dm_prompts` to `_LOADERS`
- **`accpre/collect/cli.py`** — added `"cnn_dm_300"` to the `--dataset` choices
- **`accpre/sweep/datasets.py`** — added the `cnn_dm_300` `DatasetBundle`

No file under `experiments/0505_OWT_compare/` is modified or read for input. No baseline JSON or output from `experiments/OWT_Frozen_0429/` is used (the CNN/DM workspace computes its own baselines via `job_baselines_cnndm_300.slurm`).

`data_collected/` is updated when Stages 1–2 run, with the four files listed above. Those files are not under version control.

## Smoke history (for context — not used in the full run)

A smoke test on a separate 80-prompt CNN/DM file (`stage1_pp_cnn_dm_g15_T1.pt`) validated the pipeline shape end-to-end on 2026-05-15. Smoke outputs live under `runs/_smoke_g15_T1/` and `results/_smoke_g15_T1/` (gitignored). The smoke files **are not used as input** for the final full run; the full run rebuilds everything from the canonical 300-prompt design.

## Troubleshooting

### A previous chain failed — what now?

If the chain you submitted earlier has failed jobs (or jobs stuck on `DependencyNeverSatisfied`), the safe recovery is:

```bash
# 1. List your CNN/DM jobs to find the IDs:
squeue -u "$USER" --name=cnndm300_collect,cnndm300_targets,cnndm300_baselines,cnndm300_frozen,cnndm300_joint,cnndm300_pickbest,cnndm300_eval,cnndm300_aggregate

# 2. Cancel the entire dead chain (substitute your IDs):
scancel <JC> <JT> <JB> <JF> <JJ> <JP> <JE> <JA>

# 3. Pull the latest branch to pick up any fixes.
git pull

# 4. Re-run preflight first; it will tell you what was wrong before any job is submitted.
bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --preflight-only

# 5. Once preflight passes, re-submit the chain:
SLURM_ACCOUNT=<your-account> bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh
```

### If `cnndm300_collect` fails quickly (< 1 minute)

The cause is almost always reported in the `.err` log. Capture it and share back:

```bash
# Substitute the actual collect job ID — it's the one named cnndm300_collect.
cat experiments/0505_CNNDM_compare/logs/collect_300_<job_id>.err
cat experiments/0505_CNNDM_compare/logs/collect_300_<job_id>.out
```

Common root causes:
- **`transformers` / `datasets` not in `.pydeps/`** — you skipped step 2 of Quick-start, or `.pydeps` got deleted. The shared SIF only provides torch + pyyaml + numpy/pyarrow/pandas. The cnndm_300 chain stages transformers/datasets/dill/multiprocess into `experiments/0505_CNNDM_compare/.pydeps/` via `setup_collab_env.sh`. Every slurm job sets `PYTHONNOUSERSITE=1` (no `~/.local`) and `PYTHONPATH=<repo>:<.pydeps>`, so missing `.pydeps` ⇒ instant ModuleNotFoundError. Fix:
  ```bash
  bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh
  bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --preflight-only
  SLURM_ACCOUNT=<your-account> bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --batch-preflight
  ```
- **Login preflight passes but batch fails anyway** — this used to happen when the pipeline relied on `~/.local`; the login node and the batch node have different views of HOME/UID, and `~/.local` was visible on one but not the other. Mitigation: always run the `--batch-preflight` (step 5) BEFORE submitting the full pipeline. If `--preflight-only` passes but `--batch-preflight` fails, share the `preflight_300_<id>.err` log back.
- Stale clone (missing `cnn_dm_300` in `accpre/`) — `git pull`, then `--preflight-only` again.
- Apptainer module name differs on your cluster — `module avail apptainer` to confirm.
- HF cache not writable — set `HF_HOME` to a scratch / project directory.

### Other failure modes

- **A frozen training shard times out**: re-run only the failed shards via `sbatch --account=<your-account> --array=<bad_indices> jobs/job_train_frozen_cnndm_300.slurm`. The `train_sweep.py` script has skip-on-exists so completed cells are not retrained.
- **A joint training shard times out** (more common — joint cells are long): use the recovery flow. Build a manifest of incomplete cells; `train_sweep.py --manifest <csv> --move_aside_partial` re-runs only those cells and stashes any stale `model.pt` first.
- **`pick_best` blocks on `DependencyNeverSatisfied`**: happens when the chain uses `afterok` and at least one upstream array task failed/timed-out. Manually verify completion (`find experiments/0505_CNNDM_compare/runs/joint -name train_history.json | wc -l`), then re-submit `job_pick_best_cnndm_300.slurm` without dependencies (`sbatch --account=<your-account> jobs/job_pick_best_cnndm_300.slurm`).

## Reproducibility

- `seed=42` for split RNG; `seed=0` for per-cell training. Stage-1 collection seeds RNGs from the protocol's salts (`accept_salt`, `draft_salt`, `fallback_salt`), so the records file is deterministic given the same protocol fingerprint and dataset.
- The `protocol_fp` field is stored in every record and every output JSON; mismatches between collection-time and eval-time protocols will be flagged at load.
