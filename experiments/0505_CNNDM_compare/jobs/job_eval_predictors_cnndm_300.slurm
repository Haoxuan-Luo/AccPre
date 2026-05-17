#!/bin/bash
#SBATCH --job-name=cnndm300_eval
#SBATCH --output=experiments/0505_CNNDM_compare/logs/eval_300_%A_%a.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/eval_300_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%16

# Stage 6 — V-free predictor eval over 180 cells (15 best × 2 rules × 6 taus).
# Each cell: 100 test prompts × 512 new tokens.
#
# Allowed rules: hard_threshold, prefix_product. NO relative_max.
# Per-target fallback (set by run_dir/target):
#   relmax   -> confidence_gated_sampled_first (eta=0.5)
#   alpha_q2 -> confidence_gated_sampled_first (eta=0.5)
#   dep      -> drafter_argmax_first
# Verifier is loaded only for the OFFLINE NLL/tok_succ pass.

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi -L || true

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

# --- CNNDM project-local Python deps (see job_collect for full rationale) -
CNNDM_PYDEPS="$SLURM_SUBMIT_DIR/experiments/0505_CNNDM_compare/.pydeps"
if [ ! -d "$CNNDM_PYDEPS" ]; then
    echo "[FATAL] CNNDM_PYDEPS not found: $CNNDM_PYDEPS" 1>&2
    echo "        Run: bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh" 1>&2
    exit 2
fi
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="$SLURM_SUBMIT_DIR:$CNNDM_PYDEPS"
# -----------------------------------------------------------------------

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
BEST_PER_CELL_PATH="${BEST_PER_CELL_PATH:-$RESULTS_ROOT/best_per_cell.json}"
EVAL_ROOT="${EVAL_ROOT:-$RESULTS_ROOT/eval}"
EVAL_N_PROMPTS="${EVAL_N_PROMPTS:-100}"
EVAL_LIMIT="${EVAL_LIMIT:-}"   # optional cap per shard (dry-run)
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-16}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

EVAL_LIMIT_ARG=""
if [ -n "$EVAL_LIMIT" ]; then EVAL_LIMIT_ARG="--limit $EVAL_LIMIT"; fi

echo "[eval-300] shard=$SHARD/$N_SHARDS  best=$BEST_PER_CELL_PATH  eval_root=$EVAL_ROOT"
$PYTHON $ROOT/scripts/eval_sweep.py \
    --best_per_cell $BEST_PER_CELL_PATH \
    --eval_root $EVAL_ROOT \
    --n_prompts $EVAL_N_PROMPTS \
    --dataset cnn_dm_300 \
    --gamma 15 --T 1 --max_new_tokens 512 \
    --shard $SHARD --n_shards $N_SHARDS \
    $EVAL_LIMIT_ARG

echo "[eval-300] shard $SHARD done"
date
