#!/bin/bash
#SBATCH --job-name=cnndm300_frozen
#SBATCH --output=experiments/0505_CNNDM_compare/logs/train_frozen_300_%A_%a.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/train_frozen_300_%A_%a.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%16

# Stage 3 — frozen training sweep (144 cells / 16 shards = ~9 cells/shard).
#
# MIG slice (1g.10gb) is enough for head-only training (head < 5 MB; activations
# < 100 MB; dataloader < 500 MB). The drafter is loaded only to extract hidden
# states per record (no live verifier).

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

RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"
CONFIGS_ROOT="${CONFIGS_ROOT:-$ROOT/configs/_full}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-16}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

echo "[frozen-300] shard=$SHARD / n_shards=$N_SHARDS  runs_root=$RUNS_ROOT"
$PYTHON $ROOT/scripts/train_sweep.py \
    --regime frozen \
    --shard $SHARD --n_shards $N_SHARDS \
    --runs_root $RUNS_ROOT \
    --configs_root $CONFIGS_ROOT

echo "[frozen-300] shard $SHARD done"
date
