#!/bin/bash
#SBATCH --job-name=cnndm300_joint
#SBATCH --output=experiments/0505_CNNDM_compare/logs/train_joint_300_%A_%a.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/train_joint_300_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-7%8

# Stage 3 — joint training sweep (96 cells / 8 shards = 12 cells/shard).
#
# A40 (48 GB) holds frozen drafter (~440 MB) + verifier (~3 GB) + head +
# activations comfortably. The verifier is invoked LIVE per batch to compute
# the requested target. Drafter and verifier remain frozen; only predictor
# head parameters are trained.
#
# Walltime 48 h matches the proven OWT joint config; the worst cells
# (ep10 × dep) need it. If a shard times out, the manifest-recovery flow
# in train_sweep.py (--manifest --move_aside_partial) handles re-runs cleanly.

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
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-8}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

echo "[joint-300] shard=$SHARD / n_shards=$N_SHARDS  runs_root=$RUNS_ROOT"
$PYTHON $ROOT/scripts/train_sweep.py \
    --regime joint \
    --shard $SHARD --n_shards $N_SHARDS \
    --runs_root $RUNS_ROOT \
    --configs_root $CONFIGS_ROOT

echo "[joint-300] shard $SHARD done"
date
