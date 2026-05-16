#!/bin/bash
#SBATCH --job-name=cnndm300_baselines
#SBATCH --output=experiments/0505_CNNDM_compare/logs/baselines_300_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/baselines_300_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00

# Stage 4 — CNN/DM 100 test-prompt baselines, T=1 family only.
#   1. strict_soft_T1            (l=1.0, T=1; equivalent to lossy l=1.0)
#   2. lossy_soft_T1_l1          (l=1.0)
#   3. lossy_soft_T1_l0p7        (l=0.7)
#   4. lossy_soft_T1_l0p5        (l=0.5)
#   5. lossy_soft_T1_l0p3        (l=0.3)
#
# Each lane is one verifier-side decode pass on the 100 test prompts (indices
# 200..299 from the cnn_dm_300 pool) with gamma=15, T=1, max_new_tokens=512.
# A single lane is ~1-1.5 h on A40 (300 prompts in OWT_300 took ~6 h for all 5
# lanes; CNN/DM is ~the same scale).

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi -L || true

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
BASELINES_DIR="${BASELINES_DIR:-$RESULTS_ROOT/baselines}"

mkdir -p "$BASELINES_DIR"

echo "[baselines-300] === running 5 T=1 baseline lanes on cnn_dm_300 test (100 prompts) ==="
$PYTHON $ROOT/scripts/run_baselines.py \
    --out_dir "$BASELINES_DIR" \
    --protocol_temp1 configs/protocol.yaml \
    --dataset cnn_dm_300 \
    --n_prompts 100 \
    --prompt_offset 200 \
    --gamma 15 --T 1 \
    --max_new_tokens 512 \
    --lossy_ls 1.0 0.7 0.5 0.3 \
    --include_strict_T1 \
    --lanes strict_soft_T1,lossy_soft_T1

echo "[baselines-300] === computing offline NLL fields ==="
$PYTHON $ROOT/scripts/add_baseline_nll.py \
    --baselines_dir "$BASELINES_DIR" \
    --protocol configs/protocol.yaml

echo "[baselines-300] outputs:"
ls -la "$BASELINES_DIR"/*.json
date
