#!/bin/bash
# =============================================================================
# Phase 1 validation — SLOW smoke pytest (loads real models).
#
# Validates: end-to-end pipeline smoke tests in tests/test_smoke_pipeline.py.
# Loads MDLM + GPT-2 XL once, exercises draft_verify_round, run_strict_lane,
# oracle-Q2 commit_threshold, and strict round idempotence under fixed seed.
#
# Uses a gpu-mig 2g.20gb slice (20 GB VRAM). fp32 GPT-2 XL + MDLM fits
# comfortably; drop to 1g.10gb only if switching protocol.dtype to float16.
#
# Submit from repo root AFTER phase1_test_cheap.slurm passes:
#   sbatch jobs/phase1_test_smoke.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p1_smoke
#SBATCH --output=logs/p1_test_smoke_%j.out
#SBATCH --error=logs/p1_test_smoke_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Phase 1 smoke pytest (with models) ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Make sure slow-test deps are available in user site-packages.
$CONTAINER python3 -m pip install --user --quiet pytest pyyaml

# Run ONLY the slow smoke file with the slow marker enabled.
$CONTAINER python3 -m pytest -m slow -v tests/test_smoke_pipeline.py

echo ""
echo "=== Phase 1 smoke pytest OK ==="
date
