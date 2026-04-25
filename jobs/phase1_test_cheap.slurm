#!/bin/bash
# =============================================================================
# Phase 1 validation — CHEAP (non-slow) pytest suite.
#
# Validates: canonical accept / commit / schema / faithfulness /
# matched-randomness logic using synthetic tensors. No models loaded.
# CPU-only, short timeout. Must pass before any GPU validation is run.
#
# Submit from repo root:
#   sbatch jobs/phase1_test_cheap.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p1_cheap
#SBATCH --output=logs/p1_test_cheap_%j.out
#SBATCH --error=logs/p1_test_cheap_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

echo "=== Phase 1 cheap pytest ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"

# Install pytest into user site-packages if missing (cheap no-op if present).
$CONTAINER python3 -m pip install --user --quiet pytest pyyaml

# Run the non-slow subset. pyproject.toml sets -m 'not slow' as default,
# so this is explicit for clarity / robustness.
$CONTAINER python3 -m pytest -m 'not slow' -v tests/

echo ""
echo "=== Phase 1 cheap pytest OK ==="
date
