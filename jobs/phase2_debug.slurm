#!/bin/bash
# =============================================================================
# Phase 2A debug / sanity pass.
#
# Runs `scripts/debug_phase2a.py` on CPU. Loads Stage 2A.0 records + five
# trained predictors + Phase 2A.3 online JSONs; emits a single textual
# report covering the five diagnostic sections in the review request.
#
# Submit:
#   sbatch jobs/phase2_debug.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_debug
#SBATCH --output=logs/p2_debug_%j.out
#SBATCH --error=logs/p2_debug_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

echo "=== Phase 2A debug pass ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

$CONTAINER python3 scripts/debug_phase2a.py

echo ""
echo "=== Phase 2A debug pass OK ==="
date
