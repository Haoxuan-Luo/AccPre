#!/bin/bash
# =============================================================================
# Quick Pareto overlay for existing Phase 2 online outputs.
#
# Reads Phase-1 summary.json (strict + oracle-Q2) + Phase-2 online_*.json
# per-predictor dirs, and renders the combined figure. CPU-only.
#
# Usage:
#   PHASE2_OUT_DIR=outputs/phase2_online_<jobid> \
#       PHASE1_SUMMARY_DIR=outputs/phase1_full_<jobid> \
#       sbatch jobs/phase2_pareto_overlay.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_pareto
#SBATCH --output=logs/p2_pareto_%j.out
#SBATCH --error=logs/p2_pareto_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

echo "=== Phase 2 Pareto overlay ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

: "${PHASE2_OUT_DIR:?PHASE2_OUT_DIR must be set}"
: "${PHASE1_SUMMARY_DIR:?PHASE1_SUMMARY_DIR must be set}"

PRED_DIRS=$(find "$PHASE2_OUT_DIR" -mindepth 1 -maxdepth 1 -type d | paste -sd ',')
echo "Phase 1 summary dir: $PHASE1_SUMMARY_DIR"
echo "Predictor dirs: $PRED_DIRS"

$CONTAINER python3 -m accpre.eval.pareto \
    --in_dir "$PHASE1_SUMMARY_DIR" \
    --predictor_dirs "$PRED_DIRS" \
    --out_fig "$PHASE2_OUT_DIR/pareto_phase2.png"

echo ""
echo "--- output ---"
ls -lh "$PHASE2_OUT_DIR/pareto_phase2.png"

echo ""
echo "=== Pareto overlay OK ==="
date
