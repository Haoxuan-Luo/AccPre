#!/bin/bash
# =============================================================================
# Phase 8 — strict tok/s refresh on global test-split prompt indices.
#
# Rationale: the Phase 1 strict baseline (19.75 tok/s) was measured with
# LOCAL enumerate indices 0..9 on test_split()[:10]. Every Phase 3+
# predictor lane seeds with GLOBAL indices [60..69]. Prompts are the
# same, but the RNG trajectories differ, so `× strict` in the compare
# scripts is not randomness-matched. This job measures strict tok/s
# under the SAME convention the predictor lanes use and writes
# outputs/strict_global_idx/summary.json; the compare scripts pick
# this up automatically via accpre.eval.strict_ref.
#
# Not a new experiment — just a cleaner reference.
#
# Outputs:
#   outputs/strict_global_idx/summary.json   (canonical, picked up by
#                                             compare_phase{5,6,7}.py)
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p8_strict
#SBATCH --output=logs/p8_strict_%j.out
#SBATCH --error=logs/p8_strict_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Phase 8 strict tok/s refresh (global prompt indices) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs/strict_global_idx

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

echo ""; echo "=== strict refresh: test_split()[:10], global indices [60..69] ==="
$CONTAINER python3 -m accpre.eval.strict_refresh \
    --config configs/protocol.yaml \
    --n_prompts 10 \
    --max_new_tokens 64 \
    --gamma 8 --T 2 \
    --out outputs/strict_global_idx/summary.json

echo ""; echo "--- artifact ---"
ls -la outputs/strict_global_idx/summary.json

echo ""; echo "=== Phase 8 done ==="
date
