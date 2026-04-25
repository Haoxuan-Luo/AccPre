#!/bin/bash
# =============================================================================
# Phase 20 — pure-evaluation pass.
#
# Predictor-level (Table A) is CPU-fast; verifier NLL (Table B) needs GPU
# (one GPT-2-XL forward per generated sequence). One short SLURM job.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p20
#SBATCH --output=logs/p20_%j.out
#SBATCH --error=logs/p20_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Phase 20 (pure evaluation pass) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

OUT_DIR="outputs/phase20_eval_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

$CONTAINER python3 scripts/phase20_eval.py \
    --out_dir "$OUT_DIR" \
    --max_new_tokens 64

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""; echo "=== Phase 20 done ==="
date
