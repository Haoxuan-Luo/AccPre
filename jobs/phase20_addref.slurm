#!/bin/bash
# =============================================================================
# Phase 20 follow-up: measure strict tok/s on the canonical online subset,
# then regenerate the system table with strict + oracle-Q2 reference rows.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p20a
#SBATCH --output=logs/p20a_%j.out
#SBATCH --error=logs/p20a_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail

echo "=== Phase 20 follow-up (strict + oracle rows) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs/strict_global_idx

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

# 1. Strict refresh — freshly measures strict tok/s on prompts 60..69
#    × 64 new tokens under the pretrained drafter. This is the apples-to-
#    apples number we want for the strict reference row.
echo ""; echo "=== 1. strict_refresh (10 prompts × 64 new tokens) ==="
$CONTAINER python3 -m accpre.eval.strict_refresh \
    --config configs/protocol.yaml \
    --n_prompts 10 --max_new_tokens 64 --gamma 8 --T 2 \
    --out outputs/strict_global_idx/summary.json

OUT_DIR="outputs/phase20_addref_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo ""; echo "=== 2. assemble updated table + Pareto with strict + oracle ==="
$CONTAINER python3 scripts/phase20_addref.py \
    --phase20_dir outputs/phase20_eval_11869681 \
    --oracle_json outputs/oracle_q2_ceiling/summary.json \
    --strict_refresh_json outputs/strict_global_idx/summary.json \
    --out_dir "$OUT_DIR"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
echo
echo "strict_refresh output:"
cat outputs/strict_global_idx/summary.json | head -20

echo ""; echo "=== Phase 20 follow-up done ==="
date
