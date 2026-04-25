#!/bin/bash
# =============================================================================
# Phase 2A.0 — Stage-1 collection with features.
#
# Loads MDLM + GPT-2 XL; runs `accpre.collect.cli` over all 80 pool prompts
# (train + val + test splits) to produce `data_collected/stage1.pt`, a list
# of `RoundRecord`s whose optional feature fields (drafter_hidden,
# verifier_hidden, prefix_tail, verifier_entropy/margin/top1_prob) are
# populated.
#
# Cost per prompt ≈ 2× strict-round cost (extra verifier prefix forward
# + hidden-state extraction). 80 prompts × 128 new_tokens / (8 per round)
# ≈ 16 rounds/prompt ≈ 1280 records total.
#
# Submit from repo root:
#   sbatch jobs/phase2_collect.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_collect
#SBATCH --output=logs/p2_collect_%j.out
#SBATCH --error=logs/p2_collect_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 2A.0 collection ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data_collected

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$CONTAINER python3 -m pip install --user --quiet pyyaml

OUT_PATH="data_collected/stage1.pt"

$CONTAINER python3 -m accpre.collect.cli \
    --config configs/protocol.yaml \
    --out "$OUT_PATH" \
    --splits "${SPLITS:-train,val,test}" \
    --gamma "${GAMMA:-8}" \
    --T "${T:-2}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}"

echo ""
echo "--- output ---"
ls -lh "$OUT_PATH"

echo ""
echo "=== Phase 2A.0 collection OK ==="
date
