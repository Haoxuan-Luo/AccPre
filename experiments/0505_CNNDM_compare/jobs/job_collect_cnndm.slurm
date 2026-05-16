#!/bin/bash
#SBATCH --job-name=0505_cnndm_collect
#SBATCH --output=experiments/0505_CNNDM_compare/logs/collect_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/collect_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# CNN/DailyMail stage-1 record collection at T=1, gamma=15.
#
# Outputs:
#   data_collected/stage1_pp_cnn_dm_g15_T1.pt
#
# Protocol: configs/protocol.yaml (T=1 protocol, MDLM drafter, GPT-2 XL verifier)
# Dataset:  cnn_dm (80-prompt pool, 40/20/20, prefix_len=32, seed=42)
# Per-prompt budget: max_new_tokens=512 (matches OWT_300 collection)
#
# WALLTIME RATIONALE:
#   OWT_300 collection (300 prompts × 512 tokens) took ~6 h on A100.
#   CNN/DM has 80 prompts (27%), so 4 h on A40 is generous safety margin.
#
# IDEMPOTENCY:
#   The CLI does not natively skip if --out exists. This slurm script wraps the
#   call with a pre-check that:
#     - skips collection if the output is already non-empty
#     - never overwrites data_collected/stage1_pp_cnn_dm.pt (the OLD T=2/g=8 file)
#
# This job is the DATA-COLLECTION stage of the CNN/DM workspace.
# It is NOT a training/eval job, and it does not submit a full run.

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi -L

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

OUT=data_collected/stage1_pp_cnn_dm_g15_T1.pt
OLD=data_collected/stage1_pp_cnn_dm.pt

# Guardrail: never touch the old T=2/g=8 file under any circumstances.
if [ ! -f "$OLD" ]; then
    echo "[collect] note: $OLD does not exist (no guardrail needed)"
else
    echo "[collect] guardrail: $OLD will NOT be modified (existing T=2/g=8 file is preserved)"
fi

# Skip-if-exists: bail out if a non-empty output is already present.
if [ -s "$OUT" ]; then
    bytes=$(stat -c %s "$OUT")
    echo "[collect] SKIP: $OUT already exists ($bytes bytes); not re-collecting."
    echo "[collect] If you want to force re-collection, rename or remove the file first."
    exit 0
fi

echo "[collect] === collecting CNN/DM records: T=1, gamma=15, dataset=cnn_dm ==="
$PYTHON -m accpre.collect.cli \
    --config configs/protocol.yaml \
    --dataset cnn_dm \
    --splits train,val,test \
    --gamma 15 \
    --T 1 \
    --max_new_tokens 512 \
    --out "$OUT"

# Verify output exists and is non-trivial.
if [ ! -s "$OUT" ]; then
    echo "[collect] FAILED: $OUT was not created or is empty"
    exit 1
fi

bytes=$(stat -c %s "$OUT")
echo "[collect] wrote $OUT ($bytes bytes)"

echo "[collect] done"
date
