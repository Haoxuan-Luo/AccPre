#!/bin/bash
#SBATCH --job-name=cnndm300_collect
#SBATCH --output=experiments/0505_CNNDM_compare/logs/collect_300_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/collect_300_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00

# Stage 1 — CNN/DailyMail 300-prompt stage-1 record collection at T=1, gamma=15.
#
# Output:  data_collected/stage1_pp_cnn_dm_300_g15_T1.pt
# Splits:  train (160) / val (40) / test (100), prefix_len=32, seed=42.
# Per-prompt budget: max_new_tokens=512.
#
# Walltime rationale: OWT_300 (300 prompts × 512) took ~6 h on A100; A40 is
# ~1.3× slower so 8 h is a safe budget.
#
# Idempotent: skips if output already exists and is non-empty.

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

OUT="${OUT:-data_collected/stage1_pp_cnn_dm_300_g15_T1.pt}"

if [ -s "$OUT" ]; then
    echo "[collect_300] SKIP: $OUT already exists ($(stat -c %s "$OUT") bytes)."
    exit 0
fi

echo "[collect_300] === collecting CNN/DM records: dataset=cnn_dm_300 T=1 gamma=15 ==="
$PYTHON -m accpre.collect.cli \
    --config configs/protocol.yaml \
    --dataset cnn_dm_300 \
    --splits train,val,test \
    --gamma 15 --T 1 \
    --max_new_tokens 512 \
    --out "$OUT"

if [ ! -s "$OUT" ]; then
    echo "[collect_300] FAILED: $OUT not produced or empty"; exit 1
fi
echo "[collect_300] wrote $OUT ($(stat -c %s "$OUT") bytes)"
date
