#!/bin/bash
# =============================================================================
# Phase 14 — direct threshold classifier (acc_thresh).
#
# Trains AcceptanceThresholdMultiHead (same 1A body, 3 sigmoid heads at
# τ ∈ {0.5, 0.7, 0.9}, BCE loss) and runs the online lane so tok/s is
# measured on the same hardware as the 1A reference.
#
# Same target family, same token embedding, same 1A architecture body,
# same online semantics. No calibration, no weighted MSE, no transformer
# head, no V-prefix, no length.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p14
#SBATCH --output=logs/p14_%j.out
#SBATCH --error=logs/p14_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 14 (acc_thresh) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml scikit-learn

OUT_DIR="outputs/phase14_thresh_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

CFG=configs/predictors/acc_thresh.yaml
RUN=acc_thresh
CKPT="checkpoints/${RUN}"
PRED_OUT="$OUT_DIR/${RUN}"
mkdir -p "$PRED_OUT"

echo ""; echo "=== train: $CFG ==="
$CONTAINER python3 -m accpre.train.cli --config "$CFG"

echo ""; echo "=== online: $CKPT ==="
$CONTAINER python3 -m accpre.eval.online_decode \
    --config configs/protocol.yaml \
    --checkpoint_dir "$CKPT" \
    --strict_records data_collected/stage1_pp.pt \
    --split test --n_prompts 10 --max_new_tokens 64 \
    --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
    --out_dir "$PRED_OUT"

echo ""; echo "=== comparison report (acc_thresh vs acc_1a vs oracle) ==="
$CONTAINER python3 scripts/compare_thresh.py

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la "$CKPT"

echo ""; echo "=== Phase 14 done ==="
date
