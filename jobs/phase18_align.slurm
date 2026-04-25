#!/bin/bash
# =============================================================================
# Phase 18 — minimal token-slot drafter alignment.
#
# Chain:
#   1. Train aligned drafter (drafter-only, no head).
#   2. Recollect all 80 prompts with the aligned drafter.
#   3. Retrain frozen acc_1a-style head on the recollected records.
#   4. Run the Phase 18 diagnostic: Q2 before/after, offline CF@1 for
#      both heads on both data sources.
#   5. Online decode with the aligned drafter + aligned head at
#      τ ∈ {0.5, 0.7, 0.9}.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p18
#SBATCH --output=logs/p18_%j.out
#SBATCH --error=logs/p18_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail

echo "=== Phase 18 (drafter token-slot alignment) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs data_collected

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml scikit-learn

OUT_DIR="outputs/phase18_align_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

ALIGNED_DRAFTER="checkpoints/drafter_align_tokslot/drafter.pt"
ALIGNED_DATA="data_collected/stage1_pp_aligned_${SLURM_JOB_ID}.pt"

echo ""; echo "=== 1. train aligned drafter (token-slot) ==="
$CONTAINER python3 -m accpre.train.align_cli \
    --config configs/predictors/drafter_align_tokslot.yaml

echo ""; echo "=== 2. recollect all 80 prompts with aligned drafter ==="
$CONTAINER python3 scripts/recollect_all.py \
    --drafter_state "$ALIGNED_DRAFTER" \
    --out "$ALIGNED_DATA" \
    --gamma 8 --T 2 --max_new_tokens 128

# Make the trainer find the aligned file at the canonical path.
cp "$ALIGNED_DATA" data_collected/stage1_pp_aligned.pt

echo ""; echo "=== 3. retrain frozen 1A head on aligned records ==="
$CONTAINER python3 -m accpre.train.cli \
    --config configs/predictors/acc_1a_aligned.yaml

echo ""; echo "=== 4. Phase 18 Q2 / CF@1 diagnostic ==="
$CONTAINER python3 scripts/phase18_diag.py \
    --data_before data_collected/stage1_pp.pt \
    --data_after  data_collected/stage1_pp_aligned.pt \
    --frozen_ckpt  checkpoints/acc_1a \
    --aligned_ckpt checkpoints/acc_1a_aligned \
    --out_dir "$OUT_DIR/diag"

echo ""; echo "=== 5. online decode — aligned drafter + aligned head ==="
$CONTAINER python3 -m accpre.eval.online_decode \
    --config configs/protocol.yaml \
    --checkpoint_dir checkpoints/acc_1a_aligned \
    --strict_records data_collected/stage1_pp_aligned.pt \
    --drafter_state "$ALIGNED_DRAFTER" \
    --split test --n_prompts 10 --max_new_tokens 64 \
    --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
    --out_dir "$OUT_DIR/online"

echo ""; echo "=== 6. predictor-level compare (include aligned) ==="
$CONTAINER python3 scripts/predictor_level_compare.py --include_long \
    --out_dir "$OUT_DIR/predictor_compare" || true

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/drafter_align_tokslot checkpoints/acc_1a_aligned

echo ""; echo "=== Phase 18 done ==="
date
