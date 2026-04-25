#!/bin/bash
# =============================================================================
# Phase 12 — acceptance-only feature-side directions.
#
#   1A (acc_1a): hidden_per_pos_v2 + per-position drafted-token embedding,
#                shared-weight per-position MLP head (no cross-position).
#   1B (acc_1b): same inputs as 1A but with a small per-position transformer
#                head (2 layers, d=128, 4 heads).
#
# Both V-free, acceptance only, plain MSE loss. τ grid at inference:
# {0.5, 0.7, 0.9} — no 0.3.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p12
#SBATCH --output=logs/p12_%j.out
#SBATCH --error=logs/p12_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 12 (branch 1A + 1B) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

OUT_DIR="outputs/phase12_1a1b_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

for CFG in configs/predictors/acc_1a.yaml configs/predictors/acc_1b.yaml; do
    RUN=$(basename "${CFG%.yaml}")
    echo ""; echo "=== train: $CFG ==="
    $CONTAINER python3 -m accpre.train.cli --config "$CFG"

    CKPT="checkpoints/${RUN}"
    PRED_OUT="$OUT_DIR/${RUN}"
    mkdir -p "$PRED_OUT"
    echo ""; echo "=== online: $CKPT ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$CKPT" \
        --strict_records data_collected/stage1_pp.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
done

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/acc_1a/ checkpoints/acc_1b/

echo ""; echo "=== Phase 12 done ==="
date
