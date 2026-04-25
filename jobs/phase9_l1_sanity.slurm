#!/bin/bash
# =============================================================================
# Phase 9 Step-2 — L1 sanity.
#
#   1. Train len_tx_v1 (per-position transformer committed-length head;
#      family hidden_per_pos_v1; tau-conditioned; weighted CE).
#   2. Online decode at tau in {0.3, 0.5, 0.7, 0.9} — produces
#      online_committed_length_tau_*.json with predict_L-driven L_hat
#      distributions.
#   3. scripts/sanity_l1.py — three-part plumbing check:
#        (a) training trajectory is decreasing / finite
#        (b) L_hat distribution on preds_test.pt is non-degenerate
#        (c) per-tau |mean(L_hat) - mean(L_strict)| <= 1.5 at tau in
#            {0.3, 0.5, 0.7, 0.9}
#
# Purpose: plumbing sanity for the tau-conditioned length head. This
# runs regardless of A1 outcome per the staging clarification: A1
# strength does not gate L1.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p9_l1
#SBATCH --output=logs/p9_l1_%j.out
#SBATCH --error=logs/p9_l1_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 9 Step-2 (L1 sanity) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

# -----------------------------------------------------------------------------
# Step 2a — train len_tx_v1
# -----------------------------------------------------------------------------
CFG=configs/predictors/len_tx_v1.yaml
echo ""; echo "=== train: $CFG ==="
$CONTAINER python3 -m accpre.train.cli --config "$CFG"

# -----------------------------------------------------------------------------
# Step 2b — online decode (committed-length lane)
# -----------------------------------------------------------------------------
OUT_DIR="outputs/phase9_l1_online_${SLURM_JOB_ID}"
PRED_OUT="$OUT_DIR/len_tx_v1"
mkdir -p "$PRED_OUT"
CKPT=checkpoints/len_tx_v1

echo ""; echo "=== online: $CKPT ==="
$CONTAINER python3 -m accpre.eval.online_decode \
    --config configs/protocol.yaml \
    --checkpoint_dir "$CKPT" \
    --strict_records data_collected/stage1_pp.pt \
    --split test --n_prompts 10 --max_new_tokens 64 \
    --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
    --out_dir "$PRED_OUT"

# -----------------------------------------------------------------------------
# Step 2c — sanity report (training trajectory, L_hat distribution, per-tau)
# -----------------------------------------------------------------------------
echo ""; echo "=== sanity report: scripts/sanity_l1.py ==="
$CONTAINER python3 scripts/sanity_l1.py "$PRED_OUT" \
    | tee "$OUT_DIR/sanity_l1.txt"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/len_tx_v1/

echo ""; echo "=== Phase 9 Step-2 done ==="
date
