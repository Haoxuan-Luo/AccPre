#!/bin/bash
# =============================================================================
# Phase 9 Step-1 — A1 sanity.
#
#   1. Train acc_tx_v1 (per-position transformer; family hidden_per_pos_v1;
#      feature content-equivalent to pp_base's hidden_per_pos).
#   2. Online decode at tau in {0.3, 0.5, 0.7, 0.9}.
#   3. scripts/sanity_a1.py — side-by-side against pp_base
#      (val MSE, CF@1, tok/s).
#
# Purpose: catch plumbing / shape / training errors on the simplest
# feature family (same content as pp_base), before expanding to A2/A3/A4.
#
# Not a gate for L1. The family plan runs to completion regardless of
# what this produces.
#
# Reference:
#   pp_base checkpoint  : checkpoints/acc_frz_hid_pp/
#   pp_base online      : outputs/phase3_online_pp_11799951/acc_frz_hid_pp/
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p9_a1
#SBATCH --output=logs/p9_a1_%j.out
#SBATCH --error=logs/p9_a1_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 9 Step-1 (A1 sanity) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

# -----------------------------------------------------------------------------
# Step 1a — train acc_tx_v1
# -----------------------------------------------------------------------------
CFG=configs/predictors/acc_tx_v1.yaml
echo ""; echo "=== train: $CFG ==="
$CONTAINER python3 -m accpre.train.cli --config "$CFG"

# -----------------------------------------------------------------------------
# Step 1b — online decode
# -----------------------------------------------------------------------------
OUT_DIR="outputs/phase9_a1_online_${SLURM_JOB_ID}"
PRED_OUT="$OUT_DIR/acc_tx_v1"
mkdir -p "$PRED_OUT"
CKPT=checkpoints/acc_tx_v1

echo ""; echo "=== online: $CKPT ==="
$CONTAINER python3 -m accpre.eval.online_decode \
    --config configs/protocol.yaml \
    --checkpoint_dir "$CKPT" \
    --strict_records data_collected/stage1_pp.pt \
    --split test --n_prompts 10 --max_new_tokens 64 \
    --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
    --out_dir "$PRED_OUT"

# -----------------------------------------------------------------------------
# Step 1c — sanity report against pp_base
# -----------------------------------------------------------------------------
echo ""; echo "=== sanity report: scripts/sanity_a1.py ==="
# NB pass the predictor's online subfolder (where online_*.json live),
# not the parent OUT_DIR.
$CONTAINER python3 scripts/sanity_a1.py "$PRED_OUT" \
    | tee "$OUT_DIR/sanity_a1.txt"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/acc_tx_v1/

echo ""; echo "=== Phase 9 Step-1 done ==="
date
