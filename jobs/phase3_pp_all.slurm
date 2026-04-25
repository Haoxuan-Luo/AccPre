#!/bin/bash
# =============================================================================
# Phase 3 — per-position acceptance redesign: collect → train (3) → online (3).
#
# Step 1: re-collect stage1 with `drafter_hidden_per_pos` populated.
#         Writes to data_collected/stage1_pp.pt (new file; existing
#         stage1.pt preserved).
# Step 2: train acc_frz_hid_pp, acc_frz_hid_sbce_pp, acc_jnt_hid_pp.
# Step 3: online decode all three (V-free, joint auto-re-collect for
#         the joint predictor's drafter state).
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p3_pp
#SBATCH --output=logs/p3_pp_%j.out
#SBATCH --error=logs/p3_pp_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:30:00

set -euo pipefail

echo "=== Phase 3 per-position redesign ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs data_collected

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

# --- Step 1: collection with per-pos field ---
if [ ! -f "data_collected/stage1_pp.pt" ]; then
    echo ""
    echo "=== collecting stage1_pp.pt ==="
    $CONTAINER python3 -m accpre.collect.cli \
        --config configs/protocol.yaml \
        --out data_collected/stage1_pp.pt \
        --splits train,val,test \
        --gamma 8 --T 2 --max_new_tokens 128
else
    echo "data_collected/stage1_pp.pt already exists; skipping collection"
fi

# --- Step 2: training ---
CONFIGS=(
    "configs/predictors/acc_frz_hid_pp.yaml"
    "configs/predictors/acc_frz_hid_sbce_pp.yaml"
    "configs/predictors/acc_jnt_hid_pp.yaml"
)
for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== train: $cfg ==="
    $CONTAINER python3 -m accpre.train.cli --config "$cfg"
done

# --- Step 3: online decode ---
OUT_DIR="outputs/phase3_online_pp_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

CKPTS=(
    "checkpoints/acc_frz_hid_pp"
    "checkpoints/acc_frz_hid_sbce_pp"
    "checkpoints/acc_jnt_hid_pp"
)
for ckpt in "${CKPTS[@]}"; do
    [ -f "$ckpt/model.pt" ] || { echo "SKIP $ckpt"; continue; }
    PRED_OUT="$OUT_DIR/$(basename $ckpt)"
    mkdir -p "$PRED_OUT"
    echo ""
    echo "=== online decode: $ckpt ==="
    # Use stage1_pp.pt as strict-records source (contains per-pos field).
    # For frozen predictors, online_decode filters to matching prompts.
    # For the joint predictor, online_decode auto-re-collects strict with
    # the fine-tuned drafter (both axes measured on same drafter state).
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$ckpt" \
        --strict_records data_collected/stage1_pp.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
done

echo ""
echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""
echo "=== Phase 3 per-pos done ==="
date
