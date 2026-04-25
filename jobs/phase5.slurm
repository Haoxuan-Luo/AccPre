#!/bin/bash
# =============================================================================
# Phase 5 — richer per-position features.
#
#  1. re-collect stage1_pp.pt with new drafter-side scalar fields populated
#     (drafter_entropy_j, drafter_margin_j, drafter_top1_prob_j; q_j
#     already present). Deterministic overwrite — same protocol + seeds.
#  2. train acc_frz_pp_hs  (main candidate: hidden + 4 scalars per position)
#     and acc_frz_pp_s     (ablation: 4 scalars only).
#  3. online decode both.
#
# Baseline acc_frz_hid_pp already trained; reuse its existing online output
# for the comparison (outputs/phase3_online_pp_11799951/).
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p5
#SBATCH --output=logs/p5_%j.out
#SBATCH --error=logs/p5_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00

set -euo pipefail

echo "=== Phase 5 richer per-position features ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs data_collected

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

# Step 1: re-collect stage1_pp.pt with the new scalar fields. Skip if
# file was rewritten within the last 6 hours (i.e. already updated under
# this schema).
if [ -n "$(find data_collected/stage1_pp.pt -mmin -360 2>/dev/null)" ]; then
    echo "data_collected/stage1_pp.pt is recent — skipping collection"
else
    echo ""; echo "=== re-collecting stage1_pp.pt with drafter scalars ==="
    $CONTAINER python3 -m accpre.collect.cli \
        --config configs/protocol.yaml \
        --out data_collected/stage1_pp.pt \
        --splits train,val,test \
        --gamma 8 --T 2 --max_new_tokens 128
fi

# Step 2: training.
CONFIGS=(
    "configs/predictors/acc_frz_pp_hs.yaml"
    "configs/predictors/acc_frz_pp_s.yaml"
)
for cfg in "${CONFIGS[@]}"; do
    echo ""; echo "=== train: $cfg ==="
    $CONTAINER python3 -m accpre.train.cli --config "$cfg"
done

# Step 3: online decode.
OUT_DIR="outputs/phase5_online_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
CKPTS=(
    "checkpoints/acc_frz_pp_hs"
    "checkpoints/acc_frz_pp_s"
)
for ckpt in "${CKPTS[@]}"; do
    [ -f "$ckpt/model.pt" ] || { echo "SKIP $ckpt"; continue; }
    PRED_OUT="$OUT_DIR/$(basename $ckpt)"
    mkdir -p "$PRED_OUT"
    echo ""; echo "=== online: $ckpt ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$ckpt" \
        --strict_records data_collected/stage1_pp.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
done

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""; echo "=== Phase 5 done ==="
date
