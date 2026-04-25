#!/bin/bash
# =============================================================================
# Phase 7 V1 — pp_vpn: V-prefix minimal acceptance predictor.
#
# Feature: per-position drafter hidden (γ, H) concatenated with 3 verifier-
# prefix numeric scalars broadcast to every draft position, (γ, H+3).
# Deploy mode: V-prefix (one verifier prefix-only forward per round).
#
#  1. train acc_frz_pp_vpn (no re-collection: stage1_pp.pt already has
#     drafter_hidden_per_pos AND verifier_entropy/margin/top1_prob).
#  2. online decode acc_frz_pp_vpn at τ∈{0.3,0.5,0.7,0.9}. Logs per-round
#     draft_time_s, verifier_prefix_time_s, predictor_time_s so the
#     comparison can produce a wall-clock breakdown.
#  3. run scripts/compare_phase7.py (pp_base vs pp_vpn) with the gate.
#
# Baseline pp_base (V-free) reused from Phase 3: checkpoint
# checkpoints/acc_frz_hid_pp/ and online output
# outputs/phase3_online_pp_11799951/acc_frz_hid_pp/.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p7
#SBATCH --output=logs/p7_%j.out
#SBATCH --error=logs/p7_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00

set -euo pipefail

echo "=== Phase 7 V1 V-prefix minimal (pp_vpn) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs data_collected

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

# Step 1: cheap unit tests for the new feature family (no GPU needed).
echo ""; echo "=== unit tests for hidden_per_pos_numeric ==="
$CONTAINER python3 -m pytest -x -q \
    tests/test_collect_features.py::test_hidden_per_pos_numeric_feature_dim \
    tests/test_collect_features.py::test_hidden_per_pos_numeric_shape_missing_and_broadcast \
    tests/test_collect_features.py::test_hidden_per_pos_numeric_missing_verifier_numeric_raises

# Step 2: training (no re-collection — stage1_pp.pt already has everything).
CFG=configs/predictors/acc_frz_pp_vpn.yaml
echo ""; echo "=== train: $CFG ==="
$CONTAINER python3 -m accpre.train.cli --config "$CFG"

# Step 3: online decode.
OUT_DIR="outputs/phase7_online_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
CKPT="checkpoints/acc_frz_pp_vpn"
if [ -f "$CKPT/model.pt" ]; then
    PRED_OUT="$OUT_DIR/$(basename $CKPT)"
    mkdir -p "$PRED_OUT"
    echo ""; echo "=== online: $CKPT (V-prefix) ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$CKPT" \
        --strict_records data_collected/stage1_pp.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
else
    echo "SKIP online: $CKPT/model.pt missing"
fi

# Step 4: comparison pp_base vs pp_vpn.
echo ""; echo "=== compare: pp_base vs pp_vpn ==="
PHASE7_OUT="$OUT_DIR" $CONTAINER python3 scripts/compare_phase7.py \
    | tee "$OUT_DIR/compare_phase7.txt"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""; echo "=== Phase 7 V1 done ==="
date
