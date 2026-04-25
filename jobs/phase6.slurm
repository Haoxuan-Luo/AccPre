#!/bin/bash
# =============================================================================
# Phase 6 — richer per-position drafter feature: pp_ml (multi-layer hidden).
#
#  1. re-collect stage1_pp_ml.pt populating drafter_hidden_per_pos_ml
#     (γ, 2H) = concat(penultimate, final) transformer block outputs at
#     the γ draft positions. Same protocol + matched randomness, so the
#     final-layer slice equals what stage1_pp.pt stores.
#  2. run the Phase-6 anti-drift test (layers=(-2,-1) vs default (-1,))
#     before training, to prove the new path hasn't changed drafting.
#  3. train acc_frz_pp_ml.
#  4. online decode acc_frz_pp_ml at τ∈{0.3,0.5,0.7,0.9}.
#  5. run scripts/compare_phase6.py (pp_base vs pp_ml) with the gate.
#
# Baseline acc_frz_hid_pp is reused from Phase 3: checkpoint
# checkpoints/acc_frz_hid_pp/ and online output
# outputs/phase3_online_pp_11799951/acc_frz_hid_pp/.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p6
#SBATCH --output=logs/p6_%j.out
#SBATCH --error=logs/p6_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

echo "=== Phase 6 multi-layer per-position drafter feature ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs data_collected

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

# Step 1: re-collect stage1_pp_ml.pt with multi-layer feature. Skip if
# recently written (protects against accidental re-runs).
if [ -n "$(find data_collected/stage1_pp_ml.pt -mmin -360 2>/dev/null)" ]; then
    echo "data_collected/stage1_pp_ml.pt is recent — skipping collection"
else
    echo ""; echo "=== re-collecting stage1_pp_ml.pt with (penultimate, final) hidden ==="
    $CONTAINER python3 -m accpre.collect.cli \
        --config configs/protocol.yaml \
        --out data_collected/stage1_pp_ml.pt \
        --splits train,val,test \
        --gamma 8 --T 2 --max_new_tokens 128
fi

# Step 2: anti-drift test — hard gate before training on the new feature.
echo ""; echo "=== anti-drift test (tests/test_train_infer_consistency.py) ==="
$CONTAINER python3 -m pytest -x -q -m slow \
    tests/test_train_infer_consistency.py::test_multilayer_layers_does_not_change_tokens \
    tests/test_train_infer_consistency.py::test_multilayer_final_slice_equals_single_layer

# Step 3: training.
CFG=configs/predictors/acc_frz_pp_ml.yaml
echo ""; echo "=== train: $CFG ==="
$CONTAINER python3 -m accpre.train.cli --config "$CFG"

# Step 4: online decode.
OUT_DIR="outputs/phase6_online_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
CKPT="checkpoints/acc_frz_pp_ml"
if [ -f "$CKPT/model.pt" ]; then
    PRED_OUT="$OUT_DIR/$(basename $CKPT)"
    mkdir -p "$PRED_OUT"
    echo ""; echo "=== online: $CKPT ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$CKPT" \
        --strict_records data_collected/stage1_pp_ml.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
else
    echo "SKIP online: $CKPT/model.pt missing"
fi

# Step 5: run comparison pp_base vs pp_ml.
echo ""; echo "=== compare: pp_base vs pp_ml ==="
PHASE6_OUT="$OUT_DIR" $CONTAINER python3 scripts/compare_phase6.py \
    | tee "$OUT_DIR/compare_phase6.txt"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""; echo "=== Phase 6 done ==="
date
