#!/bin/bash
# =============================================================================
# Phase 17 — longer live-Q2 joint ("did we stop too early in Phase 16?").
#
# Identical chain to Phase 16:
#   0. Sanity A — live-Q2 must match stored Q2 at init (gate).
#   1. Train acc_jnt_1a_liveq2_long (max_epochs=10 is the only delta).
#   2. Recollect full test split with fine-tuned drafter; offline CF@1;
#      Q2 conditional-variance diagnostic.
#   3. Online decode at τ ∈ {0.5, 0.7, 0.9}.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p17
#SBATCH --output=logs/p17_%j.out
#SBATCH --error=logs/p17_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail

echo "=== Phase 17 (acc_jnt_1a_liveq2_long: 10-epoch live-Q2 joint) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml scikit-learn

OUT_DIR="outputs/phase17_liveq2_long_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo ""; echo "=== 0. sanity A: live Q2 == stored Q2 at init =="
$CONTAINER python3 scripts/sanity_live_q2.py --n_records 16 --split test
SANITY_RC=$?
if [ $SANITY_RC -ne 0 ]; then
    echo "[phase17] ABORT: live-Q2 sanity test failed (rc=$SANITY_RC)"
    exit $SANITY_RC
fi

echo ""; echo "=== 1. train: configs/predictors/acc_jnt_1a_liveq2_long.yaml ==="
$CONTAINER python3 -m accpre.train.cli --config configs/predictors/acc_jnt_1a_liveq2_long.yaml

echo ""; echo "=== 2. recollect + offline CF@1 + Q2 variance diag ==="
$CONTAINER python3 scripts/eval_joint.py \
    --checkpoint_dir checkpoints/acc_jnt_1a_liveq2_long \
    --out_dir "$OUT_DIR/offline_eval" \
    --gamma 8 --T 2 --max_new_tokens 128

echo ""; echo "=== 3. online decode: tok/s at τ ∈ {0.5, 0.7, 0.9} ==="
$CONTAINER python3 -m accpre.eval.online_decode \
    --config configs/protocol.yaml \
    --checkpoint_dir checkpoints/acc_jnt_1a_liveq2_long \
    --strict_records data_collected/stage1_pp.pt \
    --split test --n_prompts 10 --max_new_tokens 64 \
    --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
    --out_dir "$OUT_DIR/online"

echo ""; echo "=== 4. predictor-level table (including long) ==="
$CONTAINER python3 scripts/predictor_level_compare.py --include_long \
    --out_dir "$OUT_DIR/predictor_compare"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/acc_jnt_1a_liveq2_long

echo ""; echo "=== Phase 17 done ==="
date
