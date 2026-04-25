#!/bin/bash
# =============================================================================
# Phase 15 — first post-audit joint (drafter + 1A head) experiment.
#
# Chain:
#   0. Sanity: full-prefix replay must bit-match stored records under
#      the pretrained, unmodified drafter + matched RNG. Gate the
#      experiment. If this fails, the joint training signal cannot be
#      trusted and we abort before any training.
#   1. Train acc_jnt_1a (5 epochs, conservative backbone_lr).
#   2. Evaluate via recollection: re-run strict over all 20 test
#      prompts with the fine-tuned drafter, compute offline CF@1 at
#      τ ∈ {0.5, 0.7, 0.9}, and dump the Q2 conditional-variance
#      diagnostic (before vs after).
#   3. Online lane: tok/s at τ ∈ {0.5, 0.7, 0.9} on the canonical 10
#      test prompts.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p15
#SBATCH --output=logs/p15_%j.out
#SBATCH --error=logs/p15_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

echo "=== Phase 15 (acc_jnt_1a: joint 1A experiment) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml scikit-learn

OUT_DIR="outputs/phase15_jnt1a_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo ""; echo "=== 0. sanity: full-prefix replay =="
$CONTAINER python3 scripts/sanity_joint_prefix.py --n_records 16 --split test
SANITY_RC=$?
if [ $SANITY_RC -ne 0 ]; then
    echo "[phase15] ABORT: sanity test failed (rc=$SANITY_RC)"
    exit $SANITY_RC
fi

echo ""; echo "=== 1. train: configs/predictors/acc_jnt_1a.yaml ==="
$CONTAINER python3 -m accpre.train.cli --config configs/predictors/acc_jnt_1a.yaml

echo ""; echo "=== 2. recollect + offline CF@1 + Q2 variance diag ==="
$CONTAINER python3 scripts/eval_joint.py \
    --checkpoint_dir checkpoints/acc_jnt_1a \
    --out_dir "$OUT_DIR/offline_eval" \
    --gamma 8 --T 2 --max_new_tokens 128

echo ""; echo "=== 3. online decode: tok/s at τ ∈ {0.5, 0.7, 0.9} ==="
$CONTAINER python3 -m accpre.eval.online_decode \
    --config configs/protocol.yaml \
    --checkpoint_dir checkpoints/acc_jnt_1a \
    --strict_records data_collected/stage1_pp.pt \
    --split test --n_prompts 10 --max_new_tokens 64 \
    --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
    --out_dir "$OUT_DIR/online"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/acc_jnt_1a

echo ""; echo "=== Phase 15 done ==="
date
