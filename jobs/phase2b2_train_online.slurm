#!/bin/bash
# =============================================================================
# Stage 2B.2 joint multitask acceptance — train + online decode.
#
# Trains acc_jnt_mt_hid (V-free) + acc_jnt_mt_hidnum (V-prefix).
# Each run uses the joint-multitask trainer: shared MDLM backbone,
# separate acceptance head (used online) and length head (aux only).
#
# Online decode auto-re-collects strict records with the fine-tuned
# drafter (measurement-gap fix from Stage 2B.1 refix).
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2b2_train
#SBATCH --output=logs/p2b2_train_%j.out
#SBATCH --error=logs/p2b2_train_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail

echo "=== Stage 2B.2 joint multitask train + online ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

# Anti-drift gate still in effect (same code path).
echo ""
echo "=== anti-drift gate ==="
$CONTAINER python3 -m pytest -m slow -v tests/test_train_infer_consistency.py

CONFIGS=(
    "configs/predictors/acc_jnt_mt_hid.yaml"
    "configs/predictors/acc_jnt_mt_hidnum.yaml"
)
for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== joint_multitask train: $cfg ==="
    $CONTAINER python3 -m accpre.train.cli --config "$cfg"
done

STRICT_RECORDS="${STRICT_RECORDS:-data_collected/stage1.pt}"
OUT_DIR="outputs/phase2b2_online_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

for ckpt in checkpoints/acc_jnt_mt_hid checkpoints/acc_jnt_mt_hidnum; do
    [ -f "$ckpt/drafter.pt" ] || { echo "SKIP $ckpt"; continue; }
    PRED_OUT="$OUT_DIR/$(basename $ckpt)"
    mkdir -p "$PRED_OUT"
    echo ""
    echo "=== online decode (joint multitask): $ckpt ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$ckpt" \
        --strict_records "$STRICT_RECORDS" \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
done

echo ""
echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""
echo "=== done ==="
date
