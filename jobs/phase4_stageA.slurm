#!/bin/bash
# =============================================================================
# Phase 4 Stage A — stronger per-position head: wide + attn.
#
# Baseline (acc_frz_hid_pp) already trained. This job trains the two new
# variants and runs online decode on both, for direct comparison with
# baseline results in outputs/phase3_online_pp_11799951/acc_frz_hid_pp/.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p4_A
#SBATCH --output=logs/p4_A_%j.out
#SBATCH --error=logs/p4_A_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 4 Stage A ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

CONFIGS=(
    "configs/predictors/acc_frz_pp_wide.yaml"
    "configs/predictors/acc_frz_pp_attn.yaml"
)
for cfg in "${CONFIGS[@]}"; do
    echo ""; echo "=== train: $cfg ==="
    $CONTAINER python3 -m accpre.train.cli --config "$cfg"
done

OUT_DIR="outputs/phase4_A_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

CKPTS=(
    "checkpoints/acc_frz_pp_wide"
    "checkpoints/acc_frz_pp_attn"
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

echo ""; echo "=== Phase 4 Stage A done ==="; date
