#!/bin/bash
# =============================================================================
# Phase 2A length weighted-CE fix — train + online decode in one job.
#
# Step 1: train len_frz_num_wce and len_frz_hidnum_wce (CPU, ~1 min).
# Step 2: online decode for both predictors × τ sweep (GPU-mig, ~6 min).
#
# Runs on gpu-mig so both can share the allocation; training on CPU but
# container is the same.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_len_wce
#SBATCH --output=logs/p2_len_wce_%j.out
#SBATCH --error=logs/p2_len_wce_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00

set -euo pipefail

echo "=== Phase 2A length weighted-CE ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs checkpoints

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

# --- train ---
for cfg in configs/predictors/len_frz_num_wce.yaml configs/predictors/len_frz_hidnum_wce.yaml; do
    echo ""
    echo "=== train $cfg ==="
    $CONTAINER python3 -m accpre.train.cli --config "$cfg"
done

# --- online decode ---
STRICT_RECORDS="${STRICT_RECORDS:-data_collected/stage1.pt}"
OUT_DIR="outputs/phase2_online_len_wce_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

for ckpt in checkpoints/len_frz_num_wce checkpoints/len_frz_hidnum_wce; do
    [ -f "$ckpt/model.pt" ] || { echo "SKIP $ckpt"; continue; }
    PRED_OUT="$OUT_DIR/$(basename $ckpt)"
    mkdir -p "$PRED_OUT"
    echo ""
    echo "=== online decode: $ckpt ==="
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
