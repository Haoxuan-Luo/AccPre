#!/bin/bash
# =============================================================================
# Phase 2A fix online decode — only the 5 new/fixed predictors.
#
# Old MSE acceptance predictors (acc_frz_{num,hid,hidnum}) are unchanged;
# their existing online outputs in outputs/phase2_online_11791962/ remain
# valid and are the "old" half of the before/after comparison.
#
# Results land in outputs/phase2_online_fix_<jobid>/.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_fix_online
#SBATCH --output=logs/p2_fix_online_%j.out
#SBATCH --error=logs/p2_fix_online_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 2A fix online decode ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

STRICT_RECORDS="${STRICT_RECORDS:-data_collected/stage1.pt}"
[ -f "$STRICT_RECORDS" ] || { echo "ERROR: $STRICT_RECORDS missing"; exit 1; }

OUT_DIR="outputs/phase2_online_fix_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

CKPTS=(
    "checkpoints/len_frz_num"
    "checkpoints/len_frz_hidnum"
    "checkpoints/acc_frz_num_sbce"
    "checkpoints/acc_frz_hid_sbce"
    "checkpoints/acc_frz_hidnum_sbce"
)
TAUS="${TAUS:-0.3,0.5,0.7,0.9}"

for ckpt in "${CKPTS[@]}"; do
    if [ ! -f "$ckpt/model.pt" ] || [ ! -f "$ckpt/config.yaml" ]; then
        echo "SKIP: missing $ckpt/model.pt or config.yaml"
        continue
    fi
    PRED_OUT="$OUT_DIR/$(basename $ckpt)"
    mkdir -p "$PRED_OUT"
    echo ""
    echo "=== online decode: $ckpt ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$ckpt" \
        --strict_records "$STRICT_RECORDS" \
        --split "${SPLIT:-test}" \
        --n_prompts "${N_PROMPTS:-10}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
        --gamma "${GAMMA:-8}" \
        --T "${T:-2}" \
        --taus "$TAUS" \
        --out_dir "$PRED_OUT"
done

echo ""
echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""
echo "=== Phase 2A fix online decode OK ==="
date
