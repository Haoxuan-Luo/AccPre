#!/bin/bash
# =============================================================================
# Stage 2B.1 measurement-gap fix — re-run online decode for the joint
# acceptance checkpoints AFTER the in-memory strict re-collection fix.
#
# No re-training. Uses existing checkpoints/acc_jnt_hid + acc_jnt_hidnum.
# For each, online_decode now re-collects strict records with the
# loaded fine-tuned drafter and uses them for CF@1 — so tok/s and
# CF@1 are measured on the same drafter state.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2b_refix
#SBATCH --output=logs/p2b_refix_%j.out
#SBATCH --error=logs/p2b_refix_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00

set -euo pipefail

echo "=== Stage 2B.1 joint online re-run (measurement-gap fix) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

OUT_DIR="outputs/phase2b_online_refix_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

for ckpt in checkpoints/acc_jnt_hid checkpoints/acc_jnt_hidnum; do
    [ -f "$ckpt/drafter.pt" ] || { echo "SKIP $ckpt (no drafter.pt)"; continue; }
    PRED_OUT="$OUT_DIR/$(basename $ckpt)"
    mkdir -p "$PRED_OUT"
    echo ""
    echo "=== online decode (re-collect strict): $ckpt ==="
    # --strict_records is still required as a CLI arg, but for joint
    # mode the re-collected records take over (see online_decode.main).
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$ckpt" \
        --strict_records data_collected/stage1.pt \
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
