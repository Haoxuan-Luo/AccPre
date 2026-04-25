#!/bin/bash
# =============================================================================
# Phase 9 Step-3 — acceptance family expansion (A2, A3, A4).
#
#   Train acc_tx_v{2,3,4} sequentially, then online-decode each at
#   tau in {0.3, 0.5, 0.7, 0.9}. No comparison here — the unified
#   table is produced later by scripts/compare_phase9.py.
#
# Features per version (for the audit trail):
#   v2 = drafter hidden + 5 local scalars         (γ, 773)  V-free
#   v3 = drafter hidden + 3 prefix-numeric bcast  (γ, 771)  V-prefix
#   v4 = drafter hidden + 5 local + verifier_hid  (γ, 2373) V-prefix
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p9_accrest
#SBATCH --output=logs/p9_accrest_%j.out
#SBATCH --error=logs/p9_accrest_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00

set -euo pipefail

echo "=== Phase 9 Step-3 (acceptance family: A2, A3, A4) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

OUT_DIR="outputs/phase9_acc_rest_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

CFGS=(
    "configs/predictors/acc_tx_v2.yaml"
    "configs/predictors/acc_tx_v3.yaml"
    "configs/predictors/acc_tx_v4.yaml"
)

for CFG in "${CFGS[@]}"; do
    RUN=$(basename "${CFG%.yaml}")
    echo ""; echo "=== train: $CFG ==="
    $CONTAINER python3 -m accpre.train.cli --config "$CFG"

    CKPT="checkpoints/${RUN}"
    PRED_OUT="$OUT_DIR/${RUN}"
    mkdir -p "$PRED_OUT"
    echo ""; echo "=== online: $CKPT ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$CKPT" \
        --strict_records data_collected/stage1_pp.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.3,0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
done

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""; echo "=== Phase 9 Step-3 done ==="
date
