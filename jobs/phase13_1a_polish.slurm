#!/bin/bash
# =============================================================================
# Phase 13 — 1A polish / ablation (acceptance, V-free only).
#
#   Variant A = current 1A (already trained as checkpoints/acc_1a; no retrain).
#   Variant B = 1A with token_emb_dim = 32 (smaller emb sanity check).
#   Variant C = 1A with separate-LN fusion (LN float feats + LN token emb
#               separately before concat; no leading LN inside the MLP body).
#
# Same target, same MSE loss, same architecture family, same τ grid
# {0.5, 0.7, 0.9}. No calibration, no threshold aux, no weighted MSE,
# no transformer sequence head, no V-prefix, no length.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p13
#SBATCH --output=logs/p13_%j.out
#SBATCH --error=logs/p13_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 13 (1A polish) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

OUT_DIR="outputs/phase13_1apolish_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# Variant A uses existing checkpoints/acc_1a + online outputs from phase12.

for CFG in configs/predictors/acc_1a_emb32.yaml configs/predictors/acc_1a_sepln.yaml; do
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
        --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
done

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/acc_1a_emb32/ checkpoints/acc_1a_sepln/

echo ""; echo "=== Phase 13 done ==="
date
