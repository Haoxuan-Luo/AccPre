#!/bin/bash
# =============================================================================
# Phase 2A fix re-train — 5 predictors:
#   - 2 length predictors re-trained under the fixed LengthMLP arch (feature
#     LayerNorm before τ concat); overwrites the buggy checkpoints.
#   - 3 soft-BCE acceptance predictors (new checkpoint names with `_sbce`
#     suffix); the original MSE checkpoints are preserved intact for
#     before/after comparison.
#
# CPU only.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_fix_train
#SBATCH --output=logs/p2_fix_train_%j.out
#SBATCH --error=logs/p2_fix_train_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail

echo "=== Phase 2A fix re-train ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints

CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

if [ ! -f "data_collected/stage1.pt" ]; then
    echo "ERROR: data_collected/stage1.pt missing — run phase2_collect.slurm first."
    exit 1
fi

# Length re-trains overwrite the old buggy checkpoints; acceptance sbce
# runs write to new checkpoint dirs.
CONFIGS=(
    "configs/predictors/len_frz_num.yaml"
    "configs/predictors/len_frz_hidnum.yaml"
    "configs/predictors/acc_frz_num_sbce.yaml"
    "configs/predictors/acc_frz_hid_sbce.yaml"
    "configs/predictors/acc_frz_hidnum_sbce.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== training from $cfg ==="
    $CONTAINER python3 -m accpre.train.cli --config "$cfg"
done

echo ""
echo "--- checkpoints ---"
ls -1 checkpoints/

echo ""
echo "=== Phase 2A fix re-train OK ==="
date
