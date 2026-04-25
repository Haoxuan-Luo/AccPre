#!/bin/bash
# =============================================================================
# Phase 2A.1 + 2A.2 — train all frozen predictors (3 acceptance + 2 length).
#
# CPU-only job. No models loaded — the frozen predictors train on the
# pre-computed features stored in `data_collected/stage1.pt`. Each run
# saves `model.pt`, `preds_val.pt`, `preds_test.pt`, `train_history.json`,
# and a copy of the config into `checkpoints/<run_name>/`.
#
# Submit from repo root AFTER `phase2_collect.slurm` produces stage1.pt:
#   sbatch jobs/phase2_train_all.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_train
#SBATCH --output=logs/p2_train_%j.out
#SBATCH --error=logs/p2_train_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Phase 2A.1 + 2A.2 predictor training ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints

CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"

$CONTAINER python3 -m pip install --user --quiet pyyaml

if [ ! -f "data_collected/stage1.pt" ]; then
    echo "ERROR: data_collected/stage1.pt not found. Run phase2_collect.slurm first."
    exit 1
fi

# Stage 2A.1 — acceptance predictors (3 families).
# Stage 2A.2 — length predictors (2 families).
CONFIGS=(
    "configs/predictors/acc_frz_num.yaml"
    "configs/predictors/acc_frz_hid.yaml"
    "configs/predictors/acc_frz_hidnum.yaml"
    "configs/predictors/len_frz_num.yaml"
    "configs/predictors/len_frz_hidnum.yaml"
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
echo "=== Phase 2A.1/2A.2 training OK ==="
date
