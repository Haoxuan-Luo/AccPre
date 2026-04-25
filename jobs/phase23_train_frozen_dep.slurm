#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p23_frzdep
#SBATCH --output=logs/p23_frzdep_%j.out
#SBATCH --error=logs/p23_frzdep_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 23 — train frozen dependence predictor (dep_frz_1a) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs checkpoints

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

$CONTAINER python3 -m accpre.train.cli \
    --config configs/predictors/dep_frz_1a.yaml

date
echo "=== done ==="
