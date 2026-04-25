#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p23_jntdep
#SBATCH --output=logs/p23_jntdep_%j.out
#SBATCH --error=logs/p23_jntdep_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail

echo "=== Phase 23 — train joint live-dep predictor (dep_jnt_live) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs checkpoints

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

$CONTAINER python3 -m accpre.train.cli \
    --config configs/predictors/dep_jnt_live.yaml

date
echo "=== done ==="
