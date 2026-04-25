#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p4A_cmp
#SBATCH --output=logs/p4A_cmp_%j.out
#SBATCH --error=logs/p4A_cmp_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
date
module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml
$CONTAINER python3 scripts/compare_stageA.py
date
