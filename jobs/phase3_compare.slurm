#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p3_cmp
#SBATCH --output=logs/p3_cmp_%j.out
#SBATCH --error=logs/p3_cmp_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
echo "=== pp-vs-pooled compare ==="; date
module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml
$CONTAINER python3 scripts/compare_pp_vs_pooled.py
echo "=== done ==="; date
