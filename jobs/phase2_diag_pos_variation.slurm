#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_pos_diag
#SBATCH --output=logs/pos_diag_%j.out
#SBATCH --error=logs/pos_diag_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail
echo "=== pos-variation diagnostic ==="
date
module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 scripts/diag_pos_variation.py
echo "=== done ==="
date
