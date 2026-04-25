#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_report_acc
#SBATCH --output=logs/report_acc_%j.out
#SBATCH --error=logs/report_acc_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
echo "=== acceptance-side unified report ==="
date
module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
CONTAINER="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml
$CONTAINER python3 scripts/report_acceptance_all.py
echo "=== done ==="
date
