#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=hw_diag
#SBATCH --output=homework/logs/diag_%j.out
#SBATCH --error=homework/logs/diag_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00

# Diagnostic: quantify how often strict's argmax-match rule and the
# ratio-based rules (lossy / threshold / confidence) diverge per position.
# Used to produce diagnostics/diagnose_rules.log + diagnostics/diagnosis.md.

set -euo pipefail
export PYTHONUNBUFFERED=1

date
module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"
mkdir -p homework/logs homework/diagnostics

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

$CONTAINER python3 -u homework/_diagnose_rules.py \
    2>&1 | tee homework/diagnostics/diagnose_rules.log

date
