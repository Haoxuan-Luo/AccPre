#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p21
#SBATCH --output=logs/p21_%j.out
#SBATCH --error=logs/p21_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00

set -euo pipefail

echo "=== Phase 21 verifier-success ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

OUT_DIR="outputs/phase21_vsucc_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

$CONTAINER python3 scripts/phase21_verifier_success.py \
    --out_dir "$OUT_DIR"

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""; echo "=== done ==="
date
