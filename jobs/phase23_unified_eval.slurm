#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p23_eval
#SBATCH --output=logs/p23_eval_%j.out
#SBATCH --error=logs/p23_eval_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail

echo "=== Phase 23 unified 20-prompt eval (with dep methods) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

OUT_DIR="outputs/phase23_unified_${SLURM_JOB_ID}"
$CONTAINER python3 scripts/phase23_unified_eval.py \
    --out_dir "$OUT_DIR" \
    --n_prompts 20 \
    --max_new_tokens 64 \
    --gamma 8 --T 2

date
echo "=== done ==="
echo "OUT_DIR=$OUT_DIR"
