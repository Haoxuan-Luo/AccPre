#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p25_lh
#SBATCH --output=logs/p25_lh_%j.out
#SBATCH --error=logs/p25_lh_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00

set -euo pipefail

echo "=== Phase 25 — long-horizon greedy sweep (20 prompts × 1024 tok) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

OUT_DIR="outputs/phase25_lh_${SLURM_JOB_ID}"
$CONTAINER python3 scripts/phase25_longhorizon.py \
    --out_dir "$OUT_DIR" \
    --protocol_yaml configs/protocol_greedy.yaml \
    --n_prompts 20 --max_new_tokens 1024 \
    --gamma 8 --T 2

date
echo "=== done ==="
echo "OUT_DIR=$OUT_DIR"
