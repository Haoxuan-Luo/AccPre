#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p20o
#SBATCH --output=logs/p20o_%j.out
#SBATCH --error=logs/p20o_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Phase 20 oracle-exec rollout ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

OUT_DIR="outputs/phase20_oracle_exec_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

$CONTAINER python3 scripts/phase20_oracle_exec.py \
    --out_dir "$OUT_DIR" \
    --n_prompts 10 --max_new_tokens 64 --taus 0.5,0.7,0.9

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""; echo "=== done ==="
date
