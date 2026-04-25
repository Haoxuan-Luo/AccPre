#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p25_fin
#SBATCH --output=logs/p25_fin_%j.out
#SBATCH --error=logs/p25_fin_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00

set -euo pipefail

echo "=== Phase 25 — finalize NLL + replay on existing decode JSONs ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

$CONTAINER python3 scripts/phase25_finalize.py \
    --decode_dir outputs/phase25_lh_11979860 \
    --protocol_yaml configs/protocol_greedy.yaml \
    --log_file logs/p25_lh_11979860.out

date
echo "=== done ==="
