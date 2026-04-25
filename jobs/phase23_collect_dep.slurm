#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p23_coll
#SBATCH --output=logs/p23_collect_%j.out
#SBATCH --error=logs/p23_collect_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

echo "=== Phase 23 — collect dep targets on pretrained drafter ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs data_collected

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

$CONTAINER python3 scripts/collect_dep_targets.py \
    --records data_collected/stage1_pp.pt \
    --out     data_collected/stage1_pp_dep.pt \
    --gamma 8 --sigma 1.0 \
    --splits train,val,test \
    --log_every 200

date
echo "=== done ==="
