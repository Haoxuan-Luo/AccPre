#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p23_sanity
#SBATCH --output=logs/p23_sanity_%j.out
#SBATCH --error=logs/p23_sanity_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail

echo "=== Phase 23 sanity — live-joint tokemb fix ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

# Invariant A (unmodified drafter: replayed tokens == stored tokens)
# Invariant B (perturbed drafter: replayed tokens drift + head q_hat changes
#              when tokemb input switches stored↔replayed)
$CONTAINER python3 scripts/sanity_joint_tokemb_fix.py \
    --split test --n_records 16 \
    --perturbed_drafter_ckpt checkpoints/acc_jnt_1a_liveq2_long/drafter.pt \
    --head_ckpt_dir          checkpoints/acc_jnt_1a_liveq2_long

date
echo "=== done ==="
