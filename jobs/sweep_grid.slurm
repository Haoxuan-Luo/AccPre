#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_sweep_grid
#SBATCH --output=logs/sweep_grid_%j.out
#SBATCH --error=logs/sweep_grid_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00

# Sequential grid sweep for ONE (family, dataset) pair.
# The sweep driver launches each grid point as a subprocess inside this
# same job, so you get one SLURM allocation per sweep.
#
# Recommended usage:
#   sbatch --export=ALL,FAMILY=frozen_alpha,DATASET=owt jobs/sweep_grid.slurm
#   sbatch --export=ALL,FAMILY=joint_alpha,DATASET=owt,TIMEOUT=36:00:00 jobs/sweep_grid.slurm
#
# The joint_* families are much slower (~1h per run at 10 epochs) so
# trim the grid file if running the full 12 points is impractical.

set -euo pipefail
export PYTHONUNBUFFERED=1

: "${FAMILY:=frozen_alpha}"
: "${DATASET:=owt}"
: "${GRID:=configs/sweep/default_grid.yaml}"
: "${SWEEP_DIR:=outputs/sweeps/${FAMILY}_${DATASET}_${SLURM_JOB_ID}}"
: "${SKIP_EXISTING:=0}"

echo "=== sweep grid ==="
echo "FAMILY=$FAMILY  DATASET=$DATASET  GRID=$GRID  SWEEP_DIR=$SWEEP_DIR"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$SWEEP_DIR"

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

EXTRA=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
    EXTRA+=(--skip_existing)
fi

$CONTAINER python3 -u scripts/sweep_grid.py \
    --family "$FAMILY" --dataset "$DATASET" \
    --sweep_dir "$SWEEP_DIR" \
    --grid "$GRID" \
    "${EXTRA[@]}"

$CONTAINER python3 -u scripts/sweep_summarize.py \
    --sweep_dir "$SWEEP_DIR"

date
echo "=== grid done ==="
echo "leaderboard: $SWEEP_DIR/leaderboard.md"
