#!/bin/bash
#SBATCH --job-name=cnndm300_aggregate
#SBATCH --output=experiments/0505_CNNDM_compare/logs/aggregate_300_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/aggregate_300_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Stage 7 — aggregate baselines + predictor evals; emit CSVs and Pareto plots.
# CPU-only.

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
BASELINES_DIR="${BASELINES_DIR:-$RESULTS_ROOT/baselines}"
EVAL_ROOT="${EVAL_ROOT:-$RESULTS_ROOT/eval}"

echo "[aggregate-300] baselines=$BASELINES_DIR  eval_root=$EVAL_ROOT  out_root=$RESULTS_ROOT"
$PYTHON $ROOT/scripts/pareto_plot.py \
    --baselines_dir $BASELINES_DIR \
    --eval_root $EVAL_ROOT \
    --out_root $RESULTS_ROOT

echo "[aggregate-300] done"
date
