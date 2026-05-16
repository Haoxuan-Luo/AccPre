#!/bin/bash
#SBATCH --job-name=cnndm300_pickbest
#SBATCH --output=experiments/0505_CNNDM_compare/logs/pick_best_300_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/pick_best_300_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Stage 5 — reduce 240 training cells to 15 best per (regime, target, arch).
# CPU-only, fast (<30 min). Produces best_per_cell.json + a markdown report.

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
BEST_PER_CELL_PATH="${BEST_PER_CELL_PATH:-$RESULTS_ROOT/best_per_cell.json}"
BEST_REPORT_PATH="${BEST_REPORT_PATH:-$RESULTS_ROOT/best_per_cell_report.md}"

mkdir -p "$RESULTS_ROOT"

echo "[pick-best-300] runs_root=$RUNS_ROOT  out=$BEST_PER_CELL_PATH"
$PYTHON $ROOT/scripts/pick_best_per_cell.py \
    --runs_root $RUNS_ROOT \
    --out_json $BEST_PER_CELL_PATH \
    --out_report $BEST_REPORT_PATH

echo "[pick-best-300] done"
date
