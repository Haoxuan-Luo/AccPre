#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p25_fl
#SBATCH --output=logs/p25_fl_%j.out
#SBATCH --error=logs/p25_fl_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# Parametric per-lane finalize. Submit with:
#   sbatch --export=ALL,LANE_METHOD=<method>,LANE_TAU=<tau> \
#          jobs/phase25_fin_lane.slurm
# where <method> in {strict, oracle_exec, frozen_1A, live_jnt_10ep,
#                    frozen_dep, live_jnt_dep}
# and   <tau>    is empty for strict, otherwise 0.5 / 0.7 / 0.9.

set -euo pipefail
export PYTHONUNBUFFERED=1

: "${LANE_METHOD:?LANE_METHOD env var must be set}"
LANE_TAU="${LANE_TAU:-}"

echo "=== Phase 25 finalize-lane ==="
echo "Job ID     : $SLURM_JOB_ID"
echo "LANE_METHOD: $LANE_METHOD"
echo "LANE_TAU   : ${LANE_TAU:-<none>}"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

DECODE_DIR="outputs/phase25_lh_11979860"
OUT_DIR="$DECODE_DIR/lanes"
mkdir -p "$OUT_DIR"

ARGS=(
    --decode_dir "$DECODE_DIR"
    --out_dir    "$OUT_DIR"
    --method     "$LANE_METHOD"
)
if [[ -n "$LANE_TAU" ]]; then
    ARGS+=(--tau "$LANE_TAU")
fi

$CONTAINER python3 -u scripts/phase25_finalize_lane.py "${ARGS[@]}"

date
echo "=== done ==="
