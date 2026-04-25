#!/bin/bash
# =============================================================================
# Phase 1 validation — SMALL end-to-end STRICT SpecDiff run.
#
# Validates: strict SpecDiff lane alone, with no oracle-Q2 sweep. Confirms
# record schema, wall-clock timer placement, and that strict records
# deserialize under the canonical loader. XL-audit is computed on strict
# output as an auxiliary sanity check (NOT on the Pareto axes).
#
# Params (override via env):
#   N_PROMPTS=10  MAX_NEW_TOKENS=64  GAMMA=8  T=2
# Outputs:
#   outputs/phase1_strict_<jobid>/strict_records.pt
#   outputs/phase1_strict_<jobid>/summary.json
#
# Submit from repo root:
#   sbatch jobs/phase1_run_strict.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p1_strict
#SBATCH --output=logs/p1_run_strict_%j.out
#SBATCH --error=logs/p1_run_strict_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Phase 1 strict small run ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

OUT_DIR="outputs/phase1_strict_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# --taus "" tells wallclock.py to skip the oracle-Q2 sweep entirely;
# only the strict lane runs, and summary.json carries strict results only.
$CONTAINER python3 -m accpre.eval.wallclock \
    --config configs/protocol.yaml \
    --n_prompts "${N_PROMPTS:-10}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
    --gamma "${GAMMA:-8}" \
    --T "${T:-2}" \
    --taus "" \
    --out_dir "$OUT_DIR"

echo ""
echo "--- artifacts ---"
ls -lh "$OUT_DIR"

echo ""
echo "=== Phase 1 strict run OK ==="
date
