#!/bin/bash
# =============================================================================
# Phase 1 validation — SMALL end-to-end ORACLE-Q2 THRESHOLD run.
#
# Validates the headline Phase 1 pipeline: strict lane (reference for CF@1)
# + oracle-Q2 threshold lane at a tau sweep + Pareto figure + auxiliary
# XL-audit table. Must run strict first because CF@1 is computed against
# stored strict records.
#
# Sanity expectations:
#   - strict CF@1 = 1.0 (strict is its own reference).
#   - oracle-Q2 CF@1 at tau = 0.0 typically low; rises toward strict L agreement
#     as tau -> 1.0 (where L_tau -> 0 and will only equal L_strict on rounds
#     where strict itself rejected at position 0).
#   - oracle-Q2 wall-clock tok/s should be <= strict tok/s (same verifier pass).
#
# Params (override via env):
#   N_PROMPTS=10  MAX_NEW_TOKENS=64  GAMMA=8  T=2  TAUS="0.3,0.5,0.7,0.9"
# Outputs:
#   outputs/phase1_full_<jobid>/strict_records.pt
#   outputs/phase1_full_<jobid>/oracle_q2_tau_<tau>_records.pt  (per tau)
#   outputs/phase1_full_<jobid>/summary.json
#   outputs/phase1_full_<jobid>/pareto.png
#
# Submit from repo root AFTER the strict job succeeds:
#   sbatch jobs/phase1_run_oracle_q2.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p1_oracle
#SBATCH --output=logs/p1_run_oracle_%j.out
#SBATCH --error=logs/p1_run_oracle_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00

set -euo pipefail

echo "=== Phase 1 oracle-Q2 small run (headline) ==="
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

OUT_DIR="outputs/phase1_full_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# Step 1: run both lanes (strict + oracle-Q2 sweep) through the canonical
# wallclock CLI. It always emits strict first, then the oracle-Q2 lane for
# each tau, then computes CF@1 against the strict records.
$CONTAINER python3 -m accpre.eval.wallclock \
    --config configs/protocol.yaml \
    --n_prompts "${N_PROMPTS:-10}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
    --gamma "${GAMMA:-8}" \
    --T "${T:-2}" \
    --taus "${TAUS:-0.3,0.5,0.7,0.9}" \
    --out_dir "$OUT_DIR"

# Step 2: Pareto (wall-clock tok/s vs CF@1) + auxiliary XL-audit panel.
$CONTAINER python3 -m accpre.eval.pareto \
    --in_dir "$OUT_DIR" \
    --out_fig "$OUT_DIR/pareto.png"

echo ""
echo "--- artifacts ---"
ls -lh "$OUT_DIR"

echo ""
echo "=== Phase 1 oracle-Q2 run OK ==="
date
