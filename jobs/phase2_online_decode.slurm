#!/bin/bash
# =============================================================================
# Phase 2A.3 — online predictor decode for all five frozen predictors.
#
# Runs each predictor through the cheap online decode loop
# (`accpre.eval.online_decode`), with the canonical commit convention:
#   - NO verifier-on-draft pass.
#   - NO strict fallback / NO bonus token.
#   - `max(1, L_hat)` draft tokens per round (progress guarantee).
#
# Deployment mode per predictor family (DESIGN_PHASE2.md §C.3):
#   - acc_frz_hid       -> V-free   (zero verifier calls)
#   - acc_frz_num       -> V-prefix (one verifier prefix pass per round)
#   - acc_frz_hidnum    -> V-prefix
#   - len_frz_num       -> V-prefix
#   - len_frz_hidnum    -> V-prefix
#
# For each predictor, iterates τ ∈ {0.3, 0.5, 0.7, 0.9} and writes one
# `online_<kind>_tau_<tau>.json` per (predictor, τ).
#
# Submit AFTER `phase2_train_all.slurm` succeeds AND after a Phase 1
# strict run has produced records on the test split:
#   STRICT_RECORDS=outputs/phase1_strict_<jobid>/strict_records.pt \
#       sbatch jobs/phase2_online_decode.slurm
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p2_online
#SBATCH --output=logs/p2_online_%j.out
#SBATCH --error=logs/p2_online_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00

set -euo pipefail

echo "=== Phase 2A.3 online predictor decode ==="
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

STRICT_RECORDS="${STRICT_RECORDS:-data_collected/stage1.pt}"
if [ ! -f "$STRICT_RECORDS" ]; then
    echo "ERROR: strict records not found at $STRICT_RECORDS."
    echo "  Run phase2_collect.slurm first, or set STRICT_RECORDS=<path>."
    exit 1
fi
echo "Using strict records: $STRICT_RECORDS"
echo "  (Stage 2A.0 records have features populated and use global"
echo "   prompt indices matching the online run; CF@1 replay is matched-randomness-correct.)"

OUT_DIR="outputs/phase2_online_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

CKPTS=(
    "checkpoints/acc_frz_num"
    "checkpoints/acc_frz_hid"
    "checkpoints/acc_frz_hidnum"
    "checkpoints/len_frz_num"
    "checkpoints/len_frz_hidnum"
)
TAUS="${TAUS:-0.3,0.5,0.7,0.9}"

for ckpt in "${CKPTS[@]}"; do
    if [ ! -f "$ckpt/model.pt" ] || [ ! -f "$ckpt/config.yaml" ]; then
        echo "SKIP: missing $ckpt/model.pt or config.yaml"
        continue
    fi
    PRED_OUT="$OUT_DIR/$(basename $ckpt)"
    mkdir -p "$PRED_OUT"
    echo ""
    echo "=== online decode: $ckpt ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$ckpt" \
        --strict_records "$STRICT_RECORDS" \
        --split "${SPLIT:-test}" \
        --n_prompts "${N_PROMPTS:-10}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-64}" \
        --gamma "${GAMMA:-8}" \
        --T "${T:-2}" \
        --taus "$TAUS" \
        --out_dir "$PRED_OUT"
done

echo ""
echo "--- Pareto overlay (strict + oracle-Q2 + predictors) ---"
# PHASE1_SUMMARY_DIR is the directory holding summary.json produced by
# phase1_run_oracle_q2.slurm (strict + oracle-Q2 on test prompts).
# It is DISTINCT from STRICT_RECORDS (which points to stage1.pt — the
# feature-populated records used for CF@1 replay).
PHASE1_SUMMARY_DIR="${PHASE1_SUMMARY_DIR:-}"
if [ -z "$PHASE1_SUMMARY_DIR" ]; then
    PHASE1_SUMMARY_DIR=$(ls -dt outputs/phase1_full_*/ 2>/dev/null | head -n 1 || true)
    PHASE1_SUMMARY_DIR="${PHASE1_SUMMARY_DIR%/}"
fi
if [ -z "$PHASE1_SUMMARY_DIR" ] || [ ! -f "$PHASE1_SUMMARY_DIR/summary.json" ]; then
    echo "WARN: no Phase-1 summary.json found; skipping Pareto overlay."
else
    echo "Using Phase-1 summary: $PHASE1_SUMMARY_DIR/summary.json"
    PRED_DIRS=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | paste -sd ',')
    $CONTAINER python3 -m accpre.eval.pareto \
        --in_dir "$PHASE1_SUMMARY_DIR" \
        --predictor_dirs "$PRED_DIRS" \
        --out_fig "$OUT_DIR/pareto_phase2.png"
fi

echo ""
echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort

echo ""
echo "=== Phase 2A.3 online decode OK ==="
date
