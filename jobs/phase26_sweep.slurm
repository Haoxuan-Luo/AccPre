#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p26_sw
#SBATCH --output=logs/p26_sw_%j.out
#SBATCH --error=logs/p26_sw_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=10:00:00

# Phase 26 — lossy baseline + confidence commit rule sweep.
# Submits 21 lanes (strict + 4 lossy + 4 predictors × 4 settings) in a
# single job; each lane writes its own online_*.json and the driver emits
# unified_table.{md,json} at the end.
#
# Env overrides (with defaults):
#   N_PROMPTS       20
#   MAX_NEW_TOKENS  1024
#   TAUS            "0.5 0.9"
#   TAU_CONFS       "0.5 0.9"
#   LOSSY_LS        "1.0 0.7 0.5 0.3"

set -euo pipefail
export PYTHONUNBUFFERED=1

: "${N_PROMPTS:=20}"
: "${MAX_NEW_TOKENS:=1024}"
: "${TAUS:=0.5 0.9}"
: "${TAU_CONFS:=0.5 0.9}"
: "${LOSSY_LS:=1.0 0.7 0.5 0.3}"
: "${OUT_DIR:=}"
: "${SKIP_EXISTING:=0}"

echo "=== Phase 26 sweep ==="
echo "Job ID         : $SLURM_JOB_ID"
echo "N_PROMPTS      : $N_PROMPTS"
echo "MAX_NEW_TOKENS : $MAX_NEW_TOKENS"
echo "TAUS           : $TAUS"
echo "TAU_CONFS      : $TAU_CONFS"
echo "LOSSY_LS       : $LOSSY_LS"
echo "OUT_DIR        : ${OUT_DIR:-<auto>}"
echo "SKIP_EXISTING  : $SKIP_EXISTING"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="outputs/phase26_sw_${SLURM_JOB_ID}"
fi
mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
    EXTRA_ARGS+=(--skip_existing)
fi

$CONTAINER python3 -u scripts/phase26_conf_sweep.py \
    --out_dir "$OUT_DIR" \
    --protocol_yaml configs/protocol_greedy.yaml \
    --n_prompts "$N_PROMPTS" --max_new_tokens "$MAX_NEW_TOKENS" \
    --gamma 8 --T 2 \
    --taus $TAUS \
    --tau_confs $TAU_CONFS \
    --lossy_ls $LOSSY_LS \
    "${EXTRA_ARGS[@]}"

$CONTAINER python3 -u scripts/phase26_plots.py \
    --table_json "$OUT_DIR/unified_table.json"

date
echo "=== done ==="
echo "OUT_DIR=$OUT_DIR"
