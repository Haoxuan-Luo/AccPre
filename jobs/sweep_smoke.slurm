#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_sweep_smoke
#SBATCH --output=logs/sweep_smoke_%j.out
#SBATCH --error=logs/sweep_smoke_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# Smoke test for the sweep framework.
# Trains a frozen_alpha predictor for 1 epoch and evaluates it on 2 prompts
# × 64 tokens. Should finish in well under an hour.
#
# Override any of these via --export=ALL,KEY=value on sbatch.

set -euo pipefail
export PYTHONUNBUFFERED=1

: "${FAMILY:=frozen_alpha}"
: "${DATASET:=owt}"
: "${LR:=1e-3}"
: "${EPOCHS:=1}"
: "${N_EVAL_PROMPTS:=2}"
: "${MAX_NEW_TOKENS:=128}"
: "${GAMMA:=15}"
: "${T:=2}"
: "${TAUS:=0.5 0.7 0.9}"
: "${RULE:=threshold}"
: "${OUT_DIR:=outputs/sweeps/smoke_${SLURM_JOB_ID}}"

echo "=== sweep smoke ==="
echo "FAMILY=$FAMILY  DATASET=$DATASET  LR=$LR  EPOCHS=$EPOCHS"
echo "N_EVAL_PROMPTS=$N_EVAL_PROMPTS  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "OUT_DIR=$OUT_DIR"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUT_DIR"

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml matplotlib

RUN_DIR="$OUT_DIR/runs/smoke"
$CONTAINER python3 -u scripts/sweep_train_eval.py \
    --family "$FAMILY" --dataset "$DATASET" \
    --run_dir "$RUN_DIR" \
    --lr "$LR" --epochs "$EPOCHS" --seed 0 \
    --n_eval_prompts "$N_EVAL_PROMPTS" --max_new_tokens "$MAX_NEW_TOKENS" \
    --gamma "$GAMMA" --T "$T" \
    --taus $TAUS --rule "$RULE"

date
echo "=== smoke done ==="
echo "summary: $RUN_DIR/summary.json"
cat "$RUN_DIR/summary.json" || true
