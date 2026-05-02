#!/bin/bash
#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_hw
#SBATCH --output=homework/logs/run_%j.out
#SBATCH --error=homework/logs/run_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00

# Homework experiment — 4-method throughput-quality comparison on OWT.
# Runs strict (1) + lossy_l (5) + threshold_lossy (9) + confidence_lossy (9)
# = 24 lanes on the same 20 OpenWebText test prompts. Outputs land under
# homework/results/ (raw JSONs + tables) and homework/figures/ (plots).
#
# Submit (from the repo root, NOT inside homework/):
#   sbatch homework/scripts/run_homework.slurm
#
# Re-run with cached lanes (e.g. after only adding new tau values):
#   sbatch --export=ALL,SKIP_EXISTING=1 homework/scripts/run_homework.slurm

set -euo pipefail
export PYTHONUNBUFFERED=1

: "${OUT_DIR:=homework/results}"
: "${GAMMA:=15}"
: "${T:=2}"
: "${N_PROMPTS:=20}"
: "${MAX_NEW_TOKENS:=1024}"
: "${LOSSY_LS:=1.0 0.7 0.5 0.3 0.1}"
: "${THRESH_TAUS:=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
: "${CONF_TAUS:=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
: "${SKIP_EXISTING:=0}"

date
module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"
mkdir -p homework/logs "$OUT_DIR"

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

echo "=== homework experiment ==="
echo "OUT_DIR=$OUT_DIR  GAMMA=$GAMMA  T=$T  N_PROMPTS=$N_PROMPTS  MAX_NEW=$MAX_NEW_TOKENS"
echo "LOSSY_LS=$LOSSY_LS"
echo "THRESH_TAUS=$THRESH_TAUS"
echo "CONF_TAUS=$CONF_TAUS"

EXTRA_FLAGS=""
if [[ "$SKIP_EXISTING" == "1" ]]; then
    EXTRA_FLAGS="--skip_existing"
fi

$CONTAINER python3 -u homework/run_homework.py \
    --out_dir "$OUT_DIR" \
    --gamma "$GAMMA" --T "$T" \
    --n_prompts "$N_PROMPTS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --lossy_ls $LOSSY_LS \
    --thresh_taus $THRESH_TAUS \
    --conf_taus $CONF_TAUS \
    $EXTRA_FLAGS 2>&1 | tee "$OUT_DIR/run.log"

echo "=== plotting ==="
$CONTAINER python3 -u homework/_make_plot.py
$CONTAINER python3 -u homework/_make_plot_final.py

date
echo "=== homework done ==="
echo "unified.md → $OUT_DIR/unified.md"
cat "$OUT_DIR/unified.md" 2>&1 || true
