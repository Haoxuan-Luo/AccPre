#!/bin/bash
#SBATCH --job-name=cnndm300_targets
#SBATCH --output=experiments/0505_CNNDM_compare/logs/targets_300_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/targets_300_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00

# Stage 2 — Build the 3 frozen target files for CNN/DM 300.
#  - relmax (verifier replay on each record's prefix+drafted tokens — GPU-heavy)
#  - q2     (no replay; repacks RoundRecord.min_pq_j — seconds)
#  - dep    (drafter logit deltas — drafter-side work, GPU-light)
#
# Verifier replay for relmax dominates. CNN/DM 300 = same scale as OWT_300
# (300 prompts, ~512 new_tokens, gamma=15). OWT_300 target-build took ~4 h on
# A100; A40 budget is 6 h.
#
# Each builder is skip-if-exists.

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi -L

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

# --- CNNDM project-local Python deps (see job_collect for full rationale) -
CNNDM_PYDEPS="$SLURM_SUBMIT_DIR/experiments/0505_CNNDM_compare/.pydeps"
if [ ! -d "$CNNDM_PYDEPS" ]; then
    echo "[FATAL] CNNDM_PYDEPS not found: $CNNDM_PYDEPS" 1>&2
    echo "        Run: bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh" 1>&2
    exit 2
fi
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="$SLURM_SUBMIT_DIR:$CNNDM_PYDEPS"
# -----------------------------------------------------------------------

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

STAGE1="${STAGE1:-data_collected/stage1_pp_cnn_dm_300_g15_T1.pt}"
TGT_RELMAX="${TGT_RELMAX:-data_collected/stage1_pp_cnn_dm_300_g15_T1_relmax.pt}"
TGT_Q2="${TGT_Q2:-data_collected/stage1_pp_cnn_dm_300_g15_T1_q2.pt}"
TGT_DEP="${TGT_DEP:-data_collected/stage1_pp_cnn_dm_300_g15_T1_dep.pt}"

if [ ! -s "$STAGE1" ]; then
    echo "[targets_300] FATAL: $STAGE1 missing — run job_collect_cnndm_300.slurm first."
    exit 1
fi

echo "[targets_300] inputs: $STAGE1 ($(stat -c %s "$STAGE1") bytes)"

if [ -s "$TGT_RELMAX" ]; then
    echo "[targets_300] SKIP relmax: $TGT_RELMAX exists"
else
    echo "[targets_300] === building relmax ==="
    $PYTHON $ROOT/scripts/build_target_relmax.py \
        --records $STAGE1 --protocol configs/protocol.yaml \
        --gamma 15 --dataset cnn_dm_300 --out $TGT_RELMAX
fi

if [ -s "$TGT_Q2" ]; then
    echo "[targets_300] SKIP q2: $TGT_Q2 exists"
else
    echo "[targets_300] === building q2 ==="
    $PYTHON $ROOT/scripts/build_target_q2.py \
        --records $STAGE1 --protocol configs/protocol.yaml \
        --gamma 15 --out $TGT_Q2
fi

if [ -s "$TGT_DEP" ]; then
    echo "[targets_300] SKIP dep: $TGT_DEP exists"
else
    echo "[targets_300] === building dep ==="
    $PYTHON $ROOT/scripts/build_target_dep.py \
        --records $STAGE1 --protocol configs/protocol.yaml \
        --gamma 15 --dataset cnn_dm_300 --out $TGT_DEP
fi

for f in "$TGT_RELMAX" "$TGT_Q2" "$TGT_DEP"; do
    if [ ! -s "$f" ]; then echo "[targets_300] FAILED: $f not produced"; exit 1; fi
done

echo "[targets_300] outputs:"
ls -la $TGT_RELMAX $TGT_Q2 $TGT_DEP
date
