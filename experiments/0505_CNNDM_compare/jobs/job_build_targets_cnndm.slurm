#!/bin/bash
#SBATCH --job-name=0505_cnndm_targets
#SBATCH --output=experiments/0505_CNNDM_compare/logs/targets_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/targets_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00

# Build the 3 frozen target files from the CNN/DM T=1, gamma=15 stage-1 records.
#  - relmax (verifier replay on each record's prefix+drafted tokens — GPU-heavy)
#  - q2     (no replay; repacks RoundRecord.min_pq_j — seconds)
#  - dep    (drafter logit deltas — drafter-side work, GPU-light)
#
# Verifier-side work for relmax dominates. CNN/DM has 80 prompts (vs 300 for
# OWT), so total runtime ~1-1.5 h on A40. Walltime budget 3 h.
#
# Each builder is skip-if-exists.
# DO NOT RUN until the collection job has completed.

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi -L

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

STAGE1=data_collected/stage1_pp_cnn_dm_g15_T1.pt
TGT_RELMAX=data_collected/stage1_pp_cnn_dm_g15_T1_relmax.pt
TGT_Q2=data_collected/stage1_pp_cnn_dm_g15_T1_q2.pt
TGT_DEP=data_collected/stage1_pp_cnn_dm_g15_T1_dep.pt

# Precondition: stage-1 records must exist.
if [ ! -s "$STAGE1" ]; then
    echo "[targets] FATAL: $STAGE1 missing or empty — run job_collect_cnndm.slurm first."
    exit 1
fi

echo "[targets] === inputs ==="
echo "  records:     $STAGE1 ($(stat -c %s $STAGE1) bytes)"
echo "  out relmax:  $TGT_RELMAX"
echo "  out q2:      $TGT_Q2"
echo "  out dep:     $TGT_DEP"

# Each builder: skip if its output already exists.

if [ -s "$TGT_RELMAX" ]; then
    echo "[targets] SKIP relmax: $TGT_RELMAX already exists"
else
    echo "[targets] === building relmax ==="
    $PYTHON $ROOT/scripts/build_target_relmax.py \
        --records $STAGE1 --protocol configs/protocol.yaml \
        --gamma 15 --dataset cnn_dm --out $TGT_RELMAX
fi

if [ -s "$TGT_Q2" ]; then
    echo "[targets] SKIP q2: $TGT_Q2 already exists"
else
    echo "[targets] === building q2 ==="
    $PYTHON $ROOT/scripts/build_target_q2.py \
        --records $STAGE1 --protocol configs/protocol.yaml \
        --gamma 15 --out $TGT_Q2
fi

if [ -s "$TGT_DEP" ]; then
    echo "[targets] SKIP dep: $TGT_DEP already exists"
else
    echo "[targets] === building dep ==="
    $PYTHON $ROOT/scripts/build_target_dep.py \
        --records $STAGE1 --protocol configs/protocol.yaml \
        --gamma 15 --dataset cnn_dm --out $TGT_DEP
fi

# Verify all 3 exist.
for f in "$TGT_RELMAX" "$TGT_Q2" "$TGT_DEP"; do
    if [ ! -s "$f" ]; then
        echo "[targets] FAILED: $f was not produced"
        exit 1
    fi
done

echo "[targets] === outputs ==="
ls -la $TGT_RELMAX $TGT_Q2 $TGT_DEP

echo "[targets] done"
date
