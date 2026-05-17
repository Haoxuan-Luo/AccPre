#!/bin/bash
#SBATCH --job-name=cnndm300_preflight
#SBATCH --output=experiments/0505_CNNDM_compare/logs/preflight_300_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/preflight_300_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

# Batch-environment preflight for the cnndm_300 chain.
#
# Why this exists:
# ---------------
# The login-node preflight (--preflight-only) was insufficient: the
# collaborator's preflight passed but the actual cnndm300_collect job died
# in 6 seconds with `ModuleNotFoundError: No module named 'transformers'`,
# because ~/.local was visible on the login node but NOT inside the
# apptainer batch context. This job mirrors the EXACT env block used by
# the other 8 cnndm_300 slurm files (project-local .pydeps + explicit
# PYTHONPATH + PYTHONNOUSERSITE=1) and runs the same preflight inside it.
# If this job exits 0, every other cnndm_300 slurm file will see the
# same env.
#
# Cost: CPU-only standard partition, 5 minutes max. Submitted only when
# the user explicitly requests it via:
#     SLURM_ACCOUNT=<your-account> bash submit_full_cnndm_300.sh --batch-preflight

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

# --- CNNDM project-local Python deps (IDENTICAL to every other cnndm_300 job) -
CNNDM_PYDEPS="$SLURM_SUBMIT_DIR/experiments/0505_CNNDM_compare/.pydeps"
if [ ! -d "$CNNDM_PYDEPS" ]; then
    echo "[FATAL] CNNDM_PYDEPS not found: $CNNDM_PYDEPS" 1>&2
    echo "        Run once on a login node:" 1>&2
    echo "          bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh" 1>&2
    exit 2
fi
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="$SLURM_SUBMIT_DIR:$CNNDM_PYDEPS"
# -----------------------------------------------------------------------

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

echo "[batch-preflight] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "[batch-preflight] CNNDM_PYDEPS=$CNNDM_PYDEPS"
echo "[batch-preflight] APPTAINERENV_PYTHONPATH=$APPTAINERENV_PYTHONPATH"
echo "[batch-preflight] APPTAINERENV_PYTHONNOUSERSITE=$APPTAINERENV_PYTHONNOUSERSITE"
echo

echo "[batch-preflight] === running preflight inside batch apptainer ==="
$PYTHON $ROOT/scripts/preflight_cnndm_300.py
RC=$?
echo
if [ $RC -eq 0 ]; then
    echo "[batch-preflight] PASS: full preflight succeeded inside the batch environment."
    echo "[batch-preflight] Safe to submit the full pipeline."
else
    echo "[batch-preflight] FAIL: preflight failed inside the batch environment (rc=$RC)."
    echo "[batch-preflight] DO NOT submit the full pipeline until this is fixed."
    exit $RC
fi
date
