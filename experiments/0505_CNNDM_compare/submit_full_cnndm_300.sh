#!/usr/bin/env bash
# Submit the full 0505_CNNDM_compare pipeline (300-prompt dataset, T=1, gamma=15).
#
# USAGE
# -----
#
#     cd <path-to-AccPre>
#     SLURM_ACCOUNT=<your-account> bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh
#
# OR equivalently with a positional argument:
#
#     bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh <your-account>
#
# The slurm files themselves do NOT carry an `#SBATCH -A` directive; the
# account is supplied here via `-A "$SLURM_ACCOUNT"` so the collaborator
# never needs to edit the 8 slurm files for that purpose.
#
# Dependency graph
# ----------------
#
#     collect ──┬─→ build_targets ──→ frozen ──┐
#               │                              ├─→ pick_best ──→ eval ──┐
#               ├──────────────────→ joint ────┘                         ├─→ aggregate
#               └─→ baselines ───────────────────────────────────────────┘
#
# No file under experiments/0505_OWT_compare/ is touched. The script does
# not depend on the submitter's username or home directory.

set -euo pipefail

# -----------------------------------------------------------------------------
# Resolve Slurm account (required).
# -----------------------------------------------------------------------------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-${1:-}}"
if [ -z "$SLURM_ACCOUNT" ]; then
    cat <<EOF 1>&2
FATAL: no Slurm account provided.
Set the SLURM_ACCOUNT environment variable, e.g.:
    SLURM_ACCOUNT=<your-account> bash $0
or pass it as the first positional argument:
    bash $0 <your-account>
EOF
    exit 1
fi

# -----------------------------------------------------------------------------
# Resolve repo root from this script's location, then verify layout.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -d "$REPO_ROOT/accpre" ] || [ ! -d "$REPO_ROOT/data_collected" ]; then
    echo "FATAL: could not locate repository root from script path." 1>&2
    echo "  SCRIPT_DIR=$SCRIPT_DIR" 1>&2
    echo "  REPO_ROOT (computed)=$REPO_ROOT" 1>&2
    echo "  Run this script from inside an AccPre checkout that has both" 1>&2
    echo "  accpre/ and data_collected/ at its top level." 1>&2
    exit 1
fi
cd "$REPO_ROOT"

JOBS=experiments/0505_CNNDM_compare/jobs

# -----------------------------------------------------------------------------
# Existence check on the 8 slurm files.
# -----------------------------------------------------------------------------
for f in "$JOBS/job_collect_cnndm_300.slurm" \
         "$JOBS/job_build_targets_cnndm_300.slurm" \
         "$JOBS/job_baselines_cnndm_300.slurm" \
         "$JOBS/job_train_frozen_cnndm_300.slurm" \
         "$JOBS/job_train_joint_cnndm_300.slurm" \
         "$JOBS/job_pick_best_cnndm_300.slurm" \
         "$JOBS/job_eval_predictors_cnndm_300.slurm" \
         "$JOBS/job_aggregate_cnndm_300.slurm"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: missing slurm file: $f" 1>&2
        exit 1
    fi
done

# Wrapper that always passes the resolved account.
_sb() { sbatch -A "$SLURM_ACCOUNT" "$@"; }

echo "=== Submitting CNN/DM 300-prompt full pipeline ==="
echo "    repo root:     $REPO_ROOT"
echo "    SLURM account: $SLURM_ACCOUNT"
echo

# Stage 1: collect (no deps)
JC=$(_sb --parsable "$JOBS/job_collect_cnndm_300.slurm")

# Stages 2 + 4: build_targets and baselines (parallel; both depend only on collect)
JT=$(_sb --parsable --dependency=afterok:$JC "$JOBS/job_build_targets_cnndm_300.slurm")
JB=$(_sb --parsable --dependency=afterok:$JC "$JOBS/job_baselines_cnndm_300.slurm")

# Stage 3a: train_frozen (needs build_targets — relmax/dep target files)
JF=$(_sb --parsable --dependency=afterok:$JT "$JOBS/job_train_frozen_cnndm_300.slurm")

# Stage 3b: train_joint (only needs records; uses live targets, not target files)
JJ=$(_sb --parsable --dependency=afterok:$JC "$JOBS/job_train_joint_cnndm_300.slurm")

# Stage 5: pick_best (needs both training jobs)
JP=$(_sb --parsable --dependency=afterok:$JF:$JJ "$JOBS/job_pick_best_cnndm_300.slurm")

# Stage 6: eval (needs pick_best)
JE=$(_sb --parsable --dependency=afterok:$JP "$JOBS/job_eval_predictors_cnndm_300.slurm")

# Stage 7: aggregate (needs eval + baselines)
JA=$(_sb --parsable --dependency=afterok:$JE:$JB "$JOBS/job_aggregate_cnndm_300.slurm")

cat <<EOF

=== Submitted CNN/DM 300-prompt full pipeline ===
  collect        JC=$JC
  build_targets  JT=$JT  (afterok:$JC)
  baselines      JB=$JB  (afterok:$JC)
  train_frozen   JF=$JF  (afterok:$JT)
  train_joint    JJ=$JJ  (afterok:$JC)
  pick_best      JP=$JP  (afterok:$JF:$JJ)
  eval           JE=$JE  (afterok:$JP)
  aggregate      JA=$JA  (afterok:$JE:$JB)

Watch progress:
  squeue -j $JC,$JT,$JB,$JF,$JJ,$JP,$JE,$JA -o "%.12i %.20j %.12T %.10M %.20R"

Cancel everything:
  scancel $JC $JT $JB $JF $JJ $JP $JE $JA
EOF
