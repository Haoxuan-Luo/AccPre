#!/usr/bin/env bash
# Submit the full 0505_CNNDM_compare pipeline (300-prompt dataset, T=1, gamma=15).
#
# USAGE
# -----
#
#     cd <path-to-AccPre>
#     SLURM_ACCOUNT=<your-account> bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh
#
# OR with a positional argument:
#
#     bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh <your-account>
#
# Preflight-only (runs every check below but submits no jobs; useful for
# verifying a fresh clone before burning queue time):
#
#     bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --preflight-only
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
# Parse args. --preflight-only is a flag; everything else is treated as the
# Slurm account if not already set via $SLURM_ACCOUNT.
# -----------------------------------------------------------------------------
PREFLIGHT_ONLY=0
POSITIONAL_ACCOUNT=""
for arg in "$@"; do
    case "$arg" in
        --preflight-only) PREFLIGHT_ONLY=1 ;;
        --*) echo "FATAL: unknown flag: $arg" 1>&2; exit 1 ;;
        *) POSITIONAL_ACCOUNT="$arg" ;;
    esac
done
SLURM_ACCOUNT="${SLURM_ACCOUNT:-$POSITIONAL_ACCOUNT}"

# Account is REQUIRED for actual submission; preflight-only mode tolerates its absence.
if [ "$PREFLIGHT_ONLY" -eq 0 ] && [ -z "$SLURM_ACCOUNT" ]; then
    cat <<EOF 1>&2
FATAL: no Slurm account provided.
Set the SLURM_ACCOUNT environment variable, e.g.:
    SLURM_ACCOUNT=<your-account> bash $0
or pass it as the first positional argument:
    bash $0 <your-account>

To run preflight only (no jobs submitted, account not required):
    bash $0 --preflight-only
EOF
    exit 1
fi

# -----------------------------------------------------------------------------
# Resolve repo root from this script's location, then verify layout.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -d "$REPO_ROOT/accpre" ]; then
    echo "FATAL: could not locate repository root from script path." 1>&2
    echo "  SCRIPT_DIR=$SCRIPT_DIR" 1>&2
    echo "  REPO_ROOT (computed)=$REPO_ROOT" 1>&2
    echo "  Run this script from inside an AccPre checkout that has" 1>&2
    echo "  accpre/ at its top level." 1>&2
    exit 1
fi
cd "$REPO_ROOT"
# data_collected/ is gitignored; create it if a fresh clone hasn't.
mkdir -p data_collected

# -----------------------------------------------------------------------------
# Identify the checkout (helps the collaborator report exactly what they ran).
# -----------------------------------------------------------------------------
echo "=== CNN/DM 300 submit ==="
echo "    repo root: $REPO_ROOT"
if command -v git >/dev/null && [ -d .git ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
    COMMIT=$(git log --oneline -1 2>/dev/null || echo "?")
    echo "    branch:    $BRANCH"
    echo "    HEAD:      $COMMIT"
fi
echo

# -----------------------------------------------------------------------------
# Locate apptainer (preflight runs INSIDE the SIF so we need it on PATH).
# -----------------------------------------------------------------------------
if ! command -v apptainer >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load apptainer 2>/dev/null || true
    fi
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "FATAL: apptainer not on PATH and 'module load apptainer' did not help." 1>&2
    echo "       Try logging in to a shell that can load the apptainer module," 1>&2
    echo "       or ask your cluster admin which module to load." 1>&2
    exit 1
fi

CONTAINERDIR=/share/resources/containers/apptainer
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"
if [ ! -r "$SIF" ]; then
    echo "FATAL: apptainer SIF not readable: $SIF" 1>&2
    exit 1
fi

# -----------------------------------------------------------------------------
# PREFLIGHT — runs inside the SIF; mirrors the runtime env the slurm jobs use.
# If anything below fails, we bail out before submitting any job.
# -----------------------------------------------------------------------------
echo "=== preflight (inside apptainer SIF) ==="
PREFLIGHT_PY=experiments/0505_CNNDM_compare/scripts/preflight_cnndm_300.py
if ! apptainer exec --nv "$SIF" python3 -u "$PREFLIGHT_PY"; then
    echo 1>&2
    echo "FATAL: preflight failed. No jobs submitted." 1>&2
    echo "Read the per-check details above and re-run after fixing." 1>&2
    exit 2
fi

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    echo
    echo "[--preflight-only] preflight passed; skipping job submission."
    exit 0
fi

# -----------------------------------------------------------------------------
# Existence check on the 8 slurm files (preflight already does this in Python,
# but keep a shell-level guard so the rest of the script can rely on them).
# -----------------------------------------------------------------------------
JOBS=experiments/0505_CNNDM_compare/jobs
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

echo
echo "=== submitting (SLURM account: $SLURM_ACCOUNT) ==="

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
