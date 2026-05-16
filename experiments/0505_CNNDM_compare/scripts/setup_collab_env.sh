#!/usr/bin/env bash
# One-time install of Python deps needed by the CNNDM pipeline.
#
# Why: the shared apptainer SIF `pytorch-2.7.0.sif` provides GPU torch but
# NOT `transformers` or `datasets`. Each user must install these once into
# their own `~/.local` (PEP 370 user-site). After this step, every later
# `apptainer exec --nv $SIF python3 ...` invocation will see them.
#
# Run this ONCE on a login node after cloning the repo. It does not
# submit any Slurm job and does not need a Slurm account.
#
#     bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh
#
# Idempotent: re-running is safe; pip will skip already-installed pins.

set -euo pipefail

# Resolve repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ ! -d "$REPO_ROOT/accpre" ]; then
    echo "FATAL: cannot find repo root from $SCRIPT_DIR" 1>&2
    echo "  (expected accpre/ at $REPO_ROOT/accpre)" 1>&2
    exit 1
fi
cd "$REPO_ROOT"

# Resolve apptainer.
if ! command -v apptainer >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load apptainer 2>/dev/null || true
    fi
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "FATAL: apptainer not on PATH and 'module load apptainer' did not help." 1>&2
    exit 1
fi

CONTAINERDIR=/share/resources/containers/apptainer
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"
if [ ! -r "$SIF" ]; then
    echo "FATAL: apptainer SIF not readable: $SIF" 1>&2
    exit 1
fi

echo "=== CNNDM env setup ==="
echo "    repo root: $REPO_ROOT"
echo "    SIF:       $SIF"
echo "    user-site: $HOME/.local"
echo

# Pinned versions match the original author's working environment.
# Top-level packages only; pip resolves their transitive deps and only
# installs what's missing (or already up-to-date) in the user-site.
PACKAGES=(
    "transformers==4.57.6"
    "datasets==4.8.4"
)

echo "=== installing into ~/.local (PEP 370 user-site) ==="
apptainer exec --nv "$SIF" python3 -m pip install --user --no-warn-script-location "${PACKAGES[@]}"

echo
echo "=== verify ==="
apptainer exec --nv "$SIF" python3 - <<'PY'
import torch, yaml
import transformers
import datasets
print(f"  torch         {torch.__version__}")
print(f"  transformers  {transformers.__version__}")
print(f"  datasets      {datasets.__version__}")
print(f"  pyyaml        {yaml.__version__}")
print("OK")
PY

echo
echo "Setup OK. Next step (still on the login node, no jobs submitted):"
echo
echo "  bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --preflight-only"
