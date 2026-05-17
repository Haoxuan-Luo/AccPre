#!/usr/bin/env bash
# One-time install of Python deps needed by the CNNDM pipeline.
#
# Why this is project-local, not --user:
# -------------------------------------
# The shared apptainer SIF `pytorch-2.7.0.sif` provides GPU torch + pyyaml
# but NOT transformers/datasets/dill/multiprocess. The earlier version of
# this script used `pip install --user`, which worked from login nodes but
# silently FAILED on batch nodes: the collaborator's `cnndm300_collect`
# job died in 6 seconds with `ModuleNotFoundError: No module named
# 'transformers'`. Root cause: ~/.local is not reliably visible inside an
# apptainer batch job (HOME can bind differently on compute nodes, and
# PYTHONNOUSERSITE may be set by the batch wrapper).
#
# The robust fix is to stage every required Python package under a
# PROJECT-LOCAL directory:
#     experiments/0505_CNNDM_compare/.pydeps/
# This directory is on the same filesystem as the rest of the repo, so it
# is always visible to apptainer (no special bind-mount needed), and every
# slurm job in the cnndm_300 chain explicitly puts it on PYTHONPATH plus
# sets PYTHONNOUSERSITE=1, so the batch environment matches the login
# environment exactly.
#
# Usage
# -----
# Run this ONCE on a login node after cloning the repo:
#
#     bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh
#
# Re-running is safe; pip skips already-present pins when --upgrade-strategy
# is only-if-needed. To force reinstall after deleting .pydeps/ manually,
# just re-run the script.

set -euo pipefail

# Resolve repo root from this script's location (script lives in
# experiments/0505_CNNDM_compare/scripts/).
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

CNNDM_PYDEPS="$REPO_ROOT/experiments/0505_CNNDM_compare/.pydeps"
mkdir -p "$CNNDM_PYDEPS"

echo "=== CNNDM env setup ==="
echo "    repo root:    $REPO_ROOT"
echo "    SIF:          $SIF"
echo "    project deps: $CNNDM_PYDEPS"
echo

# Pinned versions.
# - transformers/datasets are not in the SIF; we install them into .pydeps.
# - dill==0.3.8 and multiprocess<0.70.17 are pinned to versions compatible
#   with datasets==3.6.0.
# - numpy/pyarrow/pandas are pinned to ranges that MATCH the SIF's existing
#   versions, so pip skips them (container satisfies the pin) and the
#   container's torch-compatible builds win at import time. Without these
#   pins, transformers/datasets pull in numpy 2.4.5 / pyarrow 24 / pandas
#   3.x which would shadow the container versions and cause ABI mismatches
#   (the container's torch is compiled against numpy 1.x).
PACKAGES=(
    "transformers==4.57.6"
    "datasets==3.6.0"
    "dill==0.3.8"
    "multiprocess<0.70.17"
    "numpy<2"
    "pyarrow<20"
    "pandas<2.3"
)

echo "=== installing into project-local .pydeps (no --user) ==="
# --target: install everything into CNNDM_PYDEPS (not into the container's
#           site-packages, and not into ~/.local).
# --upgrade --upgrade-strategy only-if-needed: idempotent — on re-runs,
#           pip reuses packages that already satisfy the pin in .pydeps,
#           and respects packages already in the container's site-packages
#           (numpy 1.26.4, pyyaml, pyarrow, pandas, fsspec, filelock, tqdm,
#           aiohttp). This is critical: numpy MUST come from the container
#           because the container's torch was compiled against numpy 1.x;
#           pulling in numpy 2.x via .pydeps causes ABI mismatches.
# APPTAINERENV_PIP_CONSTRAINT="": the SIF sets PIP_CONSTRAINT=/etc/pip/constraint.txt
#           which pins dill==0.3.9. That conflicts with our pin dill==0.3.8
#           (the version compatible with datasets==3.6.0). Clearing the env
#           var inside the container disables the constraint file for this
#           pip invocation only — it does not modify the SIF.
# --no-warn-script-location: suppresses $PATH warnings (we import libraries,
#           not their CLI scripts).
APPTAINERENV_PIP_CONSTRAINT="" apptainer exec --nv "$SIF" python3 -m pip install \
    --target "$CNNDM_PYDEPS" \
    --upgrade --upgrade-strategy only-if-needed \
    --no-warn-script-location \
    "${PACKAGES[@]}"

# Defense in depth: even with the pins above, pip may pull a different
# version of a transitive dep into .pydeps that would shadow the
# container's. The container's torch is compiled against numpy 1.x, so a
# numpy 2.x in .pydeps would break every tensor.numpy() call. Similar
# concerns for pyarrow (datasets uses it) and pandas. For each package
# that is BOTH in .pydeps AND in the container at a different version,
# remove the .pydeps copy so PYTHONPATH falls through to the container.
for shadow in numpy pyarrow pandas; do
    if [ -d "$CNNDM_PYDEPS/$shadow" ]; then
        # Inspect the version that landed in .pydeps and compare to the
        # container's. If they differ on the major version, remove from
        # .pydeps so PYTHONPATH falls through to the container.
        SHADOW_VER=$(apptainer exec --nv "$SIF" python3 - <<PY
import importlib.util, sys, os
spec = importlib.util.spec_from_file_location(
    "$shadow", os.path.join("$CNNDM_PYDEPS", "$shadow", "__init__.py")
)
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
    print(getattr(m, "__version__", "?"))
except Exception:
    print("?")
PY
)
        CONTAINER_VER=$(apptainer exec --nv "$SIF" python3 -c "
import sys
sys.path = [p for p in sys.path if 'pydeps' not in p]
try:
    import $shadow as m
    print(getattr(m, '__version__', '?'))
except Exception:
    print('?')
")
        echo "[setup] $shadow: .pydeps=$SHADOW_VER  container=$CONTAINER_VER"
        if [ "$SHADOW_VER" != "$CONTAINER_VER" ] && [ "$CONTAINER_VER" != "?" ]; then
            echo "[setup] removing $CNNDM_PYDEPS/$shadow* to defer to container's $shadow $CONTAINER_VER"
            rm -rf "$CNNDM_PYDEPS/$shadow" "$CNNDM_PYDEPS/$shadow"-*.dist-info
            rm -rf "$CNNDM_PYDEPS/${shadow}.libs"  # numpy ships a .libs dir for vendored .so files
        fi
    fi
done

echo
echo "=== verify (hermetic — PYTHONNOUSERSITE=1, PYTHONPATH=.pydeps only) ==="
# Mirror the exact batch-job environment: user-site disabled, PYTHONPATH
# is repo_root + project .pydeps. If imports succeed here, they will
# succeed in every cnndm_300 slurm job.
APPTAINERENV_PYTHONNOUSERSITE=1 \
APPTAINERENV_PYTHONPATH="$REPO_ROOT:$CNNDM_PYDEPS" \
  apptainer exec --nv "$SIF" python3 - <<'PY'
import os, sys, site, importlib
print(f"  python              {sys.version.split()[0]}")
print(f"  PYTHONNOUSERSITE    {os.environ.get('PYTHONNOUSERSITE','<unset>')}")
print(f"  user_site enabled?  {site.ENABLE_USER_SITE}")
print(f"  PYTHONPATH          {os.environ.get('PYTHONPATH','<unset>')}")
print()
print("  package versions (with file paths to confirm they resolve to .pydeps):")
required_from_pydeps = ("transformers", "datasets", "dill", "multiprocess")
required_from_container = ("torch", "yaml")
for mod in required_from_container + required_from_pydeps:
    m = importlib.import_module(mod)
    where = getattr(m, "__file__", "<built-in>") or "<built-in>"
    ver = getattr(m, "__version__", "?")
    print(f"    {mod:<13} {ver:<12}  ({where})")
print()
# Strict check: the 4 .pydeps packages MUST resolve to a .pydeps path.
pydeps = os.environ.get("PYTHONPATH","").split(":")[-1]
bad = []
for mod in required_from_pydeps:
    m = importlib.import_module(mod)
    f = getattr(m, "__file__", "") or ""
    if not f.startswith(pydeps):
        bad.append(f"{mod} resolved to {f} (expected under {pydeps})")
if bad:
    print("FAIL — these packages did NOT resolve to .pydeps:")
    for b in bad:
        print(f"   {b}")
    sys.exit(1)
print("OK (transformers/datasets/dill/multiprocess all resolve to .pydeps)")
PY

echo
echo "Setup OK."
echo
echo "Next step (still on the login node, no jobs submitted):"
echo "    bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --preflight-only"
echo
echo "Then, optionally (recommended after the user-site -> batch issue), submit"
echo "a one-off batch-preflight slurm job to verify the batch environment:"
echo "    SLURM_ACCOUNT=<your-account> bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh --batch-preflight"
