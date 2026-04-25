#!/bin/bash
# Phase 1 end-to-end pipeline: strict SpecDiff + oracle-Q2 threshold baseline.
#
# Runs strict once to produce reference records + wall-clock, then runs
# oracle-Q2 online at a tau sweep. Computes CF@1 and XL-audit (auxiliary)
# from the stored records. Emits a Pareto figure and the auxiliary table.
#
# Intended to be run inside the pytorch-2.7.0 container:
#   apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif bash scripts/run_phase1.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/phase1}"
N_PROMPTS="${N_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
GAMMA="${GAMMA:-8}"
T="${T:-2}"
TAUS="${TAUS:-0.3,0.5,0.7,0.9}"

mkdir -p "$OUT_DIR"

python3 -m accpre.eval.wallclock \
    --config "$REPO_ROOT/configs/protocol.yaml" \
    --n_prompts "$N_PROMPTS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --gamma "$GAMMA" \
    --T "$T" \
    --taus "$TAUS" \
    --out_dir "$OUT_DIR"

python3 -m accpre.eval.pareto \
    --in_dir "$OUT_DIR" \
    --out_fig "$OUT_DIR/pareto.png"

echo "Phase 1 done. Outputs in: $OUT_DIR"
