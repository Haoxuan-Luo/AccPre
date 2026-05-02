#!/bin/bash
# Smoke test: 1 prompt × 64 new tokens × 1 lane per family (~5 minutes on GPU).
#
# Exits with the run_homework.py exit code; tables and JSONs land in
# homework/results_smoke/. Useful for verifying the code path end to end
# without committing to the full 6-7 hour run.
#
# Usage:
#   bash homework/scripts/run_smoke.sh                # default
#   bash homework/scripts/run_smoke.sh --gamma 8 --T 1 # custom

set -euo pipefail
HW_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$HW_DIR")"

python3 -u "$HW_DIR/run_homework.py" \
    --n_prompts 1 \
    --max_new_tokens 64 \
    --gamma 15 --T 2 \
    --lossy_ls 1.0 \
    --thresh_taus 0.5 \
    --conf_taus 0.5 \
    --out_dir "$HW_DIR/results_smoke" \
    "$@"
