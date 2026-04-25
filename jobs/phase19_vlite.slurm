#!/bin/bash
# =============================================================================
# Phase 19 — cheap verifier-feature acceptance predictor.
#
# Two variants:
#   Vlite-N: 1A body + 3 verifier-prefix numerics broadcast (hidden_per_pos_v3).
#   Vlite-H: 1A body + verifier-hidden (1600→64 learned projection broadcast)
#            (hidden_per_pos_v4).
# Both are V-prefix deploy mode: one verifier prefix forward per round.
#
# Small sweep: 2 LRs × 2 dropouts = 4 runs per variant = 8 runs total.
# max_epochs=10 with early_stop_patience=2.
#
# After training, run online decode on the best-val run per variant.
# Then do the full predictor-level comparison.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p19
#SBATCH --output=logs/p19_%j.out
#SBATCH --error=logs/p19_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail

echo "=== Phase 19 (Vlite-N / Vlite-H cheap verifier features) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml scikit-learn

OUT_DIR="outputs/phase19_vlite_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo ""; echo "=== 1. sweep: 2 variants × 2 lrs × 2 dropouts = 8 runs ==="
$CONTAINER python3 scripts/phase19_sweep.py \
    --sweep_dir "$OUT_DIR/sweep" \
    --variants vlite_n,vlite_h \
    --lrs 1e-3,5e-4 \
    --dropouts 0.1,0.2

echo ""; echo "=== 2. online decode — best of each variant at τ ∈ {0.5, 0.7, 0.9} ==="
for variant in vlite_n vlite_h; do
    BEST_CKPT=$($CONTAINER python3 -c "
import json, sys
s = json.load(open('$OUT_DIR/sweep/sweep_summary.json'))
best = s['best_by_variant'].get('$variant')
if best is None:
    print('NONE'); sys.exit(1)
print(best['checkpoint_dir'])
")
    if [ "$BEST_CKPT" = "NONE" ]; then
        echo "[phase19] no successful run for $variant"
        continue
    fi
    echo "[phase19] online $variant best=$BEST_CKPT"
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$BEST_CKPT" \
        --strict_records data_collected/stage1_pp.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
        --out_dir "$OUT_DIR/online_${variant}"
done

echo ""; echo "=== 3. predictor-level comparison (all checkpoints) ==="
$CONTAINER python3 scripts/predictor_level_compare.py --include_long \
    --out_dir "$OUT_DIR/predictor_compare" || true

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort | head -50
ls -la checkpoints | grep -E 'acc_vlite|acc_1a|acc_jnt' | head

echo ""; echo "=== Phase 19 done ==="
date
