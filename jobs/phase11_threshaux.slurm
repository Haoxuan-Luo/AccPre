#!/bin/bash
# =============================================================================
# Phase 11 — acceptance improvement pass 2: weak threshold-aware auxiliary.
#
# Train acc_tx_v2_thresh (MSE + weak threshold BCE aux at τ ∈ {0.5, 0.7, 0.9}),
# fit post-hoc temperature on its val set, then online-decode at
# τ ∈ {0.5, 0.7, 0.9} for:
#
#   acc_tx_v2_thresh            — main regression + aux
#   acc_tx_v2_thresh_tcal       — + post-hoc T (copy + temperature.json)
#
# Baseline A2 online numbers already exist from Phase 9 job 11806423.
#
# No architecture change, no V-prefix, no length. τ=0.3 dropped.
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p11
#SBATCH --output=logs/p11_%j.out
#SBATCH --error=logs/p11_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=== Phase 11 (threshold-aware aux) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

OUT_DIR="outputs/phase11_threshaux_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# --- Step 1: train with threshold aux ------------------------------------------
echo ""; echo "=== train: acc_tx_v2_thresh ==="
$CONTAINER python3 -m accpre.train.cli --config configs/predictors/acc_tx_v2_thresh.yaml

# --- Step 2: calibrate post-hoc T on a copy ------------------------------------
echo ""; echo "=== calibrate: acc_tx_v2_thresh_tcal ==="
rm -rf checkpoints/acc_tx_v2_thresh_tcal
cp -r checkpoints/acc_tx_v2_thresh checkpoints/acc_tx_v2_thresh_tcal
$CONTAINER python3 -c "
import yaml
p = 'checkpoints/acc_tx_v2_thresh_tcal/config.yaml'
cfg = yaml.safe_load(open(p))
cfg['run_name'] = 'acc_tx_v2_thresh_tcal'
yaml.safe_dump(cfg, open(p, 'w'), sort_keys=False)
"
$CONTAINER python3 scripts/calibrate_temperature.py checkpoints/acc_tx_v2_thresh_tcal

# --- Step 3: online decode at τ ∈ {0.5, 0.7, 0.9} ------------------------------
for RUN in acc_tx_v2_thresh acc_tx_v2_thresh_tcal; do
    CKPT="checkpoints/${RUN}"
    PRED_OUT="$OUT_DIR/${RUN}"
    mkdir -p "$PRED_OUT"
    echo ""; echo "=== online: $CKPT ==="
    $CONTAINER python3 -m accpre.eval.online_decode \
        --config configs/protocol.yaml \
        --checkpoint_dir "$CKPT" \
        --strict_records data_collected/stage1_pp.pt \
        --split test --n_prompts 10 --max_new_tokens 64 \
        --gamma 8 --T 2 --taus 0.5,0.7,0.9 \
        --out_dir "$PRED_OUT"
done

echo ""; echo "--- artifacts ---"
find "$OUT_DIR" -type f | sort
ls -la checkpoints/acc_tx_v2_thresh/ checkpoints/acc_tx_v2_thresh_tcal/

echo ""; echo "=== Phase 11 done ==="
date
