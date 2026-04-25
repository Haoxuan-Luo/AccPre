#!/bin/bash
# =============================================================================
# Phase 10 — acceptance improvement pass (V-free mainline).
#
# Train acc_tx_v2_wmse (weighted MSE, λ=0.5), then do post-hoc temperature
# scaling on both (a) the existing acc_tx_v2 baseline and (b) the new
# acc_tx_v2_wmse, and run online decode at τ ∈ {0.5, 0.7, 0.9} on the four
# variants we care about:
#
#   acc_tx_v2                 — baseline (already trained; just online)
#   acc_tx_v2_tcal            — baseline + post-hoc T (copy + temperature.json)
#   acc_tx_v2_wmse            — weighted MSE only
#   acc_tx_v2_wmse_tcal       — weighted MSE + post-hoc T
#
# No architecture change. No V-prefix. τ=0.3 dropped (not meaningful).
# =============================================================================

#SBATCH -A genai-spring26
#SBATCH --job-name=accpre_p10
#SBATCH --output=logs/p10_%j.out
#SBATCH --error=logs/p10_%j.err
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00

set -euo pipefail

echo "=== Phase 10 (acceptance improvement) ==="
echo "Job ID : $SLURM_JOB_ID"
date

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints outputs

CONTAINER="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif"
$CONTAINER python3 -m pip install --user --quiet pyyaml

OUT_DIR="outputs/phase10_accimp_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# -----------------------------------------------------------------------------
# Step 1 — train acc_tx_v2_wmse (weighted MSE, λ=0.5)
# -----------------------------------------------------------------------------
echo ""; echo "=== train: acc_tx_v2_wmse ==="
$CONTAINER python3 -m accpre.train.cli --config configs/predictors/acc_tx_v2_wmse.yaml

# -----------------------------------------------------------------------------
# Step 2 — fit post-hoc T on val for baseline and weighted checkpoints
# -----------------------------------------------------------------------------
echo ""; echo "=== calibrate: acc_tx_v2 (baseline) ==="
# Calibrate a copy so the original baseline ckpt stays untouched.
rm -rf checkpoints/acc_tx_v2_tcal
cp -r checkpoints/acc_tx_v2 checkpoints/acc_tx_v2_tcal
# Rename inside cfg for clarity in artifacts.
$CONTAINER python3 -c "
import yaml
p = 'checkpoints/acc_tx_v2_tcal/config.yaml'
cfg = yaml.safe_load(open(p))
cfg['run_name'] = 'acc_tx_v2_tcal'
yaml.safe_dump(cfg, open(p, 'w'), sort_keys=False)
"
$CONTAINER python3 scripts/calibrate_temperature.py checkpoints/acc_tx_v2_tcal

echo ""; echo "=== calibrate: acc_tx_v2_wmse ==="
rm -rf checkpoints/acc_tx_v2_wmse_tcal
cp -r checkpoints/acc_tx_v2_wmse checkpoints/acc_tx_v2_wmse_tcal
$CONTAINER python3 -c "
import yaml
p = 'checkpoints/acc_tx_v2_wmse_tcal/config.yaml'
cfg = yaml.safe_load(open(p))
cfg['run_name'] = 'acc_tx_v2_wmse_tcal'
yaml.safe_dump(cfg, open(p, 'w'), sort_keys=False)
"
$CONTAINER python3 scripts/calibrate_temperature.py checkpoints/acc_tx_v2_wmse_tcal

# -----------------------------------------------------------------------------
# Step 3 — online decode at τ ∈ {0.5, 0.7, 0.9} for the three new variants.
# (Baseline acc_tx_v2 was already run in job 11806423.)
# -----------------------------------------------------------------------------
for RUN in acc_tx_v2_wmse acc_tx_v2_tcal acc_tx_v2_wmse_tcal; do
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
ls -la checkpoints/acc_tx_v2_wmse/ checkpoints/acc_tx_v2_tcal/ checkpoints/acc_tx_v2_wmse_tcal/

echo ""; echo "=== Phase 10 done ==="
date
