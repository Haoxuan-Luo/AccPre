#!/bin/bash
#SBATCH --job-name=0505_cnndm_smoke_g15_T1
#SBATCH --output=experiments/0505_CNNDM_compare/logs/smoke_g15_T1_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/smoke_g15_T1_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

# CNN/DailyMail smoke on the FINAL T=1, gamma=15 records + targets.
#
# Preconditions (asserted at the top of the script):
#   - data_collected/stage1_pp_cnn_dm_g15_T1.pt        exists and non-empty
#   - data_collected/stage1_pp_cnn_dm_g15_T1_relmax.pt exists and non-empty
#   - data_collected/stage1_pp_cnn_dm_g15_T1_q2.pt     exists and non-empty
#   - data_collected/stage1_pp_cnn_dm_g15_T1_dep.pt    exists and non-empty
#
# Cells (run_smoke.py is the dispatcher):
#   - frozen × {alpha_q2, dep, relmax} × mlp_pos    (3 cells)
#   - joint  × alpha_q2 × {causal_tx, bidir_tx}     (2 cells)
# Plus 1 eval cell on frozen alpha_q2 mlp_pos (hard_threshold tau=0.5).
#
# Outputs land in isolated dirs:
#   experiments/0505_CNNDM_compare/runs/_smoke_g15_T1/
#   experiments/0505_CNNDM_compare/results/_smoke_g15_T1/

set -euo pipefail
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi -L

module load apptainer
export CONTAINERDIR=/share/resources/containers/apptainer
cd "$SLURM_SUBMIT_DIR"

ROOT=experiments/0505_CNNDM_compare
PYTHON="apptainer exec --nv $CONTAINERDIR/pytorch-2.7.0.sif python3 -u"

# Precondition checks.
for f in data_collected/stage1_pp_cnn_dm_g15_T1.pt \
         data_collected/stage1_pp_cnn_dm_g15_T1_relmax.pt \
         data_collected/stage1_pp_cnn_dm_g15_T1_q2.pt \
         data_collected/stage1_pp_cnn_dm_g15_T1_dep.pt; do
    if [ ! -s "$f" ]; then
        echo "[smoke-g15_T1] FATAL: precondition missing — $f"
        exit 1
    fi
    echo "[smoke-g15_T1] OK $f ($(stat -c %s $f) bytes)"
done

echo "[smoke-g15_T1] === sanity scripts ==="
$PYTHON $ROOT/scripts/sanity_no_relative_max.py
$PYTHON $ROOT/scripts/sanity_position_feature.py
$PYTHON $ROOT/scripts/sanity_causal_mask.py
$PYTHON $ROOT/scripts/sanity_no_fixed_targets_in_joint.py
$PYTHON $ROOT/scripts/sanity_vfree_inference.py
echo "[smoke-g15_T1] sanity scripts PASSED"

echo "[smoke-g15_T1] === smoke training cells ==="
# Frozen smoke: all 3 targets × mlp_pos (cheapest arch).
$PYTHON $ROOT/scripts/run_smoke.py --frozen_only --arch mlp_pos
# Joint smoke: alpha_q2 × {causal_tx, bidir_tx} (cheapest live target).
$PYTHON $ROOT/scripts/run_smoke.py --joint_only --target alpha_q2

echo "[smoke-g15_T1] === smoke eval cell ==="
mkdir -p $ROOT/results/_smoke_g15_T1/eval
$PYTHON $ROOT/scripts/eval_predictor.py \
    --run_dir $ROOT/runs/_smoke_g15_T1/frozen/alpha_q2/mlp_pos \
    --rule hard_threshold \
    --param 0.5 \
    --fallback auto \
    --eta 0.5 \
    --gamma 15 \
    --T 1 \
    --dataset cnn_dm \
    --n_prompts 4 \
    --max_new_tokens 64 \
    --out_path $ROOT/results/_smoke_g15_T1/eval/frozen_alpha_q2_mlp_pos__hard_threshold__tau_0p5.json

echo "[smoke-g15_T1] done"
date
