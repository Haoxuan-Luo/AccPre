#!/bin/bash
#SBATCH --job-name=0505_cnndm_smoke
#SBATCH --output=experiments/0505_CNNDM_compare/logs/smoke_%j.out
#SBATCH --error=experiments/0505_CNNDM_compare/logs/smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

# CNN/DailyMail smoke — workspace clone of experiments/0505_OWT_compare/.
#
# Runs:
#   - 5 sanity scripts (CPU-only; require torch).
#   - 4 smoke training cells:
#       frozen alpha_q2 mlp_pos          (fastest, validates frozen path)
#       frozen dep     mlp_pos           (validates dep target file loading)
#       joint  alpha_q2 causal_tx        (validates live-target + verifier)
#       joint  alpha_q2 bidir_tx         (validates bidirectional encoder)
#   - 1 eval cell (frozen alpha_q2 mlp_pos, hard_threshold tau=0.5).
#
# A40 (48 GB) for the joint cell: drafter (~440 MB) + GPT-2 XL verifier
# (~3-4 GB) + head + activations. MIG would be borderline; A40 is safe.
# Walltime ~10-25 min depending on cluster.
#
# This is the SMOKE job for the CNN/DailyMail workspace.
# It does NOT submit any full sweep; it is an end-to-end shape /
# schema verification, isolated from experiments/0505_OWT_compare/.

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

echo "[smoke] === sanity scripts ==="
$PYTHON $ROOT/scripts/sanity_no_relative_max.py
$PYTHON $ROOT/scripts/sanity_position_feature.py
$PYTHON $ROOT/scripts/sanity_causal_mask.py
$PYTHON $ROOT/scripts/sanity_no_fixed_targets_in_joint.py
$PYTHON $ROOT/scripts/sanity_vfree_inference.py
echo "[smoke] sanity scripts PASSED"

echo "[smoke] === smoke training cells ==="
# Frozen smoke cells (mlp_pos only — cheapest, validates pipeline shape).
$PYTHON $ROOT/scripts/run_smoke.py --frozen_only --arch mlp_pos
# Joint smoke cells (alpha_q2 only — cheapest live-target target).
$PYTHON $ROOT/scripts/run_smoke.py --joint_only --target alpha_q2

echo "[smoke] === smoke eval cell ==="
# Eval the freshly-trained frozen alpha_q2 mlp_pos model with hard_threshold tau=0.5.
# Smaller n_prompts and max_new_tokens than full (just exercises V-free decode + offline NLL).
mkdir -p $ROOT/results/_smoke/eval
$PYTHON $ROOT/scripts/eval_predictor.py \
    --run_dir $ROOT/runs/_smoke/frozen/alpha_q2/mlp_pos \
    --rule hard_threshold \
    --param 0.5 \
    --fallback auto \
    --eta 0.5 \
    --gamma 15 \
    --T 1 \
    --dataset cnn_dm \
    --n_prompts 4 \
    --max_new_tokens 64 \
    --out_path $ROOT/results/_smoke/eval/frozen_alpha_q2_mlp_pos__hard_threshold__tau_0p5.json

echo "[smoke] done"
date
