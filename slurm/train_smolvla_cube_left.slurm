#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-cube-left
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
##SBATCH --account=<your-account>       (or: sbatch --account=..., or SBATCH_ACCOUNT in slurm/tri_tasks.env)
# SmolVLA on cube_left ("Pick up the red cube and place it 5 cm to the left.")
# Training data: 100 cube-moving demonstrations (20 per placement x 5 placements).
# Fine-tunes the pretrained SmolVLA checkpoint (PRETRAINED, default lerobot/smolvla_base) for 100,000 optimizer steps on
# one H100 (96 GB), BF16, batch BATCH_SIZE (default 32), cosine LR decay over the 100k steps, checkpoints every 20k
# ($OUT/checkpoints/020000 ... 100000, all kept). Same job as slurm/smolvla_stack_red_on_blue.sbatch: it runs
# slurm/smolvla_red_cube_to_white_tray.sbatch (that header lists every stage and variable) with the defaults below and
# the site settings of slurm/tri_tasks.env (scratch dir, environment activation, pretrained checkpoint, dataset ids,
# batch size, account/partition). Stages: environment -> evaluation states (init_states/<task>.json) -> dataset
# -> validation (10 episodes replayed in the simulator) -> training -> optional evaluation (EVAL_CKPTS, default
# none). The dataset is used as is if present in $SCRATCH_ROOT/datasets/lerobot/<repo_id>, else
# generated: 20 successful demonstrations per placement.
# Resume: resubmitting the same command continues from $OUT/checkpoints/last (model, optimizer, scheduler and step
# from training_state/); STEPS=<more> extends the run (the LR decay is stretched to the new length).
# Inputs are the front + wrist cameras (renamed to smolvla_base's camera1/camera2) and the 6 joint positions; actions
# are absolute joint targets (degrees, gripper 0-100) at 30 fps; normalization uses the dataset's statistics and is
# saved with every checkpoint (policy_preprocessor.json / policy_postprocessor.json).
#
#   sbatch slurm/train_smolvla_cube_left.slurm                                   # from the project root
#   source slurm/tri_tasks.env && sbatch slurm/train_smolvla_cube_left.slurm     # with SBATCH_ACCOUNT / SBATCH_PARTITION set there
#   BATCH_SIZE=64 sbatch --time=20:00:00 slurm/train_smolvla_cube_left.slurm
# Log: ./liberomini-so101lm-smolvla-cube-left-<jobid>.out   Checkpoints: $OUT/checkpoints/<step>/pretrained_model
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/smolvla_red_cube_to_white_tray.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
# shellcheck disable=SC1091
source slurm/tri_tasks.env
export TASKS="${TASKS:-cube_left}"
export REPO_ID="${REPO_ID:-$CUBE_LEFT_REPO}"
export OUT="${OUT:-outputs/train/smolvla_tri_cube_left_100}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
exec bash slurm/smolvla_red_cube_to_white_tray.sbatch
