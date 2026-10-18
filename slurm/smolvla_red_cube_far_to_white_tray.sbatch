#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-far
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
# SmolVLA on red_cube_far_to_white_tray ("Pick up the red cube and put it in the white tray.") on one H100, end to end:
# the red_cube_to_white_tray job (slurm/smolvla_red_cube_to_white_tray.sbatch, run unchanged; its header lists every
# stage and variable) with this task's defaults:
#   cube at ONE far spot, 26 cm straight ahead of the shoulder's pan axis (+-1 cm, +-15 deg); same fixed white tray;
#   episodes start in the folded rest pose and every demonstration ends parked there again (rest -> home -> pick,
#   place -> home -> rest -> 2 s hold), so the policy stops after the task instead of hovering over the tray;
#   start-pose noise 2 deg (3 deg lets the folded gripper touch the shoulder or table in the simulator)
#   100 demos | BF16 | batch 32 | 100k steps, LR decay over 100k | checkpoints and evaluation at 20k, 40k, 60k, 80k,
#   100k (training ~4-5 h + evaluation ~1.5 h: 12 h limit; a resubmission resumes from the last checkpoint)
#   sbatch slurm/smolvla_red_cube_far_to_white_tray.sbatch                            # from the project root
#   WANDB=true WANDB_PROJECT=cs6283 WANDB_ENTITY=tsangb34-national-university-of-singapore-students-union \
#       sbatch slurm/smolvla_red_cube_far_to_white_tray.sbatch
#   STEPS=140000 sbatch slurm/smolvla_red_cube_far_to_white_tray.sbatch     # extend: resumes from 100k
# The first run makes init_states/red_cube_far_to_white_tray.json (commit it). Real robot afterwards (outputs/train/ on
# the Mac; start with the arm at rest): RUN=smolvla_real_cube_far_white_tray_100 CKPT=100000 \
#   TASK=red_cube_far_to_white_tray bash scripts/real_smolvla.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/smolvla_red_cube_to_white_tray.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
export TASKS="${TASKS:-red_cube_far_to_white_tray}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
export REPO_ID="${REPO_ID:-local/so101_real_cube_far_white_tray_100}"
export OUT="${OUT:-outputs/train/smolvla_real_cube_far_white_tray_100}"
export GEN_ARGS="${GEN_ARGS:---max-grasp-retries 2 --miss-prob 0.3 --miss-offset 0.018 0.026 --home-noise-deg 2}"
export STEPS="${STEPS:-100000}" SAVE_FREQ="${SAVE_FREQ:-20000}" PRECISION="${PRECISION:-bf16}"
exec bash slurm/smolvla_red_cube_to_white_tray.sbatch
