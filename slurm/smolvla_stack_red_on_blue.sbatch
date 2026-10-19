#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-stack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
# SmolVLA on stack_red_on_blue ("Put the red cube on top of the blue cube.") on one H100, end to end: the
# red_cube_to_white_tray job (slurm/smolvla_red_cube_to_white_tray.sbatch, run unchanged; its header lists every stage
# and variable) with this task's defaults:
#   two 3 cm cubes on the robot's left, the blue one always 10 cm (centre to centre) further left than the red one;
#   the pair moves as one (4 cells: 11-17 cm ahead of the shoulder's pan axis, red 6-9 cm left of the centre line, so
#   blue 16-19 cm left; each cube any yaw);
#   everything inside the front camera's view, the finished stack clear of the folded arm;
#   episodes start in the folded rest pose and every demonstration ends parked there again (rest -> home -> pick,
#   stack -> home -> rest -> 2 s hold); recovery demonstrations (30 % deliberately missed first grasps, up to 2
#   regrasps); start-pose noise 2 deg (3 deg lets the folded gripper touch the shoulder or table in the simulator)
#   100 demos | BF16 | batch 32 | 100k steps, LR decay over 100k | checkpoints and evaluation at 20k, 40k, 60k, 80k,
#   100k (training ~4-5 h + evaluation ~1.5 h: 12 h limit; a resubmission resumes from the last checkpoint)
#   sbatch slurm/smolvla_stack_red_on_blue.sbatch                                     # from the project root
#   WANDB=true WANDB_PROJECT=cs6283 WANDB_ENTITY=tsangb34-national-university-of-singapore-students-union \
#       sbatch slurm/smolvla_stack_red_on_blue.sbatch
#   STEPS=140000 sbatch slurm/smolvla_stack_red_on_blue.sbatch         # extend: resumes from 100k
# Evaluation states: init_states/stack_red_on_blue.json (made with MuJoCo 3.3.7; if it is missing, the job makes it:
# commit it). Real robot afterwards (copy $OUT to the Mac's outputs/train/; start with the arm at rest):
#   RUN=smolvla_real_stack_red_blue_100 CKPT=100000 TASK=stack_red_on_blue bash scripts/real_smolvla.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/smolvla_red_cube_to_white_tray.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
export TASKS="${TASKS:-stack_red_on_blue}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
export REPO_ID="${REPO_ID:-local/so101_real_stack_red_blue_100}"
export OUT="${OUT:-outputs/train/smolvla_real_stack_red_blue_100}"
export GEN_ARGS="${GEN_ARGS:---max-grasp-retries 2 --miss-prob 0.3 --miss-offset 0.018 0.026 --home-noise-deg 2}"
export STEPS="${STEPS:-100000}" SAVE_FREQ="${SAVE_FREQ:-20000}" PRECISION="${PRECISION:-bf16}"
exec bash slurm/smolvla_red_cube_to_white_tray.sbatch
