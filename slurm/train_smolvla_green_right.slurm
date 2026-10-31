#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-cubes-green-right
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
##SBATCH --account=<your-account>       (or: sbatch --account=..., or SBATCH_ACCOUNT in slurm/tri_tasks.env)
# SmolVLA on green_right_of_red ("Move the green cube to the right of the red cube.") on one H100, end to end, with the
# recipe of smolvla_tri_cube_flip_100: the shared SmolVLA job (slurm/smolvla_job.sbatch, run unchanged; its header lists
# every stage and variable) with the site settings of slurm/tri_tasks.env (scratch dir, environment, pretrained
# checkpoint, dataset ids, batch size, evaluation, account/partition):
#   scene so101_cubes (so101_libero/tasks/cube_tasks.py): three plain 3 cm cubes in one row 10 cm apart in front of the
#   arm (green, yellow, red from the robot's left), in 5 fixed placements; the green cube is picked top-down, carried
#   over the row and set down 6.5 cm to the robot's right of the red one; episodes start in the folded rest pose and
#   every demonstration ends parked there again (2 s hold)
#   100 demos (20 per placement) | BF16 | batch 32 | 100k steps, LR decay over 100k | checkpoints 20k, 40k, 60k, 80k,
#   100k (all kept), EACH evaluated in the simulator on 50 frozen starts (10 per placement), results in
#   $OUT/eval/<step>/results.json (training ~4-5 h + evaluation ~1.5 h: 12 h limit)
# Resume: resubmitting the same command continues from $OUT/checkpoints/last (model, optimizer, scheduler and step) and
# skips evaluations that finished; STEPS=<more> extends the run (the LR decay is stretched to the new length).
#   sbatch slurm/train_smolvla_green_right.slurm                                    # from the project root
# Real robot afterwards (copy $OUT to the Mac's outputs/train/; start with the arm at rest):
#   RUN=smolvla_cubes_green_right_100 CKPT=100000 TASK=green_right_of_red bash scripts/real_smolvla.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/smolvla_job.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
export STEPS="${STEPS:-100000}"
# shellcheck disable=SC1091
source slurm/tri_tasks.env
export TASKS="${TASKS:-green_right_of_red}"
export REPO_ID="${REPO_ID:-$GREEN_RIGHT_REPO}"
export OUT="${OUT:-outputs/train/smolvla_cubes_green_right_100}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
exec bash slurm/smolvla_job.sbatch
