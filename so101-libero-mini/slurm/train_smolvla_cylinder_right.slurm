#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-cylinder-right
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
##SBATCH --account=<your-account>       (or: sbatch --account=..., or SBATCH_ACCOUNT in slurm/tri_tasks.env)
# SmolVLA on cylinder_right_of_cube ("Put the light gray cylinder to the right of the red cube.") on one H100, end to
# end, with the recipe of smolvla_tri_cube_flip_100 except where the cylinder starts and 80k steps: the shared SmolVLA
# job (slurm/smolvla_job.sbatch, run unchanged; its header lists every stage and variable) with the so101_tri site
# settings of slurm/tri_tasks.env (scratch dir, environment, pretrained checkpoint, dataset ids, batch size, evaluation,
# account/partition):
#   scene so101_tri (docs/TRI_TASKS.md), unchanged: white tray fixed on the robot's right, red cube and green block at
#   their default placement; the gray cylinder starts ANYWHERE in a 4 x 6 cm rectangle on the robot's left-front
#   (not the five fixed placements: the policy has to look where it is); the cylinder is side-grasped (wrist camera on
#   top, 45 deg approach) as in cylinder_aside, carried over the green block and the cube and stood upright 6-9 cm to
#   the robot's right of the cube (success: >= 4 cm); episodes start in the folded rest pose and every demonstration
#   ends parked there again (2 s hold)
#   100 demos (plain: no recovery demonstrations) | BF16 | batch 32 | 80k steps, LR decay over 80k | checkpoints 20k,
#   40k, 60k, 80k (all kept), EACH evaluated in the simulator on 50 frozen starts (8-9 per 2 x 2 cm start cell),
#   results in $OUT/eval/<step>/results.json (training ~3.5-4 h + evaluation ~2 h: 12 h limit)
# Resume: resubmitting the same command continues from $OUT/checkpoints/last (model, optimizer, scheduler and step) and
# skips evaluations that finished; STEPS=<more> extends the run (the LR decay is stretched to the new length).
#   sbatch slurm/train_smolvla_cylinder_right.slurm                                 # from the project root
# Real robot afterwards (copy $OUT to the Mac's outputs/train/; start with the arm at rest):
#   RUN=smolvla_tri_cylinder_right_100 CKPT=080000 TASK=cylinder_right_of_cube bash scripts/real_smolvla.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/smolvla_job.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
export STEPS="${STEPS:-80000}"
# shellcheck disable=SC1091
source slurm/tri_tasks.env
export TASKS="${TASKS:-cylinder_right_of_cube}"
export REPO_ID="${REPO_ID:-$CYLINDER_RIGHT_REPO}"
export OUT="${OUT:-outputs/train/smolvla_tri_cylinder_right_100}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
# No predefined placements, so not tri_tasks.env's --demos-per-placement; an empty GEN_ARGS would turn on the driver's
# recovery demonstrations, so pass generate's own default explicitly.
export GEN_ARGS="--max-grasp-retries 0"
exec bash slurm/smolvla_job.sbatch
