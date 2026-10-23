#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-green-push
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
##SBATCH --account=<your-account>       (or: sbatch --account=..., or SBATCH_ACCOUNT in slurm/tri_tasks.env)
# SmolVLA on green_push_forward ("Push the green semicircular block forward 5 cm.") on one H100, end to end: the shared
# SmolVLA job (slurm/smolvla_job.sbatch, run unchanged; its header lists every stage and variable) with the so101_tri
# site settings of slurm/tri_tasks.env (scratch dir, environment, pretrained checkpoint, dataset ids, batch size,
# evaluation, account/partition):
#   scene so101_tri (docs/TRI_TASKS.md), unchanged; the demonstrations close the gripper, lower it pointing down behind
#   the green block (both fingers side by side, the wrist camera in front watching the block), push it 5 cm forward in
#   a straight line, back off and lift; episodes start in the folded rest pose and every demonstration ends parked
#   there again (2 s hold)
#   100 demos (20 per placement) | BF16 | batch 32 | 80k steps, LR decay over 80k | checkpoints 20k, 40k, 60k, 80k
#   (all kept), EACH evaluated in the simulator on 50 frozen starts (10 per placement), results in
#   $OUT/eval/<step>/results.json (training ~3.5-4 h + evaluation ~1 h: 12 h limit)
# Resume: resubmitting the same command continues from $OUT/checkpoints/last (model, optimizer, scheduler and step) and
# skips evaluations that finished; STEPS=<more> extends the run (the LR decay is stretched to the new length).
#   sbatch slurm/train_smolvla_green_push.slurm                                     # from the project root
#   BATCH_SIZE=64 sbatch --time=20:00:00 slurm/train_smolvla_green_push.slurm
# Real robot afterwards (copy $OUT to the Mac's outputs/train/; start with the arm at rest):
#   RUN=smolvla_tri_green_push_100 CKPT=080000 TASK=green_push_forward bash scripts/real_smolvla.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/smolvla_job.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
export STEPS="${STEPS:-80000}"
# shellcheck disable=SC1091
source slurm/tri_tasks.env
export TASKS="${TASKS:-green_push_forward}"
export REPO_ID="${REPO_ID:-$GREEN_PUSH_REPO}"
export OUT="${OUT:-outputs/train/smolvla_tri_green_push_100}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
exec bash slurm/smolvla_job.sbatch
