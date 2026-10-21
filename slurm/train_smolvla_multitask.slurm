#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-tri-multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
##SBATCH --account=<your-account>       (or: sbatch --account=..., or SBATCH_ACCOUNT in slurm/tri_tasks.env)
# SmolVLA multitask on all three so101_tri tasks (one policy, conditioned on the instruction) on one H100, end to end:
# the shared SmolVLA job (slurm/smolvla_job.sbatch, run unchanged; its header lists every stage and variable) with the
# so101_tri site settings of slurm/tri_tasks.env (scratch dir, environment, pretrained checkpoint, dataset ids, batch
# size, evaluation, account/partition):
#   all 300 demonstrations: the three per-task datasets (each generated first if missing) merged into MULTITASK_REPO;
#   every sample keeps its own instruction, and the three tasks are sampled equally often
#   (so101_libero.train.lerobot_train_balanced; their episodes differ in length)
#   BF16 | batch 32 | 100k steps, LR decay over 100k | checkpoints 20k, 40k, 60k, 80k, 100k (all kept), EACH evaluated
#   in the simulator on all three tasks (50 frozen starts per task, 10 per placement), results in
#   $OUT/eval/<step>/results.json (training ~4-5 h + evaluation ~4 h; if the 12 h limit stops it, resubmit)
# Resume: resubmitting the same command continues from $OUT/checkpoints/last (model, optimizer, scheduler and step) and
# skips evaluations that finished; STEPS=<more> extends the run (the LR decay is stretched to the new length).
#   sbatch slurm/train_smolvla_multitask.slurm                                      # from the project root
#   sbatch --time=16:00:00 slurm/train_smolvla_multitask.slurm                      # room for all evaluations
# Real robot afterwards (copy $OUT to the Mac's outputs/train/; start with the arm at rest; any of the three tasks):
#   RUN=smolvla_tri_v2_multitask_300 CKPT=100000 TASK=cube_flip bash scripts/real_smolvla.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/smolvla_job.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
# shellcheck disable=SC1091
source slurm/tri_tasks.env
export TASKS="${TASKS:-cylinder_aside cube_flip green_into_tray}"
export REPO_ID="${REPO_ID:-$MULTITASK_REPO}"
export OUT="${OUT:-outputs/train/smolvla_tri_v2_multitask_300}"
export MERGE_FROM="${MERGE_FROM:-cylinder_aside=$CYLINDER_ASIDE_REPO cube_flip=$CUBE_FLIP_REPO green_into_tray=$GREEN_INTO_TRAY_REPO}"
export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-python -m so101_libero.train.lerobot_train_balanced}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
exec bash slurm/smolvla_job.sbatch
