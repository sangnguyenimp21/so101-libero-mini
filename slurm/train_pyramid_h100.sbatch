#!/bin/bash
#SBATCH --job-name=so101-3c-pyramid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
##SBATCH --account=<your-account>       (or: sbatch --account=..., or SBATCH_ACCOUNT in the submitting shell)
# SmolVLA "pyramid": Place the green cube centered on top of the red and yellow cubes. (100 demos)
# From lerobot/smolvla_base (never from another task's model), 80k steps, checkpoints 020000/040000/060000/080000,
# batch 128 by default (slurm/three_cubes.env), every checkpoint evaluated in the simulator (25 episodes per task with
# the full HOME -> policy -> stop -> release -> retreat -> HOME controller) -> results/pyramid_eval.csv, BEST printed.
#   sbatch slurm/train_pyramid_h100.sbatch                        # from the project root
#   BATCH_SIZE=64 sbatch slurm/train_pyramid_h100.sbatch          # any setting of slurm/three_cubes.env
# Resubmitting the same command continues (training resumes from checkpoints/last, finished evaluations are reused).
# Stages and logs: slurm/three_cubes_job.sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f slurm/three_cubes_job.sbatch ] || { echo "submit from the project root" >&2; exit 1; }
export TASK=pyramid
exec bash slurm/three_cubes_job.sbatch
