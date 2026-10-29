#!/usr/bin/env bash
# Train the multitask SmolVLA on all 300 demonstrations (flip + sort + pyramid, task-balanced) to 80k steps (checkpoints every 20k),
# starting from the original lerobot/smolvla_base. Settings: see scripts/train_smolvla_task.sh (e.g. BATCH_SIZE=64).
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_smolvla_task.sh" multitask "$@"
