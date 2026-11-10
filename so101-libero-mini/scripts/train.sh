#!/usr/bin/env bash
# Train a LeRobot policy on a generated dataset (wraps lerobot-train with this project's defaults).
#
#   bash scripts/train.sh REPO_ID OUTPUT_DIR [extra lerobot-train flags...]
#   POLICY=act STEPS=60000 BATCH_SIZE=16 bash scripts/train.sh local/so101_mini_t1_200 outputs/train/act_t1_200
#
# Environment overrides (defaults in brackets): POLICY [act] STEPS [60000] BATCH_SIZE [16] SAVE_FREQ [10000]
# LOG_FREQ [200] NUM_WORKERS [10] SEED [1000] WANDB [false] WANDB_PROJECT [so101-libero-mini] WANDB_ENTITY [wandb default]
# WANDB_MODE [online; offline = log locally, sync later with `bash scripts/env.sh wandb sync OUTPUT_DIR/wandb/latest-run`]
# DATASET_ROOT [$HF_LEROBOT_HOME/REPO_ID]. W&B credentials: ~/.netrc (wandb login) or WANDB_API_KEY.
# An existing OUTPUT_DIR is resumed from checkpoints/last with its saved config; it continues up to STEPS if STEPS
# is set, else up to the run's original step budget.
#
# Policy presets (override any flag by appending it):
#   act        ACT, chunk 50 (1.7 s at 30 Hz), 50 actions executed per inference (eval can override)
#   diffusion  Diffusion Policy, 2 obs steps, horizon 16, 8 actions per inference, 240x240 random crop
#   smolvla    SmolVLA fine-tuned from lerobot/smolvla_base (language-conditioned; needs scripts/setup_smolvla.sh);
#              front/wrist cameras are renamed to the base model's camera1/camera2
set -euo pipefail

REPO_ID="${1:?usage: train.sh REPO_ID OUTPUT_DIR [flags...]}"
OUT="$(realpath -m "${2:?usage: train.sh REPO_ID OUTPUT_DIR [flags...]}")"  # env.sh cds to the project root
shift 2
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${POLICY:-act}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-1000}"
WANDB="${WANDB:-false}"
DATASET_ROOT="$(realpath -m "${DATASET_ROOT:-${HF_LEROBOT_HOME:-/scratch/e1583535/datasets/lerobot}/$REPO_ID}")"

case "$POLICY" in
    act) PRESET=(--policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50) ;;
    diffusion) PRESET=(--policy.type=diffusion --policy.crop_shape="[240, 240]") ;;
    smolvla) PRESET=(--policy.path=lerobot/smolvla_base
                     --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}') ;;
    *) PRESET=(--policy.type="$POLICY") ;;
esac

if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
    echo "resuming $OUT from checkpoints/last (target ${STEPS:-the original budget} steps)"
    exec bash "$PROJECT_ROOT/scripts/env.sh" lerobot-train --resume=true \
        --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" ${STEPS:+--steps="$STEPS"} \
        ${WANDB_MODE:+--wandb.mode="$WANDB_MODE"} "$@"
fi
STEPS="${STEPS:-60000}"
[ -e "$OUT" ] && { echo "$OUT exists but has no checkpoint to resume; remove it" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
exec bash "$PROJECT_ROOT/scripts/env.sh" lerobot-train \
    --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT" --dataset.video_backend=pyav \
    "${PRESET[@]}" --policy.push_to_hub=false \
    --output_dir="$OUT" --job_name="$(basename "$OUT")" \
    --steps="$STEPS" --batch_size="$BATCH_SIZE" --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ" \
    --num_workers="$NUM_WORKERS" --seed="$SEED" --eval_freq=0 \
    --wandb.enable="$WANDB" --wandb.disable_artifact=true --wandb.project="${WANDB_PROJECT:-so101-libero-mini}" \
    ${WANDB_ENTITY:+--wandb.entity="$WANDB_ENTITY"} ${WANDB_MODE:+--wandb.mode="$WANDB_MODE"} \
    "$@"
