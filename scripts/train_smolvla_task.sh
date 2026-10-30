#!/usr/bin/env bash
# Fine-tune SmolVLA on one three-cube dataset, directly on this machine's GPU (no SLURM).
#
#   bash scripts/train_smolvla_task.sh flip | sort | pyramid | stack | green_right | multitask
#   (wrappers: scripts/train_<task>_smolvla.sh, e.g. train_flip_smolvla.sh, train_stack_smolvla.sh, train_multitask_smolvla.sh)
#   BATCH_SIZE=64 bash scripts/train_flip_smolvla.sh            # any setting below, from the environment
#
# Every run starts from the ORIGINAL base checkpoint (BASE_MODEL, lerobot/smolvla_base), never from another task's
# model: flip, sort, pyramid and multitask are four independent fine-tunings. LeRobot 0.4.2's lerobot-train with the
# project's SmolVLA recipe (slurm/smolvla_job.sbatch): front/wrist cameras renamed to smolvla_base's camera1/camera2,
# VLM frozen (the action expert is trained), AdamW lr 1e-4 with warmup + cosine decay over LR_DECAY_STEPS (= STEPS:
# smolvla_base would otherwise decay over 30k steps), BF16 autocast through Accelerate (lerobot-train takes the
# precision from ACCELERATE_MIXED_PRECISION), one optimizer step per batch (lerobot-train 0.4.2 has no gradient
# accumulation), single GPU. multitask = the three 100-demo datasets merged (300 episodes; merged here if missing)
# and trained with task-balanced sampling (so101_libero/train/lerobot_train_balanced.py: each task equally often).
#
# Checkpoints: $OUTPUT_DIR/checkpoints/020000, 040000, 060000, 080000 (+ the "last" link), each with
# pretrained_model/ (weights, config, train_config.json, pre/post-processors) and training_state/ (optimizer, scheduler,
# step, RNG: needed only to resume). Never more than STEPS. Re-running the same command resumes from checkpoints/last
# (an interrupted run continues; a finished one stops immediately); an OUTPUT_DIR without any checkpoint is moved aside
# (renamed with a timestamp), never deleted. Launch settings and the full log: logs/train/<run>_<time>.{json,log}.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
source scripts/three_cubes_common.sh
_out="${OUTPUT_DIR:-}" _ds="${DATASET:-}"
# settings given explicitly (a resume passes these on; see below)
_set_batch="${BATCH_SIZE:-}" _set_save="${SAVE_FREQ:-}" _set_workers="${NUM_WORKERS:-}" _set_log="${LOG_FREQ:-}"
_set_seed="${SEED:-}" _set_base="${BASE_MODEL:-}"
three_cubes_task "${1:?usage: train_smolvla_task.sh flip|sort|pyramid|stack|green_right|multitask}"

# ------------------------------------------------------------------------------------------------ configuration
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"   # hub id (cached in $HF_HOME by setup_conda_env.sh) or a local dir
DATASET="${_ds:-$REPO_ID}"                          # LeRobot repo id under $HF_LEROBOT_HOME
OUTPUT_DIR="$(realpath -m "${_out:-$RUN_DIR}")"
EXPECT_EPISODES="${EXPECT_EPISODES:-$([ "$TASK_NAME" = multitask ] && echo 300 || echo 100)}"
STEPS="${STEPS:-80000}"                             # optimizer updates; do not train past 80k for this experiment
SAVE_FREQ="${SAVE_FREQ:-20000}"                     # -> checkpoints 020000 040000 060000 080000
BATCH_SIZE="${BATCH_SIZE:-32}"                      # 24 GB RTX A5000: 32 (see docs/THREE_CUBES.md); H100 96 GB: 64-128
LR_DECAY_STEPS="${LR_DECAY_STEPS:-$STEPS}"
DEVICE="${DEVICE:-cuda}"                            # cuda | cpu (cpu: plumbing smoke tests only)
PRECISION="${PRECISION:-$([ "$DEVICE" = cpu ] && echo no || echo bf16)}"  # Accelerate mixed precision: bf16 | fp16 | no
NUM_WORKERS="${NUM_WORKERS:-8}"                     # DataLoader workers (AV1 video decoding)
SEED="${SEED:-1000}"
LOG_FREQ="${LOG_FREQ:-200}"
WANDB="${WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-so101-three-cubes}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"                    # refuse to start below this much free disk (checkpoints ~2-3 GB each)
GPU_MEM_GB="${GPU_MEM_GB:-}"                        # cap on this run's GPU memory (GB), for a GPU shared with another
                                                    # job: if the run needs more it fails, the other job is never hit
EXTRA_ARGS=("${@:2}")                               # anything else is passed to lerobot-train

DS_ROOT="$(dataset_root "$DATASET")"
NAME="$(basename "$OUTPUT_DIR")"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROJECT_ROOT/logs/train"
LOG="$PROJECT_ROOT/logs/train/${NAME}_${STAMP}.log"

# ------------------------------------------------------------------------------------------------ checks
if [ "$TASK_NAME" = multitask ] && ! dataset_complete "$DATASET" "$EXPECT_EPISODES"; then
    SRC=() SRC_ROOTS=()
    for t in flip sort pyramid; do
        three_cubes_task "$t" >/dev/null
        dataset_complete "$REPO_ID" 100 || { echo "multitask needs the 100-demo $t dataset first: bash scripts/collect_$t.sh" >&2; exit 1; }
        SRC+=("$REPO_ID"); SRC_ROOTS+=("$(dataset_root "$REPO_ID")")
    done
    three_cubes_task multitask >/dev/null
    [ -e "$DS_ROOT" ] && { echo "$DS_ROOT exists but is incomplete: delete it" >&2; exit 1; }
    echo "==== merging ${SRC[*]} -> $DATASET (300 episodes, tasks keep their own instructions)"
    python -m so101_libero.cli.merge_datasets --repo-ids "${SRC[@]}" --roots "${SRC_ROOTS[@]}" \
        --out-repo-id "$DATASET" --out-root "$DS_ROOT"
fi
dataset_complete "$DATASET" "$EXPECT_EPISODES" || {
    echo "dataset $DS_ROOT missing or not $EXPECT_EPISODES episodes: bash scripts/collect_${TASK_NAME}.sh first" >&2; exit 1; }
python - "$BASE_MODEL" <<'PY' || { echo "base model $BASE_MODEL not available offline: bash scripts/setup_conda_env.sh" >&2; exit 1; }
import os, sys
from huggingface_hub import snapshot_download
p = sys.argv[1]
sys.exit(0 if os.path.isdir(p) or snapshot_download(p, local_files_only=True) else 1)
PY
if [ "$DEVICE" = cuda ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'" \
        || { echo "no GPU visible (CUDA_VISIBLE_DEVICES?)" >&2; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/GPU: /'
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  GPU process (pid, memory): /'
fi
FREE=$(free_gb "$OUTPUT_DIR")
echo "free disk at $(dirname "$OUTPUT_DIR"): ${FREE} GB"
if [ "$FREE" -lt "$MIN_FREE_GB" ]; then
    echo "only ${FREE} GB free: 4 SmolVLA checkpoints need ~10 GB (MIN_FREE_GB=$MIN_FREE_GB; lower it at your own risk)" >&2
    exit 1
fi

# ------------------------------------------------------------------------------------------------ launch / resume
export ACCELERATE_MIXED_PRECISION="$PRECISION"
if [ "$DEVICE" = cpu ]; then  # lerobot-train trains on Accelerate's device: hide the GPU from it as well
    export CUDA_VISIBLE_DEVICES="" ACCELERATE_USE_CPU=true
fi
LAUNCHER=(lerobot-train)
[ "$TASK_NAME" = multitask ] && LAUNCHER=(python -m so101_libero.train.lerobot_train_balanced)
if [ -n "$GPU_MEM_GB" ] && [ "$DEVICE" = cuda ]; then  # capped: same training, through so101_libero/train/capped_train.py
    export SO101_GPU_MEM_GB="$GPU_MEM_GB" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    LAUNCHER=(python -m so101_libero.train.capped_train)
    [ "$TASK_NAME" = multitask ] && LAUNCHER+=(--balanced)
    echo "GPU memory cap: $GPU_MEM_GB GB for this run"
fi
CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
if [ -f "$CONFIG" ]; then
    DONE=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" \
        "$OUTPUT_DIR/checkpoints/last/training_state/training_step.json")
    if [ "$DONE" -ge "$STEPS" ]; then
        echo "$OUTPUT_DIR already trained to step $DONE (target $STEPS): nothing to do"
        ls "$OUTPUT_DIR/checkpoints"
        exit 0
    fi
    echo "==== resume $OUTPUT_DIR from step $DONE to $STEPS"
    # lerobot-train restores every setting from the checkpoint's train_config.json; only what is passed again changes.
    # The dataset, base model and seed cannot change inside a run; batch size, checkpoint frequency, data workers and
    # log frequency can (e.g. BATCH_SIZE=16 after running out of memory) and are passed on when set explicitly.
    read -r SAVED_DS SAVED_BASE SAVED_SEED SAVED_BATCH SAVED_SAVE SAVED_WORKERS SAVED_DECAY < <(python - "$CONFIG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(c["dataset"]["repo_id"], c["policy"].get("pretrained_path") or "-", c["seed"], c["batch_size"], c["save_freq"],
      c["num_workers"], c["scheduler"]["num_decay_steps"])
PY
)
    [ "$SAVED_DS" = "$DATASET" ] || { echo "$OUTPUT_DIR was trained on $SAVED_DS, not $DATASET: use another OUTPUT_DIR" >&2; exit 1; }
    if [ -n "$_set_base" ] && [ "$SAVED_BASE" != "$BASE_MODEL" ]; then
        echo "$OUTPUT_DIR started from $SAVED_BASE, not $BASE_MODEL: use another OUTPUT_DIR" >&2; exit 1; fi
    if [ -n "$_set_seed" ] && [ "$SAVED_SEED" != "$SEED" ]; then
        echo "$OUTPUT_DIR uses seed $SAVED_SEED, not $SEED: use another OUTPUT_DIR" >&2; exit 1; fi
    TRAIN=("${LAUNCHER[@]}" --resume=true --config_path="$CONFIG" --steps="$STEPS")
    BATCH_SIZE="$SAVED_BATCH" SAVE_FREQ="$SAVED_SAVE" NUM_WORKERS="$SAVED_WORKERS" SEED="$SAVED_SEED"
    BASE_MODEL="$SAVED_BASE"
    if [ -n "$_set_batch" ] && [ "$_set_batch" != "$SAVED_BATCH" ]; then
        echo "resume: batch size $SAVED_BATCH -> $_set_batch"; BATCH_SIZE="$_set_batch"; TRAIN+=(--batch_size="$_set_batch"); fi
    if [ -n "$_set_save" ] && [ "$_set_save" != "$SAVED_SAVE" ]; then
        echo "resume: save_freq $SAVED_SAVE -> $_set_save"; SAVE_FREQ="$_set_save"; TRAIN+=(--save_freq="$_set_save"); fi
    if [ -n "$_set_workers" ]; then NUM_WORKERS="$_set_workers"; TRAIN+=(--num_workers="$_set_workers"); fi
    if [ -n "$_set_log" ]; then TRAIN+=(--log_freq="$_set_log"); fi
    [ "$SAVED_DECAY" != "$LR_DECAY_STEPS" ] && TRAIN+=(--scheduler.num_decay_steps="$LR_DECAY_STEPS")
else
    if [ -e "$OUTPUT_DIR" ]; then  # an earlier attempt stopped before its first checkpoint: keep it, aside
        echo "$OUTPUT_DIR has no checkpoint: moved to $OUTPUT_DIR.no_checkpoint_$STAMP"
        mv "$OUTPUT_DIR" "$OUTPUT_DIR.no_checkpoint_$STAMP"
    fi
    mkdir -p "$(dirname "$OUTPUT_DIR")"
    TRAIN=("${LAUNCHER[@]}"
        --dataset.repo_id="$DATASET" --dataset.root="$DS_ROOT" --dataset.video_backend=pyav
        --policy.path="$BASE_MODEL" --policy.scheduler_decay_steps="$LR_DECAY_STEPS" --policy.push_to_hub=false
        --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
        --output_dir="$OUTPUT_DIR" --job_name="$NAME"
        --steps="$STEPS" --batch_size="$BATCH_SIZE" --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ"
        --num_workers="$NUM_WORKERS" --seed="$SEED" --eval_freq=0 --policy.device="$DEVICE"
        --wandb.enable="$WANDB" --wandb.disable_artifact=true --wandb.project="$WANDB_PROJECT")
fi
TRAIN+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

python - "$PROJECT_ROOT/logs/train/${NAME}_${STAMP}.json" <<PY
import json, os, subprocess, sys
info = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_LEROBOT_HOME", "SO101_RENDER")}
info.update(task="$TASK_NAME", task_ids="$TASK_ID", base_model="$BASE_MODEL", dataset="$DATASET", dataset_root="$DS_ROOT",
            output_dir="$OUTPUT_DIR", steps=$STEPS, save_freq=$SAVE_FREQ, batch_size=$BATCH_SIZE,
            lr_decay_steps=$LR_DECAY_STEPS, precision="$PRECISION", device="$DEVICE", num_workers=$NUM_WORKERS, seed=$SEED,
            gpu_mem_gb="$GPU_MEM_GB",
            log="$LOG")
try:
    info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    info["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
except Exception:
    pass
open(sys.argv[1], "w").write(json.dumps(info, indent=2))
PY
echo "==== ${TRAIN[*]}"
echo "log: $LOG"
set +e
"${TRAIN[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
echo "==== checkpoints in $OUTPUT_DIR/checkpoints:"
ls "$OUTPUT_DIR/checkpoints" 2>/dev/null || true
if [ "$STATUS" -eq 0 ]; then
    echo "next: bash scripts/eval_${TASK_NAME}.sh   (evaluates every saved checkpoint, 25 episodes each)"
fi
exit "$STATUS"
