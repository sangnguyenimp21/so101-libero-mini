#!/usr/bin/env bash
# Run the trained SmolVLA checkpoint on the real SO-101 follower (wraps lerobot-rollout, no recording).
#
#   bash scripts/real_smolvla.sh [extra lerobot-rollout flags...]
#   bash scripts/real_smolvla.sh                         # red cube -> black box (smolvla_real_cube_box_500, the default)
#   RUN=smolvla_real_cube_300 CKPT=last TASK=red_cube_aside bash scripts/real_smolvla.sh
#   TASK="Pick up the red cylinder and place it inside the circular container." bash scripts/real_smolvla.sh
#
# TASK is a task id (red_cube_aside, red_cube_to_box [default], red_cube_to_black_box, stack_red_on_blue,
# cylinder_to_tray, cylinder_to_container, cube_to_tray, cylinder_aside, cube_flip, green_push_forward) or a full
# instruction; it must be the training instruction of RUN.
#   RUN=smolvla_real_cube_black_box_100 CKPT=100000 TASK=red_cube_to_black_box bash scripts/real_smolvla.sh
#   RUN=smolvla_real_stack_red_blue_100 CKPT=100000 TASK=stack_red_on_blue bash scripts/real_smolvla.sh
# Environment overrides (defaults in brackets):
# RUN [smolvla_real_cube_box_500] CKPT [040000]
# PORT [/dev/tty.usbmodem5B610366661] ROBOT_ID [my_follower] FRONT_CAM [3: iPhone] WRIST_CAM [0]
# FRONT_W x FRONT_H [1280x720] WRIST_W x WRIST_H [640x480] FPS [30] DURATION [45 s; 0 = until Ctrl+C]
# MAX_REL [100: max change per joint per tick, degrees; 100 = effectively uncapped] DEVICE [mps]
# FRONT_CROP / WRIST_CROP [centre square of each camera, e.g. 0,280,720,720] as TOP,LEFT,HEIGHT,WIDTH; CROP_RESIZE [1]
# Each frame is cropped to that box and resized to the training 256x256 (scripts/deploy_policy.py, written to
# outputs/deploy/); CROP_RESIZE=0 feeds the raw frames instead.
# DEBUG [1]: save what the policy sees to outputs/debug/<RUN>_<time>/: frame_<tick>.jpg (front | wrist at 256x256 +
# joint state) every DEBUG_EVERY [10] control ticks, and log.csv (state + commanded action every tick).
# Camera indices move when cameras (or the iPhone) connect: check them with `lerobot-find-cameras opencv`. With the
# iPhone (Continuity Camera) as front camera keep Center Stage / video effects off, so the view never pans or zooms.
# The robot pauses briefly at each 50-action chunk while SmolVLA plans (first call ~8 s on MPS while it warms up).
set -eo pipefail

eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate lerobot
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN="${RUN:-smolvla_tri_green_push_100}"
CKPT="${CKPT:-last}"
TASK="${TASK:-green_push_forward}"

# LAB Settings
PORT="${PORT:-/dev/tty.usbmodem5B610366661}"
ROBOT_ID="${ROBOT_ID:-my_follower_sim}"
# ROBOT_ID="${ROBOT_ID:-my_follower}"
FRONT_CAM="${FRONT_CAM:-2}"
WRIST_CAM="${WRIST_CAM:-0}"
FRONT_W="${FRONT_W:-1280}"
FRONT_H="${FRONT_H:-720}"
WRIST_W="${WRIST_W:-320}"
WRIST_H="${WRIST_H:-240}"

# HOME Settings
# PORT="${PORT:-/dev/tty.usbmodem5C821064861}"
# ROBOT_ID="${ROBOT_ID:-home_follower2}"
# FRONT_CAM="${FRONT_CAM:-2}"
# WRIST_CAM="${WRIST_CAM:-0}"
# FRONT_W="${FRONT_W:-1280}"
# FRONT_H="${FRONT_H:-720}"
# WRIST_W="${WRIST_W:-320}"
# WRIST_H="${WRIST_H:-240}"

FPS="${FPS:-30}"
DURATION="${DURATION:-60}"
MAX_REL="${MAX_REL:-100}"
DEVICE="${DEVICE:-mps}"
center() { local s=$(( $1 < $2 ? $1 : $2 )); echo "$(( ($2 - s) / 2 )),$(( ($1 - s) / 2 )),$s,$s"; }  # W H -> box
FRONT_CROP="${FRONT_CROP:-$(center "$FRONT_W" "$FRONT_H")}"
WRIST_CROP="${WRIST_CROP:-$(center "$WRIST_W" "$WRIST_H")}"
CROP_RESIZE="${CROP_RESIZE:-1}"
DEBUG="${DEBUG:-1}"
DEBUG_EVERY="${DEBUG_EVERY:-10}"

# Instructions exactly as in training (so101_libero/tasks/registry.py)
case "$TASK" in
    red_cube_aside) TASK="Pick up the red cube and place it aside." ;;
    red_cube_to_box | red_cube_to_black_box) TASK="Pick up the red cube and put it in the black box." ;;
    stack_red_on_blue) TASK="Put the red cube on top of the blue cube." ;;
    cylinder_to_tray) TASK="Pick up the red cylinder and place it inside the black tray." ;;
    cylinder_to_container) TASK="Pick up the red cylinder and place it inside the circular container." ;;
    cube_to_tray) TASK="Pick up the blue object and place it inside the black tray." ;;
    cylinder_aside) TASK="Pick up the light gray cylinder and put it aside." ;;
    cube_flip) TASK="Pick up the red cube and flip it onto its side." ;;
    green_push_forward) TASK="Push the green semicircular block forward 5 cm." ;;
esac

POLICY_PATH="$PROJECT_ROOT/outputs/train/$RUN/checkpoints/$CKPT/pretrained_model"
[ -d "$POLICY_PATH" ] || { echo "no checkpoint at $POLICY_PATH" >&2; exit 1; }
ROLLOUT=(python "$PROJECT_ROOT/scripts/rollout_tolerant.py")
if [ "$CROP_RESIZE" = 1 ]; then
    DEBUG_ARGS=()
    if [ "$DEBUG" = 1 ]; then
        DEBUG_DIR="$PROJECT_ROOT/outputs/debug/${RUN}_$(date +%Y%m%d_%H%M%S)"
        DEBUG_ARGS=(--debug-dir "$DEBUG_DIR" --debug-every "$DEBUG_EVERY")
        ROLLOUT=(python "$PROJECT_ROOT/scripts/rollout_debug.py")  # rollout_tolerant.py + the debug processor steps
        echo "debug frames and action log -> $DEBUG_DIR"
    fi
    POLICY_PATH="$(python "$PROJECT_ROOT/scripts/deploy_policy.py" "$POLICY_PATH" \
        "$PROJECT_ROOT/outputs/deploy/${RUN}_${CKPT}" --crop front="$FRONT_CROP" --crop wrist="$WRIST_CROP" \
        ${DEBUG_ARGS[@]+"${DEBUG_ARGS[@]}"})"
    echo "crop front=$FRONT_CROP wrist=$WRIST_CROP -> 256x256"
elif [ "$DEBUG" = 1 ]; then
    echo "DEBUG needs CROP_RESIZE=1 (the debug steps live in the deploy copy); running without debug output"
fi

CAMERAS="{ front: {type: opencv, index_or_path: $FRONT_CAM, width: $FRONT_W, height: $FRONT_H, fps: $FPS},
           wrist: {type: opencv, index_or_path: $WRIST_CAM, width: $WRIST_W, height: $WRIST_H, fps: $FPS} }"

echo "task: $TASK"
# lerobot-rollout replaces the checkpoint's saved rename map, so repeat the training one (front/wrist -> camera1/2).
# scripts/rollout_tolerant.py = lerobot-rollout that waits out camera stalls (> 500 ms aborts plain lerobot-rollout;
# CAMERA_STALL_MS [2000] is the longest stall it waits for).
exec "${ROLLOUT[@]}" \
    --strategy.type=base \
    --policy.path="$POLICY_PATH" \
    --policy.device="$DEVICE" \
    --device="$DEVICE" \
    --robot.type=so101_follower \
    --robot.port="$PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.use_degrees=true \
    --robot.max_relative_target="$MAX_REL" \
    --robot.cameras="$CAMERAS" \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --task="$TASK" \
    --fps="$FPS" \
    --duration="$DURATION" \
    "$@"
