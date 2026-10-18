#!/usr/bin/env bash
# Run a trained ACT checkpoint on the real SO-101 follower (wraps lerobot-rollout, no recording).
#
#   bash scripts/real_act.sh [extra lerobot-rollout flags...]
#   RUN=act_all_200 CKPT=080000 DURATION=30 bash scripts/real_act.sh --display_data=true
#
# Environment overrides (defaults in brackets): RUN [act_all_200] CKPT [last]
# PORT [/dev/tty.usbmodem5B610366661] ROBOT_ID [my_follower] FRONT_CAM [3: iPhone] WRIST_CAM [0]
# FRONT_W x FRONT_H [1280x720] WRIST_W x WRIST_H [640x480] FPS [30] DURATION [60 s; 0 = until Ctrl+C]
# MAX_REL [5: max change per arm joint per tick, degrees] GRIPPER_MAX_REL [100: same for the gripper, in its 0-100
# range; 100 = uncapped, so it can squeeze an object like in the teleop data; the servo's own 50% torque limit stays]
# DEVICE [mps]
# FRONT_CROP / WRIST_CROP [centre square of each camera, e.g. 0,280,720,720] as TOP,LEFT,HEIGHT,WIDTH; CROP_RESIZE [1]
# Each frame is cropped to that box and resized to the training 256x256 (scripts/deploy_policy.py, written to
# outputs/deploy/); CROP_RESIZE=0 feeds the raw frames instead.
# DEBUG [1]: save what the policy sees to outputs/debug/<RUN>_<time>/: frame_<tick>.jpg (front | wrist at 256x256 +
# joint state) every DEBUG_EVERY [10] control ticks, and log.csv (state + commanded action every tick).
# Camera indices move when cameras (or the iPhone) connect: check them with `lerobot-find-cameras opencv`. With the
# iPhone (Continuity Camera) as front camera keep Center Stage / video effects off, so the view never pans or zooms.
# ACT ignores language, so the run decides the task (act_t1_200_recovery: red cylinder -> black tray).
set -eo pipefail

eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate lerobot
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${RUN:-act_legalaspro}"
CKPT="${CKPT:-last}"
PORT="${PORT:-/dev/tty.usbmodem5B610366661}"
# PORT="${PORT:-/dev/tty.usbmodem5C821064861}"
ROBOT_ID="${ROBOT_ID:-my_follower_sim}"
# ROBOT_ID="${ROBOT_ID:-my_follower}"
# ROBOT_ID="${ROBOT_ID:-home_follower2}"
FRONT_CAM="${FRONT_CAM:-3}"
WRIST_CAM="${WRIST_CAM:-0}"
FRONT_W="${FRONT_W:-1280}"
FRONT_H="${FRONT_H:-720}"
# FRONT_W="${WRIST_W:-640}"
# FRONT_H="${WRIST_H:-480}"
WRIST_W="${WRIST_W:-640}"
WRIST_H="${WRIST_H:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-60}"
MAX_REL="${MAX_REL:-100}"
GRIPPER_MAX_REL="${GRIPPER_MAX_REL:-100}"
DEVICE="${DEVICE:-mps}"
center() { local s=$(( $1 < $2 ? $1 : $2 )); echo "$(( ($2 - s) / 2 )),$(( ($1 - s) / 2 )),$s,$s"; }  # W H -> box
FRONT_CROP="${FRONT_CROP:-$(center "$FRONT_W" "$FRONT_H")}"
WRIST_CROP="${WRIST_CROP:-$(center "$WRIST_W" "$WRIST_H")}"
CROP_RESIZE="${CROP_RESIZE:-1}"
DEBUG="${DEBUG:-1}"
DEBUG_EVERY="${DEBUG_EVERY:-10}"

POLICY_PATH="$PROJECT_ROOT/outputs/train/$RUN/checkpoints/$CKPT/pretrained_model"
[ -d "$POLICY_PATH" ] || { echo "no checkpoint at $POLICY_PATH" >&2; exit 1; }
ROLLOUT=(lerobot-rollout)
if [ "$CROP_RESIZE" = 1 ]; then
    DEBUG_ARGS=()
    if [ "$DEBUG" = 1 ]; then
        DEBUG_DIR="$PROJECT_ROOT/outputs/debug/${RUN}_$(date +%Y%m%d_%H%M%S)"
        DEBUG_ARGS=(--debug-dir "$DEBUG_DIR" --debug-every "$DEBUG_EVERY")
        ROLLOUT=(python "$PROJECT_ROOT/scripts/rollout_debug.py")  # lerobot-rollout + the debug processor steps
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

exec "${ROLLOUT[@]}" \
    --strategy.type=base \
    --policy.path="$POLICY_PATH" \
    --policy.device="$DEVICE" \
    --device="$DEVICE" \
    --robot.type=so101_follower \
    --robot.port="$PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.use_degrees=true \
    --robot.max_relative_target="{shoulder_pan: $MAX_REL, shoulder_lift: $MAX_REL, elbow_flex: $MAX_REL, \
        wrist_flex: $MAX_REL, wrist_roll: $MAX_REL, gripper: $GRIPPER_MAX_REL}" \
    --robot.cameras="$CAMERAS" \
    --fps="$FPS" \
    --duration="$DURATION" \
    "$@"
