#!/usr/bin/env bash
# Run the trained SmolVLA checkpoint on the real SO-101 follower (wraps lerobot-rollout, no recording).
#
#   bash scripts/real_smolvla.sh [extra lerobot-rollout flags...]
#   bash scripts/real_smolvla.sh                         # red cube -> aside (smolvla_real_cube_300, the default)
#   RUN=smolvla_all_200 TASK=cube_to_tray DURATION=30 bash scripts/real_smolvla.sh
#   TASK="Pick up the red cylinder and place it inside the circular container." bash scripts/real_smolvla.sh
#
# TASK is a task id (red_cube_aside [default], red_cube_to_box, cylinder_to_tray, cylinder_to_container, cube_to_tray) or a full
# instruction; it must be the training instruction of RUN. Environment overrides (defaults in brackets):
# RUN [smolvla_real_cube_300] CKPT [last]
# PORT [/dev/tty.usbmodem5B610366661] ROBOT_ID [my_follower] FRONT_CAM [3: iPhone] WRIST_CAM [0]
# FRONT_W x FRONT_H [1280x720] WRIST_W x WRIST_H [640x480] FPS [30] DURATION [60 s; 0 = until Ctrl+C]
# MAX_REL [5: max change per joint per tick, degrees] DEVICE [mps]
# FRONT_CROP / WRIST_CROP [centre square of each camera, e.g. 0,280,720,720] as TOP,LEFT,HEIGHT,WIDTH; CROP_RESIZE [1]
# Each frame is cropped to that box and resized to the training 256x256 (scripts/deploy_policy.py, written to
# outputs/deploy/); CROP_RESIZE=0 feeds the raw frames instead.
# Camera indices move when cameras (or the iPhone) connect: check them with `lerobot-find-cameras opencv`. With the
# iPhone (Continuity Camera) as front camera keep Center Stage / video effects off, so the view never pans or zooms.
# The robot pauses briefly at each 50-action chunk while SmolVLA plans (first call ~8 s on MPS while it warms up).
set -eo pipefail

eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate lerobot
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${RUN:-smolvla_real_cube_300}"
CKPT="${CKPT:-last}"
TASK="${TASK:-red_cube_aside}"
PORT="${PORT:-/dev/tty.usbmodem5B610366661}"
ROBOT_ID="${ROBOT_ID:-my_follower}"
FRONT_CAM="${FRONT_CAM:-1}"
WRIST_CAM="${WRIST_CAM:-0}"
FRONT_W="${FRONT_W:-1280}"
FRONT_H="${FRONT_H:-720}"
WRIST_W="${WRIST_W:-640}"
WRIST_H="${WRIST_H:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-60}"
MAX_REL="${MAX_REL:-5}"
DEVICE="${DEVICE:-mps}"
center() { local s=$(( $1 < $2 ? $1 : $2 )); echo "$(( ($2 - s) / 2 )),$(( ($1 - s) / 2 )),$s,$s"; }  # W H -> box
FRONT_CROP="${FRONT_CROP:-$(center "$FRONT_W" "$FRONT_H")}"
WRIST_CROP="${WRIST_CROP:-$(center "$WRIST_W" "$WRIST_H")}"
CROP_RESIZE="${CROP_RESIZE:-1}"

# Instructions exactly as in training (so101_libero/tasks/registry.py)
case "$TASK" in
    red_cube_aside) TASK="Pick up the red cube and place it aside." ;;
    red_cube_to_box) TASK="Pick up the red cube and put it in the black box." ;;
    cylinder_to_tray) TASK="Pick up the red cylinder and place it inside the black tray." ;;
    cylinder_to_container) TASK="Pick up the red cylinder and place it inside the circular container." ;;
    cube_to_tray) TASK="Pick up the blue object and place it inside the black tray." ;;
esac

POLICY_PATH="$PROJECT_ROOT/outputs/train/$RUN/checkpoints/$CKPT/pretrained_model"
[ -d "$POLICY_PATH" ] || { echo "no checkpoint at $POLICY_PATH" >&2; exit 1; }
if [ "$CROP_RESIZE" = 1 ]; then
    POLICY_PATH="$(python "$PROJECT_ROOT/scripts/deploy_policy.py" "$POLICY_PATH" \
        "$PROJECT_ROOT/outputs/deploy/${RUN}_${CKPT}" --crop front="$FRONT_CROP" --crop wrist="$WRIST_CROP")"
    echo "crop front=$FRONT_CROP wrist=$WRIST_CROP -> 256x256"
fi

CAMERAS="{ front: {type: opencv, index_or_path: $FRONT_CAM, width: $FRONT_W, height: $FRONT_H, fps: $FPS},
           wrist: {type: opencv, index_or_path: $WRIST_CAM, width: $WRIST_W, height: $WRIST_H, fps: $FPS} }"

echo "task: $TASK"
# lerobot-rollout replaces the checkpoint's saved rename map, so repeat the training one (front/wrist -> camera1/2)
exec lerobot-rollout \
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
