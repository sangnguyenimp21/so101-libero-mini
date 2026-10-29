#!/usr/bin/env bash
# Real SO-101 inference with the full episode logic: move to HOME -> the operator sets up the cubes (ENTER) -> SmolVLA
# -> completion detected -> policy stopped, queued actions discarded -> safe release -> safe retreat -> HOME.
#
#   bash scripts/run_real.sh --task flip --checkpoint outputs/flip_red/checkpoints/060000/pretrained_model
#   bash scripts/run_real.sh --task sort --checkpoint <ckpt> --permutation GRY --layout-for-setup 0 --detector vision
#   bash scripts/run_real.sh --task pyramid --checkpoint <ckpt> --no-send        # dry run: never commands the arm
# Hardware defaults (change them for your rig, or put them in REAL_ARGS): --port /dev/ttyACM0 --robot-id my_follower
# --front-camera 2 --wrist-camera 0 --front-size 1280x720 --wrist-size 640x480. Completion on the real arm:
# --detector behavior (default: the learned behaviour finished, from gripper and arm signals) or vision (+ a camera
# check of the result at HOME). The red-cube flip cannot be verified by a camera (plain cube): see docs/THREE_CUBES.md.
# Safety: joint limits, <= 6 deg / step per joint (--max-step-deg), LeRobot max_relative_target 5 deg vs the measured
# position, non-finite / out-of-range actions and camera or bus errors stop the policy; Ctrl-C stops it, the arm holds,
# and you are asked before it moves (release / retreat / HOME). Defaults added unless given: --episodes 1,
# --out outputs/run_real/<task>_<time>. Run it from a terminal: every episode waits for ENTER and every automatic move
# to HOME is confirmed (without a terminal it refuses to start unless --unattended). A dry run (--no-send) observes and
# infers and never moves the arm. After a Ctrl-C, a failed or interrupted parking move, or an episode that did not end
# at HOME, the session stops and the arm holds. Environment on the robot machine: the so101lm recipe (lerobot 0.4.2,
# mujoco 3.3.7: the task definitions import MuJoCo) plus the motor driver, pip install "lerobot[feetech]==0.4.2";
# another conda env: ENV_ACTIVATE="conda activate <env>".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
# shellcheck disable=SC2206
ARGS=(${REAL_ARGS:-} "$@")
has() { local a; for a in "${ARGS[@]}"; do [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0; done; return 1; }
TASK=real
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in --task) TASK="${ARGS[$((i + 1))]:-real}" ;; --task=*) TASK="${ARGS[$i]#--task=}" ;; esac
done
has --episodes || ARGS+=(--episodes 1)
has --out || ARGS+=(--out "outputs/run_real/${TASK}_$(date +%Y%m%d_%H%M%S)")
exec python -m so101_libero.cli.run_episodes --robot real "${ARGS[@]}"
