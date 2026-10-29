#!/usr/bin/env bash
# Evaluate every saved checkpoint (20k, 40k, 60k, 80k) of one three-cube run in the simulator, with the full episode
# logic (HOME -> SmolVLA -> success detected -> policy stopped, queued actions discarded -> release -> retreat -> HOME).
#
#   bash scripts/eval_task.sh flip | sort | pyramid | multitask      (or scripts/eval_{flip,sort,pyramid,multitask}.sh)
#
# 25 episodes per checkpoint and task: the frozen start states init_states/<task>.json (5 layouts x 5 trials; sort:
# every layout x starting permutation once). Multitask: every checkpoint on all three tasks.
# Per checkpoint: <run dir>/eval/<step>/<task_id>/{episodes.csv,episodes.json,summary.json,videos/} (a complete one is
# reused; FORCE=1 re-runs it after moving the old one aside). All checkpoints: results/<task>_eval.csv / .json, and the
# summary table with the best checkpoint (highest success rate; ties -> the earlier checkpoint).
#
# Settings (environment, defaults in brackets):
EPISODES="${EPISODES:-25}"        # per checkpoint and task
WORKERS="${WORKERS:-4}"           # evaluation processes (each loads SmolVLA: ~2-2.5 GB GPU; lowered to what fits)
VIDEOS="${VIDEOS:-5}"             # videos of the first N episodes of each checkpoint
CHECKPOINTS="${CHECKPOINTS:-all}" # "all" or steps, e.g. "060000 080000"
DEVICE="${DEVICE:-cuda}"          # policy device
FORCE="${FORCE:-0}"               # 1: re-run existing evaluations (old results moved aside, never deleted)
#   RUN_DIR [outputs/<flip_red|sort_ryg|pyramid|multitask>]  RESULTS_DIR [results]  SO101_RENDER [auto: EGL]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
_run="${RUN_DIR:-}"
source scripts/three_cubes_common.sh
three_cubes_task "${1:?usage: eval_task.sh flip|sort|pyramid|multitask}"
RUN_DIR="${_run:-$RUN_DIR}"

if [ ! -d "$RUN_DIR/checkpoints" ]; then
    echo "no checkpoints in $RUN_DIR: train first (bash scripts/train_${TASK_NAME}_smolvla.sh)" >&2
    exit 1
fi
TASK_ARG="$TASK_NAME"
[ "$TASK_NAME" = multitask ] && TASK_ARG=all
ARGS=(--run-dir "$RUN_DIR" --task "$TASK_ARG" --episodes "$EPISODES" --workers "$WORKERS" --videos "$VIDEOS"
      --device "$DEVICE" --results-dir "$RESULTS_DIR" --results-name "$TASK_NAME")
if [ "$CHECKPOINTS" != all ]; then
    # shellcheck disable=SC2206
    ARGS+=(--checkpoints $CHECKPOINTS)
fi
[ "$FORCE" = 1 ] && ARGS+=(--force)
echo "==== evaluate $RUN_DIR on $TASK_ARG: $EPISODES episodes per checkpoint, $WORKERS workers, renderer $SO101_RENDER"
python -m so101_libero.cli.eval_three_cubes "${ARGS[@]}" "${@:2}"
