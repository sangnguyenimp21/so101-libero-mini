#!/usr/bin/env bash
# Shared settings of the three-cube SmolVLA pipeline (sourced by the scripts below; not run on its own).
# One place for names and paths:
#
#   task       task id        dataset (LeRobot repo id)        run dir (checkpoints)   eval csv
#   flip       flip_red_cube  local/smolvla_flip_red           outputs/flip_red        results/flip_eval.csv
#   sort       sort_ryg       local/smolvla_sort_ryg           outputs/sort_ryg        results/sort_eval.csv
#   pyramid    build_pyramid  local/smolvla_pyramid            outputs/pyramid         results/pyramid_eval.csv
#   multitask  (all three)    local/smolvla_multitask_300      outputs/multitask       results/multitask_eval.csv
#
# Datasets live in $HF_LEROBOT_HOME/<repo id> (default ~/so101lm/datasets/lerobot, set by scripts/local_env.sh).
# Checkpoints: <run dir>/checkpoints/020000|040000|060000|080000/pretrained_model (LeRobot's naming).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$PROJECT_ROOT/outputs}"

three_cubes_task() {  # NAME -> TASK_NAME TASK_ID REPO_ID RUN_DIR DEMOS_PER_PLACEMENT EVAL_CSV
    case "$1" in
        flip | flip_red | flip_red_cube)
            TASK_NAME=flip TASK_ID=flip_red_cube REPO_ID=local/smolvla_flip_red RUN_DIR="$OUTPUTS_DIR/flip_red"
            DEMOS_PER_PLACEMENT=20 ;;  # 5 layouts x 20 = 100
        sort | sort_ryg)
            TASK_NAME=sort TASK_ID=sort_ryg REPO_ID=local/smolvla_sort_ryg RUN_DIR="$OUTPUTS_DIR/sort_ryg"
            DEMOS_PER_PLACEMENT=4 ;;  # 25 placements (5 layouts x 5 permutations) x 4 = 100 (20 per layout)
        pyramid | build_pyramid)
            TASK_NAME=pyramid TASK_ID=build_pyramid REPO_ID=local/smolvla_pyramid RUN_DIR="$OUTPUTS_DIR/pyramid"
            DEMOS_PER_PLACEMENT=20 ;;
        multitask)
            TASK_NAME=multitask TASK_ID="flip_red_cube sort_ryg build_pyramid" REPO_ID=local/smolvla_multitask_300
            RUN_DIR="$OUTPUTS_DIR/multitask" DEMOS_PER_PLACEMENT= ;;
        *) echo "unknown task '$1' (flip | sort | pyramid | multitask)" >&2; return 1 ;;
    esac
    EVAL_CSV="$RESULTS_DIR/${TASK_NAME}_eval.csv"
    export TASK_NAME TASK_ID REPO_ID RUN_DIR DEMOS_PER_PLACEMENT EVAL_CSV
}

dataset_root() {  # REPO_ID -> its directory
    echo "${HF_LEROBOT_HOME:?source scripts/local_env.sh first}/$1"
}

dataset_complete() {  # REPO_ID [EPISODES]: generated to the end (side file written) with that many episodes
    local root
    root="$(dataset_root "$1")"
    [ -f "$root/meta/so101_libero.json" ] && [ -f "$root/meta/info.json" ] || return 1
    python - "$root" "${2:-100}" <<'PY'
import json, sys
info = json.load(open(sys.argv[1] + "/meta/info.json"))
sys.exit(0 if info["total_episodes"] == int(sys.argv[2]) else 1)
PY
}

free_gb() {  # free space (GB) of the filesystem holding PATH (or its first existing parent)
    local d="$1"
    while [ ! -d "$d" ]; do d="$(dirname "$d")"; done
    df -BG --output=avail "$d" | tail -1 | tr -dc 0-9
}
