#!/usr/bin/env bash
# Collect exactly 100 scripted-expert demonstrations of one three-cube task into a LeRobot dataset, then validate it.
#
#   bash scripts/collect_task.sh flip | sort | pyramid | stack | green_right   (or the wrappers scripts/collect_<task>.sh)
#
# 20 demonstrations per layout (sort: 4 per layout x starting permutation, 25 combinations). Every attempt is executed in
# the physics simulator and saved only if the expert succeeded (the goal predicate holds at the end); a failed attempt
# is retried with the next seed. An existing complete dataset is kept (never overwritten); a partial one stops the
# script (delete it, or OVERWRITE=1).
# Settings (environment, defaults in brackets): WORKERS [8] REPO_ID [see three_cubes_common.sh] OVERWRITE [0]
#   VALIDATE [1: LeRobot loader + open-loop replay of 10 episodes] VCODEC [libsvtav1] SO101_RENDER [auto]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
source scripts/three_cubes_common.sh
_repo="${REPO_ID:-}"
three_cubes_task "${1:?usage: collect_task.sh flip|sort|pyramid|stack|green_right}"
[ "$TASK_NAME" != multitask ] || { echo "the multitask dataset is merged: bash scripts/train_multitask_smolvla.sh" >&2; exit 1; }
REPO_ID="${_repo:-$REPO_ID}"
WORKERS="${WORKERS:-8}"
VALIDATE="${VALIDATE:-1}"
ROOT="$(dataset_root "$REPO_ID")"

echo "==== collect $TASK_ID: $DEMOS_PER_PLACEMENT per placement (100 demos) -> $ROOT ($WORKERS workers, renderer $SO101_RENDER)"
if dataset_complete "$REPO_ID" 100 && [ "${OVERWRITE:-0}" != 1 ]; then
    echo "dataset $ROOT is complete (100 episodes): kept (OVERWRITE=1 regenerates it)"
else
    ARGS=()
    if [ -e "$ROOT" ]; then
        [ "${OVERWRITE:-0}" = 1 ] || { echo "$ROOT exists but is incomplete: delete it or set OVERWRITE=1" >&2; exit 1; }
        ARGS+=(--overwrite)
    fi
    python -m so101_libero.cli.generate --tasks "$TASK_ID" --demos-per-placement "$DEMOS_PER_PLACEMENT" \
        --workers "$WORKERS" --repo-id "$REPO_ID" --vcodec "${VCODEC:-libsvtav1}" "${ARGS[@]+"${ARGS[@]}"}" 2>&1 \
        | grep --line-buffered -v -e "^Svt" -e "^SVT" -e "moov atom"
    dataset_complete "$REPO_ID" 100 || { echo "generation did not reach 100 episodes ($ROOT)" >&2; exit 2; }
fi
if [ "$VALIDATE" = 1 ]; then
    python -m so101_libero.cli.validate_dataset --repo-id "$REPO_ID" --replay-episodes 10
fi
python -m so101_libero.cli.check_dataset --repo-id "$REPO_ID" --expect-episodes 100
