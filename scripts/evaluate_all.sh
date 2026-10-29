#!/usr/bin/env bash
# Evaluate the 20k/40k/60k/80k checkpoints of every three-cube run that exists: flip, sort, pyramid, and the multitask
# run if outputs/multitask/checkpoints exists. A missing run is reported and skipped; one failing evaluation never
# stops the others. Ends with all summaries and best checkpoints again (from results/*_eval.json).
#
#   bash scripts/evaluate_all.sh                     # settings as in scripts/eval_task.sh (EPISODES, WORKERS, ...)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/local_env.sh
source scripts/three_cubes_common.sh
NAMES=(flip sort pyramid)
[ -d "$OUTPUTS_DIR/multitask/checkpoints" ] && NAMES+=(multitask)
STATUS=0
DONE=()
for t in "${NAMES[@]}"; do
    three_cubes_task "$t"
    if [ ! -d "$RUN_DIR/checkpoints" ]; then
        echo "==== $t: no checkpoints in $RUN_DIR (not trained yet): skipped"
        continue
    fi
    if bash scripts/eval_task.sh "$t"; then
        DONE+=("$t")
    else
        echo "==== $t: EVALUATION FAILED OR INCOMPLETE (see above)"
        STATUS=1
        DONE+=("$t")
    fi
done
echo
echo "================================ summaries ================================"
for t in "${DONE[@]+"${DONE[@]}"}"; do
    three_cubes_task "$t"
    TASK_ARG="$TASK_NAME"
    [ "$TASK_NAME" = multitask ] && TASK_ARG=all
    python -m so101_libero.cli.eval_three_cubes --run-dir "$RUN_DIR" --task "$TASK_ARG" --summary-only \
        --results-dir "$RESULTS_DIR" --results-name "$TASK_NAME" || STATUS=1
done
[ ${#DONE[@]} -eq 0 ] && echo "nothing evaluated: no run has checkpoints yet (train first)"
exit "$STATUS"
