#!/bin/bash
# The three-cube SmolVLA job (one task, one H100 96 GB, end to end). Submit a wrapper, not this file:
#   sbatch slurm/train_flip_h100.sbatch       sbatch slurm/train_sort_h100.sbatch
#   sbatch slurm/train_pyramid_h100.sbatch    sbatch slurm/train_multitask_h100.sbatch
# Settings: slurm/three_cubes.env (BATCH_SIZE, NUM_WORKERS, EVAL_*, ...), overridable from the submitting shell.
# Every stage skips what is already done, so resubmitting the same command after a timeout continues:
#  1. runtime: conda env via scripts/setup_conda_env.sh under a lock (the first job on a cluster: ~15 min, internet)
#  2. renderer: EGL if tests/egl_cuda_workers_selftest.py passes on this node (EGL next to CUDA aborted on some
#     nodes), else OSMesa (CPU)
#  3. dataset(s): used as is when complete (e.g. rsync'ed from the machine that made them), else generated here
#     (scripts/collect_task.sh: 100 demonstrations + replay validation) under a per-dataset lock; the multitask job
#     waits for, or makes, all three and merges them (scripts/train_smolvla_task.sh)
#  4. training: scripts/train_smolvla_task.sh (from lerobot/smolvla_base, 80k steps, checkpoints every 20k, BF16;
#     resumes from <run dir>/checkpoints/last)
#  5. evaluation: every checkpoint with the full episode controller (scripts/eval_task.sh, 25 episodes per task), each
#     one as soon as it is saved while training continues on the same GPU, then any missing one, then the summary
#     table with BEST (results/<task>_eval.csv / .json)
# Logs: ./liberomini-<job name>-<job id>.out (this script) and logs/slurm/<task>_<job id>/ (per-stage logs).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f so101_libero/__init__.py ] || { echo "submit from the project root" >&2; exit 1; }
: "${TASK:?set TASK=flip|sort|pyramid|multitask (submit a wrapper, e.g. sbatch slurm/train_flip_h100.sbatch)}"
# shellcheck disable=SC1091
source slurm/three_cubes.env
PROJECT_ROOT="$PWD"
JOB="${SLURM_JOB_ID:-local$$}"
LOGDIR="$PROJECT_ROOT/logs/slurm/${TASK}_$JOB"
mkdir -p "$LOGDIR"
echo "==== $(hostname) | job $JOB | task $TASK | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) | $(date)"
echo "batch $BATCH_SIZE, $STEPS steps, checkpoints every $SAVE_FREQ, $NUM_WORKERS data workers, eval $EVAL_EPISODES episodes x $EVAL_WORKERS workers"

# ---- 1. runtime (the lock stops two jobs from installing into the same env at once)
mkdir -p "$HOME/.cache"
(
    flock 9
    ENV_NAME="$ENV_NAME" SCRATCH_ROOT="$SCRATCH_ROOT" CONDA_ROOT="$CONDA_ROOT" bash scripts/setup_conda_env.sh
) 9>"$HOME/.cache/so101lm_setup_$ENV_NAME.lock"
export SO101_TMPDIR="/tmp/${USER:-so101}-so101lm-$JOB"
trap 'rm -rf "$SO101_TMPDIR"' EXIT

# ---- 2. renderer
if [ "$EVAL_RENDER" = auto ]; then
    if (SO101_RENDER=egl; source scripts/local_env.sh && timeout 600 python tests/egl_cuda_workers_selftest.py 2) \
            > "$LOGDIR/egl_selftest.log" 2>&1; then
        EVAL_RENDER=egl
    else
        EVAL_RENDER=osmesa
        echo "EGL failed its self-test on $(hostname) ($LOGDIR/egl_selftest.log): rendering with OSMesa (CPU)"
    fi
fi
export SO101_RENDER="$EVAL_RENDER"
# shellcheck disable=SC1091
source scripts/local_env.sh
# shellcheck disable=SC1091
source scripts/three_cubes_common.sh
echo "renderer: $SO101_RENDER | python $(python -c 'import sys; print(sys.version.split()[0])') | free disk: $(free_gb "$PROJECT_ROOT/outputs") GB"
if [ "${REQUIRE_CUDA:-1}" = 1 ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('cuda:', torch.cuda.get_device_name(0))"
fi

# ---- 3. dataset(s)
ensure_dataset() {  # NAME -> its complete 100-episode dataset, generated here (under a lock) when missing
    local name="$1" root
    three_cubes_task "$name"
    root="$(dataset_root "$REPO_ID")"
    mkdir -p "$(dirname "$root")"
    exec 8>"$root.lock"
    echo "==== dataset $REPO_ID: waiting for its lock (another job may be generating it) $(date)"
    flock 8
    if dataset_complete "$REPO_ID" 100; then
        echo "==== dataset $root is complete: used as is"
    elif [ -e "$root" ]; then
        echo "$root exists but is incomplete (an interrupted generation): delete it (rm -rf $root) and resubmit" >&2
        exit 1
    else
        echo "==== generating $REPO_ID ($GEN_WORKERS workers) $(date)"
        WORKERS="$GEN_WORKERS" bash scripts/collect_task.sh "$name" 2>&1 | tee "$LOGDIR/collect_$name.log" \
            | grep --line-buffered -E "demos saved|REJECTED|== result|Result:|WARNING|Error"
    fi
    flock -u 8
    exec 8>&-
}
if [ "$TASK" = multitask ]; then
    for t in flip sort pyramid; do ensure_dataset "$t"; done
else
    ensure_dataset "$TASK"
fi
three_cubes_task "$TASK"
if [ "$TASK" != multitask ]; then
    bash scripts/check_dataset.sh "$TASK" > "$LOGDIR/check_dataset.log" 2>&1 \
        || { cat "$LOGDIR/check_dataset.log"; echo "dataset check failed" >&2; exit 1; }
    tail -n 1 "$LOGDIR/check_dataset.log"
fi

# ---- 4 + 5. training, with each checkpoint evaluated as soon as it is saved
evaluate() {  # [STEP ...]: evaluate these checkpoints (all saved ones when none is given)
    CHECKPOINTS="${*:-all}" EPISODES="$EVAL_EPISODES" WORKERS="$EVAL_WORKERS" VIDEOS="$EVAL_VIDEOS" \
        DEVICE="$EVAL_DEVICE" RUN_DIR="$RUN_DIR" bash scripts/eval_task.sh "$TASK" ${EVAL_ARGS:-}
}
STOP="$LOGDIR/training_finished"
watch_checkpoints() {  # background: evaluate each complete checkpoint once, until training has finished
    declare -A done_steps=()
    local d s
    while :; do
        for d in "$RUN_DIR"/checkpoints/[0-9]*; do
            # LeRobot writes pretrained_model/ first and training_state/ after it: the policy is complete by then
            [ -f "$d/training_state/training_step.json" ] || continue
            s="$(basename "$d")"
            [ -n "${done_steps[$s]:-}" ] && continue
            done_steps[$s]=1
            echo "==== evaluating checkpoint $s while training continues $(date)"
            if evaluate "$s" > "$LOGDIR/eval_$s.log" 2>&1; then
                grep -E "^ *[0-9]+k? +[0-9]+/[0-9]+|BEST" "$LOGDIR/eval_$s.log" | tail -n 3 || true
            else
                echo "evaluation of $s failed ($LOGDIR/eval_$s.log); retried after training"
            fi
        done
        [ -f "$STOP" ] && break
        sleep 60
    done
}
WATCH_PID=
if [ "$EVAL_DURING_TRAINING" = 1 ]; then
    watch_checkpoints &
    WATCH_PID=$!
fi
echo "==== training $TASK -> $RUN_DIR $(date)"
set +e
bash scripts/train_smolvla_task.sh "$TASK" ${TRAIN_ARGS:-}
TRAIN_STATUS=$?
set -e
touch "$STOP"
if [ -n "$WATCH_PID" ]; then
    echo "==== waiting for the running checkpoint evaluation $(date)"
    wait "$WATCH_PID" || true
fi
if compgen -G "$RUN_DIR/checkpoints/[0-9]*" > /dev/null; then
    echo "==== evaluation of every saved checkpoint (completed ones are reused) $(date)"
    set +e
    evaluate 2>&1 | tee "$LOGDIR/eval_final.log" | grep -v -E "^\{|EGL|eglGetCurrentContext|Exception ignored"
    EVAL_STATUS=${PIPESTATUS[0]}
    set -e
else
    EVAL_STATUS=0
fi
echo "==== done $(date): training exit $TRAIN_STATUS, evaluation exit $EVAL_STATUS"
echo "checkpoints: $RUN_DIR/checkpoints | results: $RESULTS_DIR/${TASK_NAME}_eval.csv | logs: $LOGDIR"
[ "$TRAIN_STATUS" -eq 0 ] || exit "$TRAIN_STATUS"
exit "$EVAL_STATUS"
