#!/bin/bash
#SBATCH --job-name=so101lm-smolvla-wtray
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=liberomini-%x-%j.out
#SBATCH --error=liberomini-%x-%j.out
# SmolVLA on red_cube_to_white_tray ("Pick up the red cube and put it in the white tray.") on one H100, end to end.
# The recipe of slurm/smolvla_red_cube_to_box.sbatch (that run made real progress) with this plan:
#   robot SO-101 | object 3 cm red cube | target fixed white tray (inner 14 x 9 cm, 6 cm walls, centre 15 cm ahead and
#   15 cm to the robot's left) | 100 successful demonstrations | cube start: 10 cells (2 x 5 grid over x 0.16-0.24,
#   y -0.10..+0.06 m; cell = seed % 10, uniform inside) | cameras front + wrist | lerobot/smolvla_base | 1 x H100 96 GB
#   | BF16 | batch 32, no gradient accumulation (lerobot-train 0.4.2 has none: one optimizer step per batch) | 40k steps
#   | checkpoints and evaluation at 10k, 20k, 30k, 40k | extend from physical results (below)
#   sbatch slurm/smolvla_red_cube_to_white_tray.sbatch                                # from the project root
#   WANDB=true WANDB_PROJECT=cs6283 WANDB_ENTITY=tsangb34-national-university-of-singapore-students-union \
#       sbatch slurm/smolvla_red_cube_to_white_tray.sbatch
#   STEPS=60000 sbatch slurm/smolvla_red_cube_to_white_tray.sbatch      # extend: resumes from 40k, LR re-decays to 60k
# Every stage skips what is already done, so resubmitting the same command after a timeout carries on:
#  1. runtime: scripts/setup_conda_env.sh under a lock (conda env so101lm, OSMesa, SmolVLA weights; first run ~15 min)
#  2. evaluation states init_states/<task>.json: made here if missing (50 expert-verified states, ~5 min). They must be
#     made with the cluster's MuJoCo 3.3.7 (the scene hash depends on it): commit the new file from here
#  3. dataset $SCRATCH/datasets/lerobot/$REPO_ID: used as is if complete, else generated (EGL if a multi-worker
#     self-test passes, else OSMesa; 100 demos ~10 / ~30 min; the expert solves ~95 % of starts, failures are
#     discarded); refused if it missed its demo target or is an interrupted leftover (delete it); validated (10
#     episodes replayed) unless a matching PASS is cached
#  4. training with scripts/train.sh's smolvla preset, scheduler_decay_steps = STEPS (smolvla_base decays over 30k
#     steps whatever --steps says) and BF16 autocast (lerobot-train 0.4.2 runs under Accelerate, which takes the
#     precision from ACCELERATE_MIXED_PRECISION; printed before training). Resumes from $OUT/checkpoints/last; a
#     resume with DECAY_STEPS different from the checkpoint's schedule (e.g. STEPS=60000) re-stretches the cosine
#     decay to DECAY_STEPS, so the extra steps do not all run at the 2.5e-6 floor
#  5. evaluation of EVAL_CKPTS on the frozen states, OSMesa rendering + policy on the GPU (~15 min per checkpoint)
# Variables [defaults]: ENV_NAME [so101lm] SCRATCH_ROOT [~/so101lm: datasets, HF cache, OSMesa] CONDA_ROOT [~/miniconda3]
#   TASKS [red_cube_to_white_tray] NUM_DEMOS [100] REPO_ID [local/so101_real_cube_white_tray_100] GEN_WORKERS [10]
#   GEN_RENDER [auto | egl | osmesa] GEN_ARGS [--max-grasp-retries 2 --miss-prob 0.3 --miss-offset 0.018 0.026
#   --home-noise-deg 3: recovery demonstrations, as in the box run] OUT [outputs/train/smolvla_real_cube_white_tray_100]
#   STEPS [40000] DECAY_STEPS [= STEPS] BATCH_SIZE [32] SAVE_FREQ [10000] PRECISION [bf16 | fp16 | no] NUM_WORKERS [10]
#   SEED [1000] TRAIN_ARGS [extra lerobot-train flags] EVAL_TASKS [= TASKS] EVAL_EPISODES [50] EVAL_VIDEOS [10]
#   EVAL_CKPTS [all | last | none] EVAL_WORKERS [4] EVAL_EGL [0] EVAL_ARGS [extra evaluate flags] WANDB [false]
#   WANDB_PROJECT [so101-libero-mini] WANDB_ENTITY WANDB_MODE [online if credentials exist and api.wandb.ai is
#   reachable, else offline]
#   PRETRAINED [lerobot/smolvla_base: hub id (cached by setup_conda_env.sh; jobs run offline) or a local
#   pretrained_model dir] DATASET_ROOT [$SCRATCH_ROOT/datasets/lerobot/$REPO_ID] TRAIN_LAUNCHER [lerobot-train;
#   multitask: python -m so101_libero.train.lerobot_train_balanced] MERGE_FROM ["task=repo_id ...": build the dataset
#   by merging these per-task datasets (each generated first with GEN_ARGS if missing) instead of generating it]
#   ENV_ACTIVATE [unset: conda env ENV_NAME via scripts/setup_conda_env.sh; set: a shell command that activates an
#   existing environment instead, e.g. "source /path/to/venv/bin/activate"; no setup is run then]
# Log: ./liberomini-<job-name>-<jobid>.out. Checkpoints: $OUT/checkpoints/<step>/pretrained_model (copy $OUT to the Mac's
# outputs/train/ for scripts/real_smolvla.sh: RUN=smolvla_real_cube_white_tray_100 CKPT=040000
# TASK=red_cube_to_white_tray ROBOT_ID=my_follower_sim). Results: $OUT/eval/<ckpt>/results.json. Exits non-zero if any
# stage failed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f so101_libero/__init__.py ] || { echo "submit from the project root: sbatch slurm/$(basename "$0")" >&2; exit 1; }
PROJECT_ROOT="$PWD"
ENV_NAME="${ENV_NAME:-so101lm}"
SCRATCH="${SCRATCH_ROOT:-$HOME/so101lm}"  # must be shared by all nodes (not /tmp): later jobs reuse it
OSMESA_PREFIX="${OSMESA_PREFIX:-$SCRATCH/opt/osmesa}"
TASKS="${TASKS:-red_cube_to_white_tray}"
NUM_DEMOS="${NUM_DEMOS:-100}"
REPO_ID="${REPO_ID:-local/so101_real_cube_white_tray_100}"
GEN_WORKERS="${GEN_WORKERS:-10}"
GEN_RENDER="${GEN_RENDER:-auto}"
GEN_ARGS="${GEN_ARGS:---max-grasp-retries 2 --miss-prob 0.3 --miss-offset 0.018 0.026 --home-noise-deg 3}"
OUT="$(realpath -m "${OUT:-outputs/train/smolvla_real_cube_white_tray_100}")"
STEPS="${STEPS:-40000}"
DECAY_STEPS="${DECAY_STEPS:-$STEPS}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-1000}"
EVAL_TASKS="${EVAL_TASKS:-$TASKS}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_VIDEOS="${EVAL_VIDEOS:-10}"
EVAL_CKPTS="${EVAL_CKPTS:-all}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
WANDB="${WANDB:-false}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_base}"
read -r -a LAUNCHER <<< "${TRAIN_LAUNCHER:-lerobot-train}"
MERGE_FROM="${MERGE_FROM:-}"
DS="$(realpath -m "${DATASET_ROOT:-$SCRATCH/datasets/lerobot/$REPO_ID}")"

{ mkdir -p "$SCRATCH" && [ -w "$SCRATCH" ]; } 2>/dev/null \
    || { echo "SCRATCH_ROOT=$SCRATCH is not writable on $(hostname): set SCRATCH_ROOT to a large writable dir" >&2; exit 1; }
echo "==== host $(hostname) | job ${SLURM_JOB_ID:-n/a} | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) | $(date)"
avail() { local d="$1"; while [ ! -d "$d" ]; do d="$(dirname "$d")"; done; df -h --output=avail "$d" | tail -1 | tr -d " "; }
echo "free space: SCRATCH_ROOT=$SCRATCH $(avail "$SCRATCH") | OUT=$OUT $(avail "$OUT")"

# ---- 1. runtime. The lock stops two jobs from installing into the same env at once.
if [ -n "${ENV_ACTIVATE:-}" ]; then
    echo "==== activating the given environment: $ENV_ACTIVATE"
    set +u
    eval "$ENV_ACTIVATE"
    set -u
else
    mkdir -p "$HOME/.cache"
    (
        flock 9
        ENV_NAME="$ENV_NAME" SCRATCH_ROOT="$SCRATCH" OSMESA_PREFIX="$OSMESA_PREFIX" bash scripts/setup_conda_env.sh
    ) 9>"$HOME/.cache/so101lm_setup_$ENV_NAME.lock"
    set +u
    # shellcheck disable=SC1091
    source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    set -u
fi
# What scripts/env.sh sets up on Hopper: project on PYTHONPATH, caches on scratch, offline hub, small thread pools
# (OpenBLAS would start one thread per core in every worker; 2 also keeps regeneration bit-identical), node-local temp.
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HOME="$SCRATCH/cache" HF_LEROBOT_HOME="$SCRATCH/datasets/lerobot" TORCH_HOME="$SCRATCH/cache/torch" HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=2 OPENBLAS_THREAD_TIMEOUT=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="/tmp/${USER:-so101}-so101lm-${SLURM_JOB_ID:-$$}"
export TEMP="$TMPDIR" TMP="$TMPDIR"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Offscreen MuJoCo rendering: egl (GPU; the node's libEGL + NVIDIA EGL driver) or osmesa (CPU; only libOSMesa from
# the OSMesa prefix is loaded, see so101_libero/_gl.py). glvnd's libEGL needs a vendor file naming the NVIDIA driver;
# if the node has the driver but no file, point it at one (what scripts/setup_egl_shim.sh does on Hopper).
render() {
    case "$1" in
        egl) export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl; unset SO101_OSMESA_LIB ;;
        osmesa) export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa SO101_OSMESA_LIB="$OSMESA_PREFIX/lib/libOSMesa.so.8" \
                       LP_NUM_THREADS="${LP_NUM_THREADS:-2}" ;;
        *) echo "renderer must be egl or osmesa (got $1)" >&2; exit 1 ;;
    esac
    export SO101_RENDER="$1"
}
if ! ls /usr/share/glvnd/egl_vendor.d/*nvidia*.json /etc/glvnd/egl_vendor.d/*nvidia*.json >/dev/null 2>&1 \
        && [[ "$(/sbin/ldconfig -p 2>/dev/null)" == *libEGL_nvidia.so.0* ]]; then
    printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n' > "$TMPDIR/10_nvidia.json"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$TMPDIR/10_nvidia.json"
fi

# ---- 2. evaluation states: a new task has none yet. make_init_states skips tasks whose file exists; a new file is
# printed so it can be committed (it only matches the scene hash of this MuJoCo, 3.3.7).
for f in $(python -c "import sys; from so101_libero.tasks.registry import get_suite
print(' '.join(f'init_states/{t.task_id}.json' for n in sys.argv[1:] for t in get_suite(n)))" $EVAL_TASKS); do
    [ -f "$f" ] && continue
    echo "==== make the evaluation states $f (50 expert-verified starts) $(date)"
    # shellcheck disable=SC2086
    python -m so101_libero.cli.make_init_states --tasks $EVAL_TASKS
    echo "NEW FILE $PROJECT_ROOT/$f: commit it (git add $f), every later evaluation of this task uses it"
    break
done
# A different MuJoCo changes the compiled scene: evaluation would refuse the frozen states after hours of training.
python - "$EVAL_TASKS" <<'PY'
import sys

import torch
from so101_libero.cli.make_init_states import load_init_states, scene_hash
from so101_libero.tasks.registry import get_suite

assert torch.cuda.is_available(), "no CUDA device visible"
for task in (t for name in sys.argv[1].split() for t in get_suite(name)):
    if scene_hash(task) != load_init_states(task.task_id)["scene_hash"]:
        sys.exit(f"{task.task_id}: the scene differs from init_states/ (MuJoCo version?); evaluation would refuse it")
print("scene hashes match init_states/; cuda:", torch.cuda.get_device_name(0))
PY

# ---- 3. dataset
gen_render() {  # EGL aborts the interpreter in worker processes on some nodes (hopper-30): test it with spawned workers first.
    if [ "$GEN_RENDER" = auto ]; then
        render egl
        timeout 600 python tests/egl_workers_selftest.py "$GEN_WORKERS" \
            || { echo "EGL fails in worker processes on $(hostname): rendering with OSMesa (CPU) instead"; render osmesa; }
    else
        render "$GEN_RENDER"
    fi
}
generate() {  # generate TASKS REPO_ID ROOT: generated datasets that missed their demo target stop the job
    # One job at a time per dataset (e.g. the multitask job and a single-task job submitted together): the lock holder
    # generates it, a job that waited for the lock finds it complete and uses it.
    mkdir -p "$(dirname "$3")"
    exec 8>"$3.lock"
    echo "==== waiting for the generation lock of $3 (another job may be generating it) $(date)"
    flock 8
    if [ -f "$3/meta/so101_libero.json" ]; then
        echo "==== dataset $3 was generated by another job, using it"
        flock -u 8
        return
    fi
    [ -e "$3" ] && { echo "$3 is incomplete (no meta/so101_libero.json): delete it and resubmit" >&2; exit 1; }
    [ -n "${SO101_RENDER:-}" ] || gen_render
    echo "==== generate $NUM_DEMOS demos per task of [$1] -> $3 (renderer $SO101_RENDER, $GEN_WORKERS workers) $(date)"
    # shellcheck disable=SC2086
    python -m so101_libero.cli.generate --tasks $1 --num-demos "$NUM_DEMOS" --workers "$GEN_WORKERS" \
        --repo-id "$2" --root "$3" $GEN_ARGS 2>&1 | grep --line-buffered -v -e "^Svt" -e "^SVT" -e "moov atom"
    flock -u 8
}
check_target() {
    python - "$1" <<'PY'
import json, sys

m = json.load(open(sys.argv[1] + "/meta/so101_libero.json"))
short = {t: v["saved"] for t, v in m.get("stats", {}).items() if v["saved"] < m["num_demos_per_task"]}
sys.exit(f"{sys.argv[1]} missed its demo target: {short}" if short else 0)
PY
}
unset SO101_RENDER
if [ -f "$DS/meta/so101_libero.json" ]; then
    echo "==== dataset $DS exists, not regenerating"
elif [ -n "$MERGE_FROM" ]; then
    [ -e "$DS" ] && { echo "$DS is incomplete (no meta/so101_libero.json): delete it and resubmit" >&2; exit 1; }
    SRC_IDS=() SRC_ROOTS=()
    for pair in $MERGE_FROM; do
        src_task="${pair%%=*}" src_id="${pair#*=}" src_root="$SCRATCH/datasets/lerobot/${pair#*=}"
        [ -f "$src_root/meta/so101_libero.json" ] || generate "$src_task" "$src_id" "$src_root"
        check_target "$src_root"
        SRC_IDS+=("$src_id"); SRC_ROOTS+=("$src_root")
    done
    echo "==== merge [${SRC_IDS[*]}] -> $DS $(date)"
    python -m so101_libero.cli.merge_datasets --repo-ids "${SRC_IDS[@]}" --roots "${SRC_ROOTS[@]}" \
        --out-repo-id "$REPO_ID" --out-root "$DS"
else
    generate "$TASKS" "$REPO_ID" "$DS"
fi
check_target "$DS"
# A cached PASS counts only if it is for this dataset root and content (episode/frame counts) and newer than it.
CHECK="outputs/dataset_checks/$(basename "$DS")/validation.json"
if ! python - "$CHECK" "$DS" 2>/dev/null <<'PY'
import json, os, sys

check, ds = sys.argv[1:]
c, info = json.load(open(check)), json.load(open(ds + "/meta/info.json"))
ok = (c["errors"] == [] and os.path.realpath(c["root"]) == os.path.realpath(ds)
      and c["episodes"] == info["total_episodes"] and c["frames"] == info["total_frames"]
      and os.path.getmtime(check) > os.path.getmtime(ds + "/meta/info.json"))
sys.exit(0 if ok else 1)
PY
then
    echo "==== validate $DS"
    python -m so101_libero.cli.validate_dataset --repo-id "$REPO_ID" --root "$DS" --replay-episodes 10
fi

# ---- 4. training
if [ "$WANDB" = true ] && [ -z "${WANDB_MODE:-}" ]; then
    # No credentials or no route to the internet: log offline and `wandb sync` from the login node afterwards.
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs api.wandb.ai "$HOME/.netrc"; then
        export WANDB_MODE=offline; echo "no W&B credentials (wandb login or WANDB_API_KEY): W&B logs offline"
    elif curl -sS -m 20 -o /dev/null https://api.wandb.ai; then export WANDB_MODE=online
    else export WANDB_MODE=offline; echo "api.wandb.ai unreachable from $(hostname): W&B logs offline"; fi
fi
export WANDB_PROJECT="${WANDB_PROJECT:-so101-libero-mini}" WANDB_ENTITY WANDB_MODE
# lerobot-train builds its Accelerator without a precision argument: Accelerate reads it from this variable.
export ACCELERATE_MIXED_PRECISION="$PRECISION"
python -c "from accelerate import Accelerator; print('accelerate mixed precision:', Accelerator().mixed_precision)"
CONFIG="$OUT/checkpoints/last/pretrained_model/train_config.json"
if [ -f "$CONFIG" ]; then
    echo "==== resume $OUT from checkpoints/last (target $STEPS steps) $(date)"
    TRAIN=("${LAUNCHER[@]}" --resume=true --config_path="$CONFIG" --steps="$STEPS")
    # A resume keeps the checkpoint's scheduler (train_config.json "scheduler", not policy.scheduler_decay_steps):
    # extending past it would run every extra step at the LR floor, so stretch the decay when DECAY_STEPS differs.
    SAVED_DECAY=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['scheduler']['num_decay_steps'])" "$CONFIG")
    if [ "$SAVED_DECAY" != "$DECAY_STEPS" ]; then
        echo "LR decay: checkpoint schedule $SAVED_DECAY steps -> $DECAY_STEPS"
        TRAIN+=(--scheduler.num_decay_steps="$DECAY_STEPS")
    fi
else
    [ -e "$OUT" ] && { echo "$OUT exists but has no checkpoint to resume: remove it" >&2; exit 1; }
    mkdir -p "$(dirname "$OUT")"
    echo "==== train SmolVLA ($PRETRAINED) on $REPO_ID -> $OUT ($STEPS steps, LR decay over $DECAY_STEPS," \
         "batch $BATCH_SIZE, $PRECISION, ${LAUNCHER[*]}) $(date)"
    # scripts/train.sh's smolvla preset; front/wrist cameras are renamed to smolvla_base's camera1/camera2
    TRAIN=("${LAUNCHER[@]}"
        --dataset.repo_id="$REPO_ID" --dataset.root="$DS" --dataset.video_backend=pyav
        --policy.path="$PRETRAINED" --policy.scheduler_decay_steps="$DECAY_STEPS" --policy.push_to_hub=false
        --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
        --output_dir="$OUT" --job_name="$(basename "$OUT")"
        --steps="$STEPS" --batch_size="$BATCH_SIZE" --save_freq="$SAVE_FREQ" --log_freq=200
        --num_workers="$NUM_WORKERS" --seed="$SEED" --eval_freq=0
        --wandb.enable="$WANDB" --wandb.disable_artifact=true --wandb.project="$WANDB_PROJECT")
    if [ -n "${WANDB_ENTITY:-}" ]; then TRAIN+=(--wandb.entity="$WANDB_ENTITY"); fi
fi
if [ -n "${WANDB_MODE:-}" ]; then TRAIN+=(--wandb.mode="$WANDB_MODE"); fi
# shellcheck disable=SC2206
TRAIN+=(${TRAIN_ARGS:-})
"${TRAIN[@]}"

# ---- 5. evaluation (a failed evaluation is reported, the others still run)
set +e
# EGL contexts next to CUDA abort on some nodes; OSMesa and EGL images differ by < 1/255.
if [ "${EVAL_EGL:-0}" = 1 ]; then render egl; else render osmesa; fi
ckpt_name() { if [ "$1" = last ]; then echo "last_$(tr -dc 0-9 < "$OUT/checkpoints/last/training_state/training_step.json")"; else echo "$1"; fi; }
case "$EVAL_CKPTS" in
    all) CKPTS=$(find "$OUT/checkpoints" -mindepth 1 -maxdepth 1 -type d ! -name last -printf '%f\n' | sort) ;;
    none) CKPTS="" ;;
    *) CKPTS=last ;;
esac
echo "==== evaluate [${CKPTS//$'\n'/ }] on [$EVAL_TASKS], $EVAL_EPISODES episodes each, renderer $SO101_RENDER $(date)"
FAILED=0
for c in $CKPTS; do
    name=$(ckpt_name "$c")
    python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/eval/$name/results.json')).get('complete', True) else 1)" \
        2>/dev/null && { echo "eval $name exists"; continue; }
    # shellcheck disable=SC2086
    python -m so101_libero.cli.evaluate --policy "$OUT/checkpoints/$c/pretrained_model" --tasks $EVAL_TASKS \
        --episodes "$EVAL_EPISODES" --videos "$EVAL_VIDEOS" --workers "$EVAL_WORKERS" --out "$OUT/eval/$name" \
        ${EVAL_ARGS:-} > "$OUT/eval_$name.log" 2>&1
    status=$?
    grep -E "WARNING|Error" "$OUT/eval_$name.log" | head -5
    tail -n 25 "$OUT/eval_$name.log"
    [ "$status" -eq 0 ] || { echo "EVALUATION OF $c FAILED (exit $status), log $OUT/eval_$name.log"; FAILED=1; }
done
echo "==== success rates"
python -m so101_libero.cli.summarize "$OUT"/eval/*/results.json 2>/dev/null
# Post the success rates + videos to the training run. Offline, W&B would open a second offline run with the same
# id, so the evaluation is posted after `wandb sync` from the login node instead (commands printed below).
if [ "$WANDB" = true ] && [ -n "$CKPTS" ]; then
    if [ "${WANDB_MODE:-online}" = offline ]; then
        echo "==== W&B was offline. From the login node (conda activate $ENV_NAME, in $PROJECT_ROOT):"
        echo "  wandb sync $OUT/wandb/offline-run-*"
    fi
    for c in $CKPTS; do
        name=$(ckpt_name "$c")
        [ -f "$OUT/eval/$name/results.json" ] || continue
        if [ "${WANDB_MODE:-online}" = offline ]; then
            echo "  WANDB_PROJECT=$WANDB_PROJECT WANDB_ENTITY=${WANDB_ENTITY:-} python scripts/wandb_log_eval.py" \
                 "--run-dir $OUT --results $OUT/eval/$name/results.json --label $name"
        else
            python scripts/wandb_log_eval.py --run-dir "$OUT" --results "$OUT/eval/$name/results.json" --label "$name" \
                --videos 4 || echo "wandb eval logging failed (non-fatal)"
        fi
    done
fi
echo "==== DONE $(date)"
exit "$FAILED"
