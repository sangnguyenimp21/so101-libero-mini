#!/usr/bin/env bash
# Run a command inside the project runtime: the Singularity PyTorch image + the lerobot042-act venv,
# with caches on /scratch and offscreen MuJoCo rendering (EGL on GPU nodes).
#
#   bash scripts/env.sh python -m so101_libero.cli.check_env
#   bash scripts/env.sh                       # interactive shell in the runtime
#
# Rendering (SO101_RENDER=auto|egl|osmesa|none): auto = EGL on GPU nodes, OSMesa (CPU, scripts/setup_osmesa.sh)
# elsewhere. With OSMesa, import so101_libero before mujoco (it preloads libOSMesa; see so101_libero/_gl.py).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
IMAGE="${SO101_IMAGE:-/app1/common/singularity-img/hopper/pytorch/pytorch_2.7.0_cuda_12.8.sif}"
VENV="${SO101_VENV:-$SCRATCH/virtualenvs/lerobot042-act}"

if [ -z "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then
    if ! command -v singularity >/dev/null 2>&1; then
        set +u
        # shellcheck disable=SC1091
        [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
        module load singularity >/dev/null 2>&1 || true
        set -u
    fi
    command -v singularity >/dev/null 2>&1 || { echo "singularity not found (module load singularity)" >&2; exit 1; }
    NV=()
    [ -e /dev/nvidia0 ] && NV=(--nv)
    FWD=()
    for v in SCRATCH_ROOT SO101_VENV NCPUS PBS_JOBID WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT HF_HUB_OFFLINE \
             CUDA_VISIBLE_DEVICES SO101_RENDER SO101_OSMESA_LIB LP_NUM_THREADS SO101_TMPDIR USER PYTHONFAULTHANDLER MUJOCO_EGL_DEVICE_ID SO101_OVERLAY SO101_TORCH_THREADS \
             OPENBLAS_NUM_THREADS OPENBLAS_THREAD_TIMEOUT OMP_NUM_THREADS MKL_NUM_THREADS; do
        [ -n "${!v+x}" ] && FWD+=(--env "$v=${!v}")
    done
    exec singularity exec -e "${NV[@]}" "${FWD[@]}" "$IMAGE" bash --norc "$PROJECT_ROOT/scripts/env.sh" "$@"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Optional overlay with the packages SmolVLA needs that the venv lacks (transformers, tokenizers, num2words,
# docopt; scripts/setup_smolvla.sh). It contains nothing the venv already has, so it shadows nothing.
OVERLAY="${SO101_OVERLAY:-$SCRATCH/opt/so101_overlay}"
[ -d "$OVERLAY/transformers" ] && export PYTHONPATH="$PYTHONPATH:$OVERLAY"
export HF_HOME="$SCRATCH/cache" HF_LEROBOT_HOME="$SCRATCH/datasets/lerobot"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCH_HOME="$SCRATCH/cache/torch"
# Thread pools: numpy's OpenBLAS would start 64 threads per process (heavy spinning with many workers), and its
# 1-thread solve path rounds differently from the multi-thread one used to generate the datasets; 2 threads keeps
# regeneration bit-identical and the CPU cost low. (A 1-CPU cpuset forces 1 thread whatever is set here.)
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OPENBLAS_THREAD_TIMEOUT="${OPENBLAS_THREAD_TIMEOUT:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
# Node-local temp dir: on NFS, multiprocessing's cleanup trips over .nfs* files (harmless but noisy tracebacks).
export TMPDIR="${SO101_TMPDIR:-/tmp/${USER:-so101}-so101}"
export TEMP="$TMPDIR" TMP="$TMPDIR"
mkdir -p "$TMPDIR"
# Offscreen OpenGL for MuJoCo. SO101_RENDER=auto (default) picks egl on GPU nodes, else osmesa.
#  egl:    the image has no EGL libraries; --nv binds the host's libEGL + NVIDIA EGL driver but not
#          libGLdispatch or the EGL vendor file, so $SCRATCH/opt/egl_shim provides them (scripts/setup_egl_shim.sh).
#  osmesa: CPU rendering (Mesa llvmpipe) from a separate conda prefix, scripts/setup_osmesa.sh. Only
#          libOSMesa is loaded (so101_libero/_gl.py preloads $SO101_OSMESA_LIB), nothing is added
#          to LD_LIBRARY_PATH, so the container's own libraries are untouched.
EGL_SHIM="$SCRATCH/opt/egl_shim"
OSMESA_LIB="${SO101_OSMESA_LIB:-$SCRATCH/opt/osmesa/lib/libOSMesa.so.8}"
RENDER="${SO101_RENDER:-auto}"
if [ "$RENDER" = auto ]; then
    if [ -e /dev/nvidia0 ] && [ -f "$EGL_SHIM/10_nvidia.json" ]; then RENDER=egl
    elif [ -f "$OSMESA_LIB" ]; then RENDER=osmesa
    else RENDER=none; fi
fi
case "$RENDER" in
    egl)
        export LD_LIBRARY_PATH="$EGL_SHIM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_SHIM/10_nvidia.json" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ;;
    osmesa)
        # llvmpipe defaults to one thread per core; 4 is ~as fast for 256x256 frames and leaves cores free
        export SO101_OSMESA_LIB="$OSMESA_LIB" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS="${LP_NUM_THREADS:-4}" ;;
    none) unset MUJOCO_GL ;;
    *) echo "SO101_RENDER must be auto, egl, osmesa or none (got $RENDER)" >&2; exit 1 ;;
esac
export SO101_RENDER="$RENDER"
cd "$PROJECT_ROOT"
if [ $# -eq 0 ]; then exec bash --norc -i; else exec "$@"; fi
