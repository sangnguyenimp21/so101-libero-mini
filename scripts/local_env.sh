#!/usr/bin/env bash
# Run a command in the project runtime on a workstation / GPU server: no SLURM, no Singularity. The counterpart of
# scripts/env.sh (Hopper image) for machines that use the conda env made by scripts/setup_conda_env.sh
# (lerobot[smolvla] 0.4.2, transformers 4.57.6, mujoco 3.3.7, OSMesa, smolvla_base weights).
#
#   bash scripts/local_env.sh python -m so101_libero.cli.check_env
#   bash scripts/local_env.sh                      # interactive shell in the runtime
#   source scripts/local_env.sh                    # from another bash script: sets up this shell instead
#
# Settings (environment, defaults in brackets):
#   ENV_NAME [so101lm]  CONDA_ROOT [~/miniconda3]  SCRATCH_ROOT [~/so101lm: HF cache, datasets, OSMesa]
#   ENV_ACTIVATE        a shell command that activates another environment instead of the conda env
#   SO101_RENDER [auto | egl | osmesa | none]: offscreen MuJoCo rendering. auto = egl if the NVIDIA EGL driver is
#                       present (~5 ms per control step with both cameras on an RTX A5000, vs ~110-150 ms with osmesa),
#                       else osmesa (CPU). EGL next to CUDA aborted on some cluster nodes (README); if that happens here,
#                       use SO101_RENDER=osmesa. Evaluate with the renderer the training data was made with (evaluate
#                       warns otherwise; the two differ by < 1/255 per pixel on average).
# First-time setup (needs internet, ~15 min, ~10 GB):  bash scripts/setup_conda_env.sh
# Datasets: $SCRATCH_ROOT/datasets/lerobot/<repo-id> (= $HF_LEROBOT_HOME).

_so101_sourced=0
(return 0 2>/dev/null) && _so101_sourced=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/so101lm}"
ENV_NAME="${ENV_NAME:-so101lm}"
OSMESA_PREFIX="${OSMESA_PREFIX:-$SCRATCH_ROOT/opt/osmesa}"

if [ -n "${ENV_ACTIVATE:-}" ]; then
    # "conda activate <env>" needs conda's shell functions, which a non-interactive bash does not have: load them first
    _conda_sh="${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    _so101_nounset=0; case "$-" in *u*) _so101_nounset=1; set +u ;; esac
    # shellcheck disable=SC1090
    [ -f "$_conda_sh" ] && source "$_conda_sh"
    eval "$ENV_ACTIVATE"
    [ "$_so101_nounset" = 1 ] && set -u
elif [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    _conda_sh="${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    if [ ! -f "$_conda_sh" ]; then
        echo "conda not found at ${CONDA_ROOT:-$HOME/miniconda3} (set CONDA_ROOT or ENV_ACTIVATE)" >&2
        if [ "$_so101_sourced" = 1 ]; then return 1; else exit 1; fi
    fi
    # conda's shell functions read unset variables
    _so101_nounset=0; case "$-" in *u*) _so101_nounset=1; set +u ;; esac
    # shellcheck disable=SC1090
    source "$_conda_sh"
    if ! conda activate "$ENV_NAME" 2>/dev/null; then
        echo "conda env '$ENV_NAME' missing: run  bash scripts/setup_conda_env.sh  first" >&2
        if [ "$_so101_sourced" = 1 ]; then return 1; else exit 1; fi
    fi
    [ "$_so101_nounset" = 1 ] && set -u
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export HF_HOME="$SCRATCH_ROOT/cache" HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$SCRATCH_ROOT/datasets/lerobot}"
export TORCH_HOME="$SCRATCH_ROOT/cache/torch" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Small BLAS pools: numpy would start one OpenBLAS thread per core in every worker process; 2 threads also keep
# dataset regeneration bit-identical (README, Reproducibility).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OPENBLAS_THREAD_TIMEOUT="${OPENBLAS_THREAD_TIMEOUT:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TMPDIR="${SO101_TMPDIR:-/tmp/${USER:-so101}-so101lm}"
export TEMP="$TMPDIR" TMP="$TMPDIR"
mkdir -p "$TMPDIR"

_render="${SO101_RENDER:-auto}"
if [ "$_render" = auto ]; then
    if [[ "$(/sbin/ldconfig -p 2>/dev/null)" == *libEGL_nvidia.so.0* ]]; then _render=egl
    elif [ -f "$OSMESA_PREFIX/lib/libOSMesa.so.8" ]; then _render=osmesa
    else _render=none; fi
fi
case "$_render" in
    egl)
        export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
        unset SO101_OSMESA_LIB
        # glvnd's libEGL needs a vendor file naming the NVIDIA driver; point it at one if the system has none.
        if ! compgen -G "/usr/share/glvnd/egl_vendor.d/*nvidia*.json" >/dev/null \
                && ! compgen -G "/etc/glvnd/egl_vendor.d/*nvidia*.json" >/dev/null; then
            printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n' \
                > "$TMPDIR/10_nvidia.json"
            export __EGL_VENDOR_LIBRARY_FILENAMES="$TMPDIR/10_nvidia.json"
        fi ;;
    osmesa)
        if [ ! -f "$OSMESA_PREFIX/lib/libOSMesa.so.8" ]; then
            echo "OSMesa missing at $OSMESA_PREFIX (bash scripts/setup_conda_env.sh installs it)" >&2
            if [ "$_so101_sourced" = 1 ]; then return 1; else exit 1; fi
        fi
        # llvmpipe would start one thread per core per process; 4 is about as fast for 256x256 frames
        export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa SO101_OSMESA_LIB="$OSMESA_PREFIX/lib/libOSMesa.so.8" \
            LP_NUM_THREADS="${LP_NUM_THREADS:-4}" ;;
    none) unset MUJOCO_GL ;;
    *) echo "SO101_RENDER must be osmesa, egl, auto or none (got $_render)" >&2
       if [ "$_so101_sourced" = 1 ]; then return 1; else exit 1; fi ;;
esac
export SO101_RENDER="$_render"

if [ "$_so101_sourced" = 0 ]; then
    cd "$PROJECT_ROOT"
    if [ $# -eq 0 ]; then exec bash --norc -i; else exec "$@"; fi
fi
