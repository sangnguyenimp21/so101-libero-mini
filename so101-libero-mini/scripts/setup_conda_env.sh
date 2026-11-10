#!/usr/bin/env bash
# One-time (needs internet): a conda runtime for clusters without Hopper's Singularity image + venv, used by the
# SLURM jobs in slurm/. Idempotent: finished steps are skipped, so every job calls it (under a lock).
#  1. conda env $ENV_NAME [so101lm], conda-forge only: python 3.10 + ffmpeg 7.1.1, then from pip lerobot[smolvla]
#     0.4.2 (the dataset format and API this project is written for; it brings torch 2.7.1), transformers 4.57.6 (as
#     scripts/setup_smolvla.sh pins on Hopper) and mujoco 3.3.7. That MuJoCo reproduces the scene hashes in
#     init_states/; newer ones write different MJCF (evaluation then refuses the frozen states) and drop
#     mjENBL_MULTICCD, which sim/scene.py sets.
#  2. $OSMESA_PREFIX [$SCRATCH/opt/osmesa; $SCRATCH = $SCRATCH_ROOT, default ~/so101lm]: conda-forge mesalib 25.0.4
#     (the last series with OSMesa), the CPU-rendering fallback, in its own prefix as in scripts/setup_osmesa.sh: only
#     libOSMesa is loaded (so101_libero/_gl.py).
#  3. lerobot/smolvla_base and its VLM backbone into $HF_HOME ($SCRATCH/cache); the jobs run with HF_HUB_OFFLINE=1.
#   bash scripts/setup_conda_env.sh        (overrides: ENV_NAME, SCRATCH_ROOT, OSMESA_PREFIX, CONDA_ROOT [~/miniconda3])
set -euo pipefail
ENV_NAME="${ENV_NAME:-so101lm}"
SCRATCH="${SCRATCH_ROOT:-$HOME/so101lm}"
OSMESA_PREFIX="${OSMESA_PREFIX:-$SCRATCH/opt/osmesa}"
export PIP_CACHE_DIR="$SCRATCH/pip-cache" HF_HOME="$SCRATCH/cache" HF_HUB_OFFLINE=0
CF=(--override-channels -c conda-forge)  # conda-forge only: no Anaconda channel terms to accept
set +u  # conda's shell functions and activation scripts read unset variables
# shellcheck disable=SC1091
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then  # not -q: no SIGPIPE under pipefail
    echo "[setup] creating conda env $ENV_NAME"
    conda create -y -q -n "$ENV_NAME" "${CF[@]}" python=3.10 ffmpeg=7.1.1
fi
conda activate "$ENV_NAME"
set -u
# The version check also catches an env left half-installed by an interrupted setup.
if ! python -c "
import importlib.metadata as m, num2words, threadpoolctl
assert (m.version('lerobot'), m.version('transformers'), m.version('mujoco')) == ('0.4.2', '4.57.6', '3.3.7')
" 2>/dev/null; then
    echo "[setup] installing lerobot[smolvla] 0.4.2, transformers 4.57.6, mujoco 3.3.7 into $ENV_NAME"
    pip install -q "lerobot[smolvla]==0.4.2" "transformers==4.57.6" "mujoco==3.3.7" threadpoolctl pytest
fi

if [ ! -f "$OSMESA_PREFIX/lib/libOSMesa.so.8" ]; then
    echo "[setup] installing OSMesa into $OSMESA_PREFIX"
    conda create -y -q -p "$OSMESA_PREFIX" "${CF[@]}" mesalib=25.0.4
fi

python - <<'PY'
from huggingface_hub import snapshot_download

skip = ["onnx/*", "*.onnx"]  # SmolVLM2's ONNX exports (4.8 GB): SmolVLA loads model.safetensors only
for repo in ("lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    try:
        path = snapshot_download(repo, local_files_only=True, ignore_patterns=skip)
    except Exception:  # noqa: BLE001  (not cached yet)
        path = snapshot_download(repo, ignore_patterns=skip)
    print("[setup]", repo, "->", path)
PY
python -c "import importlib.metadata as m, torch; print('[setup] ready: lerobot', m.version('lerobot'), '| torch',
torch.__version__, '| mujoco', m.version('mujoco'), '| transformers', m.version('transformers'))"
