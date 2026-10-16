#!/usr/bin/env bash
# One-time (login node, needs internet): what SmolVLA needs on top of the lerobot042-act venv, without touching it.
#  1. transformers/tokenizers/num2words/docopt into an overlay dir ($SCRATCH/opt/so101_overlay) that scripts/env.sh
#     appends to PYTHONPATH. --no-deps + pinned versions: nothing already in the venv is installed, upgraded or
#     shadowed (`pip install lerobot[smolvla]` would pull numpy 2 in via rerun-sdk; do not use it).
#  2. the pretrained weights into the HF cache: lerobot/smolvla_base and its VLM backbone.
#   bash scripts/setup_smolvla.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
OVERLAY="${SO101_OVERLAY:-$SCRATCH/opt/so101_overlay}"
SO101_RENDER=none HF_HUB_OFFLINE=0 bash "$ROOT/scripts/env.sh" bash -c "
unset PIP_CONSTRAINT  # the container pins packaging==23.2, which conflicts with lerobot
PIP_CACHE_DIR=$SCRATCH/pip-cache pip install -q --no-deps --target '$OVERLAY' \
    'transformers==4.57.6' 'tokenizers==0.22.2' 'num2words==0.5.14' 'docopt==0.6.2'
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ('lerobot/smolvla_base', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'):
    print(repo, '->', snapshot_download(repo))
PY"
