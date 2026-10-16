#!/usr/bin/env bash
# One-time: what EGL rendering inside the Singularity image needs on GPU nodes. `--nv` binds the host's libEGL and
# NVIDIA EGL driver but not libGLdispatch.so.0 (glvnd) or an EGL vendor file; this puts copies in $SCRATCH/opt/egl_shim,
# which scripts/env.sh uses automatically on GPU nodes.
#   bash scripts/setup_egl_shim.sh      (any Hopper node with /usr/lib64/libGLdispatch.so.0, e.g. the login node)
set -euo pipefail
SHIM="${SCRATCH_ROOT:-/scratch/e1583535}/opt/egl_shim"
if [ -f "$SHIM/10_nvidia.json" ] && [ -e "$SHIM/lib/libGLdispatch.so.0" ]; then echo "EGL shim already at $SHIM"; exit 0; fi
[ -e /usr/lib64/libGLdispatch.so.0 ] || { echo "host /usr/lib64/libGLdispatch.so.0 not found" >&2; exit 1; }
mkdir -p "$SHIM/lib"
cp -aL /usr/lib64/libGLdispatch.so.0 "$SHIM/lib/libGLdispatch.so.0"
printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : { "library_path" : "libEGL_nvidia.so.0" }\n}\n' > "$SHIM/10_nvidia.json"
echo "EGL shim written to $SHIM"
