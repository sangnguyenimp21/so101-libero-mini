#!/usr/bin/env bash
# One-time: CPU offscreen rendering (Mesa llvmpipe / OSMesa) for nodes without a GPU (login node, CPU jobs).
# Installs conda-forge mesalib into its own prefix ($SCRATCH/opt/osmesa); the venv is not touched.
#   bash scripts/setup_osmesa.sh
set -euo pipefail
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
PREFIX="${OSMESA_PREFIX:-$SCRATCH/opt/osmesa}"
MAMBA="${MAMBA:-$SCRATCH/opt/micromamba/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$SCRATCH/opt/micromamba/root}"
if [ -f "$PREFIX/lib/libOSMesa.so.8" ]; then echo "OSMesa already installed: $PREFIX/lib/libOSMesa.so.8"; exit 0; fi
# mesalib 25.0.x is the last series with the OSMesa frontend (removed in Mesa 25.1).
"$MAMBA" create -y -q -p "$PREFIX" -c conda-forge "mesalib=25.0.4"
ls -l "$PREFIX/lib/libOSMesa.so.8"
