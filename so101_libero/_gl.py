"""Offscreen OpenGL setup that has to happen before the first ``import mujoco``.

``scripts/env.sh`` sets ``MUJOCO_GL`` (``egl`` on GPU nodes, ``osmesa`` elsewhere). For OSMesa it points
``SO101_OSMESA_LIB`` at a conda-forge libOSMesa instead of adding that prefix to ``LD_LIBRARY_PATH`` (which
would shadow the container's own libraries). Loading it here by absolute path, with RTLD_GLOBAL, lets
PyOpenGL's later ``dlopen("libOSMesa.so.8")`` find the already-loaded library by its soname.
"""

import ctypes
import os
import sys

if os.environ.get("MUJOCO_GL") == "osmesa" and os.environ.get("SO101_OSMESA_LIB"):
    if "mujoco" in sys.modules:
        raise ImportError("import so101_libero before mujoco when MUJOCO_GL=osmesa (libOSMesa must load first)")
    ctypes.CDLL(os.environ["SO101_OSMESA_LIB"], mode=ctypes.RTLD_GLOBAL)
