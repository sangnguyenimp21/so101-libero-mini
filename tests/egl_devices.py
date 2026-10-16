"""Diagnose EGL devices vs the CUDA device of this job (GPU nodes): which EGL device is our GPU?"""
import ctypes
import os
import subprocess

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), " MUJOCO_EGL_DEVICE_ID =",
      os.environ.get("MUJOCO_EGL_DEVICE_ID"))
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
import so101_libero  # noqa: F401,E402
from mujoco.egl import egl_ext as EGL  # noqa: E402

devices = EGL.eglQueryDevicesEXT()
print("EGL devices:", len(devices))
try:
    from OpenGL.EGL.EXT.device_query import eglQueryDeviceAttribEXT
    from OpenGL.EGL.NV.device_cuda import EGL_CUDA_DEVICE_NV
except Exception as e:  # pragma: no cover
    eglQueryDeviceAttribEXT = None
    print("no device_query binding:", e)
for i, dev in enumerate(devices):
    cuda = None
    if eglQueryDeviceAttribEXT is not None:
        try:
            val = ctypes.c_ssize_t(-1)
            ok = eglQueryDeviceAttribEXT(dev, EGL_CUDA_DEVICE_NV, ctypes.byref(val))
            cuda = val.value if ok else f"query failed ({EGL.eglGetError():#x})"
        except Exception as e:
            cuda = f"error {type(e).__name__}: {e}"
    disp = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, dev, None)
    init = None
    try:
        init = bool(EGL.eglInitialize(disp, None, None))
    except Exception as e:
        init = f"error {type(e).__name__}"
    print(f"  EGL device {i}: cuda_device_attr={cuda} initializes={init}")
import torch  # noqa: E402

print("torch cuda:", torch.cuda.is_available(), torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
