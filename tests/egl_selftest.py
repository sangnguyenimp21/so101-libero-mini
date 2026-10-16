"""GPU-node check that EGL rendering survives CUDA work in the same process (it aborts on some node setups).

    bash scripts/env.sh python tests/egl_selftest.py && echo EGL_OK
Run it as its own process: an EGL failure aborts the interpreter, which a caller can only see as an exit code.
"""
import so101_libero  # noqa: F401
import torch

from so101_libero.env import EnvConfig, SO101TabletopEnv

env = SO101TabletopEnv("cylinder_to_tray", EnvConfig())
obs, _ = env.reset(seed=0)
x = torch.randn(1024, 1024, device="cuda")
for _ in range(20):
    x = (x @ x).clamp(-1, 1)  # CUDA work between renders, like a policy rollout
    obs, *_ = env.step(obs["observation.state"])
torch.cuda.synchronize()
env.close()
print("EGL + CUDA self-test passed")
