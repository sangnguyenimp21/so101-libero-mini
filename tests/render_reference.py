"""Render fixed states with the active backend (MUJOCO_GL) to compare OSMesa (CPU) and EGL (GPU) images.

    bash scripts/env.sh python tests/render_reference.py            # writes outputs/render_ref/<backend>.npz
    bash scripts/env.sh python tests/render_reference.py --compare  # pixel statistics between saved backends
"""
import argparse
import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs" / "render_ref"

ap = argparse.ArgumentParser()
ap.add_argument("--compare", action="store_true")
args = ap.parse_args()
if args.compare:
    a, b = np.load(OUT / "osmesa.npz"), np.load(OUT / "egl.npz")
    for k in a.files:
        d = np.abs(a[k].astype(np.int16) - b[k].astype(np.int16))
        print(f"{k:24s} mean |diff| {d.mean():6.2f}/255  p99 {np.percentile(d, 99):5.1f}  "
              f"pixels differing >16: {100 * (d.max(-1) > 16).mean():5.2f}%")
    raise SystemExit
import so101_libero  # noqa: E402,F401
from so101_libero.env import EnvConfig, SO101TabletopEnv  # noqa: E402
from so101_libero.expert.pick_place import PickPlaceExpert  # noqa: E402
from so101_libero.expert.rollout import expert_rng  # noqa: E402

env = SO101TabletopEnv("cylinder_to_tray", EnvConfig())
frames = {}
for seed in (0, 1):
    obs, _ = env.reset(seed=seed)
    ex = PickPlaceExpert(env, env.task.plan, expert_rng(seed))
    for t in range(120):
        if t in (0, 60, 119):
            for cam in ("front", "wrist"):
                frames[f"s{seed}_t{t}_{cam}"] = env.render_camera(cam)
        env.step(ex.act())
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"{os.environ.get('MUJOCO_GL')}.npz", **frames)
env.close()
print("saved", OUT / f"{os.environ.get('MUJOCO_GL')}.npz")
