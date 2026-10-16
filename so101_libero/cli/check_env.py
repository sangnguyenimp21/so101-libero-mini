"""Validate the runtime: packages, GPU/CUDA, offscreen rendering, scene, IK, LeRobot APIs.

    bash scripts/env.sh python -m so101_libero.cli.check_env [--out outputs/check_env]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/check_env"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report, problems = {}, []

    import so101_libero  # noqa: F401  (GL setup)

    print("== packages")
    for name in ("mujoco", "lerobot", "torch", "torchvision", "gymnasium", "numpy", "av", "datasets", "PIL"):
        try:
            v = getattr(importlib.import_module(name), "__version__", "?")
        except Exception as e:  # noqa: BLE001
            v = f"MISSING ({e})"
            problems.append(f"{name} missing")
        report[name] = v
        print(f"   {name:12s} {v}")
    for opt in ("transformers",):
        try:
            importlib.import_module(opt)
            print(f"   {opt:12s} available (SmolVLA possible)")
        except Exception:
            print(f"   {opt:12s} not installed (only needed for SmolVLA)")

    print("== GPU / CUDA")
    import torch

    cuda = torch.cuda.is_available()
    report["cuda"] = cuda
    print(f"   cuda available: {cuda}" + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} device(s))"
                                          if cuda else " (training needs a GPU node: qsub pbs/*.pbs)"))

    print("== rendering")
    backend = os.environ.get("MUJOCO_GL")
    report["render_backend"] = backend
    print(f"   MUJOCO_GL={backend} (SO101_RENDER={os.environ.get('SO101_RENDER')})")
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    if backend:
        try:
            env = SO101TabletopEnv("cylinder_to_tray", EnvConfig())
            obs, _ = env.reset(seed=0)
            t = time.time()
            for _ in range(5):
                env.render()
            ms = (time.time() - t) / 5 * 1e3
            from PIL import Image

            Image.fromarray(env.render()).save(args.out / "front_wrist.png")
            env.close()
            print(f"   rendered front+wrist {obs['observation.images.front'].shape}: {ms:.1f} ms per pair "
                  f"-> {args.out / 'front_wrist.png'}")
            report["render_ms_per_step"] = round(ms, 1)
        except Exception as e:  # noqa: BLE001
            problems.append(f"rendering failed: {e}")
            print(f"   FAIL rendering: {e}")
    else:
        print("   rendering disabled (no GPU and no OSMesa: run scripts/setup_osmesa.sh)")

    print("== simulation + IK")
    import numpy as np

    from so101_libero.expert.ik import IKSolver, top_down_frame

    env = SO101TabletopEnv("cylinder_to_tray", EnvConfig(render=False))
    env.reset(seed=0)
    ik = IKSolver(env.h)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(20):
        r, a = rng.uniform(0.13, 0.22), rng.uniform(-1.0, 1.0)
        target = np.array([ik.pan_axis_xy[0] + r * np.cos(a), r * np.sin(a), rng.uniform(0.02, 0.06)])
        res = ik.solve_multi(target, top_down_frame(a), [np.r_[a, -0.5, 0.7, 1.2, 0.0]])
        pos, _ = ik.fk(res.q)
        errs.append(np.linalg.norm(pos - target) if res.success else np.inf)
    worst = float(np.max(errs))
    report["ik_worst_err_mm"] = worst * 1e3
    print(f"   timestep {env.model.opt.timestep * 1e3:.3f} ms, {env.cfg.substeps} substeps per 30 Hz step; "
          f"IK 20 random top-down targets: worst FK error {worst * 1e3:.3f} mm")
    if worst > 1e-3:
        problems.append("IK self-test failed")

    print("== LeRobot")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        from lerobot.policies.factory import make_pre_post_processors  # noqa: F401

        print("   dataset + policy factory imports ok")
    except Exception as e:  # noqa: BLE001
        problems.append(f"lerobot import: {e}")
    hub = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser() / "hub" / "checkpoints"
    has_r18 = any(p.name.startswith("resnet18") for p in hub.glob("*.pth"))
    print(f"   ImageNet resnet18 for ACT cached in {hub}: {has_r18}")
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        print("   torchcodec importable")
    except Exception:
        print("   torchcodec unusable here -> datasets are read with video_backend=pyav (handled by the scripts)")
    print(f"   HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} HF_LEROBOT_HOME={os.environ.get('HF_LEROBOT_HOME')}")

    report["problems"] = problems
    (args.out / "check_env.json").write_text(json.dumps(report, indent=2, default=str))
    print("== result:", "OK" if not problems else f"PROBLEMS: {problems}")


if __name__ == "__main__":
    main()
