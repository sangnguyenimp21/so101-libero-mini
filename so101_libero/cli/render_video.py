"""Render scripted-expert episodes to MP4 (front | wrist side by side) plus a key-frame contact sheet.

    bash scripts/env.sh python -m so101_libero.cli.render_video --task cylinder_to_tray --seeds 0 1 2 --out docs/media
    python -m so101_libero.cli.render_video --task cube_flip --seeds 1400000 --placement 0 --out docs/media/tri_tasks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import so101_libero  # noqa: F401  (GL setup before mujoco)
import numpy as np
from PIL import Image

from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.expert.rollout import expert_rng, make_expert
from so101_libero.utils.video import label, write_mp4


def render_expert(task: str, seed: int, out_dir: Path, size: int, max_steps: int = 900,
                  placement: int | None = None) -> dict:
    env = SO101TabletopEnv(task, EnvConfig(image_size=(size, size), terminate_on_success=False, max_steps=max_steps))
    env.reset(seed=seed, options={"placement_id": placement} if placement is not None else None)
    expert = make_expert(env, env.task.plan, expert_rng(seed))
    frames, keyframes, last_phase = [], [], None
    info = {"is_success": False}
    while True:
        frame = env.render()
        if expert.phase != last_phase:
            keyframes.append(label(frame, expert.phase))
            last_phase = expert.phase
        frames.append(label(frame, f"{env.task.task_id} seed {seed} | {expert.phase} | success={info['is_success']}"))
        action = expert.act()
        if action is None:
            break
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    frames.append(label(env.render(), f"{env.task.task_id} seed {seed} | end | success={info['is_success']}"))
    keyframes.append(label(env.render(), f"end success={info['is_success']}"))
    stem = f"expert_{task}_seed{seed}" + (f"_p{placement}" if placement is not None else "")
    write_mp4(frames, out_dir / f"{stem}.mp4", fps=env.cfg.control_hz)
    sheet = np.concatenate([np.concatenate(keyframes[i:i + 2], 1) if i + 1 < len(keyframes) else
                            np.concatenate([keyframes[i], np.zeros_like(keyframes[i])], 1)
                            for i in range(0, len(keyframes), 2)], 0)
    Image.fromarray(sheet).save(out_dir / f"{stem}_keyframes.png")
    env.close()
    return {"task": task, "seed": seed, "placement": placement, "success": bool(info["is_success"]), "failure": expert.failure,
            "steps": len(frames) - 1, "video": str(out_dir / f"{stem}.mp4")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="cylinder_to_tray")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/videos"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--placement", type=int, default=None, help="predefined placement id (so101_tri tasks)")
    ap.add_argument("--max-steps", type=int, default=900)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(json.dumps(render_expert(args.task, seed, args.out, args.size, max_steps=args.max_steps,
                                       placement=args.placement)))


if __name__ == "__main__":
    main()
