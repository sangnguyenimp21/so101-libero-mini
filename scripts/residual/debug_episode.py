"""Run one frozen cube_flip episode in-process with visible progress.

This diagnostic avoids the evaluator's worker process so model-loading and
rollout stalls can be located from the terminal output.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model"),
    )
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(
        f"startup: MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
        f"HF_HOME={os.environ.get('HF_HOME')} device={args.device}",
        flush=True,
    )

    import so101_libero  # noqa: F401  (must precede MuJoCo under OSMesa)
    from so101_libero.cli.make_init_states import load_init_states
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.policy.runner import PolicyRunner
    from so101_libero.tasks.registry import get_task

    task = get_task("cube_flip")
    state = load_init_states(task.task_id)["states"][args.state_index]

    print("loading policy...", flush=True)
    started = time.perf_counter()
    runner = PolicyRunner(args.checkpoint, device=args.device)
    print(f"policy loaded in {time.perf_counter() - started:.1f}s: {runner.describe()}", flush=True)

    print("creating EGL environment and resetting frozen state...", flush=True)
    env = SO101TabletopEnv(task, EnvConfig(max_steps=task.max_steps, terminate_on_success=True, render=True))
    obs, info = env.reset(seed=state["seed"], options={"init_state": state["init_state"]})
    torch.manual_seed(state["seed"])
    runner.reset()
    print(f"reset complete: seed={state['seed']} info={info}", flush=True)

    episode_started = time.perf_counter()
    try:
        for step in range(1, task.max_steps + 1):
            act_started = time.perf_counter()
            action = runner.act(obs, task.language)
            act_ms = 1e3 * (time.perf_counter() - act_started)

            step_started = time.perf_counter()
            obs, _, terminated, truncated, info = env.step(action)
            env_ms = 1e3 * (time.perf_counter() - step_started)

            if step == 1 or step % 50 == 0 or terminated or truncated:
                lift_cm = 100 * env.world.max_lift.get("red_cube", 0.0)
                print(
                    f"step={step:3d} elapsed={time.perf_counter() - episode_started:7.1f}s "
                    f"act={act_ms:7.1f}ms env={env_ms:7.1f}ms "
                    f"lift={lift_cm:4.1f}cm success={info['is_success']}",
                    flush=True,
                )

            if terminated or truncated:
                print(
                    f"finished: steps={step} success={info['is_success']} "
                    f"failure={info.get('failure')} truncated={truncated}",
                    flush=True,
                )
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
