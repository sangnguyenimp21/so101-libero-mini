"""Benchmark simulation, rendering, and SmolVLA rollout costs for residual RL.

Run EGL and OSMesa in separate processes because MuJoCo selects its rendering
backend at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": round(1e3 * float(arr.mean()), 3),
        "median_ms": round(1e3 * float(np.median(arr)), 3),
        "min_ms": round(1e3 * float(arr.min()), 3),
        "max_ms": round(1e3 * float(arr.max()), 3),
    }


def benchmark_env(render: bool, steps: int) -> dict:
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    env = SO101TabletopEnv(
        "cube_flip",
        EnvConfig(render=render, terminate_on_success=False, max_steps=steps + 10),
    )
    try:
        obs, _ = env.reset(options={"placement_id": 0})
        action = obs["observation.state"].copy()
        samples = []
        for _ in range(steps):
            started = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(action)
            samples.append(time.perf_counter() - started)
            if terminated or truncated:
                raise RuntimeError("stationary timing rollout ended unexpectedly")
        return {"render": render, "steps": steps, **summary(samples)}
    finally:
        env.close()


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_act(runner, obs: dict, instruction: str) -> tuple[np.ndarray, float]:
    cuda_sync()
    started = time.perf_counter()
    action = runner.act(obs, instruction)
    cuda_sync()
    return action, time.perf_counter() - started


def benchmark_policy(checkpoint: Path, chunk_repeats: int, state_index: int) -> dict:
    import psutil
    import torch

    from so101_libero.cli.make_init_states import load_init_states
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.policy.runner import PolicyRunner
    from so101_libero.tasks.registry import get_task

    task = get_task("cube_flip")
    state = load_init_states(task.task_id)["states"][state_index]
    env = SO101TabletopEnv(task, EnvConfig(render=True, terminate_on_success=True, max_steps=task.max_steps))
    process = psutil.Process()
    rss_before = process.memory_info().rss
    cuda_sync()
    free_before, total_gpu = torch.cuda.mem_get_info()

    print("loading policy...", flush=True)
    load_started = time.perf_counter()
    runner = PolicyRunner(checkpoint, device="cuda")
    cuda_sync()
    load_s = time.perf_counter() - load_started
    rss_after = process.memory_info().rss
    free_after, _ = torch.cuda.mem_get_info()
    print(f"policy loaded in {load_s:.1f}s", flush=True)

    try:
        obs, _ = env.reset(seed=state["seed"], options={"init_state": state["init_state"]})
        torch.manual_seed(state["seed"])
        runner.reset()

        _, warmup_s = timed_act(runner, obs, task.language)
        print(f"first chunk warm-up: {warmup_s:.3f}s", flush=True)
        for _ in range(runner.cfg.n_action_steps - 1):
            runner.act(obs, task.language)

        chunk_samples = []
        for index in range(chunk_repeats):
            _, elapsed = timed_act(runner, obs, task.language)
            chunk_samples.append(elapsed)
            print(f"warmed chunk {index + 1}/{chunk_repeats}: {elapsed:.3f}s", flush=True)
            for _ in range(runner.cfg.n_action_steps - 1):
                runner.act(obs, task.language)

        obs, _ = env.reset(seed=state["seed"], options={"init_state": state["init_state"]})
        torch.manual_seed(state["seed"])
        runner.reset()
        episode_started = time.perf_counter()
        inference_s = 0.0
        steps = 0
        while True:
            action, act_s = timed_act(runner, obs, task.language)
            inference_s += act_s
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if steps % 50 == 0 or terminated or truncated:
                print(
                    f"episode step={steps} elapsed={time.perf_counter() - episode_started:.2f}s "
                    f"success={info['is_success']}",
                    flush=True,
                )
            if terminated or truncated:
                break
        episode_s = time.perf_counter() - episode_started

        return {
            "checkpoint": str(checkpoint),
            "policy_load_s": round(load_s, 3),
            "first_chunk_warmup_s": round(warmup_s, 3),
            "warmed_chunk": {"samples": chunk_repeats, **summary(chunk_samples)},
            "episode": {
                "state_index": state_index,
                "seed": state["seed"],
                "steps": steps,
                "success": bool(info["is_success"]),
                "wall_s": round(episode_s, 3),
                "policy_act_total_s": round(inference_s, 3),
            },
            "memory": {
                "rss_increase_gib": round((rss_after - rss_before) / 2**30, 3),
                "gpu_increase_gib": round((free_before - free_after) / 2**30, 3),
                "gpu_total_gib": round(total_gpu / 2**30, 3),
                "max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("env", "policy", "all"), default="all")
    parser.add_argument("--env-steps", type=int, default=100)
    parser.add_argument("--chunk-repeats", type=int, default=3)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--training-episodes", type=int, default=2_000)
    parser.add_argument("--episode-horizon", type=int, default=450)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    backend = os.environ.get("MUJOCO_GL")
    if backend not in ("egl", "osmesa"):
        parser.error("set MUJOCO_GL=egl or MUJOCO_GL=osmesa")

    import so101_libero  # noqa: F401  (must be imported before MuJoCo under OSMesa)

    result: dict = {"backend": backend}
    if args.mode in ("env", "all"):
        print(f"benchmarking {args.env_steps} physics-only steps...", flush=True)
        result["env_step_no_render"] = benchmark_env(False, args.env_steps)
        print(f"benchmarking {args.env_steps} two-camera rendered steps...", flush=True)
        result["env_step_render"] = benchmark_env(True, args.env_steps)
    if args.mode in ("policy", "all"):
        result["policy"] = benchmark_policy(args.checkpoint, args.chunk_repeats, args.state_index)
        episode_s = result["policy"]["episode"]["wall_s"]
        measured_steps = result["policy"]["episode"]["steps"]
        horizon_s = episode_s * args.episode_horizon / measured_steps
        gpu_per_env = result["policy"]["memory"]["gpu_increase_gib"]
        rss_per_env = result["policy"]["memory"]["rss_increase_gib"]
        result["training_estimate"] = {
            "episodes": args.training_episodes,
            "measured_episode_steps": measured_steps,
            "measured_length_one_env_hours": round(args.training_episodes * episode_s / 3600, 2),
            "full_horizon_steps": args.episode_horizon,
            "full_horizon_estimated_s_per_episode": round(horizon_s, 3),
            "full_horizon_one_env_hours": round(args.training_episodes * horizon_s / 3600, 2),
            "parallel_envs": args.parallel_envs,
            "measured_length_ideal_parallel_hours": round(
                args.training_episodes * episode_s / args.parallel_envs / 3600, 2
            ),
            "full_horizon_ideal_parallel_hours": round(
                args.training_episodes * horizon_s / args.parallel_envs / 3600, 2
            ),
            "estimated_gpu_for_parallel_envs_gib": round(gpu_per_env * args.parallel_envs, 2),
            "estimated_rss_for_parallel_envs_gib": round(rss_per_env * args.parallel_envs, 2),
            "note": "Parallel times assume linear scaling; shared-GPU contention must be measured before training.",
        }

    print(json.dumps(result, indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
