"""Train a PPO residual policy over frozen SmolVLA.

The default configuration uses eight subprocess environments, a [256, 256]
actor and critic, a zero-initialized actor mean, and initial log standard
deviation -1.  Evaluation uses frozen 2 cm axis and 1.5 cm diagonal states,
whose seeds are disjoint from residual training seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from so101_libero.residual.residual_env import ResidualConfig, ResidualSO101Env

DEFAULT_POLICY = Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model")
DEFAULT_STATES = Path("docs/residual/placements")
DEFAULT_BASELINE = Path("docs/residual/results/baseline_shifted_140/results.json")
DEFAULT_RUN_DIR = Path("docs/residual/results/ppo")
TARGET_SETS = ("shift_2cm_axis", "shift_1p5cm_diagonal")


def load_eval_states(states_dir: Path, baseline_path: Path, count: int) -> list[dict]:
    """Take a balanced prefix from each target set and attach saved outcomes."""
    if count < len(TARGET_SETS) or count % len(TARGET_SETS):
        raise ValueError(f"eval episode count must be a multiple of {len(TARGET_SETS)}")
    per_set = count // len(TARGET_SETS)
    baseline = {
        (row["set_id"], int(row["state_index"])): row
        for row in json.loads(baseline_path.read_text())["episodes"]
    }
    rows = []
    for set_id in TARGET_SETS:
        states = json.loads((states_dir / f"{set_id}.json").read_text())["states"][:per_set]
        for state in states:
            saved = baseline[(set_id, int(state["index"]))]
            rows.append(
                {
                    **state,
                    "set_id": set_id,
                    "baseline_success": bool(saved["success"]),
                    "baseline_outcome": saved["outcome"],
                }
            )
    return rows


def evaluate_residual(model: PPO, env: ResidualSO101Env, states: list[dict]) -> dict:
    episodes = []
    for state in states:
        seed = int(state["seed"])
        torch.manual_seed(seed)
        obs, _ = env.reset(
            seed=seed,
            options={"init_state": state["init_state"], "offset_m": state["offset_m"]},
        )
        total_reward, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        episodes.append(
            {
                "set_id": state["set_id"],
                "state_index": int(state["index"]),
                "seed": seed,
                "offset_label": state["offset_label"],
                "baseline_success": state["baseline_success"],
                "residual_success": bool(info["is_success"]),
                "lifted": bool(info["lifted"]),
                "steps": steps,
                "total_reward": round(total_reward, 6),
            }
        )
    successes = sum(row["residual_success"] for row in episodes)
    baseline_successes = sum(row["baseline_success"] for row in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "baseline_successes": baseline_successes,
        "baseline_success_rate": baseline_successes / len(episodes),
        "lifted": sum(row["lifted"] for row in episodes),
        "episode_results": episodes,
    }


class FrozenEvaluationCallback(BaseCallback):
    def __init__(self, env: ResidualSO101Env, states: list[dict], eval_freq: int, out_dir: Path):
        super().__init__(verbose=1)
        self.eval_env = env
        self.states = states
        self.eval_freq = eval_freq
        self.out_dir = out_dir
        self.next_eval = eval_freq
        self.best_successes = -1
        self.path = out_dir / "evaluations.jsonl"

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps < self.next_eval:
            return True
        result = evaluate_residual(self.model, self.eval_env, self.states)
        record = {"timesteps": self.num_timesteps, "wall_time": time.time(), **result}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.logger.record("eval/success_rate", result["success_rate"])
        self.logger.record("eval/successes", result["successes"])
        self.logger.record("eval/lifted", result["lifted"])
        print(
            f"evaluation at {self.num_timesteps}: {result['successes']}/{result['episodes']} successes, "
            f"{result['lifted']}/{result['episodes']} lifted",
            flush=True,
        )
        if result["successes"] > self.best_successes:
            self.best_successes = result["successes"]
            self.model.save(self.out_dir / "best_model")
        while self.next_eval <= self.num_timesteps:
            self.next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


def env_factory(rank: int, args) -> callable:
    def make():
        return ResidualSO101Env(
            ResidualConfig(
                checkpoint=args.policy,
                device=args.base_device,
                worker_id=rank,
                num_workers=args.num_envs,
                train_seed_start=args.train_seed_start,
                use_cube_pose=not args.no_cube_pose,
                early_arm_scale_deg=args.early_arm_scale,
                late_arm_scale_deg=args.late_arm_scale,
                early_gripper_scale=args.gripper_scale,
                late_gripper_scale=args.gripper_scale,
                distance_progress_scale=args.distance_progress_scale,
                focus_probability=args.focus_probability,
            )
        )

    return make


def initialize_actor(model: PPO, log_std: float) -> None:
    """Make the deterministic initial residual exactly zero."""
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.log_std.fill_(log_std)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--states-dir", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--log-std", type=float, default=-1.0)
    parser.add_argument("--base-device", default="cuda")
    parser.add_argument("--ppo-device", default="cpu")
    parser.add_argument("--train-seed-start", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--early-arm-scale", type=float, default=5.0)
    parser.add_argument("--late-arm-scale", type=float, default=2.5)
    parser.add_argument("--gripper-scale", type=float, default=5.0)
    parser.add_argument("--distance-progress-scale", type=float, default=1.0)
    parser.add_argument("--focus-probability", type=float, default=0.75)
    parser.add_argument("--no-cube-pose", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("MUJOCO_GL") != "egl":
        raise SystemExit("run on a GPU node with MUJOCO_GL=egl")
    if args.n_steps * args.num_envs % args.batch_size:
        raise SystemExit("n_steps * num_envs must be divisible by batch_size")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + "\n"
    )
    eval_states = load_eval_states(args.states_dir, args.baseline, args.eval_episodes)

    print(f"creating {args.num_envs} residual environments", flush=True)
    train_env = VecMonitor(
        SubprocVecEnv([env_factory(rank, args) for rank in range(args.num_envs)], start_method="spawn"),
        filename=str(args.run_dir / "monitor.csv"),
        info_keywords=("lifted",),
    )
    eval_env = ResidualSO101Env(
        ResidualConfig(
            checkpoint=args.policy,
            device=args.base_device,
            use_cube_pose=not args.no_cube_pose,
            early_arm_scale_deg=args.early_arm_scale,
            late_arm_scale_deg=args.late_arm_scale,
            early_gripper_scale=args.gripper_scale,
            late_gripper_scale=args.gripper_scale,
            distance_progress_scale=args.distance_progress_scale,
            focus_probability=args.focus_probability,
        )
    )
    if args.resume:
        model = PPO.load(args.resume, env=train_env, device=args.ppo_device, tensorboard_log=str(args.run_dir / "tb"))
        print(f"resumed model from {args.resume}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device=args.ppo_device,
            tensorboard_log=str(args.run_dir / "tb"),
            verbose=1,
        )
        initialize_actor(model, args.log_std)

    initial = evaluate_residual(model, eval_env, eval_states)
    (args.run_dir / "initial_evaluation.json").write_text(json.dumps(initial, indent=2) + "\n")
    print(
        f"initial deterministic evaluation: residual {initial['successes']}/{initial['episodes']}, "
        f"saved baseline {initial['baseline_successes']}/{initial['episodes']}",
        flush=True,
    )
    if args.resume is None and abs(initial["successes"] - initial["baseline_successes"]) > 1:
        train_env.close()
        eval_env.close()
        raise SystemExit("initial zero-mean residual differs from baseline by more than one episode")

    eval_callback = FrozenEvaluationCallback(eval_env, eval_states, args.eval_freq, args.run_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(args.run_dir / "checkpoints"),
        name_prefix="ppo_residual",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList([checkpoint_callback, eval_callback]),
            tb_log_name="PPO",
            reset_num_timesteps=args.resume is None,
            progress_bar=False,
        )
        model.save(args.run_dir / "final_model")
    finally:
        try:
            train_env.close()
        except (BrokenPipeError, EOFError):
            # A worker traceback is the useful error if a subprocess died.
            pass
        # The evaluation callback closes this at normal training end. Closing twice is safe.
        eval_env.close()


if __name__ == "__main__":
    main()
