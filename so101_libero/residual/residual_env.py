"""Gymnasium wrapper for residual RL on top of frozen SmolVLA.

The wrapper owns an unchanged SO101TabletopEnv and PolicyRunner.  Its action is
dimensionless in [-1, 1]^6; the applied absolute target is the current frozen
SmolVLA target plus a time-dependent per-joint scale times that action.

Cube pose and the expert-derived grasp point are privileged simulator signals.
This environment is therefore simulation-only and must be reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.expert.rollout import expert_rng, make_expert
from so101_libero.policy.runner import PolicyRunner
from so101_libero.residual.placements import (
    TRAIN_SEED_MAX,
    TRAIN_SEED_MIN,
    shifted_init_state,
    weighted_training_offset,
)
from so101_libero.tasks.registry import get_task


@dataclass(frozen=True)
class ResidualConfig:
    checkpoint: str | Path
    device: str = "cuda"
    early_steps: int = 50
    early_arm_scale_deg: float = 5.0
    late_arm_scale_deg: float = 2.5
    early_gripper_scale: float = 5.0
    late_gripper_scale: float = 5.0
    lift_bonus: float = 0.2
    success_bonus: float = 1.0
    distance_progress_scale: float = 1.0
    focus_probability: float = 0.75
    use_cube_pose: bool = True
    image_size: tuple[int, int] = (256, 256)
    train_seed_start: int = TRAIN_SEED_MIN
    worker_id: int = 0
    num_workers: int = 1

    def __post_init__(self) -> None:
        if self.early_steps < 0:
            raise ValueError("early_steps must be non-negative")
        if min(self.early_arm_scale_deg, self.late_arm_scale_deg) < 0:
            raise ValueError("arm scales must be non-negative")
        if min(self.early_gripper_scale, self.late_gripper_scale) < 0:
            raise ValueError("gripper scales must be non-negative")
        if self.num_workers < 1 or not 0 <= self.worker_id < self.num_workers:
            raise ValueError("worker_id must be in [0, num_workers)")
        if not TRAIN_SEED_MIN <= self.train_seed_start <= TRAIN_SEED_MAX:
            raise ValueError("train_seed_start is outside the reserved training range")


class ResidualSO101Env(gym.Env):
    """Residual-control environment with a frozen SmolVLA base policy."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: ResidualConfig):
        super().__init__()
        self.cfg = config
        self.task = get_task("cube_flip")
        self.base_env = SO101TabletopEnv(
            self.task,
            EnvConfig(
                image_size=config.image_size,
                max_steps=self.task.max_steps,
                terminate_on_success=True,
            ),
        )
        self.runner = PolicyRunner(config.checkpoint, device=config.device)
        for parameter in self.runner.policy.parameters():
            parameter.requires_grad_(False)
        self.runner.policy.eval()

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        obs_dim = 6 + 6 + 1 + (7 if config.use_cube_pose else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._base_lo = self.base_env.action_space.low.astype(np.float32)
        self._base_hi = self.base_env.action_space.high.astype(np.float32)
        self._base_mid = (self._base_lo + self._base_hi) / 2.0
        self._base_half = np.maximum((self._base_hi - self._base_lo) / 2.0, 1e-6)

        self._episode_number = 0
        self._obs: dict[str, np.ndarray] | None = None
        self._base_action: np.ndarray | None = None
        self._episode_step = 0
        self._episode_seed: int | None = None
        self._offset_m: tuple[float, float] | None = None
        self._grasp_target = np.zeros(3, dtype=np.float64)
        self._previous_distance = 0.0
        self._lift_awarded = False

    @property
    def unwrapped_base(self) -> SO101TabletopEnv:
        return self.base_env

    def _next_training_seed(self) -> int:
        seed = self.cfg.train_seed_start + self.cfg.worker_id + self._episode_number * self.cfg.num_workers
        self._episode_number += 1
        if seed > TRAIN_SEED_MAX:
            raise RuntimeError("exhausted the reserved residual training seed range")
        return seed

    def _plan_grasp_target(self, seed: int) -> np.ndarray:
        """Use the unchanged scripted expert to select the feasible cube grasp."""
        expert = make_expert(self.base_env, self.task.plan, expert_rng(seed))
        obj_pos = self.base_env.world.pos(self.task.plan.obj).copy()
        expert._approach(obj_pos, expert._face_psis(), expert._u(expert.pregrasp_z), expert.grasp_z[0])
        return np.array([expert._grasp_xy[0], expert._grasp_xy[1], expert.grasp_z[0]], dtype=np.float64)

    def _distance_to_grasp(self) -> float:
        return float(np.linalg.norm(self.base_env.world.tcp_pos() - self._grasp_target))

    def _cube_pose(self) -> np.ndarray:
        obj = self.task.plan.obj
        body_id = self.base_env.h.item_body[obj]
        position = self.base_env.world.pos(obj)
        quaternion = self.base_env.data.xquat[body_id]
        return np.concatenate([position, quaternion]).astype(np.float32)

    def _normalize_joints(self, value: np.ndarray) -> np.ndarray:
        return ((np.asarray(value, dtype=np.float32) - self._base_mid) / self._base_half).astype(np.float32)

    def _residual_observation(self) -> np.ndarray:
        assert self._obs is not None and self._base_action is not None
        pieces = [
            self._normalize_joints(self._obs["observation.state"]),
            self._normalize_joints(self._base_action),
            np.array([(self._episode_step % 50) / 49.0], dtype=np.float32),
        ]
        if self.cfg.use_cube_pose:
            pieces.append(self._cube_pose())
        return np.concatenate(pieces).astype(np.float32)

    def current_scale(self) -> np.ndarray:
        early = self._episode_step < self.cfg.early_steps
        arm = self.cfg.early_arm_scale_deg if early else self.cfg.late_arm_scale_deg
        gripper = self.cfg.early_gripper_scale if early else self.cfg.late_gripper_scale
        return np.array([arm, arm, arm, arm, arm, gripper], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        if "init_state" in options:
            episode_seed = int(seed if seed is not None else options.pop("episode_seed", 0))
            init_state = options["init_state"]
            self._offset_m = tuple(options.get("offset_m", (np.nan, np.nan)))
        else:
            # VecEnv passes the PPO/model seed into the first reset.  It is not
            # a placement seed: all training placements must come from the
            # reserved, evaluation-disjoint 20M series owned by this worker.
            episode_seed = self._next_training_seed()
            dx, dy = weighted_training_offset(episode_seed, self.cfg.focus_probability)
            init_state = shifted_init_state(dx, dy)
            self._offset_m = (dx, dy)
        self._episode_seed = episode_seed
        self._obs, info = self.base_env.reset(seed=episode_seed, options={"init_state": init_state})
        self.runner.reset()
        self._episode_step = 0
        self._lift_awarded = False
        self._grasp_target = self._plan_grasp_target(episode_seed)
        self._previous_distance = self._distance_to_grasp()
        self._base_action = self.runner.act(self._obs, self.task.language)
        info.update(
            {
                "residual_training_seed": episode_seed,
                "offset_m": self._offset_m,
                "grasp_target_m": self._grasp_target.copy(),
                "privileged_cube_pose": self.cfg.use_cube_pose,
            }
        )
        return self._residual_observation(), info

    def step(self, action):
        if self._base_action is None or self._obs is None:
            raise RuntimeError("reset() must be called before step()")
        residual = np.clip(np.asarray(action, dtype=np.float32).reshape(6), -1.0, 1.0)
        scale = self.current_scale()
        final_action = np.clip(self._base_action + scale * residual, self._base_lo, self._base_hi)
        self._obs, _, terminated, truncated, info = self.base_env.step(final_action)
        self._episode_step += 1

        distance = self._distance_to_grasp()
        distance_reward = self.cfg.distance_progress_scale * (self._previous_distance - distance)
        self._previous_distance = distance
        lifted_now = self.base_env.world.lifted(self.task.plan.obj, 0.02)
        lift_reward = self.cfg.lift_bonus if lifted_now and not self._lift_awarded else 0.0
        self._lift_awarded = self._lift_awarded or lifted_now
        success_reward = self.cfg.success_bonus if info["is_success"] else 0.0
        reward = float(distance_reward + lift_reward + success_reward)

        info.update(
            {
                "base_action": self._base_action.copy(),
                "final_action": final_action.copy(),
                "residual_action": residual.copy(),
                "residual_scale": scale.copy(),
                "distance_to_grasp_m": distance,
                "reward_distance": float(distance_reward),
                "reward_lift": float(lift_reward),
                "reward_success": float(success_reward),
                "lifted": bool(self._lift_awarded),
                "episode_seed": self._episode_seed,
                "offset_m": self._offset_m,
            }
        )
        if not (terminated or truncated):
            self._base_action = self.runner.act(self._obs, self.task.language)
            observation = self._residual_observation()
        else:
            # The terminal observation still describes the action just applied.
            observation = self._residual_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()

