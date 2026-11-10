"""PPO environment for episode-fixed joint calibration offsets on original placements."""
from dataclasses import dataclass
from pathlib import Path
import gymnasium as gym
import numpy as np
from so101_libero.policy.runner import PolicyRunner
from so101_libero.residual.calibration import CalibrationOffsetEnv
from so101_libero.tasks.registry import get_task

@dataclass(frozen=True)
class CalibrationResidualConfig:
    checkpoint: str | Path = 'checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model'
    limit_deg: float = 0.0
    device: str = 'cuda'
    train_seed_start: int = 22_000_000
    worker_id: int = 0
    num_workers: int = 1

class CalibrationResidualEnv(gym.Env):
    """True-state feedback; no cube pose, expert target, or image in PPO input."""
    def __init__(self, config: CalibrationResidualConfig):
        super().__init__()
        self.cfg = config
        self.env = CalibrationOffsetEnv(config.limit_deg)
        self.runner = PolicyRunner(config.checkpoint, device=config.device)
        for p in self.runner.policy.parameters():
            p.requires_grad_(False)
        self.runner.policy.eval()
        self.task = get_task('cube_flip')
        self.action_space = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (31,), dtype=np.float32)
        self.lo = self.env.action_space.low.astype(np.float32)
        self.hi = self.env.action_space.high.astype(np.float32)
        self.mid = (self.lo + self.hi) / 2
        self.half = np.maximum((self.hi - self.lo) / 2, 1e-6)
        self.scale = np.array([config.limit_deg + 2] * 5 + [2], dtype=np.float32)
        self.episode_number = 0
        self.base_action = None
        self.obs = None
        self.previous_executed = None
        self.previous_residual = np.zeros(6, np.float32)
        self.step_number = 0
        self.lift_awarded = False
        self.initial_state = None

    def _normalize(self, x):
        return (np.asarray(x, np.float32) - self.mid) / self.half

    def _vector(self):
        state = self.obs['observation.state'].astype(np.float32)
        prev = self.previous_executed if self.previous_executed is not None else state
        return np.concatenate([self._normalize(state), self._normalize(self.base_action),
            self._normalize(prev), (prev - state) / self.half,
            np.array([(self.step_number % 50) / 50], np.float32), self.previous_residual]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        if 'init_state' in options:
            episode_seed = int(seed)
        else:
            episode_seed = self.cfg.train_seed_start + self.cfg.worker_id + self.episode_number * self.cfg.num_workers
            self.episode_number += 1
            if not 22_000_000 <= episode_seed < 23_000_000:
                raise RuntimeError('training seed range exhausted')
            options['placement_id'] = episode_seed % 5
        self.obs, info = self.env.reset(seed=episode_seed, options=options)
        self.runner.reset()
        self.base_action = self.runner.act(self.obs, self.task.language)
        self.previous_executed = None
        self.previous_residual = np.zeros(6, np.float32)
        self.step_number = 0
        self.lift_awarded = False
        self.initial_state = self.obs['observation.state'].copy()
        return self._vector(), info

    def step(self, action):
        residual = np.clip(np.asarray(action, np.float32).reshape(6), -1, 1)
        commanded = np.clip(self.base_action + self.scale * residual, self.lo, self.hi)
        self.obs, _, done, truncated, info = self.env.step(commanded)
        self.previous_executed = info['executed_target'].copy()
        self.previous_residual = residual.copy()
        self.step_number += 1
        lifted = bool(self.env.unwrapped.world.lifted('red_cube', 0.02))
        bonus = 0.2 if lifted and not self.lift_awarded else 0.0
        self.lift_awarded |= lifted
        success = bool(info['is_success'])
        reward = bonus + float(success)
        info.update({'base_action': self.base_action.copy(), 'residual_action': residual.copy(),
            'residual_scale': self.scale.copy(), 'lifted': self.lift_awarded,
            'max_arm_motion_deg': float(np.max(np.abs(self.obs['observation.state'][:5] - self.initial_state[:5])))})
        if not (done or truncated):
            self.base_action = self.runner.act(self.obs, self.task.language)
        return self._vector(), reward, done, truncated, info

    def close(self):
        self.env.close()
