"""Episode-fixed actuation bias for cube_flip; simulator and observations stay unchanged."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from so101_libero.env import SO101TabletopEnv, EnvConfig

BIAS_STREAM = 81_773


def sample_bias(seed: int, limit_deg: float) -> np.ndarray:
    if limit_deg < 0:
        raise ValueError('limit_deg must be nonnegative')
    bias = np.zeros(6, dtype=np.float32)
    bias[:5] = np.random.default_rng([int(seed), BIAS_STREAM]).uniform(-limit_deg, limit_deg, 5)
    return bias


class CalibrationOffsetEnv(gym.Wrapper):
    """Command u becomes clip(u+b), with b fixed across an episode.

    The gripper's bias is always zero. The wrapped env's observation, including
    measured joint state and camera images, is returned without modification.
    """
    def __init__(self, limit_deg: float, env: SO101TabletopEnv | None = None):
        super().__init__(env or SO101TabletopEnv('cube_flip', EnvConfig()))
        self.limit_deg = float(limit_deg)
        self.bias = np.zeros(6, dtype=np.float32)
        self.episode_seed = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if seed is None:
            raise ValueError('explicit episode seed required for frozen bias')
        self.episode_seed = int(seed)
        supplied = options.pop('calibration_bias_deg', None)
        self.bias = sample_bias(seed, self.limit_deg) if supplied is None else np.asarray(supplied, dtype=np.float32)
        if self.bias.shape != (6,) or self.bias[5] != 0 or np.any(np.abs(self.bias[:5]) > self.limit_deg + 1e-6):
            raise ValueError('invalid calibration bias')
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, {**info, 'calibration_bias_deg': self.bias.copy(), 'episode_seed': self.episode_seed}

    def step(self, action):
        commanded = np.asarray(action, dtype=np.float32)
        executed = np.clip(commanded + self.bias, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        return obs, reward, terminated, truncated, {**info, 'calibration_bias_deg': self.bias.copy(),
            'commanded_target': commanded.copy(), 'executed_target': executed.copy(), 'episode_seed': self.episode_seed}
