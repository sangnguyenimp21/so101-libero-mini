"""Calibration residual with only commanded targets and measured joint feedback.

The base wrapper still records the hidden offset for diagnostics. This class
never places that offset or the simulator's post-offset target in the PPO
observation.
"""
import numpy as np

from so101_libero.residual.calibration_residual_env import CalibrationResidualEnv


class NoLeakCalibrationResidualEnv(CalibrationResidualEnv):
    def step(self, action):
        residual = np.clip(np.asarray(action, np.float32).reshape(6), -1, 1)
        commanded = np.clip(self.base_action + self.scale * residual, self.lo, self.hi).copy()
        _, reward, done, truncated, info = super().step(action)
        # The parent stores info['executed_target'] = commanded + hidden bias.
        # Replace it before building the observation returned to PPO.
        self.previous_executed = commanded
        return self._vector(), reward, done, truncated, info
