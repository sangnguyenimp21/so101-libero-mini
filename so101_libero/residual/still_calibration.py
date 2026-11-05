"""Non-learning joint-bias estimate from the first steady command/position window.

Uses only the target sent by the controller and the measured joint positions.
The simulator's hidden bias and its executed-target diagnostic are never inputs.
"""
from collections import deque
import numpy as np


class FirstStillCalibrator:
    def __init__(self, limit_deg: float, window: int = 10,
                 position_span_deg: float = 0.3, command_span_deg: float = 0.5):
        self.limit_deg = float(limit_deg)
        self.window = int(window)
        self.position_span_deg = float(position_span_deg)
        self.command_span_deg = float(command_span_deg)
        self.samples = deque(maxlen=window)
        self.bias_hat = np.zeros(6, dtype=np.float32)
        self.estimated_at_step = None

    def observe(self, command, measured_state, step: int) -> bool:
        """Observe the state after applying `command`; estimate once, at first still window."""
        if self.estimated_at_step is not None:
            return False
        command = np.asarray(command, dtype=np.float32)[:5].copy()
        measured = np.asarray(measured_state, dtype=np.float32)[:5].copy()
        self.samples.append((command, measured))
        if len(self.samples) < self.window:
            return False
        commands = np.stack([x[0] for x in self.samples])
        positions = np.stack([x[1] for x in self.samples])
        if (np.max(np.ptp(commands, axis=0)) > self.command_span_deg or
                np.max(np.ptp(positions, axis=0)) > self.position_span_deg):
            return False
        # At rest, measured position ~= commanded target + hidden actuation bias.
        # Thus b_hat = position - command, and future commands subtract b_hat.
        self.bias_hat[:5] = np.clip(np.median(positions - commands, axis=0),
                                    -self.limit_deg, self.limit_deg)
        self.estimated_at_step = int(step)
        return True

    def compensate(self, base_action, low, high):
        return np.clip(np.asarray(base_action, dtype=np.float32) - self.bias_hat,
                       low, high)
