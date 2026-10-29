"""The SO-101 HOME pose and the one move that reaches it.

HOME is defined ONCE, in the three-cube suite's scene configuration (so101_libero/tasks/three_cubes.py
``ThreeCubeScene.home_deg``): every simulated episode starts there and the episode finisher returns there. It is the
project's HOME_DEG arm pose [0, -60, 35, 85, 0] (gripper 13 cm above the table over the workspace centre, tilted 30 deg
forward, clear of every cube) with the gripper at 10/100. Units: LeRobot so101_follower with use_degrees=True (degrees
for the 5 arm joints, gripper 0 = closed .. 100 = open), the policy's units.

Overrides, for a real arm whose comfortable home differs: ``--home-pose 0,-60,35,85,0,10`` on the command-line tools,
or the environment variable SO101_HOME_POSE. The policies were trained on episodes that start at the default HOME;
an override changes only where the deterministic finisher parks the arm and where the real episode starts.
"""

from __future__ import annotations

import os

import numpy as np

HOME_ENV = "SO101_HOME_POSE"


def default_home() -> np.ndarray:
    from so101_libero.tasks.three_cubes import SCENE

    return np.asarray(SCENE.home_deg, dtype=np.float64)


def parse_pose(spec) -> np.ndarray:
    """'0,-60,35,85,0,10' or a sequence of 6 numbers -> (6,) float array."""
    vals = [float(v) for v in str(spec).replace(" ", "").split(",")] if isinstance(spec, str) else [float(v) for v in spec]
    if len(vals) != 6 or not np.all(np.isfinite(vals)):
        raise ValueError(f"a pose needs 6 finite values (5 arm joints in degrees + gripper 0..100), got {spec!r}")
    return np.asarray(vals, dtype=np.float64)


def resolve_home(override=None) -> np.ndarray:
    """HOME in policy units: ``override`` if given, else $SO101_HOME_POSE, else the suite's HOME."""
    if override is not None:
        return parse_pose(override)
    if os.environ.get(HOME_ENV):
        return parse_pose(os.environ[HOME_ENV])
    return default_home()


def min_jerk_path(q_from, q_to, fps: float, max_joint_speed: float = 40.0, min_seconds: float = 1.5) -> np.ndarray:
    """Joint targets (n, 6) from ``q_from`` to ``q_to`` (the last row is ``q_to``): one min-jerk segment whose duration
    gives an AVERAGE speed of at most ``max_joint_speed`` deg/s on every arm joint (min-jerk peaks are 1.875 x that),
    at least ``min_seconds``."""
    a, b = np.asarray(q_from, float), np.asarray(q_to, float)
    seconds = max(min_seconds, float(np.max(np.abs(b[:5] - a[:5]))) / max(max_joint_speed, 1e-6))
    n = max(2, int(round(seconds * fps)))
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
    return a + s[:, None] * (b - a)


def home_error(q, home) -> float:
    """Largest arm-joint distance (degrees) between ``q`` and ``home`` (the gripper is not compared)."""
    return float(np.max(np.abs(np.asarray(q, float)[:5] - np.asarray(home, float)[:5])))
