"""SO-101 constants and the conversion between simulator joint angles and LeRobot units.

Datasets, policies and the environment's public observation/action interface all use the units of LeRobot's
real ``so101_follower`` with ``use_degrees=True``: the five arm joints in degrees and the gripper in 0..100
(0 = closed, 100 = open). The simulator works in radians internally; ``sim_to_lerobot`` / ``lerobot_to_sim``
are the only places where the two meet.

The Menagerie model derives from TheRobotStudio ``so101_new_calib.xml``, whose joint zero is the pose the
LeRobot calibration maps to 0 degrees (middle of each joint's range), so arm angles convert 1:1. The gripper's
joint range [-10, 100] degrees is mapped linearly onto 0..100.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINT_NAMES = JOINT_NAMES[:5]
# Feature names used by LeRobot's so101_follower for observation.state and action.
MOTOR_NAMES = tuple(f"{j}.pos" for j in JOINT_NAMES)
NUM_JOINTS = len(JOINT_NAMES)
GRIPPER_INDEX = 5

# Joint range of the gripper in so101.xml (radians): -10 deg (closed past contact) .. 100 deg (wide open).
GRIPPER_RANGE_RAD = (-0.174533, 1.7453292)


def gripper_rad_to_lerobot(q: np.ndarray | float) -> np.ndarray | float:
    lo, hi = GRIPPER_RANGE_RAD
    return (np.asarray(q) - lo) / (hi - lo) * 100.0


def gripper_lerobot_to_rad(g: np.ndarray | float) -> np.ndarray | float:
    lo, hi = GRIPPER_RANGE_RAD
    return lo + np.asarray(g) / 100.0 * (hi - lo)


def sim_to_lerobot(q: np.ndarray) -> np.ndarray:
    """Joint angles in radians (..., 6) -> LeRobot units (..., 6): degrees x5, gripper 0..100."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)
    out[..., :5] = np.degrees(q[..., :5])
    out[..., 5] = gripper_rad_to_lerobot(q[..., 5])
    return out.astype(np.float32)


def lerobot_to_sim(a: np.ndarray) -> np.ndarray:
    """LeRobot units (..., 6) -> joint angles in radians (..., 6)."""
    a = np.asarray(a, dtype=np.float64)
    out = np.empty_like(a)
    out[..., :5] = np.radians(a[..., :5])
    out[..., 5] = gripper_lerobot_to_rad(a[..., 5])
    return out


# Start ("home") pose: gripper above the workspace centre, tilted down towards the table, jaws slightly open.
HOME_DEG = np.array([0.0, -60.0, 35.0, 85.0, 0.0, 0.0])  # joint angles in degrees (sim convention)
HOME_QPOS = np.radians(HOME_DEG)
# Folded rest pose in LeRobot units (gripper 0..100), for tasks that start and end at rest. The real arm rests at
# about [0, -101, 88, 63..73, -4, 3] (my_follower_sim calibration); the simulator's lift limit (-100) and its boxy
# forearm/shoulder colliders (contact from elbow ~90) need it a little less folded: jaws 2.6 cm above the table,
# collision-free under +-2 deg start noise.
REST_DEG = np.array([0.0, -94.0, 82.0, 68.0, 0.0, 3.0])
