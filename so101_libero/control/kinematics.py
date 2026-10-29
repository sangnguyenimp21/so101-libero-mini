"""SO-101 forward kinematics and a small IK for the TCP, in numpy only (no MuJoCo).

The episode finisher lifts the gripper straight up after a task (``safe_retreat``) on the real arm as well as in the
simulator; this module itself needs only numpy (the real-robot runner as a whole still imports the task definitions,
which need MuJoCo). The chain below is the compiled Menagerie model
(assets/robotstudio_so101/so101.xml, as composed by so101_libero/sim/scene.py), read from MuJoCo once:
every arm joint is a hinge about its body's z axis at the body origin, so a body's pose is
parent * translate(pos) * rotate(quat) * rot_z(q). tests/test_three_cubes.py checks it against MuJoCo's FK.

Joint values are in policy units: degrees for the five arm joints (sim zero = LeRobot calibration zero).
"""

from __future__ import annotations

import math

import numpy as np

# (body, pos in the parent frame, quat (w, x, y, z) in the parent frame); joint k rotates body k about its z axis.
CHAIN = (
    ("shoulder", (0.038835, 0.0, 0.0624), (0.0, 0.0, -1.0, 0.0)),
    ("upper_arm", (-0.030399, -0.018278, -0.0542), (0.5, -0.5, -0.5, -0.5)),
    ("lower_arm", (-0.11257, -0.028, 0.0), (0.70710678, 0.0, 0.0, 0.70710678)),
    ("wrist", (-0.1349, 0.0052, 0.0), (0.70710678, 0.0, 0.0, -0.70710678)),
    ("gripper", (0.0, -0.0611, 0.0181), (0.01720911, -0.01720911, 0.70689734, 0.70689734)),
)
TCP_IN_GRIPPER = np.array([0.005, 0.0, -0.089])  # so101_libero/sim/scene.py TCP_POS


def _arm_range_deg() -> np.ndarray:
    """Joint ranges of the 5 arm joints in degrees: the runners' single definition (scripts/real/run_policy.py)."""
    from scripts.real.run_policy import JOINT_LIMITS

    return np.asarray(JOINT_LIMITS[:5], dtype=float)


def _quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


_LINKS = tuple((np.asarray(p, float), _quat_mat(q)) for _, p, q in CHAIN)


def fk(q_arm_deg) -> tuple[np.ndarray, np.ndarray]:
    """TCP position (3,) and orientation (3, 3) in the world (robot base) frame for the 5 arm joints in degrees."""
    q = np.radians(np.asarray(q_arm_deg, dtype=float)[:5])
    pos, rot = np.zeros(3), np.eye(3)
    for (p, r), qi in zip(_LINKS, q):
        pos = pos + rot @ p
        c, s = math.cos(qi), math.sin(qi)
        rot = rot @ r @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pos + rot @ TCP_IN_GRIPPER, rot


def _rot_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) taking ``current`` to ``target``."""
    r = target @ current.T
    angle = math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:  # 180 degrees: axis from the symmetric part
        axis = np.sqrt(np.clip((np.diag(r) + 1.0) / 2.0, 0.0, None))
        n = np.linalg.norm(axis)
    return axis / n * angle


def ik(target_pos, target_rot, q_init_deg, iters: int = 100, tol_pos: float = 5e-4, tol_rot: float = 0.02,
       rot_weight: float = 0.25, damping: float = 1e-4, margin_deg: float = 1.0) -> tuple[np.ndarray, bool]:
    """Damped least squares on the 5 arm joints (numerical Jacobian): (q in degrees, converged)."""
    rng = _arm_range_deg()
    lo, hi = rng[:, 0] + margin_deg, rng[:, 1] - margin_deg
    q = np.clip(np.asarray(q_init_deg, float)[:5].copy(), lo, hi)
    w = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
    h = 1e-3  # degrees, finite-difference step
    for _ in range(iters):
        pos, rot = fk(q)
        e = np.concatenate([np.asarray(target_pos) - pos, _rot_error(np.asarray(target_rot), rot)])
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
            return q, True
        jac = np.zeros((6, 5))
        for k in range(5):
            dq = q.copy()
            dq[k] += h
            p2, r2 = fk(dq)
            jac[:3, k] = (p2 - pos) / math.radians(h)
            jac[3:, k] = _rot_error(r2, rot) / math.radians(h)
        jw, ew = jac * w[:, None], e * w
        step = jw.T @ np.linalg.solve(jw @ jw.T + damping * np.eye(6), ew)  # radians
        n = np.linalg.norm(step)
        if n > 0.2:
            step *= 0.2 / n
        q = np.clip(q + np.degrees(step), lo, hi)
    pos, rot = fk(q)
    ok = np.linalg.norm(np.asarray(target_pos) - pos) < tol_pos and np.linalg.norm(_rot_error(np.asarray(target_rot), rot)) < tol_rot
    return q, bool(ok)


def min_jerk(n: int) -> np.ndarray:
    """n interpolation fractions in (0, 1] along a minimum-jerk profile."""
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def vertical_lift_path(q_arm_deg, dz: float, n_steps: int) -> tuple[np.ndarray, float]:
    """Joint path (n, 5) that lifts the TCP straight up by ``dz`` metres keeping its orientation (min-jerk profile).
    Returns the reachable prefix and the height actually gained (the path stops where the IK no longer converges)."""
    p0, r0 = fk(q_arm_deg)
    q, out = np.asarray(q_arm_deg, float)[:5].copy(), []
    for s in min_jerk(max(1, n_steps)):
        q_next, ok = ik(p0 + np.array([0.0, 0.0, s * dz]), r0, q)
        if not ok or np.max(np.abs(q_next - q)) > 10.0:  # no IK branch switches
            break
        q = q_next
        out.append(q.copy())
    gained = float(fk(out[-1])[0][2] - p0[2]) if out else 0.0
    return np.array(out).reshape(-1, 5), gained
