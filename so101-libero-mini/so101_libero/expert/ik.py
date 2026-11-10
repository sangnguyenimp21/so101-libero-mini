"""Inverse kinematics for the SO-101 tool centre point (TCP).

Damped least squares on the 5 arm joints with a 6-d task error (position + orientation). With the gripper
pointing straight down the problem is exactly determined for this arm: shoulder_pan sets the azimuth,
shoulder_lift / elbow_flex set reach and height, wrist_flex makes the approach axis vertical and wrist_roll
sets the yaw of the jaws. Solutions are checked by forward kinematics; nothing downstream trusts an
unconverged solve.

TCP frame (site "tcp" on the wrist-roll body): -z = approach direction (out of the finger tips),
+x = closing direction (from the fixed jaw towards the moving jaw).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from so101_libero.sim.scene import SceneHandles


def grasp_frame(approach: np.ndarray, closing_yaw: float) -> np.ndarray:
    """Rotation matrix of the TCP for a given approach direction and (horizontal) closing-direction yaw."""
    a = np.asarray(approach, float)
    a = a / np.linalg.norm(a)
    c = np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.0])
    c = c - a * (c @ a)  # closing direction must be perpendicular to the approach
    c /= np.linalg.norm(c)
    z = -a
    y = np.cross(z, c)
    return np.stack([c, y, z], axis=1)


def top_down_frame(closing_yaw: float) -> np.ndarray:
    return grasp_frame(np.array([0.0, 0.0, -1.0]), closing_yaw)


def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that takes ``current`` to ``target``."""
    err = target @ current.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, err.flatten())
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class IKResult:
    q: np.ndarray  # (5,) arm joint angles, radians
    success: bool
    pos_err: float  # metres
    rot_err: float  # radians
    iterations: int


class IKSolver:
    """DLS IK on a private MjData (never touches the simulation state)."""

    def __init__(self, handles: SceneHandles, joint_margin: float = 0.02):
        self.m = handles.model
        self.d = mujoco.MjData(self.m)
        self.site = handles.tcp_site
        self.qadr = handles.joint_qpos_adr[:5]
        self.dadr = handles.joint_dof_adr[:5]
        self.lo = handles.joint_range[:5, 0] + joint_margin
        self.hi = handles.joint_range[:5, 1] - joint_margin
        self._jacp = np.zeros((3, self.m.nv))
        self._jacr = np.zeros((3, self.m.nv))
        # The pan axis is vertical through the shoulder body origin. The wrist-roll axis (not the TCP, which is 5 mm
        # off it along the closing direction) stays in the vertical plane through the pan axis, so tilted approaches
        # must lean in that plane: see tilted_frame.
        mujoco.mj_kinematics(self.m, self.d)
        self.pan_axis_xy = self.d.xpos[self.m.body("shoulder").id][:2].copy()

    def azimuth(self, xy: np.ndarray) -> float:
        """Azimuth of a point around the pan axis (0 = straight ahead, +x)."""
        v = np.asarray(xy, float)[:2] - self.pan_axis_xy
        return math.atan2(v[1], v[0])

    def tilted_frame(self, pos: np.ndarray, tilt: float, yaw: float) -> np.ndarray:
        """TCP frame tilted outward by ``tilt`` in the arm plane, i.e. aimed from the wrist-roll axis point next to
        the TCP (the frame then is exactly reachable by the 5-DoF arm also for non-radial jaw yaws)."""
        from so101_libero.sim.scene import TCP_POS

        rot = grasp_frame(self.approach_dir(pos, tilt), yaw)
        if tilt == 0.0:
            return rot
        for _ in range(3):
            rot = grasp_frame(self.approach_dir(np.asarray(pos)[:2] - TCP_POS[0] * rot[:2, 0], tilt), yaw)
        return rot

    def radius(self, xy: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(xy, float)[:2] - self.pan_axis_xy))

    def approach_dir(self, xy: np.ndarray, tilt: float) -> np.ndarray:
        """Approach direction tilted by ``tilt`` (rad) from vertical, finger tips pointing away from the base."""
        az = self.azimuth(xy)
        return np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)])

    def fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q_arm
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 200,
        tol_pos: float = 5e-4,
        tol_rot: float = 1e-2,
        rot_weight: float = 0.25,
        damping: float = 1e-4,
        max_step: float = 0.25,
    ) -> IKResult:
        q = np.clip(np.asarray(q_init, float)[:5], self.lo, self.hi)
        w = np.array([1, 1, 1, rot_weight, rot_weight, rot_weight])
        pos_err = rot_err = float("inf")
        for it in range(max_iters):
            pos, rot = self.fk(q)
            e = np.concatenate([target_pos - pos, orientation_error(target_rot, rot)])
            pos_err, rot_err = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if pos_err < tol_pos and rot_err < tol_rot:
                return IKResult(q, True, pos_err, rot_err, it)
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self._jacp, self._jacr, self.site)
            J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]]) * w[:, None]
            ew = e * w
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), ew)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        return IKResult(q, False, pos_err, rot_err, max_iters)

    def solve_multi(self, target_pos, target_rot, seeds, **kw) -> IKResult:
        """Try several seeds; return the first success (seeds are tried in order), else the best failure."""
        best = None
        for s in seeds:
            r = self.solve(target_pos, target_rot, s, **kw)
            if r.success:
                return r
            if best is None or r.pos_err + 0.05 * r.rot_err < best.pos_err + 0.05 * best.rot_err:
                best = r
        return best
