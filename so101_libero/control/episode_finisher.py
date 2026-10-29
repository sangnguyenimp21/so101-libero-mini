"""The deterministic end of every episode, after the policy has stopped (success, timeout or abort):

    stop policy -> discard its queued actions -> safe release (only if needed) -> safe retreat -> return HOME

One implementation for the three tasks, the simulator and the real arm (``robot`` is a so101_libero/control/robots.py
backend). Nothing here is learned: SmolVLA only solves the manipulation; this module parks the arm.

safe_release_if_needed  Nothing happens if the jaws are open and free (commanded and measured open, no object felt):
    after a success that is always the case (every success rule requires a released cube), so a finished arrangement is
    never disturbed. Otherwise the arm first holds still for ``stabilize_s`` (the jaws finish whatever they were doing),
    the state is judged again, and unless the jaws are now open and free they are opened slowly (``release_s``) while
    the arm does not move: an object still held after a timeout or an abort is put down where it is held (no automatic
    recovery). Opening jaws that hold nothing is harmless.
safe_retreat  The TCP goes straight up (numpy IK, orientation kept; so101_libero/control/kinematics.py) by
    ``retreat_dz``, and at least to ``retreat_min_z`` above the table, at ``retreat_speed``: the open jaws slide off the
    cubes vertically instead of sweeping across them. If the arm is already that high, nothing moves.
return_home   One min-jerk joint move from the last command to HOME (average joint speed <= ``home_max_joint_speed``),
    then ``home_settle_s`` of holding; success = every arm joint within ``home_tol_deg`` of HOME.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from so101_libero.control.home import home_error, min_jerk_path
from so101_libero.control.kinematics import fk, min_jerk, vertical_lift_path


@dataclass
class FinisherConfig:
    stabilize_s: float = 0.3  # hold still before opening the gripper
    release_opening: float = 30.0  # gripper command of a release (policy units; the demonstrations release at ~27)
    release_s: float = 0.6  # gripper opening time (min-jerk)
    release_settle_s: float = 0.4  # after opening, before moving
    open_threshold: float = 20.0  # measured gripper >= this = open
    retreat_dz: float = 0.05  # m: lift the TCP at least this much ...
    retreat_min_z: float = 0.10  # m: ... and at least to this height above the table
    retreat_speed: float = 0.05  # m/s (average)
    home_max_joint_speed: float = 40.0  # deg/s average per joint (min-jerk peaks 1.875 x)
    home_min_s: float = 1.5
    home_settle_s: float = 0.5
    home_tol_deg: float = 3.0


class EpisodeFinisher:
    def __init__(self, robot, home, cfg: FinisherConfig | None = None, log=print, on_step=None):
        self.robot = robot
        self.home = np.asarray(home, dtype=np.float64)
        self.cfg = cfg or FinisherConfig()
        self.log = log
        self.on_step = on_step  # called after every control step the finisher commands (videos)

    # ------------------------------------------------------------------------------------------------ helpers
    def _send(self, target) -> None:
        self.robot.send(target)
        if self.on_step is not None:
            self.on_step()

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.robot.fps)))

    def _hold(self, seconds: float) -> None:
        for _ in range(self._steps(seconds)):
            self._send(self.robot.last_sent)

    # ------------------------------------------------------------------------------------------------ stages
    def stop_policy(self, policy, reason: str) -> dict:
        """No further ``act`` call is made after this (the controller's loop has ended); recorded for the log."""
        self.log(f"STOP_POLICY ({reason})")
        return {"reason": reason}

    def discard_pending_actions(self, policy) -> int:
        """Drop the rest of the current action chunk: it is never executed."""
        n = int(policy.discard_pending())
        self.log(f"discarded {n} queued action(s) of the current chunk")
        return n

    def _open_and_free(self, q) -> bool:
        c = self.cfg
        return (self.robot.holding_object() is False and float(q[5]) >= c.open_threshold
                and float(self.robot.last_sent[5]) >= c.open_threshold)

    def safe_release_if_needed(self) -> dict:
        c = self.cfg
        q = self.robot.read_joints()
        out = {"holding_before": self.robot.holding_object(), "gripper_before": round(float(q[5]), 2), "needed": False}
        if self._open_and_free(q):
            out["reason"] = "jaws open and free: nothing to release"
            self.log(f"SAFE_RELEASE: not needed ({out['reason']})")
            return out
        self._hold(c.stabilize_s)  # stop and let the jaws settle, then judge the settled state
        q = self.robot.read_joints()
        out["holding_after_stabilize"] = self.robot.holding_object()
        if self._open_and_free(q):
            out["reason"] = "jaws open and free after stabilizing"
            self.log(f"SAFE_RELEASE: not needed ({out['reason']})")
            return out
        out["needed"] = True
        self.log("SAFE_RELEASE: the jaws may hold an object: opening them slowly with the arm still")
        start = self.robot.last_sent.astype(np.float64).copy()
        g0 = float(start[5])
        g1 = max(g0, c.release_opening)
        for s in min_jerk(self._steps(c.release_s)):
            target = start.copy()
            target[5] = g0 + s * (g1 - g0)
            self._send(target)
        self._hold(c.release_settle_s)
        q = self.robot.read_joints()
        out.update(holding_after=self.robot.holding_object(), gripper_after=round(float(q[5]), 2))
        return out

    def safe_retreat(self) -> dict:
        c = self.cfg
        q_cmd = self.robot.last_sent.astype(np.float64)
        z0 = float(fk(self.robot.read_joints()[:5])[0][2])
        out = {"tcp_z_before": round(z0, 4), "performed": False}
        dz = max(c.retreat_dz, c.retreat_min_z - z0)
        if z0 >= c.retreat_min_z - 0.005:  # already up there (within 5 mm)
            out["reason"] = f"TCP already {100 * z0:.1f} cm above the table"
            self.log(f"SAFE_RETREAT: not needed ({out['reason']})")
            return out
        n = max(10, int(round(dz / c.retreat_speed * self.robot.fps)))
        path, gained = vertical_lift_path(q_cmd[:5], dz, n)
        for q_arm in path:
            self._send(np.r_[q_arm, q_cmd[5]])
        self._hold(0.2)
        z1 = float(fk(self.robot.read_joints()[:5])[0][2])
        out.update(performed=bool(len(path)), dz_requested=round(dz, 4), dz_planned=round(gained, 4),
                   tcp_z_after=round(z1, 4), steps=len(path))
        if len(path) < n:
            out["note"] = "the straight lift left the arm's reach early; the remaining height comes with the move home"
        self.log(f"SAFE_RETREAT: TCP {100 * z0:.1f} -> {100 * z1:.1f} cm ({len(path)} steps)")
        return out

    def return_home(self) -> dict:
        c = self.cfg
        path = min_jerk_path(self.robot.last_sent, self.home, self.robot.fps, c.home_max_joint_speed, c.home_min_s)
        for q in path:
            self._send(q)
        self._hold(c.home_settle_s)
        err = home_error(self.robot.read_joints(), self.home)
        ok = err <= c.home_tol_deg
        self.log(f"RETURN_HOME: {'reached' if ok else 'NOT reached'} (max arm joint error {err:.2f} deg)")
        return {"success": bool(ok), "max_error_deg": round(err, 3), "steps": int(len(path))}

    def move_to_home(self) -> dict:
        """The same move at the start of an episode (the arm may be anywhere, e.g. folded at rest)."""
        err = home_error(self.robot.read_joints(), self.home)
        if err <= self.cfg.home_tol_deg:
            self.log(f"MOVE_TO_HOME: already at HOME (max arm joint error {err:.2f} deg)")
            return {"success": True, "max_error_deg": round(err, 3), "steps": 0}
        return self.return_home()

    def config(self) -> dict:
        return {"home": self.home.tolist(), **asdict(self.cfg)}
