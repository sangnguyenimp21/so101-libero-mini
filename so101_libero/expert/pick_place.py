"""Scripted pick-and-place expert with privileged state (object poses, contacts).

Phases: approach (joint-space) -> descend -> close -> lift -> transport -> lower -> release -> retreat -> hold.
Every Cartesian segment is solved with IK at each control step and checked (no unconverged solution is ever
sent); durations follow a joint-speed cap and min-jerk timing, so commands are smooth and executable on the
real arm. Each phase is planned when it starts, from the current state, and the expert checks the physics at
phase boundaries (grasp formed, object lifted, object still held, object placed). Grasps are purely
frictional: nothing is welded, teleported or attached.

Actions are absolute joint targets in LeRobot units, exactly what the environment (and a learned policy)
sends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from so101_libero.env import SO101TabletopEnv
from so101_libero.expert.ik import IKSolver, grasp_frame, wrap_angle
from so101_libero.sim.robot import HOME_QPOS, gripper_rad_to_lerobot, lerobot_to_sim
from so101_libero.tasks.base import SIDES, PickPlace, PlaceAt, PlaceRelative, StackOn

DEG = math.pi / 180


@dataclass
class ExpertConfig:
    max_joint_speed: float = 75 * DEG  # rad/s, sets segment durations (real-arm friendly)
    min_segment_time: float = 0.5  # s
    pregrasp_height: tuple[float, float] = (0.065, 0.085)  # TCP height above the table
    grasp_height: tuple[float, float] = (0.017, 0.021)  # TCP height when closing (finger tips ~6-9 mm up)
    approach_offset: float = 0.006  # TCP shifted towards the fixed jaw: the fixed jaw clears the object
    open_deg: float = 32.0  # gripper angle while approaching (jaw gap ~5.3 cm)
    close_deg: float = -10.0  # commanded when closing (past contact: the actuator squeezes)
    release_deg: float = 20.0  # opened only this far inside containers (jaw gap ~4.2 cm)
    carry_height: tuple[float, float] = (0.075, 0.09)
    # The held object's bottom stays this far above the target container's walls while carried: carry_height is
    # raised by whatever it lacks (nothing for walls up to ~3 cm; the 6 cm white tray lifts it by ~2.5 cm).
    wall_clearance: float = 0.02
    # Inside tall containers the object is released with its bottom at most this far below the rim: release_height is
    # raised by whatever it lacks (nothing for walls up to ~4 cm), so the jaws stay clear of the walls.
    release_depth: float = 0.025
    park_hold_time: float = 2.0  # tasks with return_to_start: final hold at the start pose (the policy learns to stay)
    release_height: tuple[float, float] = (0.036, 0.044)  # TCP height when opening inside the container
    table_release_gap: tuple[float, float] = (0.003, 0.006)  # object bottom above the table (or the base) when placing
    place_jitter: float = 0.005  # random offset of table placements (m)
    stack_jitter: float = 0.003  # random offset (m, per axis) of the held object's centre from the base's when stacking
    retreat_height: float = 0.10
    place_margin: float = 0.012  # release point stays this far inside the inner walls (beyond the object)
    close_time: float = 0.5
    settle_time: float = 0.25
    hold_time: float = 0.4  # after the retreat, before the episode ends
    time_scale: tuple[float, float] = (0.9, 1.15)  # per-segment duration randomization
    max_tilt_deg: float = 45.0
    yaw_clearance: float = 0.012  # min distance from the open jaws to other items at the grasp
    max_tracking_error_deg: float = 8.0  # arm joint |command - measured| beyond this for 3 steps = blocked
    # True reproduces the expert of datasets generated before 2026-10-17 01:45 bit for bit: IK fallback seeds with
    # the mirrored pan sign and tilted approaches aimed at the TCP instead of the wrist-roll axis (both fixed since;
    # about 10 % of those demos would come out slightly different, all equally valid).
    legacy_ik: bool = False
    # Recovery demonstrations (off by default): a failed grasp, or an object dropped right after lifting, is retried
    # up to max_grasp_retries times (re-open, rise, re-plan from the object's current pose); with probability
    # miss_prob the first attempt is deliberately offset sideways by miss_offset (m) so the data shows recoveries.
    max_grasp_retries: int = 0
    miss_prob: float = 0.0
    miss_offset: tuple[float, float] = (0.012, 0.02)
    # Offsets and openings above are tuned for a 3 cm wide object. For a wider/narrower one (per-episode sizes) the
    # approach and miss offsets shift by the difference in half width (the fixed jaw keeps its clearance) and the
    # release opening grows by this many degrees per mm of extra width (never shrinks below release_deg).
    nominal_half_width: float = 0.015
    release_deg_per_mm: float = 1.1  # jaw gap grows ~0.9 mm per degree around 20-32 deg


def min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class PlanError(RuntimeError):
    pass


class PickPlaceExpert:
    """Call ``act()`` once per control step; it returns the action or None once ``done``."""

    def __init__(self, env: SO101TabletopEnv, plan: PickPlace | PlaceRelative | PlaceAt | StackOn, rng: np.random.Generator,
                 cfg: ExpertConfig | None = None):
        self.env, self.plan, self.rng = env, plan, rng
        self.cfg = cfg or ExpertConfig()
        self.ik = IKSolver(env.h)  # path following: 1 deg from the joint limits
        # Segment end poses are chosen 4 deg inside the joint limits, so the path between them has slack (an end
        # pose on a limit makes nearby poses infeasible and the plan brittle).
        self.ik_goal = IKSolver(env.h, joint_margin=4 * DEG)
        self.ws = env.world
        self.dt = env.dt
        self.obj = env.h.objects[plan.obj]
        self.container = env.h.containers[plan.target] if isinstance(plan, PickPlace) else None
        self._dw = self.obj.grasp_width / 2 - self.cfg.nominal_half_width  # 0.0 exactly for 3 cm objects
        self.phase = "start"
        self.done = False
        self.failure: str | None = None
        self.log: list[tuple[int, str]] = []
        self._queue: list[np.ndarray] = []  # pending actions (LeRobot units)
        self._step = 0
        self._q = env.data.qpos[env.h.joint_qpos_adr].copy()  # last commanded joint targets (rad)
        self._yaw_offset = 0.0  # jaw closing yaw relative to the arm azimuth, fixed after the grasp
        self.max_ik_err = 0.0
        self._scratch = mujoco.MjData(env.model)  # for collision checks of planned configurations
        self._blocked_steps = 0
        self.retries = 0
        self.deliberate_miss = False

    # ----------------------------------------------------------------------------------------- public
    def act(self) -> np.ndarray | None:
        if self._tracking_blocked():
            self._fail("blocked")
        while not self._queue and not self.done:
            try:
                self._advance()
            except PlanError as e:
                self._fail(f"plan:{e}")
        if self.done:
            return None
        self._step += 1
        return self._queue.pop(0)

    # ---------------------------------------------------------------------------------------- phases
    def _advance(self) -> None:
        c, ws = self.cfg, self.ws
        p = self.phase
        if p == "start" and self.env.task.start_deg is not None:  # e.g. from the folded rest pose: unfold first
            self._enter("unfold")
            self._joint_move(HOME_QPOS[:5], self._q[5])
        elif p in ("start", "unfold"):
            self._enter("approach")
            self._plan_approach()
        elif p == "approach":
            self._enter("descend")
            g = self._grasp_target
            self._grasp_z = self._u(c.grasp_height)  # TCP height above the object's bottom while it is held
            self._cartesian(np.r_[g[:2], self._grasp_z], self._grasp_yaw, self._open)
        elif p == "descend":
            self._enter("close")
            self._gripper_move(self._closed, c.close_time)
            self._hold(c.settle_time)
        elif p == "close":
            if not ws.grasped(self.plan.obj):
                return self._retry_or_fail("grasp_failed")
            self._enter("lift")
            tcp = ws.tcp_pos()
            self._cartesian(np.r_[tcp[:2], self._carry_z()], self._yaw_now(), self._closed)
        elif p == "lift":
            if not self._held():
                return self._retry_or_fail("dropped_during_lift")
            self._enter("transport")
            self._plan_transport()
        elif p == "transport":
            if not self._held():
                return self._fail("dropped_during_transport")
            self._enter("lower")
            tgt = self._place_target
            if self.container is not None:
                z = self._release_z()
            elif isinstance(self.plan, StackOn):  # the object's bottom just above the base's top
                z = self._base_top + self._u(c.table_release_gap) + self._held_z
            else:  # on the table: the object's bottom just above the table top
                z = self._grasp_z + self._u(c.table_release_gap)
            self._cartesian(np.r_[tgt[:2], z], self._yaw_at(tgt), self._closed)
        elif p == "lower":
            if not self._held(min_bottom=0.005 if self.container is not None else 0.0):
                return self._fail("dropped_while_lowering")
            self._enter("release")
            self._gripper_move(self._release, 0.4)
            self._hold(c.settle_time)
        elif p == "regrasp":
            self._enter("approach")
            self._plan_approach(retry=True)
        elif p == "release":
            self._enter("retreat")
            tcp = ws.tcp_pos()
            self._cartesian(np.r_[tcp[:2], c.retreat_height], self._yaw_now(), self._release)
        elif p == "retreat" and self.env.task.return_to_start:  # park: back to home, then the start pose
            self._enter("return")
            start = lerobot_to_sim(np.asarray(self.env.task.start_deg if self.env.task.start_deg is not None
                                              else np.r_[np.degrees(HOME_QPOS[:5]), 0.0]))
            self._joint_move(HOME_QPOS[:5], start[5])
            self._joint_move(start[:5], start[5])
        elif p == "return":
            self._enter("hold")
            self._hold(c.park_hold_time)
        elif p == "retreat":
            self._enter("hold")
            self._hold(c.hold_time)
        elif p == "hold":
            self.done = True
            self.phase = "done"
        else:
            raise AssertionError(p)

    def _retry_or_fail(self, reason: str) -> None:
        """Recover from a failed grasp: open, rise to the pre-grasp height, then approach the object again."""
        if self.retries >= self.cfg.max_grasp_retries:
            return self._fail(reason)
        self.retries += 1
        self._enter("regrasp")
        self.log[-1] = (self._step, f"regrasp ({reason})")
        self._gripper_move(self._open, 0.4)
        tcp = self.ws.tcp_pos()
        self._cartesian(np.r_[tcp[:2], max(tcp[2], self.cfg.pregrasp_height[0])], self._yaw_now(), self._open)

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self.log.append((self._step, phase))

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self.done = True
        self._queue.clear()
        self.log.append((self._step, f"FAIL {reason}"))

    # ---------------------------------------------------------------------------------------- planning
    @property
    def _open(self) -> float:
        return self.cfg.open_deg * DEG

    @property
    def _closed(self) -> float:
        return self.cfg.close_deg * DEG

    @property
    def _release(self) -> float:
        return (self.cfg.release_deg + max(0.0, 2e3 * self._dw) * self.cfg.release_deg_per_mm) * DEG

    @property
    def _approach_offset(self) -> float:
        return self.cfg.approach_offset + self._dw

    def _u(self, rng_pair: tuple[float, float]) -> float:
        return float(self.rng.uniform(*rng_pair))

    def _plan_approach(self, retry: bool = False) -> None:
        c = self.cfg
        obj = self.ws.pos(self.plan.obj)
        yaw = self._choose_grasp_yaw(obj)
        closing = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        # Shift the TCP towards the fixed jaw so that the fixed jaw clears the object on the way down; the moving
        # jaw then pushes the object against the fixed jaw while closing.
        self._grasp_target = obj - closing * self._approach_offset
        if not retry and c.miss_prob > 0 and self.rng.uniform() < c.miss_prob:
            # Deliberate miss for recovery data: shift along the jaw width (not along the closing axis, where the
            # jaws would land on the object), so the jaws close beside it or on its edge.
            lateral = np.array([-closing[1], closing[0], 0.0]) * self.rng.choice([-1.0, 1.0])
            self._grasp_target = self._grasp_target + lateral * (self._u(c.miss_offset) + self._dw)
            self.deliberate_miss = True
        self._grasp_yaw = yaw
        pre = np.r_[self._grasp_target[:2], self._u(c.pregrasp_height)]
        _, q_pre = self._solve(pre, yaw)
        self._joint_move(q_pre, self._open)

    def _carry_z(self) -> float:
        """TCP height while carrying: the sampled carry height, shifted up (range kept) until the object's bottom
        (``_grasp_z`` below the TCP) clears the container walls by ``wall_clearance``."""
        c = self.cfg
        z = self._u(c.carry_height)
        if self.container is not None:
            z += max(0.0, self.container.wall_height + c.wall_clearance + self._grasp_z - c.carry_height[0])
        return z

    def _release_z(self) -> float:
        """TCP height when opening inside the container: the sampled release height, shifted up (range kept) until the
        object's bottom is at most ``release_depth`` below the rim."""
        c = self.cfg
        z = self._u(c.release_height)
        return z + max(0.0, self.container.wall_height - c.release_depth + self._grasp_z - c.release_height[0])

    def _plan_transport(self) -> None:
        if isinstance(self.plan, PlaceRelative):
            return self._plan_transport_relative()
        if isinstance(self.plan, PlaceAt):
            return self._plan_transport_at()
        if isinstance(self.plan, StackOn):
            return self._plan_transport_stack()
        c = self.cfg
        center = self.ws.pos(self.plan.target)
        rot = self.ws.rot(self.plan.target)
        # Random release point inside the container, away from the walls (the object hangs at the TCP).
        o = self.obj.footprint_radius + c.place_margin
        if self.container.shape == "tray":
            lim = np.array(self.container.inner_size) - o
            local = self.rng.uniform(-lim, lim) * 0.6
        else:
            r = (self.container.inner_size[0] - o) * 0.6 * math.sqrt(self.rng.uniform())
            a = self.rng.uniform(-math.pi, math.pi)
            local = np.array([r * math.cos(a), r * math.sin(a)])
        tgt = center + rot[:, :2] @ local
        # The moving jaw opens on the +closing side: shift the release point towards the fixed-jaw side.
        yaw = self._yaw_at(tgt)
        tgt[:2] -= 0.01 * np.array([math.cos(yaw), math.sin(yaw)])
        self._place_target = tgt
        tcp = self.ws.tcp_pos()
        self._arc(tcp, np.r_[tgt[:2], tcp[2]], self._closed)

    def _plan_transport_relative(self) -> None:
        """Table spot ``distance`` from the reference object on the requested side; it and the jaws must be clear."""
        p = self.plan
        axis, sign = SIDES[p.side]
        tgt = self.ws.pos(p.ref)
        tgt[axis] += sign * p.distance
        tgt[:2] += self.rng.uniform(-self.cfg.place_jitter, self.cfg.place_jitter, 2)
        for name in (*self.env.h.objects, *self.env.h.containers):
            if name == p.obj:
                continue
            item = self.env.h.objects.get(name) or self.env.h.containers[name]
            gap = float(np.linalg.norm(self.ws.pos(name)[:2] - tgt[:2])) - item.footprint_radius - self.obj.footprint_radius
            if gap < 0.008:
                raise PlanError(f"target spot occupied by {name}")
        if self._jaw_clearance(tgt, self._yaw_at(tgt), moving_reach=0.046) < 0.004:
            raise PlanError("jaws would hit an item at the target spot")
        self._place_target = tgt
        tcp = self.ws.tcp_pos()
        self._arc(tcp, np.r_[tgt[:2], tcp[2]], self._closed)

    def _plan_transport_at(self) -> None:
        """Random table spot inside the plan's region (shrunk by its margin) that is clear of other items and
        reachable at carry and release height; up to 20 candidates, else the episode fails (recorded)."""
        p, c = self.plan, self.cfg
        lo = np.array([p.region.x[0], p.region.y[0]]) + p.margin
        hi = np.array([p.region.x[1], p.region.y[1]]) - p.margin
        if np.any(hi < lo):
            raise PlanError("placement region smaller than its margin")
        tcp = self.ws.tcp_pos()
        z_release = self._grasp_z + c.table_release_gap[1]
        for _ in range(20):
            tgt = np.r_[self.rng.uniform(lo, hi), 0.0]
            yaw = self._yaw_at(tgt)
            if self._spot_blocked(tgt) or self._jaw_clearance(tgt, yaw, moving_reach=0.046) < 0.004:
                continue
            try:
                self._solve(np.r_[tgt[:2], tcp[2]], yaw)
                self._solve(np.r_[tgt[:2], z_release], yaw)
            except PlanError:
                continue
            self._place_target = tgt
            self._arc(tcp, np.r_[tgt[:2], tcp[2]], self._closed)
            return
        raise PlanError("no reachable free spot in the placement region")

    def _plan_transport_stack(self) -> None:
        """Over the base object, so that the held object's centre lands on the base's centre (+- ``stack_jitter``).
        Where the object sits in the jaws is measured now (after the lift) in the jaw frame and carried over to the
        jaw yaw at the release point; so is the TCP height above the object's bottom (any slip included)."""
        tcp, obj = self.ws.tcp_pos(), self.ws.pos(self.plan.obj)
        rot = self.ws.d.site_xmat[self.env.h.tcp_site].reshape(3, 3)
        yaw_now = math.atan2(rot[1, 0], rot[0, 0])
        c, s = math.cos(-yaw_now), math.sin(-yaw_now)
        dx, dy = obj[:2] - tcp[:2]
        held = np.array([c * dx - s * dy, s * dx + c * dy])  # object centre - TCP, jaw frame
        self._held_z = float(tcp[2] - (obj[2] - self.obj.half_height))
        base = self.ws.pos(self.plan.base)
        self._base_top = float(base[2] + self.env.h.objects[self.plan.base].half_height)
        goal = base[:2] + self.rng.uniform(-self.cfg.stack_jitter, self.cfg.stack_jitter, 2)
        tgt = np.r_[goal, 0.0]
        for _ in range(3):  # the jaw yaw follows the target's azimuth
            yaw = self._yaw_at(tgt)
            c, s = math.cos(yaw), math.sin(yaw)
            tgt[:2] = goal - np.array([c * held[0] - s * held[1], s * held[0] + c * held[1]])
        self._place_target = tgt
        self._arc(tcp, np.r_[tgt[:2], tcp[2]], self._closed)

    def _spot_blocked(self, tgt: np.ndarray, min_gap: float = 0.008) -> bool:
        """Would the object placed at ``tgt`` overlap (within ``min_gap``) another item?"""
        for name in (*self.env.h.objects, *self.env.h.containers):
            if name == self.plan.obj:
                continue
            item = self.env.h.objects.get(name) or self.env.h.containers[name]
            gap = float(np.linalg.norm(self.ws.pos(name)[:2] - tgt[:2])) - item.footprint_radius - self.obj.footprint_radius
            if gap < min_gap:
                return True
        return False

    def _choose_grasp_yaw(self, obj_pos: np.ndarray) -> float:
        """Jaw closing direction: aligned with a cube face (any yaw for a cylinder), preferring small wrist roll
        (close to radial), open jaws clear of the other items, and a collision-free grasp configuration
        (the wrist-camera mount can hit the shoulder when the object is close to the base)."""
        az = self.ik.azimuth(obj_pos)
        if self.obj.yaw_symmetry > 0:
            cands = self.ws.yaw(self.plan.obj) + np.arange(-4, 5) * self.obj.yaw_symmetry
        else:
            cands = az + np.radians(np.arange(-120, 121, 15))
        scored = []
        for yaw in cands:
            dev = abs(wrap_angle(yaw - az))
            if dev <= 125 * DEG:
                scored.append((self._jaw_clearance(obj_pos, yaw) < self.cfg.yaw_clearance, round(dev, 6), float(yaw)))
        scored.sort()
        for _, _, yaw in scored:
            closing = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            grasp = obj_pos - closing * self._approach_offset
            try:
                _, q_grasp = self._solve(np.r_[grasp[:2], self.cfg.grasp_height[0]], yaw)
                _, q_pre = self._solve(np.r_[grasp[:2], self.cfg.pregrasp_height[1]], yaw)
            except PlanError:
                continue
            if self._collision(q_grasp, self._open) is None and self._collision(q_pre, self._open) is None:
                self._yaw_offset = wrap_angle(yaw - az)
                return yaw
        raise PlanError("no collision-free grasp yaw")

    def _collision(self, q_arm: np.ndarray, grip: float) -> str | None:
        """Robot contact (with itself, the table or a non-target item) in configuration ``q_arm``, else None."""
        h, d = self.env.h, self._scratch
        d.qpos[:] = self.env.data.qpos
        d.qpos[h.joint_qpos_adr] = np.r_[q_arm, grip]
        mujoco.mj_kinematics(self.env.model, d)
        mujoco.mj_collision(self.env.model, d)
        target = h.item_geoms[self.plan.obj]
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = h.model.geom_bodyid[c.geom1], h.model.geom_bodyid[c.geom2]
            r1, r2 = b1 in h.robot_bodies, b2 in h.robot_bodies
            if not (r1 or r2) or c.geom1 in target or c.geom2 in target:
                continue
            return f"{h.model.body(b1).name}-{h.model.body(b2).name}"
        return None

    def _jaw_clearance(self, obj_pos: np.ndarray, yaw: float, moving_reach: float = 0.058) -> float:
        """Smallest distance from the open jaws (two segments across the closing direction) to other items.
        ``moving_reach``: outer face of the moving jaw from the TCP (0.058 at the 32 deg approach opening, ~0.046 at
        the 20 deg release opening)."""
        cdir = np.array([math.cos(yaw), math.sin(yaw)])
        lat = np.array([-cdir[1], cdir[0]])
        tcp = obj_pos[:2] - cdir * self._approach_offset
        jaws = [tcp - cdir * 0.018, tcp + cdir * moving_reach]  # fixed jaw outer face, moving jaw outer face
        best = float("inf")
        for name in (*self.env.h.objects, *self.env.h.containers):
            if name == self.plan.obj:
                continue
            item = self.env.h.objects.get(name) or self.env.h.containers[name]
            p = self.ws.pos(name)[:2]
            for j in jaws:
                for s in (-0.015, 0.0, 0.015):
                    best = min(best, float(np.linalg.norm(j + lat * s - p)) - item.footprint_radius)
        return best

    def _yaw_at(self, pos: np.ndarray) -> float:
        return self.ik.azimuth(pos) + self._yaw_offset

    def _yaw_now(self) -> float:
        return self._yaw_at(self.ws.tcp_pos())

    def _solve(self, pos: np.ndarray, yaw: float, seed: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """IK with the least outward tilt that converges (vertical whenever reachable): (tilt, q)."""
        seed = self._q[:5] if seed is None else seed
        pan = (1.0 if self.cfg.legacy_ik else -1.0) * self.ik.azimuth(pos)  # fallback seeds facing the target
        seeds = [seed, np.r_[pan, -0.5, 0.7, 1.2, seed[4]], np.r_[pan, 0.0, 0.0, 1.5, seed[4]]]
        for tilt in np.radians(np.arange(0, self.cfg.max_tilt_deg + 1e-6, 2.5)):
            r = self.ik_goal.solve_multi(pos, self._frame(pos, tilt, yaw), seeds)
            if r.success:
                self.max_ik_err = max(self.max_ik_err, r.pos_err)
                return float(tilt), r.q
        raise PlanError(f"unreachable pos={np.round(pos, 3).tolist()}")

    def _frame(self, pos: np.ndarray, tilt: float, yaw: float) -> np.ndarray:
        if self.cfg.legacy_ik:
            return grasp_frame(self.ik.approach_dir(pos, tilt), yaw)
        return self.ik.tilted_frame(pos, float(tilt), yaw)

    def _tilt_of(self, q_arm: np.ndarray) -> float:
        _, rot = self.ik.fk(q_arm)
        return math.acos(float(np.clip(rot[2, 2], -1.0, 1.0)))  # TCP z = -approach; 0 when pointing down

    # ---------------------------------------------------------------------------------------- segments
    def _duration(self, q0: np.ndarray, q1: np.ndarray) -> int:
        t = max(self.cfg.min_segment_time, float(np.max(np.abs(q1 - q0))) / self.cfg.max_joint_speed)
        t *= self._u(self.cfg.time_scale)
        return max(2, int(round(t / self.dt)))

    def _emit(self, q_arm: np.ndarray, grip: float) -> None:
        q = np.r_[q_arm, grip]
        self._queue.append(np.r_[np.degrees(q[:5]), gripper_rad_to_lerobot(q[5])].astype(np.float32))
        self._q = q

    def _joint_move(self, q_goal: np.ndarray, grip_goal: float) -> None:
        q0 = self._q.copy()
        n = self._duration(q0[:5], q_goal)
        for s in min_jerk(n):
            self._emit(q0[:5] + s * (q_goal - q0[:5]), q0[5] + s * (grip_goal - q0[5]))

    def _path(self, pose_at, length: float, max_speed: float, grip: float) -> None:
        """Follow ``pose_at(s) -> (pos, yaw)``, s in [0, 1], with IK at every control step.

        The tilt of the gripper goes smoothly from the current tilt to the least feasible tilt at the goal; if
        an intermediate pose is unreachable, a smooth bump (zero at both ends) is added to the tilt profile.
        No per-step tilt search, so the joint targets cannot jump.
        """
        pos1, yaw1 = pose_at(1.0)
        t1, q_goal = self._solve(pos1, yaw1)
        t0 = self._tilt_of(self._q[:5])
        n = self._duration(self._q[:5], q_goal)
        n = max(n, int(round(length / max_speed / self.dt)))
        g0 = self._q[5]
        for extra in np.radians([0.0, 5.0, 10.0, 15.0, 20.0, 30.0]):
            q, qs = self._q[:5].copy(), []
            for sv in min_jerk(n):
                pos, yaw = pose_at(sv)
                tilt = t0 + sv * (t1 - t0) + extra * 4 * sv * (1 - sv)
                r = self.ik.solve(pos, self._frame(pos, tilt, yaw), q)
                if not r.success:
                    break
                self.max_ik_err = max(self.max_ik_err, r.pos_err)
                q = r.q
                qs.append(q)
            else:
                for sv, q in zip(min_jerk(n), qs):
                    self._emit(q, g0 + sv * (grip - g0))
                self._check_smooth()
                return
        raise PlanError(f"no feasible tilt profile to {np.round(pos1, 3).tolist()}")

    def _cartesian(self, goal: np.ndarray, yaw_goal: float, grip: float) -> None:
        """Straight TCP line from the current commanded pose (<= 12 cm/s), jaw yaw interpolated."""
        p0, _ = self.ik.fk(self._q[:5])
        yaw0 = self._yaw_from_q(self._q[:5])
        dyaw = wrap_angle(yaw_goal - yaw0)
        self._path(lambda sv: (p0 + sv * (goal - p0), yaw0 + sv * dyaw), float(np.linalg.norm(goal - p0)), 0.12, grip)

    def _arc(self, start: np.ndarray, goal: np.ndarray, grip: float) -> None:
        """Constant-height move interpolating radius and azimuth around the pan axis (<= 15 cm/s)."""
        r0, r1 = self.ik.radius(start), self.ik.radius(goal)
        a0 = self.ik.azimuth(start)
        da = wrap_angle(self.ik.azimuth(goal) - a0)
        c = self.ik.pan_axis_xy

        def pose_at(sv):
            a, r = a0 + sv * da, r0 + sv * (r1 - r0)
            pos = np.array([c[0] + r * math.cos(a), c[1] + r * math.sin(a), start[2] + sv * (goal[2] - start[2])])
            return pos, a + self._yaw_offset

        self._path(pose_at, float(np.linalg.norm(goal - start)), 0.15, grip)

    def _gripper_move(self, grip_goal: float, duration: float) -> None:
        q0 = self._q.copy()
        for s in min_jerk(max(2, int(round(duration / self.dt)))):
            self._emit(q0[:5], q0[5] + s * (grip_goal - q0[5]))

    def _hold(self, duration: float) -> None:
        for _ in range(max(1, int(round(duration / self.dt)))):
            self._emit(self._q[:5], self._q[5])

    def _yaw_from_q(self, q_arm: np.ndarray) -> float:
        _, rot = self.ik.fk(q_arm)
        return math.atan2(rot[1, 0], rot[0, 0])

    def _check_smooth(self) -> None:
        """Reject plans with joint-target jumps (e.g. an IK branch switch)."""
        qs = np.array([np.radians(a[:5]) for a in self._queue])
        if len(qs) > 1 and np.max(np.abs(np.diff(qs, axis=0))) > 2.5 * self.cfg.max_joint_speed * self.dt:
            raise PlanError(f"joint jump {np.degrees(np.max(np.abs(np.diff(qs, axis=0)))):.1f} deg/step")

    # ----------------------------------------------------------------------------------------- checks
    def _tracking_blocked(self) -> bool:
        """The arm stopped following its commands (unplanned collision)."""
        h = self.env.h
        err = np.abs(self.env.data.qpos[h.joint_qpos_adr[:5]] - self.env.data.ctrl[h.actuator_ids[:5]])
        self._blocked_steps = self._blocked_steps + 1 if np.max(err) > self.cfg.max_tracking_error_deg * DEG else 0
        return self._blocked_steps >= 3

    def _held(self, min_bottom: float = 0.005) -> bool:
        """Target object pinched by both jaws, its bottom at least ``min_bottom`` above the table."""
        z = self.ws.pos(self.plan.obj)[2] - self.obj.half_height
        return self.ws.grasped(self.plan.obj) and z > min_bottom
