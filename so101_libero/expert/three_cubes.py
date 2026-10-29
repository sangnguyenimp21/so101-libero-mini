"""Scripted experts for the ``so101_three_cubes`` tasks (so101_libero/tasks/three_cubes.py).

The project's expert machinery throughout (damped-least-squares IK at every control step with checked convergence,
min-jerk timing under the joint-speed cap, collision-checked poses, physics checks at every phase boundary; grasps are
purely frictional). Every demonstration starts at HOME and ends right after the manipulation: release, a short retreat
straight up, a 1 s hold. Returning HOME is NOT part of the demonstrations: at run time the deterministic episode
finisher (so101_libero/control/) does it after the success detector has stopped the policy.

  FlipRedExpert  (FlipInPlace)  the one-grasp flip of the so101_tri cube_flip expert (approach tilted 45 deg outward,
                                lift, 180 deg wrist roll about the approach axis, set the cube down at its start), with the
                                grasp side and roll direction fixed for every demonstration
  PyramidExpert  (StackOnPair)  top-down pick of green, carried over the red/yellow midpoint, the wrist turned on the way
                                so the green cube lands square on the base, released 3-6 mm above it
  SortExpert     (SortIntoRow)  red, then yellow, then green: each picked with the jaws closing along the row (the moving
                                jaw on the outer side, away from the other cubes) and put on its target with the jaws
                                closing front-back (never reaching between neighbours in the row)
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from so101_libero.env import SO101TabletopEnv
from so101_libero.expert.ik import wrap_angle
from so101_libero.expert.pick_place import DEG, ExpertConfig, PlanError
from so101_libero.expert.tri_tasks import CubeFlipExpert, _TriExpert
from so101_libero.sim.robot import HOME_QPOS, gripper_rad_to_lerobot, lerobot_to_sim
from so101_libero.tasks.base import FlipInPlace, SortIntoRow, StackOnPair, row_targets

FINAL_HOLD_S = 1.0  # every demonstration ends holding still this long after its retreat (the policy learns to stop)
AT_HOME_TOL = 2 * DEG  # the arm counts as already at its start pose (no unfold move) within this


def _at_start(env: SO101TabletopEnv, q_arm: np.ndarray) -> bool:
    """Is the arm at the task's start pose (the suite's HOME), so the demonstration can go straight to the cube?"""
    start = HOME_QPOS if env.task.start_deg is None else lerobot_to_sim(np.asarray(env.task.start_deg, dtype=float))
    return bool(np.max(np.abs(np.asarray(q_arm)[:5] - start[:5])) < AT_HOME_TOL)


class FlipRedExpert(CubeFlipExpert):
    """``CubeFlipExpert`` with ONE motion: grasp side ``plan.psi_deg`` and roll direction ``plan.roll_dir`` are fixed (the
    base expert picks whichever is feasible), no unfold from HOME, and after the release a retreat straight up and a
    1 s hold instead of parking at rest."""

    def __init__(self, env: SO101TabletopEnv, plan: FlipInPlace, rng: np.random.Generator,
                 cfg: ExpertConfig | None = None):
        super().__init__(env, plan, rng, cfg)
        self._psi_fixed = math.radians(plan.psi_deg)
        self._roll_fixed = float(np.sign(plan.roll_dir) or 1.0)

    def _face_psis(self) -> list[float]:
        return [self._psi_fixed]

    def _extra_checks(self, psi: float, grasp_xy: np.ndarray, q_grasp: np.ndarray) -> None:
        """The base check for the one roll direction of the plan (no fallback to the other direction)."""
        pos = np.r_[grasp_xy, self.roll_z[1]]
        q = self._solve_frame(pos, self._frame(pos, phi=0.0, psi=psi), q_grasp)
        for k in range(1, 13):
            q = self._solve_frame(pos, self._frame(pos, phi=self._roll_fixed * math.pi * k / 12, psi=psi), q)
            hit = self._collision_any(q, self._closed, ignore=(self.plan.obj,))
            if hit:
                raise PlanError(f"roll collides ({hit})")
        self._release_tcp_z = self._lowest_release_z(grasp_xy, self._roll_fixed * math.pi, psi, q)
        self._roll_dir = self._roll_fixed

    def _advance(self) -> None:
        p = self.phase
        if p == "start" and _at_start(self.env, self._q):
            self._enter("approach")
            self._approach(self.ws.pos(self.plan.obj), self._face_psis(), self._u(self.pregrasp_z), self.grasp_z[0])
        elif p == "retreat":
            self._enter("hold")
            self._hold(FINAL_HOLD_S)
        else:
            super()._advance()  # start (unfold if not at HOME), approach ... release, retreat, hold -> done


class PyramidExpert(_TriExpert):
    """Pick ``plan.obj`` top-down and put it centred on the midpoint of ``plan.bases``, square to the table."""

    place_yaw: float = 0.0  # world yaw of the jaws' closing direction over the base (cube faces along the table axes)

    def __init__(self, env: SO101TabletopEnv, plan: StackOnPair, rng: np.random.Generator,
                 cfg: ExpertConfig | None = None):
        super().__init__(env, plan, rng, cfg, hold_time=FINAL_HOLD_S)

    def _advance(self) -> None:
        c, ws = self.cfg, self.ws
        p = self.phase
        if p == "start" and _at_start(self.env, self._q):
            self._enter("approach")
            self._plan_approach()
        elif p == "lift":
            if not self._held():
                return self._fail("dropped_during_lift")
            self._enter("transport")
            self._plan_transport_pair()
        elif p == "transport":
            if not self._held():
                return self._fail("dropped_during_transport")
            self._enter("lower")
            z = self._base_top + self._u(c.table_release_gap) + self._held_z  # object bottom just above the base
            self._cartesian(np.r_[self._place_target[:2], z], self._yaw_at(self._place_target), self._closed)
        else:
            super()._advance()

    def _plan_transport_pair(self) -> None:
        """Arc at carry height to over the bases' midpoint (+- stack_jitter), turning the jaws to ``place_yaw`` on the
        way; where the object sits in the jaws (measured after the lift) is carried over, so its centre lands on the
        goal."""
        held, yaw_now, self._held_z = self._held_offset()
        b0, b1 = (self.ws.pos(b) for b in self.plan.bases)
        self._base_top = float(max(b0[2] + self.env.h.objects[self.plan.bases[0]].half_height,
                                   b1[2] + self.env.h.objects[self.plan.bases[1]].half_height))
        goal = (b0[:2] + b1[:2]) / 2 + self.rng.uniform(-self.cfg.stack_jitter, self.cfg.stack_jitter, 2)
        tgt = np.r_[self._tcp_for(goal, held, self.place_yaw), 0.0]
        self._place_target = tgt
        tcp = self.ws.tcp_pos()
        self._arc_yaw(tcp, np.r_[tgt[:2], tcp[2]], yaw_now, self.place_yaw, self._closed)
        self._yaw_offset = wrap_angle(self.place_yaw - self.ik.azimuth(tgt))  # lower / release / retreat keep it


@dataclasses.dataclass(frozen=True)
class _OneCube:
    """Plan of one sort step (the attribute the pick-and-place machinery reads)."""

    obj: str


class _RowStepExpert(_TriExpert):
    """One sort step: pick ``obj`` with the jaws closing along the row (world yaw +-90 deg: the moving jaw on the side
    away from the row's centre line) and put it on ``target`` with the jaws closing front-back (``place_yaw`` = 0, the
    moving jaw outward). Starts from the previous step's last command and ends after its retreat."""

    place_yaw: float = 0.0
    row_jitter: float = 0.003  # m, per axis: random offset of each placement from its row target
    min_neighbour_dist: float = 0.035  # m between the goal and any other cube's centre (one cube + 5 mm)

    def __init__(self, env: SO101TabletopEnv, obj: str, target: np.ndarray, centre_y: float, q_start: np.ndarray,
                 rng: np.random.Generator, cfg: ExpertConfig):
        super().__init__(env, _OneCube(obj), rng, cfg)
        self._q = np.asarray(q_start, dtype=float).copy()
        self.target = np.asarray(target, dtype=float)
        self.centre_y = float(centre_y)

    def _choose_grasp_yaw(self, obj_pos: np.ndarray) -> float:
        """+90 deg (moving jaw on the robot's left) for a cube left of the centre line, else -90 deg; the other side only
        if that one is unreachable or collides."""
        first = math.pi / 2 if obj_pos[1] > self.centre_y + 0.02 else -math.pi / 2
        az = self.ik.azimuth(obj_pos)
        for yaw in (first, -first):
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
        raise PlanError("no collision-free grasp across the row direction")

    def _plan_approach(self, retry: bool = False) -> None:
        """Joint move to the pre-grasp, via a raised waypoint if the direct joint path would touch a cube."""
        c = self.cfg
        obj = self.ws.pos(self.plan.obj)
        yaw = self._choose_grasp_yaw(obj)
        closing = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        self._grasp_target = obj - closing * self._approach_offset
        self._grasp_yaw = yaw
        pre = np.r_[self._grasp_target[:2], self._u(c.pregrasp_height)]
        _, q_pre = self._solve(pre, yaw)
        try:
            self._check_joint_path(q_pre, self._open)
        except PlanError:
            tcp = self.ws.tcp_pos()
            up = np.r_[tcp[:2], max(tcp[2], 0.10) + 0.04]
            _, q_up = self._solve(up, self._yaw_now())
            self._check_joint_path(q_up, self._open)
            self._joint_move(q_up, self._open)
            self._check_joint_path(q_pre, self._open)
        self._joint_move(q_pre, self._open)

    def _advance(self) -> None:
        c, ws = self.cfg, self.ws
        p = self.phase
        if p == "start":
            self._enter("approach")
            self._plan_approach()
        elif p == "lift":
            if not self._held():
                return self._fail("dropped_during_lift")
            self._enter("transport")
            held, yaw_now, self._held_z = self._held_offset()
            goal = self.target + self.rng.uniform(-self.row_jitter, self.row_jitter, 2)
            tgt = np.r_[self._tcp_for(goal, held, self.place_yaw), 0.0]
            for other in self.env.h.objects:  # a neighbour in the row stands 5 cm away; anything closer is in the way
                if other != self.plan.obj and np.linalg.norm(ws.pos(other)[:2] - goal) < self.min_neighbour_dist:
                    raise PlanError(f"row target of {self.plan.obj} is occupied by {other}")
            self._place_target = tgt
            tcp = ws.tcp_pos()
            self._arc_yaw(tcp, np.r_[tgt[:2], tcp[2]], yaw_now, self.place_yaw, self._closed)
            self._yaw_offset = wrap_angle(self.place_yaw - self.ik.azimuth(tgt))
        elif p == "transport":
            if not self._held():
                return self._fail("dropped_during_transport")
            self._enter("lower")
            z = self._held_z + self._u(c.table_release_gap)  # the cube's bottom just above the table
            self._cartesian(np.r_[self._place_target[:2], z], self._yaw_at(self._place_target), self._closed)
        elif p == "retreat":  # the step ends after its retreat; the next cube (or the final hold) follows
            self.done = True
            self.phase = "done"
        else:
            super()._advance()  # approach -> descend -> close -> lift; lower -> release -> retreat


class SortExpert:
    """Sort the cubes into the row, one ``_RowStepExpert`` per cube in row order (red, yellow, green), then hold.

    Same interface as the other experts (``act``, ``phase``, ``log``, ``failure``, ``done``, ``max_ik_err``,
    ``deliberate_miss``) plus ``active_obj``, the cube being handled (grasp monitoring)."""

    def __init__(self, env: SO101TabletopEnv, plan: SortIntoRow, rng: np.random.Generator,
                 cfg: ExpertConfig | None = None):
        self.env, self.plan, self.rng = env, plan, rng
        self.cfg = cfg or ExpertConfig()
        self.ws = env.world
        self.targets = row_targets(env.world, plan.objs, plan.targets, plan.slot_centroid)
        self.centre_y = float(np.mean(self.targets[:, 1]))
        self.log: list[tuple[int, str]] = []
        self.failure: str | None = None
        self.done = False
        self.deliberate_miss = False
        self.max_ik_err = 0.0
        self._k = 0
        self._sub: _RowStepExpert | None = None
        self._q = env.data.qpos[env.h.joint_qpos_adr].copy()
        self._step = 0
        self._hold: list[np.ndarray] = []
        self._holding = False

    @property
    def active_obj(self) -> str | None:
        return self.plan.objs[self._k] if self._k < len(self.plan.objs) else None

    @property
    def phase(self) -> str:
        if self._sub is not None:
            return self._sub.phase
        return "hold" if self._holding else ("done" if self.done else "start")

    def act(self) -> np.ndarray | None:
        while not self.done:
            if self._sub is None and self._k < len(self.plan.objs):
                self._sub = _RowStepExpert(self.env, self.plan.objs[self._k], self.targets[self._k], self.centre_y,
                                           self._q, self.rng, self.cfg)
            if self._sub is not None:
                n_log = len(self._sub.log)
                action = self._sub.act()
                for t, ph in self._sub.log[n_log:]:
                    self.log.append((self._step, f"{self._sub.plan.obj}:{ph}"))
                self.max_ik_err = max(self.max_ik_err, self._sub.max_ik_err)
                if action is not None:
                    self._q = self._sub._q.copy()
                    self._step += 1
                    return action
                if self._sub.failure is not None:
                    self.failure = f"{self._sub.plan.obj}:{self._sub.failure}"
                    self.done = True
                    return None
                self._sub, self._k = None, self._k + 1
                continue
            if not self._holding:  # every cube is on its target: hold still
                self._holding = True
                self.log.append((self._step, "hold"))
                hold = np.r_[np.degrees(self._q[:5]), gripper_rad_to_lerobot(self._q[5])].astype(np.float32)
                self._hold = [hold] * max(1, int(round(FINAL_HOLD_S / self.env.dt)))
            if self._hold:
                self._step += 1
                return self._hold.pop()
            self.done = True
        return None


def make_three_cube_expert(env: SO101TabletopEnv, plan, rng: np.random.Generator, cfg: ExpertConfig | None = None):
    if isinstance(plan, FlipInPlace):
        return FlipRedExpert(env, plan, rng, cfg)
    if isinstance(plan, StackOnPair):
        return PyramidExpert(env, plan, rng, cfg)
    if isinstance(plan, SortIntoRow):
        return SortExpert(env, plan, rng, cfg)
    return None
