"""Scripted experts for the ``so101_tri`` tasks (so101_libero/tasks/tri_tasks.py), built on ``PickPlaceExpert``.

Same machinery as the pick-and-place expert: damped-least-squares IK at every control step with checked convergence,
min-jerk timing under the joint-speed cap, collision-checked poses, and physics checks at every phase boundary (grasped,
lifted, still held, placed). Everything moves through joint-position actions and contact physics; nothing is welded,
attached or teleported. Every episode starts in the folded rest pose (the unfold is collision-checked against the
objects) and the demonstration parks there again and holds 2 s, as in stack_red_on_blue.

  CylinderAsideExpert (PlaceUprightAt) top-down grasp at the cylinder's middle, carry it to a free reachable spot in the
                                       aside zone and stand it there; the gripper keeps one orientation while it holds
                                       the cylinder (any tilt change would tilt the 6 cm cylinder and topple it).
  CubeFlipExpert      (TipOver)        grasp the cube across two vertical faces with the approach tilted 45 degrees
                                       outward, lift, roll the wrist 180 degrees about the approach axis (the cube turns
                                       90 degrees: its white top face ends up facing the robot) and release it over its
                                       start, ~2 cm above the table (after the roll the wrist-camera mount hangs below
                                       the TCP's height, so the arm cannot go lower).
  BoxPlaceExpert      (PlaceInBox)     top-down grasp across the green block's flat end faces, carry over the 5 cm walls,
                                       turn the jaws along the tray's long side, lower inside and release just above the
                                       floor at a reachable spot (searched on a grid inside the tray).
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import mujoco
import numpy as np

from so101_libero.env import SO101TabletopEnv
from so101_libero.expert.ik import wrap_angle
from so101_libero.expert.pick_place import DEG, ExpertConfig, PickPlaceExpert, PlanError, min_jerk
from so101_libero.sim.robot import HOME_QPOS, lerobot_to_sim
from so101_libero.sim.scene import TCP_POS
from so101_libero.tasks.base import PlaceInBox, PlaceUprightAt, TipOver, _gap_to_tray


class _TriExpert(PickPlaceExpert):
    """Shared helpers: rectangle-aware clearances for trays, collision-checked joint moves (the unfold from rest
    included), world-yaw arcs and held-offset placement."""

    def __init__(self, env: SO101TabletopEnv, plan, rng: np.random.Generator, cfg: ExpertConfig | None = None,
                 **overrides):
        cfg = dataclasses.replace(cfg or ExpertConfig(), **overrides)
        super().__init__(env, plan, rng, cfg)

    # ------------------------------------------------------------------------------------------ geometry
    def _item_gap(self, name: str, xy: np.ndarray, radius: float = 0.0) -> float:
        """Distance from a disc (centre ``xy``, ``radius``) to item ``name``'s footprint (trays: walls included)."""
        h = self.env.h
        p = self.ws.pos(name)
        if name in h.containers and h.containers[name].shape == "tray":
            return _gap_to_tray(h.containers[name], (p[0], p[1], self.ws.yaw(name)), (xy[0], xy[1])) - radius
        item = h.objects.get(name) or h.containers[name]
        return float(np.linalg.norm(p[:2] - np.asarray(xy)[:2])) - item.footprint_radius - radius

    def _jaw_clearance(self, obj_pos: np.ndarray, yaw: float, moving_reach: float = 0.058) -> float:
        cdir = np.array([math.cos(yaw), math.sin(yaw)])
        lat = np.array([-cdir[1], cdir[0]])
        tcp = np.asarray(obj_pos)[:2] - cdir * self._approach_offset
        best = float("inf")
        for name in (*self.env.h.objects, *self.env.h.containers):
            if name == self.plan.obj:
                continue
            for j in (tcp - cdir * 0.018, tcp + cdir * moving_reach):
                for s in (-0.015, 0.0, 0.015):
                    best = min(best, self._item_gap(name, j + lat * s))
        return best

    def _spot_blocked(self, tgt: np.ndarray, min_gap: float = 0.004) -> bool:
        r = min(self.obj.footprint_radius, self.obj.grasp_width / 2 + 0.003)
        return any(self._item_gap(n, tgt[:2], r) < min_gap
                   for n in (*self.env.h.objects, *self.env.h.containers) if n != self.plan.obj)

    def _collision_any(self, q_arm: np.ndarray, grip: float, ignore: tuple[str, ...] = ()) -> str | None:
        """Robot contact with itself, the table or ANY item (except ``ignore``) in configuration ``q_arm``."""
        h, d = self.env.h, self._scratch
        d.qpos[:] = self.env.data.qpos
        d.qpos[h.joint_qpos_adr] = np.r_[q_arm, grip]
        mujoco.mj_kinematics(self.env.model, d)
        mujoco.mj_collision(self.env.model, d)
        skip = set().union(*(h.item_geoms[n] for n in ignore)) if ignore else set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = h.model.geom_bodyid[c.geom1], h.model.geom_bodyid[c.geom2]
            if not (b1 in h.robot_bodies or b2 in h.robot_bodies) or c.geom1 in skip or c.geom2 in skip:
                continue
            return f"{h.model.body(b1).name}-{h.model.body(b2).name}"
        return None

    def _check_joint_path(self, q_goal: np.ndarray, grip: float, ignore: tuple[str, ...] = (), n: int = 12) -> None:
        q0 = self._q[:5]
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            hit = self._collision_any(q0 + s * (q_goal - q0), grip, ignore)
            if hit:
                raise PlanError(f"joint move collides ({hit})")

    def _advance(self) -> None:
        if self.phase == "start" and self.env.task.start_deg is not None:
            self._check_joint_path(HOME_QPOS[:5], self._q[5])  # the unfold from rest must clear every object
        super()._advance()

    # ------------------------------------------------------------------------------------------ motion
    def _arc_yaw(self, start: np.ndarray, goal: np.ndarray, yaw0: float, yaw1: float, grip: float) -> None:
        """Constant-height arc around the pan axis (like ``_arc``) with the jaw yaw interpolated in the world."""
        r0, r1 = self.ik.radius(start), self.ik.radius(goal)
        a0 = self.ik.azimuth(start)
        da = wrap_angle(self.ik.azimuth(goal) - a0)
        dyaw = wrap_angle(yaw1 - yaw0)
        c = self.ik.pan_axis_xy

        def pose_at(sv):
            a, r = a0 + sv * da, r0 + sv * (r1 - r0)
            pos = np.array([c[0] + r * math.cos(a), c[1] + r * math.sin(a), start[2] + sv * (goal[2] - start[2])])
            return pos, yaw0 + sv * dyaw

        self._path(pose_at, float(np.linalg.norm(goal - start)), 0.15, grip)

    def _held_offset(self) -> tuple[np.ndarray, float, float]:
        """Where the object sits in the jaws now: (centre - TCP in the jaw frame (xy), jaw yaw, TCP height above the
        object's bottom). Used to put the object's centre (not the TCP) on a target."""
        tcp, obj = self.ws.tcp_pos(), self.ws.pos(self.plan.obj)
        rot = self.ws.d.site_xmat[self.env.h.tcp_site].reshape(3, 3)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = obj[:2] - tcp[:2]
        return np.array([c * dx - s * dy, s * dx + c * dy]), yaw, float(tcp[2] - (obj[2] - self.obj.half_height))

    @staticmethod
    def _tcp_for(goal_xy: np.ndarray, held: np.ndarray, yaw: float) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        return np.asarray(goal_xy)[:2] - np.array([c * held[0] - s * held[1], s * held[0] + c * held[1]])


class BoxPlaceExpert(_TriExpert):
    """Pick the object and release it inside an open box (fixed or free) with the jaws along the box's long side."""

    release_floor_gap: tuple[float, float] = (0.005, 0.009)  # object bottom above the box floor when releasing
    fixed_jaw_reach: float = 0.026  # outer face of the fixed jaw from the TCP, with margin (along the closing axis)
    moving_jaw_reach: float = 0.05  # outer face of the moving jaw at the release opening, with margin
    finger_half_width: float = 0.016  # across the closing axis
    wall_margin: float = 0.005

    def __init__(self, env, plan: PlaceInBox, rng, cfg=None):
        # The 1.75 cm tall block is grasped a little lower than a 3 cm cube: the fixed jaw's lowest point (15 mm below
        # the TCP) stays 2.5-4.5 mm above the table, the pads cover the end faces from ~1 cm up to the top.
        super().__init__(env, plan, rng, cfg, grasp_height=(0.0175, 0.0195))
        self.container = env.h.containers[plan.target]
        self._release_yaw: float | None = None

    def _yaw_at(self, pos: np.ndarray) -> float:
        return self._release_yaw if self._release_yaw is not None else super()._yaw_at(pos)

    def _release_z(self) -> float:
        return self.container.base_thickness + self._grasp_z + self._u(self.release_floor_gap)

    def _plan_transport(self) -> None:
        box, held_now = self.plan.target, self._held_offset()
        held, yaw_now, held_z = held_now
        self._grasp_z = held_z
        centre, rot = self.ws.pos(box), self.ws.rot(box)
        box_yaw = math.atan2(rot[1, 0], rot[0, 0])
        hx, hy = self.container.inner_size
        long_axis = box_yaw + (math.pi / 2 if hy >= hx else 0.0)
        half_long, half_short = max(hx, hy), min(hx, hy)
        yaws = sorted((long_axis, long_axis + math.pi), key=lambda y: abs(wrap_angle(y - yaw_now)))
        # Release spots for the object's centre along the long side (u) and across it (v), box frame; the jaws must stay
        # inside the walls. Tried nearest to a randomly jittered spot first (demonstrations differ), then the rest.
        reach_obj = self.obj.footprint_radius
        v_lim = half_short - max(self.finger_half_width, reach_obj) - self.wall_margin
        tcp = self.ws.tcp_pos()
        z_carry, z_release = tcp[2], self._release_z()
        for yaw in yaws:
            sign = math.cos(wrap_angle(yaw - long_axis))  # +1: closing along +long side
            # TCP along the long side: [-half_long + fixed reach, half_long - moving reach] (closing towards +u)
            u_lo, u_hi = -half_long + self.fixed_jaw_reach + self.wall_margin, half_long - self.moving_jaw_reach - self.wall_margin
            if sign < 0:
                u_lo, u_hi = -u_hi, -u_lo
            us = np.linspace(u_lo, u_hi, 7)
            vs = np.linspace(-v_lim, v_lim, 5) if v_lim > 0 else np.zeros(1)
            pref = np.array([self.rng.uniform(u_lo, u_hi), self.rng.uniform(-max(v_lim, 0), max(v_lim, 0))])
            cands = sorted(((u, v) for u in us for v in vs), key=lambda uv: np.hypot(uv[0] - pref[0], uv[1] - pref[1]))
            for u, v in cands:
                ax = np.array([math.cos(long_axis), math.sin(long_axis)])
                tcp_xy = centre[:2] + ax * u + np.array([-ax[1], ax[0]]) * v
                obj_goal = tcp_xy + (self._tcp_for(np.zeros(2), -held, yaw))  # centre of the held object there
                local = rot[:2, :2].T @ (obj_goal - centre[:2])
                if not self.container.inside_xy(local, reach_obj * 0.5 + 0.003):
                    continue
                try:
                    self._solve(np.r_[tcp_xy, z_carry], yaw)
                    _, q_rel = self._solve(np.r_[tcp_xy, z_release], yaw)
                except PlanError:
                    continue
                if self._collision(q_rel, self._release) is not None:
                    continue
                self._release_yaw = yaw
                self._place_target = np.r_[tcp_xy, 0.0]
                self._arc_yaw(tcp, np.r_[tcp_xy, z_carry], yaw_now, yaw, self._closed)
                return
        raise PlanError("no reachable release spot inside the box")


class _HeldFrameExpert(_TriExpert):
    """Pick-and-place in ONE family of gripper frames: the approach tilted ``tilt`` outward in the arm plane (aimed
    from the wrist-roll axis, which stays in the vertical plane through the pan axis, so every frame is exactly
    reachable by the 5-DoF arm), the closing direction at yaw ``azimuth + psi`` (horizontal) rolled by ``phi`` about
    the approach. While holding the object only the position (and, deliberately, phi) changes, so the object's
    orientation never drifts."""

    retreat_z: float = 0.11

    def __init__(self, env, plan, rng, cfg=None, **overrides):
        super().__init__(env, plan, rng, cfg, **overrides)
        self.tilt = 0.0
        self._psi = 0.0
        self._phi = 0.0
        self.start_xy = env.world.init_pos[plan.obj][:2].copy()

    # ------------------------------------------------------------------------------------------ frames and IK
    def _frame(self, pos: np.ndarray, phi: float | None = None, tilt: float | None = None,
               psi: float | None = None) -> np.ndarray:
        phi = self._phi if phi is None else phi
        tilt = self.tilt if tilt is None else tilt
        psi = self._psi if psi is None else psi
        pos = np.asarray(pos, float)
        pt = pos[:2]
        for _ in range(4):
            az = self.ik.azimuth(pt)
            a = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)])
            c0 = np.array([math.cos(az + psi), math.sin(az + psi), 0.0])
            c0 = c0 - a * (c0 @ a)
            c0 /= np.linalg.norm(c0)
            c = math.cos(phi) * c0 + math.sin(phi) * np.cross(a, c0)
            pt = pos[:2] - TCP_POS[0] * c[:2]
        z = -a
        return np.stack([c, np.cross(z, c), z], axis=1)

    def _solve_frame(self, pos: np.ndarray, rot: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
        """Goal IK (4 deg inside the joint limits): ``seed`` first, then generic seeds facing the target."""
        pan = -self.ik.azimuth(pos)
        seeds = [] if seed is None else [np.asarray(seed, float)]
        seeds += [self._q[:5]] + [np.r_[pan, l, e, w, r] for l in (-0.5, 0.0, 0.5, 1.0) for e in (0.0, 0.6, 1.2)
                                  for w in (-0.5, 0.6, 1.4) for r in (-1.6, 0.0, 1.6)]
        for s in seeds:
            r = self.ik_goal.solve(np.asarray(pos, float), rot, s, max_iters=300)
            if r.success:
                self.max_ik_err = max(self.max_ik_err, r.pos_err)
                return r.q
        raise PlanError(f"unreachable pos={np.round(pos, 3).tolist()}")

    def _follow(self, frame_at, n: int, grip0: float, grip1: float) -> None:
        """Follow ``frame_at(s) -> (pos, rot)`` with IK at every control step (continuous from the current joints)."""
        q, qs = self._q[:5].copy(), []
        for sv in min_jerk(n):
            pos, rot = frame_at(sv)
            r = self.ik.solve(pos, rot, q, max_iters=300)
            if not r.success:
                raise PlanError(f"path unreachable at {np.round(pos, 3).tolist()}")
            self.max_ik_err = max(self.max_ik_err, r.pos_err)
            q = r.q
            qs.append(q)
        for sv, q in zip(min_jerk(n), qs):
            self._emit(q, grip0 + sv * (grip1 - grip0))
        self._check_smooth()

    def _steps(self, q_goal: np.ndarray, length: float, max_speed: float = 0.10) -> int:
        return max(self._duration(self._q[:5], q_goal), int(round(length / max_speed / self.dt)))

    # ------------------------------------------------------------------------------------------ segments
    def _line(self, goal: np.ndarray, grip: float) -> None:
        """Straight TCP line in the current frame family, jaws going to ``grip``."""
        p0, _ = self.ik.fk(self._q[:5])
        goal = np.asarray(goal, float)
        q_goal = self._solve_frame(goal, self._frame(goal), self._q[:5])
        n = self._steps(q_goal, float(np.linalg.norm(goal - p0)))
        self._follow(lambda sv: (p0 + sv * (goal - p0), self._frame(p0 + sv * (goal - p0))), n, self._q[5], grip)

    def _arc_to(self, goal: np.ndarray, grip: float) -> None:
        """Arc around the pan axis (radius, azimuth and height interpolated) in the current frame family."""
        p0, _ = self.ik.fk(self._q[:5])
        r0, r1 = self.ik.radius(p0), self.ik.radius(goal)
        a0 = self.ik.azimuth(p0)
        da = wrap_angle(self.ik.azimuth(goal) - a0)
        c = self.ik.pan_axis_xy

        def pos_at(sv):
            a, r = a0 + sv * da, r0 + sv * (r1 - r0)
            return np.array([c[0] + r * math.cos(a), c[1] + r * math.sin(a), p0[2] + sv * (goal[2] - p0[2])])

        q_goal = self._solve_frame(goal, self._frame(goal), self._q[:5])
        n = self._steps(q_goal, float(np.linalg.norm(goal - p0)) * 1.2)
        self._follow(lambda sv: (pos_at(sv), self._frame(pos_at(sv))), n, self._q[5], grip)

    def _approach(self, obj_pos: np.ndarray, psis, pre_z: float, grasp_z: float) -> None:
        """Pick the closing direction (from ``psis``, most clearance first) whose pre-grasp and grasp are reachable and
        collision-free, then move there (joint space, collision-checked; via a raised waypoint if needed)."""
        az = self.ik.azimuth(obj_pos)
        ranked = sorted(psis, key=lambda p: -self._jaw_clearance(obj_pos, az + p))
        last = None
        for psi in ranked:
            yaw = az + psi
            closing = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            g = np.asarray(obj_pos, float)[:2] - closing[:2] * self._approach_offset
            try:
                pre, grasp = np.r_[g, pre_z], np.r_[g, grasp_z]
                q_pre = self._solve_frame(pre, self._frame(pre, psi=psi))
                q_grasp = self._solve_frame(grasp, self._frame(grasp, psi=psi), q_pre)
                for q in (q_pre, q_grasp):
                    hit = self._collision(q, self._open)
                    if hit:
                        raise PlanError(f"grasp pose collides ({hit})")
                self._extra_checks(psi, g, q_grasp)
                try:
                    self._check_joint_path(q_pre, self._open)
                except PlanError:
                    tcp = self.ws.tcp_pos()
                    up = np.r_[tcp[:2], max(tcp[2], 0.1) + 0.04]
                    q_up = self._solve_frame(up, self._frame(up), self._q[:5])
                    self._check_joint_path(q_up, self._open)
                    self._joint_move(q_up, self._open)
                    self._check_joint_path(q_pre, self._open)
            except PlanError as e:
                last = e
                continue
            self._psi = psi
            self._grasp_xy = g
            self._joint_move(q_pre, self._open)
            return
        raise PlanError(f"no feasible grasp ({last})")

    def _extra_checks(self, psi: float, grasp_xy: np.ndarray, q_grasp: np.ndarray) -> None:
        """Task-specific feasibility checks of a grasp candidate (raise PlanError to reject it)."""

    def _park(self) -> None:
        """Back to the start pose via home (the demonstration ends parked, as in stack_red_on_blue)."""
        start = lerobot_to_sim(np.asarray(self.env.task.start_deg))
        self._joint_move(HOME_QPOS[:5], start[5])
        self._joint_move(start[:5], start[5])

    def _object_bottom(self) -> float:
        """Lowest point of the held object (a box: its lowest corner; else centre - half height)."""
        p, rot = self.ws.pos(self.plan.obj), self.ws.rot(self.plan.obj)
        if self.obj.shape == "box":
            half = np.asarray(self.obj.size)
            return float(min((p + rot @ (np.array(s) * half))[2] for s in itertools.product((-1, 1), repeat=3)))
        return float(p[2] - self.obj.half_height)


class CylinderAsideExpert(_HeldFrameExpert):
    """Stand a tall object at a free reachable spot inside a region, holding it at its middle in one orientation."""

    pregrasp_z: tuple[float, float] = (0.10, 0.11)  # finger tips >= 2.8 cm above the 6 cm top before descending
    grasp_z: tuple[float, float] = (0.029, 0.031)  # the cylinder's middle
    carry_z: tuple[float, float] = (0.08, 0.09)  # the bottom 5-6 cm above the table while carried
    place_gap: tuple[float, float] = (0.002, 0.004)  # bottom above the table when released
    tilts_deg: tuple[float, ...] = (0.0, 15.0, 30.0)  # least tilt for which every pose of the plan is reachable

    def __init__(self, env, plan: PlaceUprightAt, rng, cfg=None):
        super().__init__(env, plan, rng, cfg)

    def _choose_target_and_tilt(self) -> None:
        """A free spot in the region (shrunk by the plan's margin) and the least tilt at which the start and the spot
        are reachable at every height of the plan; up to 40 random spots."""
        p = self.plan
        lo = np.array([p.region.x[0], p.region.y[0]]) + p.margin
        hi = np.array([p.region.x[1], p.region.y[1]]) - p.margin
        obj = self.ws.pos(p.obj)
        for _ in range(40):
            spot = self.rng.uniform(lo, hi)
            if self._spot_blocked(np.r_[spot, 0.0], min_gap=0.02):
                continue
            for tilt in np.radians(self.tilts_deg):
                # top-down: the closing yaw is free; tilted: only tangential closing keeps the pads vertical
                psis = [0.0, math.pi / 2, math.pi / 4, -math.pi / 4] if tilt == 0 else [math.pi / 2, -math.pi / 2]
                feasible = [psi for psi in psis if self._reachable_at(tilt, psi, obj[:2], spot)]
                if feasible:
                    self.tilt, self._psis, self._target = float(tilt), feasible, spot
                    return
        raise PlanError("no reachable free spot in the aside zone")

    def _reachable_at(self, tilt: float, psi: float, start_xy: np.ndarray, spot: np.ndarray) -> bool:
        """Every pose of the plan (pre-grasp, grasp, carry, place, retreat) reachable at this tilt and closing yaw."""
        try:
            for xy, z in ((start_xy, self.pregrasp_z[1]), (start_xy, self.grasp_z[0]), (start_xy, self.carry_z[1]),
                          (spot, self.carry_z[1]), (spot, self.grasp_z[0]), (spot, self.retreat_z)):
                pos = np.r_[xy, z]
                self._solve_frame(pos, self._frame(pos, tilt=tilt, psi=psi))
        except PlanError:
            return False
        return True

    def _advance(self) -> None:
        c, ws, obj = self.cfg, self.ws, self.plan.obj
        p = self.phase
        if p == "start":
            self._check_joint_path(HOME_QPOS[:5], self._q[5])  # the unfold from rest must clear every object
            self._choose_target_and_tilt()
            self._enter("unfold")
            self._joint_move(HOME_QPOS[:5], self._q[5])
        elif p == "unfold":
            self._enter("approach")
            self._approach(ws.pos(obj), self._psis, self._u(self.pregrasp_z), self.grasp_z[0])
        elif p == "approach":
            self._enter("descend")
            self._line(np.r_[self._grasp_xy, self._u(self.grasp_z)], self._open)
        elif p == "descend":
            self._enter("close")
            self._gripper_move(self._closed, c.close_time)
            self._hold(c.settle_time)
        elif p == "close":
            if not ws.grasped(obj):
                return self._fail("grasp_failed")
            self._enter("lift")
            tcp = ws.tcp_pos()
            self._line(np.r_[tcp[:2], self._u(self.carry_z)], self._closed)
        elif p == "lift":
            if not self._held():
                return self._fail("dropped_during_lift")
            self._enter("transport")
            held, _, held_z = self._held_offset()
            self._grasp_z = held_z
            goal = self._target
            tgt = np.r_[goal, 0.0]
            for _ in range(3):  # the jaw yaw follows the target's azimuth
                tgt[:2] = self._tcp_for(goal, held, self.ik.azimuth(tgt) + self._psi)
            self._place_target = tgt
            self._arc_to(np.r_[tgt[:2], ws.tcp_pos()[2]], self._closed)
        elif p == "transport":
            if not self._held():
                return self._fail("dropped_during_transport")
            self._enter("lower")
            self._line(np.r_[self._place_target[:2], self._grasp_z + self._u(self.place_gap)], self._closed)
        elif p == "lower":
            if not self._held(min_bottom=0.0):
                return self._fail("dropped_while_lowering")
            self._enter("release")
            self._gripper_move(self._release, 0.4)
            self._hold(c.settle_time)
        elif p == "release":
            self._enter("retreat")
            tcp = ws.tcp_pos()
            self._line(np.r_[tcp[:2], self.retreat_z], self._release)
        elif p == "retreat":
            self._enter("return")
            self._park()
        elif p == "return":
            self._enter("hold")
            self._hold(c.park_hold_time)
        elif p == "hold":
            self.done = True
            self.phase = "done"
        else:
            raise AssertionError(p)


class CubeFlipExpert(_HeldFrameExpert):
    """Turn a box onto a neighbouring face in place with one grasp and a 180 degree wrist roll."""

    pregrasp_z: tuple[float, float] = (0.07, 0.08)  # finger tips >= 3 cm above the 3 cm top before descending
    grasp_z: tuple[float, float] = (0.0165, 0.0185)  # TCP at the cube's middle (the fixed jaw's lowest point ~6 mm up)
    roll_z: tuple[float, float] = (0.09, 0.10)  # the cube's corners sweep 2.6 cm around the TCP while rolling
    place_gap: tuple[float, float] = (0.002, 0.004)  # bottom above the table when released (if the arm allows)
    # After the roll the wrist-camera mount hangs ~1.2 cm below the TCP's height on the robot side, so the cube cannot be
    # lowered to the table from its middle: it is released from the lowest TCP height at which the arm (release
    # opening, cube ignored) stays this far clear of the table and the tray, a ~2 cm drop onto its new face.
    release_clearance: float = 0.003

    def __init__(self, env, plan: TipOver, rng, cfg=None):
        super().__init__(env, plan, rng, cfg)
        self.tilt = math.radians(plan.tilt_deg)
        self._roll_dir = 1.0

    def _extra_checks(self, psi: float, grasp_xy: np.ndarray, q_grasp: np.ndarray) -> None:
        """The roll at ``roll_z`` must be reachable and collision-free (jaws closed, the cube ignored) in one of the two
        directions, and so must the way back down in the rolled frame and the release there (the roll swings the
        wrist-camera mount to the other side of the gripper, ~8 cm towards the robot at cube height); remember which."""
        pos = np.r_[grasp_xy, self.roll_z[1]]
        last = None
        for direction in (1.0, -1.0):
            try:
                q = self._solve_frame(pos, self._frame(pos, phi=0.0, psi=psi), q_grasp)
                for k in range(1, 13):
                    q = self._solve_frame(pos, self._frame(pos, phi=direction * math.pi * k / 12, psi=psi), q)
                    hit = self._collision_any(q, self._closed, ignore=(self.plan.obj,))
                    if hit:
                        raise PlanError(f"roll collides ({hit})")
                self._release_tcp_z = self._lowest_release_z(grasp_xy, direction * math.pi, psi, q)
            except PlanError as e:
                last = e
                continue
            self._roll_dir = direction
            return
        raise PlanError(f"no feasible roll ({last})")

    def _lowest_release_z(self, xy: np.ndarray, phi: float, psi: float, q_seed: np.ndarray) -> float:
        """Lowest TCP height (2 mm steps from the roll height down) whose rolled pose, ``release_clearance`` lower, is
        collision-free with the jaws closed and at the release opening (the cube ignored)."""
        best, q = None, q_seed
        for z in np.arange(self.roll_z[1], self.grasp_z[0] - 1e-9, -0.002):
            probe = np.r_[xy, z - self.release_clearance]
            try:
                q = self._solve_frame(probe, self._frame(probe, phi=phi, psi=psi), q)
            except PlanError:
                break
            if (self._collision_any(q, self._closed, ignore=(self.plan.obj,))
                    or self._collision_any(q, self._release, ignore=(self.plan.obj,))):
                break
            best = float(z)
        if best is None:
            raise PlanError("no collision-free release height after the roll")
        return best

    def _advance(self) -> None:
        c, ws, obj = self.cfg, self.ws, self.plan.obj
        p = self.phase
        if p == "start":
            self._check_joint_path(HOME_QPOS[:5], self._q[5])  # the unfold from rest must clear every object
            self._enter("unfold")
            self._joint_move(HOME_QPOS[:5], self._q[5])
        elif p == "unfold":
            self._enter("approach")
            self._approach(ws.pos(obj), self._face_psis(), self._u(self.pregrasp_z), self.grasp_z[0])
        elif p == "approach":
            self._enter("descend")
            self._line(np.r_[self._grasp_xy, self._u(self.grasp_z)], self._open)
        elif p == "descend":
            self._enter("close")
            self._gripper_move(self._closed, c.close_time)
            self._hold(c.settle_time)
        elif p == "close":
            if not ws.grasped(obj):
                return self._fail("grasp_failed")
            self._enter("lift")
            tcp = ws.tcp_pos()
            self._line(np.r_[tcp[:2], self._u(self.roll_z)], self._closed)
        elif p == "lift":
            if not self._held(min_bottom=0.02):
                return self._fail("dropped_during_lift")
            self._enter("roll")
            tcp = ws.tcp_pos()
            q_end = self._solve_frame(tcp, self._frame(tcp, phi=self._roll_dir * math.pi), self._q[:5])
            n = self._duration(self._q[:5], q_end)
            self._follow(lambda sv: (tcp, self._frame(tcp, phi=self._roll_dir * math.pi * sv)), n, self._q[5],
                         self._q[5])
            self._phi = self._roll_dir * math.pi
        elif p == "roll":
            if not self._held(min_bottom=0.01):
                return self._fail("dropped_during_roll")
            self._enter("lower")
            # Put the cube's centre back over its start, its lowest point place_gap above the table if the arm allows
            # (else the TCP stops at the lowest collision-free release height).
            tcp, cube = ws.tcp_pos(), ws.pos(obj)
            goal = tcp + np.r_[self.start_xy - cube[:2], -(self._object_bottom() - self._u(self.place_gap))]
            goal[2] = max(goal[2], self._release_tcp_z)
            self._line(goal, self._closed)
        elif p == "lower":
            if not self._held(min_bottom=0.0):
                return self._fail("dropped_while_lowering")
            self._enter("release")
            self._gripper_move(self._release, 0.4)
            self._hold(c.settle_time)
        elif p == "release":
            self._enter("retreat")
            tcp = ws.tcp_pos()
            self._line(np.r_[tcp[:2], self.retreat_z], self._release)
        elif p == "retreat":
            self._enter("return")
            self._park()
        elif p == "return":
            self._enter("hold")
            self._hold(c.park_hold_time)
        elif p == "hold":
            self.done = True
            self.phase = "done"
        else:
            raise AssertionError(p)

    def _face_psis(self) -> list[float]:
        """Tangential closing directions (+-90 deg from the azimuth): with a tilted approach only these keep the pads
        vertical, i.e. flat on two faces of a cube turned to face the pan axis."""
        return [math.pi / 2, -math.pi / 2]


def make_tri_expert(env: SO101TabletopEnv, plan, rng: np.random.Generator, cfg: ExpertConfig | None = None):
    if isinstance(plan, PlaceUprightAt):
        return CylinderAsideExpert(env, plan, rng, cfg)
    if isinstance(plan, TipOver):
        return CubeFlipExpert(env, plan, rng, cfg)
    if isinstance(plan, PlaceInBox):
        return BoxPlaceExpert(env, plan, rng, cfg)
    return None
