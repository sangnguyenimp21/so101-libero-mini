"""Scripted experts for the ``so101_tri`` tasks (so101_libero/tasks/tri_tasks.py), built on ``PickPlaceExpert``.

Same machinery as the pick-and-place expert: damped-least-squares IK at every control step with checked convergence,
min-jerk timing under the joint-speed cap, collision-checked grasp poses, and physics checks at every phase boundary
(grasped, lifted, still held, placed). Everything moves through joint-position actions and contact physics; nothing is
welded, attached or teleported.

  CubeMoveExpert    (MoveBy)          top-down grasp across two faces, carry, put the cube down ``offset`` from where it
                                      started with the jaw yaw (so the cube's yaw) unchanged in the world.
  BoxPlaceExpert    (PlaceInBox)      top-down grasp across the green block's flat end faces, carry over the 5 cm walls,
                                      turn the jaws along the box's long side, lower inside and release just above the
                                      floor at a reachable spot (searched on a grid inside the box).
  CylinderFlipExpert (FlipViaStation) three grasps: top-down at the middle -> stand it at the station; horizontal side
                                      grasp at the middle -> lift, roll the wrist 180 deg about the horizontal approach
                                      axis, stand it on its other end; top-down at the middle -> back to its start.
Stages are told apart by the arm pose and the jaw opening (station release 20 deg, side release 40 deg, approach 32
deg), so a memoryless policy can tell where it is although the cylinder looks the same either way up. Every task starts
in the folded rest pose (as stack_red_on_blue); the cube and box demonstrations park there again, the flip
demonstration ends hovering above the cylinder (a parked final frame would look exactly like the first).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from so101_libero.env import SO101TabletopEnv
from so101_libero.expert.ik import wrap_angle
from so101_libero.expert.pick_place import DEG, ExpertConfig, PickPlaceExpert, PlanError, min_jerk
from so101_libero.sim.robot import HOME_QPOS
from so101_libero.sim.scene import TCP_POS
from so101_libero.tasks.base import FlipViaStation, MoveBy, PlaceInBox, _gap_to_tray


class _TriExpert(PickPlaceExpert):
    """Shared helpers: rectangle-aware clearances for trays (the fixed box), world-yaw arcs and held-offset placement.
    Start (unfold from the rest pose) and end (park at the rest pose, 2 s hold) are the base expert's, as in
    stack_red_on_blue."""

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


class CubeMoveExpert(_TriExpert):
    """Pick the object and put it down ``plan.offset`` from where it started, its yaw unchanged in the world."""

    def __init__(self, env, plan: MoveBy, rng, cfg=None):
        super().__init__(env, plan, rng, cfg)

    def _yaw_at(self, pos: np.ndarray) -> float:
        return self._grasp_yaw if hasattr(self, "_grasp_yaw") else super()._yaw_at(pos)

    def _arc(self, start: np.ndarray, goal: np.ndarray, grip: float) -> None:
        self._arc_yaw(start, goal, self._grasp_yaw, self._grasp_yaw, grip)

    def _plan_transport(self) -> None:
        p, c = self.plan, self.cfg
        held, yaw, held_z = self._held_offset()
        goal = self.ws.init_pos[p.obj][:2] + np.asarray(p.offset) + self.rng.uniform(-p.jitter, p.jitter, 2)
        if self._spot_blocked(goal):
            raise PlanError("target spot occupied")
        tgt = np.r_[self._tcp_for(goal, held, yaw), 0.0]
        if self._jaw_clearance(goal, yaw, moving_reach=0.046) < 0.004:
            raise PlanError("jaws would hit an item at the target spot")
        tcp = self.ws.tcp_pos()
        self._grasp_z = held_z  # the lower phase puts the object's bottom table_release_gap above the table
        self._solve(np.r_[tgt[:2], held_z + c.table_release_gap[1]], yaw)
        self._place_target = tgt
        self._arc(tcp, np.r_[tgt[:2], tcp[2]], self._closed)


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


@dataclass
class FlipConfig:
    # While the cylinder is held top-down its axis must stay vertical, so the gripper keeps ONE tilt from grasp to
    # release: 30 deg outward in the arm plane, jaws closing tangentially (pads vertical). Tilt 0 is not reachable at the
    # station (top-down reach at 3 cm height ends ~26.5 cm from the pan axis; the side grasp starts at ~29 cm).
    top_tilt_deg: float = 30.0
    top_grasp_z: tuple[float, float] = (0.029, 0.031)  # TCP height of the top grasps (the cylinder's middle)
    top_pregrasp_z: tuple[float, float] = (0.10, 0.11)  # finger tips above the 6 cm top before descending
    carry_z: tuple[float, float] = (0.10, 0.11)  # TCP while carrying: the bottom 4-5 cm above the table
    place_gap: tuple[float, float] = (0.002, 0.004)  # bottom above the table when released
    side_pregrasp_z: tuple[float, float] = (0.095, 0.105)  # horizontal fingers above the 6 cm top
    side_grasp_z: tuple[float, float] = (0.029, 0.031)  # TCP height of the side grasp (the middle)
    side_offset: float = 0.004  # TCP towards the fixed jaw before closing: the fixed jaw clears the cylinder
    flip_z: tuple[float, float] = (0.10, 0.11)  # TCP while rolling: the ends sweep 3.4 cm around it
    side_open_deg: float = 40.0  # side release opening (tells this stage apart from the others)
    # After the 180 deg roll the wrist-camera mount (4-8 cm beside the roll axis, on the gripper's y side) hangs BELOW
    # the cylinder, so the flipped cylinder cannot be set down from its middle: it is released from the lowest TCP
    # height at which the robot stays ``mount_clearance`` above the table (~8.5 cm: a ~5.5 cm drop onto its end). The
    # moving jaw snaps open while the TCP slides ``release_slide`` away from the fixed jaw, so no pad drags on the
    # cylinder (a dragging pad spins it, and a spinning drop topples it).
    mount_clearance: float = 0.006
    release_open_time: float = 0.12
    release_slide: float = 0.004
    drop_settle_time: float = 0.6
    station_release_deg: float = 20.0
    retreat_z: float = 0.11
    jitter: float = 0.0015  # placement jitter (m, per axis) at the station and at the start position
    max_speed: float = 0.10  # m/s along Cartesian segments (min-jerk peak 1.875x)


class CylinderFlipExpert(_TriExpert):
    """Flip an upright cylinder end over end via a station where a horizontal side grasp is reachable."""

    final_hold: float = 2.0  # s hovering at the end (the policy learns to stop)

    def __init__(self, env, plan: FlipViaStation, rng, cfg=None, flip: FlipConfig | None = None):
        self.fc = flip or FlipConfig()
        super().__init__(env, plan, rng, cfg, release_deg=self.fc.station_release_deg)
        self.start_xy = env.world.init_pos[plan.obj][:2].copy()
        self.tilt = self.fc.top_tilt_deg * DEG
        self._top_offset = math.pi / 2  # jaw yaw = azimuth + this (tangential closing; sign chosen per grasp)
        self._sigma = 1.0  # side grasp closing direction: +/- the tangential direction
        self._roll_dir = 1.0
        self._flipped = False

    # ------------------------------------------------------------------------------------------ frames
    def _top_frame(self, pos: np.ndarray) -> np.ndarray:
        """Fixed-tilt grasp frame: approach tilted ``tilt`` outward in the arm plane, closing tangential."""
        return self.ik.tilted_frame(pos, self.tilt, self.ik.azimuth(pos) + self._top_offset)

    def _side_frame(self, pos: np.ndarray, phi: float, sigma: float | None = None) -> np.ndarray:
        """TCP frame with a horizontal approach along the arm's radial direction (aimed from the wrist-roll axis, which
        stays in the vertical plane through the pan axis) and the closing direction rolled by ``phi`` about it from
        sigma * tangential (phi = pi: the opposite side, the 180 degree flip)."""
        sigma = self._sigma if sigma is None else sigma
        pt = np.asarray(pos, float)[:2]
        for _ in range(3):
            az = self.ik.azimuth(pt)
            a = np.array([math.cos(az), math.sin(az), 0.0])
            t = sigma * np.array([-math.sin(az), math.cos(az), 0.0])
            closing = math.cos(phi) * t + math.sin(phi) * np.array([0.0, 0.0, 1.0])
            pt = np.asarray(pos, float)[:2] - TCP_POS[0] * closing[:2]
        z = -a
        return np.stack([closing, np.cross(z, closing), z], axis=1)

    def _solve_frame(self, pos: np.ndarray, rot: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
        """Goal IK (4 deg inside the joint limits): ``seed`` first, then generic seeds facing the target."""
        pan = -self.ik.azimuth(pos)
        seeds = [] if seed is None else [np.asarray(seed, float)]
        seeds += [self._q[:5]] + [np.r_[pan, l, e, w, r] for l in (-0.5, 0.0, 0.5, 1.0) for e in (0.0, 0.6, 1.2)
                                  for w in (-1.0, 0.0, 1.0) for r in (-1.6, 0.0, 1.6)]
        for s in seeds:
            r = self.ik_goal.solve(pos, rot, s, max_iters=300)
            if r.success:
                self.max_ik_err = max(self.max_ik_err, r.pos_err)
                return r.q
        raise PlanError(f"unreachable pos={np.round(pos, 3).tolist()}")

    def _solve_side(self, pos, phi, sigma=None, seed=None) -> np.ndarray:
        return self._solve_frame(np.asarray(pos, float), self._side_frame(pos, phi, sigma), seed)

    # ------------------------------------------------------------------------------------------ motion
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

    def _steps(self, q_goal: np.ndarray, length: float) -> int:
        return max(self._duration(self._q[:5], q_goal), int(round(length / self.fc.max_speed / self.dt)))

    def _top_line(self, goal: np.ndarray, grip: float) -> None:
        """Straight TCP line at the fixed top tilt, jaws going to ``grip``."""
        p0, _ = self.ik.fk(self._q[:5])
        q_goal = self._solve_frame(goal, self._top_frame(goal), self._q[:5])
        n = self._steps(q_goal, float(np.linalg.norm(goal - p0)))
        self._follow(lambda sv: (p0 + sv * (goal - p0), self._top_frame(p0 + sv * (goal - p0))), n, self._q[5], grip)

    def _top_arc(self, goal: np.ndarray, grip: float) -> None:
        """Arc around the pan axis (radius, azimuth and height interpolated) at the fixed top tilt."""
        p0, _ = self.ik.fk(self._q[:5])
        r0, r1 = self.ik.radius(p0), self.ik.radius(goal)
        a0 = self.ik.azimuth(p0)
        da = wrap_angle(self.ik.azimuth(goal) - a0)
        c = self.ik.pan_axis_xy

        def pos_at(sv):
            a, r = a0 + sv * da, r0 + sv * (r1 - r0)
            return np.array([c[0] + r * math.cos(a), c[1] + r * math.sin(a), p0[2] + sv * (goal[2] - p0[2])])

        q_goal = self._solve_frame(goal, self._top_frame(goal), self._q[:5])
        n = self._steps(q_goal, float(np.linalg.norm(goal - p0)) * 1.2)
        self._follow(lambda sv: (pos_at(sv), self._top_frame(pos_at(sv))), n, self._q[5], grip)

    def _side_line(self, goal: np.ndarray, grip: float) -> None:
        """Straight TCP line with the side frame (phi = pi after the flip, else 0), jaws going to ``grip``."""
        p0, _ = self.ik.fk(self._q[:5])
        phi = math.pi * self._roll_dir if self._flipped else 0.0
        q_goal = self._solve_side(goal, phi, seed=self._q[:5])
        n = self._steps(q_goal, float(np.linalg.norm(goal - p0)))
        self._follow(lambda sv: (p0 + sv * (goal - p0), self._side_frame(p0 + sv * (goal - p0), phi)), n,
                     self._q[5], grip)

    def _approach_top(self, obj_pos: np.ndarray) -> None:
        """Pick the tangential closing side with the most clearance that is reachable and collision-free, then move
        (joint space, collision-checked; via a raised waypoint if needed) to the pre-grasp above ``obj_pos``."""
        c, fc = self.cfg, self.fc
        az = self.ik.azimuth(obj_pos)
        options = []
        for off in (math.pi / 2, -math.pi / 2):
            yaw = az + off
            self._top_offset = off
            options.append((self._jaw_clearance(obj_pos, yaw), off))
        options.sort(reverse=True)
        last = None
        for _, off in options:
            self._top_offset = off
            yaw = az + off
            closing = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            g = np.asarray(obj_pos, float)[:2] - closing[:2] * self._approach_offset
            try:
                pre = np.r_[g, self._u(fc.top_pregrasp_z)]
                q_pre = self._solve_frame(pre, self._top_frame(pre))
                grasp = np.r_[g, fc.top_grasp_z[0]]
                q_grasp = self._solve_frame(grasp, self._top_frame(grasp), q_pre)
                for q in (q_pre, q_grasp):
                    hit = self._collision(q, self._open)
                    if hit:
                        raise PlanError(f"grasp pose collides ({hit})")
                try:
                    self._check_joint_path(q_pre, self._open)
                except PlanError:
                    tcp = self.ws.tcp_pos()
                    up = np.r_[tcp[:2], max(tcp[2], 0.1) + 0.04]
                    rot = self._side_frame(up, math.pi * self._roll_dir) if self._flipped else self._top_frame(up)
                    q_up = self._solve_frame(up, rot, self._q[:5])
                    self._check_joint_path(q_up, self._open)
                    self._joint_move(q_up, self._open)
                    self._check_joint_path(q_pre, self._open)
            except PlanError as e:
                last = e
                continue
            self._grasp_xy = g
            self._joint_move(q_pre, self._open)
            return
        raise PlanError(f"no feasible top grasp ({last})")

    def _carry_to(self, goal_xy: np.ndarray) -> None:
        """Arc to where the held cylinder's centre lands on ``goal_xy`` (the grasp offset carried over)."""
        held, yaw_now, held_z = self._held_offset()
        self._grasp_z = held_z
        tgt = np.r_[goal_xy, 0.0]
        for _ in range(3):  # the jaw yaw follows the target's azimuth
            yaw = self.ik.azimuth(tgt) + self._top_offset
            tgt[:2] = self._tcp_for(goal_xy, held, yaw)
        self._place_target = tgt
        tcp = self.ws.tcp_pos()
        self._top_arc(np.r_[tgt[:2], tcp[2]], self._closed)

    # ------------------------------------------------------------------------------------------ phases
    def _advance(self) -> None:
        c, fc, ws, obj = self.cfg, self.fc, self.ws, self.plan.obj
        p = self.phase
        stage2 = p.endswith("2")
        if p == "start" and self.env.task.start_deg is not None:  # from the folded rest pose: unfold first
            self._enter("unfold")
            self._joint_move(HOME_QPOS[:5], self._q[5])
        elif p in ("start", "unfold", "side_retreat"):
            if p == "side_retreat" and not self._standing(upside_down=True):
                return self._fail("flip_failed")
            self._enter("approach2" if p == "side_retreat" else "approach")
            self._approach_top(ws.pos(obj))
        elif p in ("approach", "approach2"):
            self._enter("descend2" if stage2 else "descend")
            self._top_line(np.r_[self._grasp_xy, self._u(fc.top_grasp_z)], self._open)
        elif p in ("descend", "descend2"):
            self._enter("close2" if stage2 else "close")
            self._gripper_move(self._closed, c.close_time)
            self._hold(c.settle_time)
        elif p in ("close", "close2"):
            if not ws.grasped(obj):
                return self._fail("regrasp_failed" if stage2 else "grasp_failed")
            self._enter("lift2" if stage2 else "lift")
            tcp = ws.tcp_pos()
            self._top_line(np.r_[tcp[:2], self._u(fc.carry_z)], self._closed)
        elif p in ("lift", "lift2"):
            if not self._held():
                return self._fail("dropped_during_lift")
            self._enter("transport2" if stage2 else "transport")
            goal = (self.start_xy if stage2 else np.asarray(self.plan.station)) + self.rng.uniform(-fc.jitter, fc.jitter, 2)
            self._carry_to(goal)
        elif p in ("transport", "transport2"):
            if not self._held():
                return self._fail("dropped_during_transport")
            self._enter("lower2" if stage2 else "lower")
            self._top_line(np.r_[self._place_target[:2], self._grasp_z + self._u(fc.place_gap)], self._closed)
        elif p in ("lower", "lower2"):
            if not self._held(min_bottom=0.0):
                return self._fail("dropped_while_lowering")
            self._enter("release2" if stage2 else "release")
            self._gripper_move(self._release, 0.4)
            self._hold(c.settle_time)
        elif p in ("release", "release2"):
            self._enter("retreat2" if stage2 else "retreat")
            tcp = ws.tcp_pos()
            self._top_line(np.r_[tcp[:2], fc.retreat_z], self._release)
        elif p == "retreat":
            if not self._standing(upside_down=False):
                return self._fail("not_standing_at_station")
            self._enter("side_approach")
            self._plan_side_approach()
        elif p == "side_approach":
            self._enter("side_descend")
            self._side_line(np.r_[self._side_grasp_xy, self._u(fc.side_grasp_z)], self._open)
        elif p == "side_descend":
            self._enter("side_close")
            self._gripper_move(self._closed, c.close_time)
            self._hold(c.settle_time)
        elif p == "side_close":
            if not ws.grasped(obj):
                return self._fail("side_grasp_failed")
            self._enter("side_lift")
            tcp = ws.tcp_pos()
            self._side_line(np.r_[tcp[:2], self._u(fc.flip_z)], self._closed)
        elif p == "side_lift":
            if not self._held(min_bottom=0.02):
                return self._fail("dropped_during_side_lift")
            self._enter("flip")
            self._plan_flip()
        elif p == "flip":
            if not self._held(min_bottom=0.01):
                return self._fail("dropped_during_flip")
            self._flipped = True
            self._enter("side_lower")
            tcp = ws.tcp_pos()
            self._side_line(np.r_[tcp[:2], self._release_height(tcp)], self._closed)
        elif p == "side_lower":
            if not self._held(min_bottom=0.0):
                return self._fail("dropped_while_side_lowering")
            self._enter("side_release")
            self._snap_release()
            self._hold(fc.drop_settle_time)
        elif p == "side_release":
            self._enter("side_retreat")
            tcp = ws.tcp_pos()
            self._side_line(np.r_[tcp[:2], fc.retreat_z], fc.side_open_deg * DEG)
        elif p == "retreat2":
            # End hovering above the cylinder with the jaws open, NOT parked at the start pose: the cylinder looks the
            # same either way up, so a parked final frame would equal the first one (go vs. stop for the policy).
            self._enter("hold")
            self._hold(self.final_hold)
        elif p == "hold":
            self.done = True
            self.phase = "done"
        else:
            raise AssertionError(p)

    def _standing(self, upside_down: bool) -> bool:
        ws, obj = self.ws, self.plan.obj
        up = ws.up_z(obj)
        ok_axis = up < -math.cos(15 * DEG) if upside_down else up > math.cos(15 * DEG)
        return ok_axis and abs(ws.pos(obj)[2] - self.obj.half_height) < 0.004 and not ws.touching_robot(obj)

    def _plan_side_approach(self) -> None:
        """Choose the closing side and roll direction for which the whole side sequence is reachable, then move (joint
        space, collision-checked) to the horizontal pre-grasp above the cylinder."""
        fc = self.fc
        cyl = self.ws.pos(self.plan.obj)
        az = self.ik.azimuth(cyl)
        z_pre, z_grasp, z_flip = self._u(fc.side_pregrasp_z), fc.side_grasp_z[0], fc.flip_z[1]
        last = None
        for sigma in (1.0, -1.0):
            t = sigma * np.array([-math.sin(az), math.cos(az)])
            g = cyl[:2] - t * fc.side_offset
            for roll_dir in (1.0, -1.0):
                try:
                    q_pre = self._solve_side(np.r_[g, z_pre], 0.0, sigma)
                    q = self._solve_side(np.r_[g, z_grasp], 0.0, sigma, seed=q_pre)
                    hit = self._collision_any(q, self._open, ignore=(self.plan.obj,))
                    if hit:
                        raise PlanError(f"side grasp pose collides ({hit})")
                    q = self._solve_side(np.r_[cyl[:2], z_flip], 0.0, sigma, seed=q)
                    for k in range(1, 7):  # the roll, in steps, continuous from the previous solution
                        q = self._solve_side(np.r_[cyl[:2], z_flip], roll_dir * math.pi * k / 6, sigma, seed=q)
                    hit = self._collision_any(q_pre, self._open)
                    if hit:
                        raise PlanError(f"side pre-grasp collides ({hit})")
                    self._check_joint_path(q_pre, self._open)
                except PlanError as e:
                    last = e
                    continue
                self._sigma, self._roll_dir = sigma, roll_dir
                self._side_grasp_xy = g
                self._joint_move(q_pre, self._open)
                return
        raise PlanError(f"no feasible side grasp at the station ({last})")

    def _release_height(self, tcp: np.ndarray) -> float:
        """Lowest TCP height (2.5 mm steps) at which the flipped side pose keeps every robot part ``mount_clearance``
        above the table and clear of the other items (the held cylinder excepted)."""
        phi = math.pi * self._roll_dir
        q = self._q[:5]
        for z in np.arange(0.07, 0.13, 0.0025):
            q = self._solve_side(np.r_[tcp[:2], z - self.fc.mount_clearance], phi, seed=q)
            if self._collision_any(q, self._closed, ignore=(self.plan.obj,)) is None:
                return float(z)
        raise PlanError("no collision-free release height after the flip")

    def _snap_release(self) -> None:
        """Open the jaws fast while sliding the TCP away from the fixed jaw (along the closing direction)."""
        p0, _ = self.ik.fk(self._q[:5])
        phi = math.pi * self._roll_dir
        closing = self._side_frame(p0, phi)[:, 0]
        goal = p0 + closing * self.fc.release_slide
        n = max(2, int(round(self.fc.release_open_time / self.dt)))
        self._follow(lambda sv: (p0 + sv * (goal - p0), self._side_frame(p0 + sv * (goal - p0), phi)), n,
                     self._q[5], self.fc.side_open_deg * DEG)

    def _plan_flip(self) -> None:
        """Roll 180 degrees about the horizontal approach axis with the TCP (the cylinder's middle) fixed; timed by the
        joint-speed cap like every other segment."""
        tcp = self.ws.tcp_pos()
        q_end = self._solve_side(tcp, self._roll_dir * math.pi, seed=self._q[:5])
        n = self._duration(self._q[:5], q_end)
        self._follow(lambda sv: (tcp, self._side_frame(tcp, self._roll_dir * math.pi * sv)), n, self._q[5], self._q[5])


def make_tri_expert(env: SO101TabletopEnv, plan, rng: np.random.Generator, cfg: ExpertConfig | None = None):
    if isinstance(plan, MoveBy):
        return CubeMoveExpert(env, plan, rng, cfg)
    if isinstance(plan, PlaceInBox):
        return BoxPlaceExpert(env, plan, rng, cfg)
    if isinstance(plan, FlipViaStation):
        return CylinderFlipExpert(env, plan, rng, cfg)
    return None
