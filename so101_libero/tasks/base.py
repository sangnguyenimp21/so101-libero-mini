"""LIBERO-style task framework: layouts (objects + placement regions), goal predicates and tasks.

A ``Layout`` says which items exist and where each may start (axis-aligned table regions plus a yaw range,
like LIBERO's BDDL ``:regions``/``:init``). A ``Task`` adds a language instruction, goal predicates that must
all hold (``:goal``), the expert plan used to generate demonstrations, and the episode horizon. Tasks that
share a layout share one compiled MuJoCo model, so a multi-task policy sees the same scene and must use the
instruction to know what to do (LIBERO-Goal style).

Predicates read simulator state through ``WorldState``. They are only used for success detection and by the
scripted expert; policies never see that state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from so101_libero.sim.objects import ContainerSpec, ObjectSpec
from so101_libero.sim.scene import SceneConfig, SceneHandles


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle on the table (world frame, metres) and a yaw range (radians)."""

    x: tuple[float, float]
    y: tuple[float, float]
    yaw: tuple[float, float] = (-math.pi, math.pi)

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        return (float(rng.uniform(*self.x)), float(rng.uniform(*self.y)), float(rng.uniform(*self.yaw)))


@dataclass(frozen=True)
class Layout:
    """The items in a scene and the region each one starts in."""

    name: str
    objects: tuple[ObjectSpec, ...]
    containers: tuple[ContainerSpec, ...]
    regions: dict[str, Region]
    clearance: float = 0.02  # minimum gap between item footprints at the start
    max_tries: int = 1000

    def scene_config(self, **kw) -> SceneConfig:
        return SceneConfig(objects=self.objects, containers=self.containers, **kw)

    def sample(self, rng: np.random.Generator) -> dict[str, tuple[float, float, float]]:
        """Sample a collision-free start pose (x, y, yaw) for every item (containers first: they are larger)."""
        items = [*self.containers, *self.objects]
        for _ in range(self.max_tries):
            poses: dict[str, tuple[float, float, float]] = {}
            ok = True
            for it in items:
                x, y, yaw = self.regions[it.name].sample(rng)
                for other, (ox, oy, _) in poses.items():
                    o = next(i for i in items if i.name == other)
                    if math.hypot(x - ox, y - oy) < it.footprint_radius + o.footprint_radius + self.clearance:
                        ok = False
                        break
                if not ok:
                    break
                poses[it.name] = (x, y, yaw)
            if ok:
                return poses
        raise RuntimeError(f"layout {self.name}: no collision-free placement in {self.max_tries} tries")


class WorldState:
    """Read-only view of the privileged simulator state used by predicates and the expert."""

    def __init__(self, handles: SceneHandles, data: mujoco.MjData):
        self.h = handles
        self.d = data
        self.init_pos: dict[str, np.ndarray] = {}  # settled item positions at reset (set by the env)

    def pos(self, item: str) -> np.ndarray:
        return self.d.xpos[self.h.item_body[item]].copy()

    def rot(self, item: str) -> np.ndarray:
        return self.d.xmat[self.h.item_body[item]].reshape(3, 3).copy()

    def yaw(self, item: str) -> float:
        r = self.rot(item)
        return math.atan2(r[1, 0], r[0, 0])

    def linvel(self, item: str) -> np.ndarray:
        adr = self.h.item_dof_adr[item]
        return self.d.qvel[adr:adr + 3].copy()

    def upright(self, item: str, max_tilt_deg: float = 30.0) -> bool:
        return self.rot(item)[2, 2] >= math.cos(math.radians(max_tilt_deg))

    def shift_xy(self, item: str) -> float:
        """Horizontal displacement since the reset (0 if unknown)."""
        if item not in self.init_pos:
            return 0.0
        return float(np.linalg.norm(self.pos(item)[:2] - self.init_pos[item][:2]))

    def at_rest(self, item: str, max_speed: float = 0.05) -> bool:
        return float(np.linalg.norm(self.linvel(item))) < max_speed

    def resting_on_table(self, item: str) -> bool:
        """Container standing upright on the table, not held by the robot."""
        return abs(self.pos(item)[2]) < 0.005 and self.upright(item) and not self.touching_robot(item)

    def tcp_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.h.tcp_site].copy()

    def contacts(self, item: str) -> tuple[bool, bool, bool]:
        """(touching the fixed-jaw pads, touching the moving jaw, touching any other robot part)."""
        geoms = self.h.item_geoms[item]
        fixed = moving = other = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in geoms:
                g = g2
            elif g2 in geoms:
                g = g1
            else:
                continue
            if g in self.h.fixed_pad_geoms:
                fixed = True
            elif g in self.h.moving_jaw_geoms:
                moving = True
            elif self.h.model.geom_bodyid[g] in self.h.robot_bodies:
                other = True
        return fixed, moving, other

    def touching_robot(self, item: str) -> bool:
        return any(self.contacts(item))

    def grasped(self, item: str) -> bool:
        """Pinched between both jaws."""
        fixed, moving, _ = self.contacts(item)
        return fixed and moving


# --------------------------------------------------------------------------------------------- predicates
@dataclass(frozen=True)
class InContainer:
    """``obj`` rests inside ``container``: the container stands upright on the table (not held), the object's
    centre is within the inner area and above the floor, its bottom on the floor (not held above it, not on a
    wall top), and the object is released and at rest."""

    obj: str
    container: str
    margin: float = 0.003  # centre must be this far inside the inner walls
    max_speed: float = 0.05  # m/s

    def __call__(self, ws: WorldState) -> bool:
        c = ws.h.containers[self.container]
        o = ws.h.objects[self.obj]
        if not ws.resting_on_table(self.container):
            return False  # tipped, flipped, lifted or held container
        local = ws.rot(self.container).T @ (ws.pos(self.obj) - ws.pos(self.container))
        if not c.inside_xy(local[:2], self.margin):
            return False
        if local[2] < c.base_thickness:
            return False  # centre below the floor: e.g. the container was put over the object
        bottom = local[2] - o.half_height
        if bottom > c.base_thickness + 0.01 or local[2] > c.wall_height + o.half_height:
            return False  # still above the floor of the container (e.g. held over it or resting on a wall)
        if ws.touching_robot(self.obj):
            return False
        return ws.at_rest(self.obj, self.max_speed) and ws.at_rest(self.container, self.max_speed)

    def describe(self) -> str:
        return f"In({self.obj}, {self.container})"


def inside_any_container(ws: WorldState, obj: str) -> str | None:
    """Name of the container whose footprint (walls included) holds ``obj`` below its rim, else None."""
    p = ws.pos(obj)
    hh = ws.h.objects[obj].half_height
    for name, c in ws.h.containers.items():
        local = ws.rot(name).T @ (p - ws.pos(name))
        if c.inside_xy(local[:2], margin=-c.wall_thickness) and local[2] < c.wall_height + hh:
            return name
    return None


@dataclass(frozen=True)
class OnTable:
    """``obj`` rests directly on the table top, outside every container (2 mm height tolerance: an object on a
    container floor sits ~4.7 mm higher)."""

    obj: str

    def __call__(self, ws: WorldState) -> bool:
        if inside_any_container(ws, self.obj) is not None:
            return False
        return abs(ws.pos(self.obj)[2] - ws.h.objects[self.obj].half_height) < 0.002

    def describe(self) -> str:
        return f"OnTable({self.obj})"


# side -> (world axis, sign): robot frame, +x = in front of the robot, +y = the robot's left
SIDES = {"left": (1, 1), "right": (1, -1), "front": (0, 1), "behind": (0, -1)}


@dataclass(frozen=True)
class RelativeTo:
    """``obj`` rests on the table on the given side of ``ref``, which stayed where it was (<= ``max_ref_shift``), on
    the table, untouched (robot frame: +y = left, +x = front). Moving the reference instead of the object, or holding
    it, does not count. (Initial states where the goal already holds are resampled by the environment.)"""

    obj: str
    ref: str
    side: str  # "left" | "right" | "front" | "behind"
    min_dist: float = 0.04
    max_lateral: float = 0.06
    max_speed: float = 0.05
    max_ref_shift: float = 0.01
    min_obj_shift: float = 0.0  # optionally require the object itself to have moved

    def __call__(self, ws: WorldState) -> bool:
        d = ws.pos(self.obj) - ws.pos(self.ref)
        axis, sign = SIDES[self.side]
        along, lateral = sign * d[axis], abs(d[1 - axis])
        if along < self.min_dist or lateral > self.max_lateral:
            return False
        for item in (self.obj, self.ref):
            if not OnTable(item)(ws) or ws.touching_robot(item) or not ws.at_rest(item, self.max_speed):
                return False
        if ws.init_pos and ws.shift_xy(self.ref) > self.max_ref_shift:
            return False
        if ws.init_pos and self.min_obj_shift > 0 and ws.shift_xy(self.obj) < self.min_obj_shift:
            return False
        return True

    def describe(self) -> str:
        return f"{self.side.capitalize()}Of({self.obj}, {self.ref})"


# ------------------------------------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class PickPlace:
    """Expert plan: pick ``obj`` and release it above ``target`` (a container name)."""

    obj: str
    target: str


@dataclass(frozen=True)
class PlaceRelative:
    """Expert plan: pick ``obj`` and put it on the table ``distance`` m from ``ref`` on ``side`` (robot frame)."""

    obj: str
    ref: str
    side: str
    distance: float = 0.065


@dataclass(frozen=True)
class Task:
    task_id: str
    language: str
    layout: Layout
    goal: tuple  # predicates; all must hold
    plan: PickPlace | PlaceRelative | None = None
    max_steps: int = 450  # 15 s at 30 Hz
    success_hold_steps: int = 5  # goal must hold this many consecutive control steps
    tags: tuple[str, ...] = field(default_factory=tuple)

    def check(self, ws: WorldState) -> bool:
        return all(p(ws) for p in self.goal)

    def describe_goal(self) -> str:
        return " and ".join(p.describe() for p in self.goal)
