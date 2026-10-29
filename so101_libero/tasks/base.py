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

    def sample(self, rng: np.random.Generator, seed: int | None = None) -> tuple[float, float, float]:
        return (float(rng.uniform(*self.x)), float(rng.uniform(*self.y)), float(rng.uniform(*self.yaw)))


@dataclass(frozen=True)
class GridRegion:
    """A rectangle split into ``nx`` x ``ny`` equal cells: an episode starts in cell ``seed % (nx * ny)`` (uniform
    inside it), so consecutive seeds -- the demonstrations and the evaluation states -- cover every cell equally.
    Cell k is row k // ny along x (near to far), column k % ny along y (right to left)."""

    x: tuple[float, float]
    y: tuple[float, float]
    nx: int
    ny: int
    yaw: tuple[float, float] = (-math.pi, math.pi)

    def cell(self, k: int) -> Region:
        i, j = divmod(k, self.ny)
        dx, dy = (self.x[1] - self.x[0]) / self.nx, (self.y[1] - self.y[0]) / self.ny
        return Region(x=(self.x[0] + i * dx, self.x[0] + (i + 1) * dx), y=(self.y[0] + j * dy, self.y[0] + (j + 1) * dy),
                      yaw=self.yaw)

    def sample(self, rng: np.random.Generator, seed: int | None = None) -> tuple[float, float, float]:
        n = self.nx * self.ny
        return self.cell(seed % n if seed is not None else int(rng.integers(n))).sample(rng)


@dataclass(frozen=True)
class RelativeRegion:
    """Start at a fixed offset (``dx``, ``dy``: world frame, metres) from the start position of item ``ref``, with an
    own yaw range: items that keep their spacing while the group moves. ``ref`` must be placed first (containers come
    before objects, objects in the layout's order)."""

    ref: str
    dx: float
    dy: float
    yaw: tuple[float, float] = (-math.pi, math.pi)


def _gap_to_tray(c: ContainerSpec, pose: tuple[float, float, float], xy: tuple[float, float]) -> float:
    """Distance from a point to the outer rectangle (walls included) of tray ``c`` at ``pose`` (0 inside it)."""
    cx, cy, yaw = pose
    dx, dy = xy[0] - cx, xy[1] - cy
    lx, ly = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    ox, oy = abs(lx) - (c.inner_size[0] + c.wall_thickness), abs(ly) - (c.inner_size[1] + c.wall_thickness)
    return math.hypot(max(ox, 0.0), max(oy, 0.0))


@dataclass(frozen=True)
class Layout:
    """The items in a scene and the region each one starts in."""

    name: str
    objects: tuple[ObjectSpec, ...]
    containers: tuple[ContainerSpec, ...]
    regions: dict[str, Region | GridRegion | RelativeRegion]
    clearance: float = 0.02  # minimum gap between item footprints at the start
    max_tries: int = 1000
    # False: every item is a circle of its footprint radius (a 15 x 10 cm tray blocks a 9 cm radius). True: trays are
    # their outer rectangle, so objects may start next to the long sides.
    tray_rectangles: bool = False

    def scene_config(self, **kw) -> SceneConfig:
        return SceneConfig(objects=self.objects, containers=self.containers, **kw)

    @property
    def num_placements(self) -> int:
        """Number of predefined initial placements (0: continuous start regions)."""
        return 0

    def choose_placement(self, rng: np.random.Generator, placement_id: int | None = None) -> int | None:
        """The placement of an episode: ``placement_id`` if given, else drawn from ``rng`` (layouts with predefined
        placements); None for continuous layouts, which draw nothing here."""
        if placement_id is not None:
            raise ValueError(f"layout {self.name} has no predefined placements (got placement_id={placement_id})")
        return None

    def sample_sizes(self, rng: np.random.Generator) -> dict[str, list[float]]:
        """Episode size of every object with a ``scale_range`` (no random draw at all for fixed-size layouts)."""
        sizes = {}
        for o in self.objects:
            if o.scale_range is not None:
                scale = float(rng.uniform(*o.scale_range))
                sizes[o.name] = [float(v) * scale for v in o.size]
        return sizes

    def sample(self, rng: np.random.Generator, sizes: dict[str, list[float]] | None = None,
               seed: int | None = None, placement_id: int | None = None) -> dict[str, tuple[float, float, float]]:
        """Sample a collision-free start pose (x, y, yaw) for every item (containers first: they are larger), with
        object footprints at the episode's ``sizes``. ``seed`` (the episode seed) picks the cell of a GridRegion."""
        if placement_id is not None:
            raise ValueError(f"layout {self.name} has no predefined placements")
        sizes = sizes or {}
        items = [*self.containers, *(o.with_size(sizes[o.name]) if o.name in sizes else o for o in self.objects)]
        for _ in range(self.max_tries):
            poses: dict[str, tuple[float, float, float]] = {}
            ok = True
            for it in items:
                region = self.regions[it.name]
                if isinstance(region, RelativeRegion):
                    if region.ref not in poses:
                        raise ValueError(f"layout {self.name}: {it.name} starts relative to {region.ref}, "
                                         f"which must be placed before it")
                    rx, ry, _ = poses[region.ref]
                    x, y, yaw = rx + region.dx, ry + region.dy, float(rng.uniform(*region.yaw))
                else:
                    x, y, yaw = region.sample(rng, seed)
                for other, (ox, oy, oyaw) in poses.items():
                    o = next(i for i in items if i.name == other)
                    if self.tray_rectangles and isinstance(o, ContainerSpec) and o.shape == "tray":
                        overlap = _gap_to_tray(o, (ox, oy, oyaw), (x, y)) < it.footprint_radius + self.clearance
                    else:
                        overlap = math.hypot(x - ox, y - oy) < it.footprint_radius + o.footprint_radius + self.clearance
                    if overlap:
                        ok = False
                        break
                if not ok:
                    break
                poses[it.name] = (x, y, yaw)
            if ok:
                return poses
        raise RuntimeError(f"layout {self.name}: no collision-free placement in {self.max_tries} tries")


@dataclass(frozen=True)
class FixedPlacementLayout(Layout):
    """A layout with a small set of predefined initial placements and nothing random in them: no position noise, no
    rotation, no per-object offsets. ``placements[k]`` gives the exact start pose (x, y, yaw) of every free item;
    containers with a ``fixed_pose`` are part of the scene itself and appear in no placement. An episode uses the
    placement passed to ``reset(options={"placement_id": k})``, else one drawn from the reset seed."""

    placements: tuple[dict[str, tuple[float, float, float]], ...] = ()
    placement_names: tuple[str, ...] = ()

    @property
    def num_placements(self) -> int:
        return len(self.placements)

    def choose_placement(self, rng: np.random.Generator, placement_id: int | None = None) -> int:
        if placement_id is None:
            return int(rng.integers(len(self.placements)))
        if not 0 <= int(placement_id) < len(self.placements):
            raise ValueError(f"layout {self.name}: placement_id must be in [0, {len(self.placements)}), got {placement_id}")
        return int(placement_id)

    def sample(self, rng: np.random.Generator, sizes: dict[str, list[float]] | None = None,
               seed: int | None = None, placement_id: int | None = None) -> dict[str, tuple[float, float, float]]:
        if sizes:
            raise ValueError(f"layout {self.name}: objects have fixed sizes")
        if placement_id is None:
            raise ValueError(f"layout {self.name}: choose_placement first (the env passes the placement id)")
        return {name: tuple(float(v) for v in pose) for name, pose in self.placements[placement_id].items()}


class WorldState:
    """Read-only view of the privileged simulator state used by predicates and the expert."""

    def __init__(self, handles: SceneHandles, data: mujoco.MjData):
        self.h = handles
        self.d = data
        self.init_pos: dict[str, np.ndarray] = {}  # settled item positions at reset (set by the env)
        self.init_rot: dict[str, np.ndarray] = {}  # settled item orientations (3x3) at reset (set by the env)
        # Per-episode history, updated by the environment after every control step (``track``):
        #   max_lift: highest point (m) each object's bottom reached above the table WHILE PINCHED BETWEEN THE JAWS;
        #   held_xy:  the object's last table position while it was pinched (where it was released from).
        # "Pick up and place" goals read them: pushing an object, flicking it, or dropping it elsewhere and shoving it
        # into place do not count.
        self.max_lift: dict[str, float] = {}
        self.held_xy: dict[str, np.ndarray] = {}

    def track(self) -> None:
        """Update the per-episode history (called by the environment after each control step)."""
        for name, spec in self.h.objects.items():
            if not self.grasped(name):
                continue
            p = self.d.xpos[self.h.item_body[name]]
            self.held_xy[name] = p[:2].copy()
            z = float(p[2]) - spec.half_height
            if z > self.max_lift.get(name, 0.0):
                self.max_lift[name] = z

    def lifted(self, item: str, min_height: float = 0.02) -> bool:
        """Was ``item`` ever carried ``min_height`` above the table in this episode (needs ``track``)?"""
        return self.max_lift.get(item, 0.0) >= min_height

    def release_offset(self, item: str) -> float | None:
        """Horizontal distance (m) between where ``item`` was last held and where it is now; None if never held."""
        if item not in self.held_xy:
            return None
        return float(np.linalg.norm(self.pos(item)[:2] - self.held_xy[item]))

    def pos(self, item: str) -> np.ndarray:
        return self.d.xpos[self.h.item_body[item]].copy()

    def rot(self, item: str) -> np.ndarray:
        return self.d.xmat[self.h.item_body[item]].reshape(3, 3).copy()

    def yaw(self, item: str) -> float:
        r = self.rot(item)
        return math.atan2(r[1, 0], r[0, 0])

    def up_z(self, item: str) -> float:
        """World z component of the item's body z axis: +1 upright, -1 upside down."""
        return float(self.d.xmat[self.h.item_body[item]][8])

    def tilt_from_start(self, item: str) -> float:
        """Angle (rad) between the item's body z axis now and at the reset (0 if unknown)."""
        if item not in self.init_rot:
            return 0.0
        return math.acos(float(np.clip(self.rot(item)[:, 2] @ self.init_rot[item][:, 2], -1.0, 1.0)))

    def linvel(self, item: str) -> np.ndarray:
        if self.h.is_fixed(item):
            return np.zeros(3)
        adr = self.h.item_dof_adr[item]
        return self.d.qvel[adr:adr + 3].copy()

    def upright(self, item: str, max_tilt_deg: float = 30.0) -> bool:
        """Standing on its base (containers, cylinders) within ``max_tilt_deg``; a box may rest on any of its faces."""
        rot = self.rot(item)
        spec = self.h.objects.get(item)
        if spec is not None and spec.shape == "box":
            return float(np.max(np.abs(rot[2, :]))) >= math.cos(math.radians(max_tilt_deg))
        return rot[2, 2] >= math.cos(math.radians(max_tilt_deg))

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

    def touching(self, a: str, b: str) -> bool:
        """Is item ``a`` in contact with item ``b`` (``b`` may also be "table")?"""
        ga = self.h.item_geoms[a]
        gb = frozenset((self.h.table_geom,)) if b == "table" else self.h.item_geoms[b]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 in ga and c.geom2 in gb) or (c.geom2 in ga and c.geom1 in gb):
                return True
        return False

    def flat(self, item: str, max_tilt_deg: float) -> bool:
        """A box resting on one of its faces: some body axis within ``max_tilt_deg`` of vertical."""
        return float(np.max(np.abs(self.rot(item)[2, :]))) >= math.cos(math.radians(max_tilt_deg))

    def rotation_from_start(self, item: str) -> float:
        """Angle (rad) of the rotation that takes the item's start orientation to its current one (0 if unknown)."""
        if item not in self.init_rot:
            return 0.0
        r = self.rot(item) @ self.init_rot[item].T
        return math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))

    def start_top_turn(self, item: str) -> float:
        """Angle (rad) between world up and the face normal that pointed up at the start (0 if unknown): 0 = that face
        is still on top, pi/2 = it faces sideways, pi = upside down."""
        if item not in self.init_rot:
            return 0.0
        n = self.rot(item) @ (self.init_rot[item].T @ np.array([0.0, 0.0, 1.0]))
        return math.acos(float(np.clip(n[2], -1.0, 1.0)))

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


@dataclass(frozen=True)
class OnTableInRegion:
    """``obj`` was picked up (carried in the jaws >= ``min_lift`` above the table at some point in the episode), was
    released within ``max_release_offset`` of where it now rests, and rests on the table inside ``region`` (its centre
    within the rectangle), on a face (boxes) / upright, released, at rest, at least ``min_shift`` from where it
    started, with the gripper's TCP at least ``min_tcp_distance`` away (the arm has let go and moved off). A region
    placement goal ("put it aside / over there") without a reference object."""

    obj: str
    region: Region
    region_name: str = "zone"
    min_lift: float = 0.02
    min_shift: float = 0.04
    max_speed: float = 0.05
    max_tilt_deg: float = 30.0
    max_release_offset: float = 0.05  # carried into the zone, not dropped elsewhere and shoved
    min_tcp_distance: float = 0.04  # m between the TCP and the object's centre

    def __call__(self, ws: WorldState) -> bool:
        p = ws.pos(self.obj)
        if not (self.region.x[0] <= p[0] <= self.region.x[1] and self.region.y[0] <= p[1] <= self.region.y[1]):
            return False
        if not OnTable(self.obj)(ws) or not ws.upright(self.obj, self.max_tilt_deg):
            return False
        if ws.touching_robot(self.obj) or not ws.at_rest(self.obj, self.max_speed):
            return False
        if float(np.linalg.norm(ws.tcp_pos() - p)) < self.min_tcp_distance:
            return False
        if ws.init_pos and ws.shift_xy(self.obj) < self.min_shift:
            return False
        off = ws.release_offset(self.obj)
        if off is None or off > self.max_release_offset:
            return False
        return ws.lifted(self.obj, self.min_lift)

    def describe(self) -> str:
        return f"OnTableIn({self.obj}, {self.region_name})"


@dataclass(frozen=True)
class StackedOn:
    """``obj`` rests on top of ``base`` (two boxes, each lying on a face): obj's bottom on base's top (within
    ``max_gap``), its centre over base's top face (horizontal distance between the centres <= ``max_offset``); base
    rests directly on the table (it may have been nudged); both upright, released (no robot contact) and at rest, with
    the gripper's TCP at least ``min_tcp_distance`` from obj (the jaws have let go and moved off: an evaluation stops
    at success, so a stack knocked over while the open jaws are still around it must not count)."""

    obj: str
    base: str
    max_offset: float = 0.012  # m: for 3 cm cubes obj's centre stays over base's top face
    max_gap: float = 0.004  # m between obj's bottom and base's top
    max_tilt_deg: float = 20.0
    max_speed: float = 0.05
    min_tcp_distance: float = 0.04  # m between the TCP and obj's centre

    def __call__(self, ws: WorldState) -> bool:
        po, pb = ws.pos(self.obj), ws.pos(self.base)
        if float(np.linalg.norm(po[:2] - pb[:2])) > self.max_offset:
            return False
        if float(np.linalg.norm(ws.tcp_pos() - po)) < self.min_tcp_distance:
            return False
        gap = (po[2] - ws.h.objects[self.obj].half_height) - (pb[2] + ws.h.objects[self.base].half_height)
        if abs(gap) > self.max_gap or not OnTable(self.base)(ws):
            return False
        for item in (self.obj, self.base):
            if not ws.upright(item, self.max_tilt_deg) or ws.touching_robot(item) or not ws.at_rest(item, self.max_speed):
                return False
        return True

    def describe(self) -> str:
        return f"On({self.obj}, {self.base})"


def _released(ws: WorldState, obj: str, max_speed: float, min_tcp_distance: float) -> bool:
    """No robot contact, at rest, and the gripper's TCP at least ``min_tcp_distance`` from the object's centre."""
    if ws.touching_robot(obj) or not ws.at_rest(obj, max_speed):
        return False
    return float(np.linalg.norm(ws.tcp_pos() - ws.pos(obj))) >= min_tcp_distance


@dataclass(frozen=True)
class PushedBy:
    """``obj`` was pushed, not picked up (never carried in the jaws higher than ``max_lift``), by ``offset`` (world xy,
    m) from where it started: its displacement along the offset is within ``along_tol`` of the offset's length and at
    most ``lateral_tol`` across it; it still lies on its base (``max_tilt_deg``) directly on the table, at rest, and the
    gripper has let go and moved off (no contact, TCP at least ``min_tcp_distance`` from its centre)."""

    obj: str
    offset: tuple[float, float]
    along_tol: float = 0.01
    lateral_tol: float = 0.015
    max_tilt_deg: float = 20.0
    max_lift: float = 0.01
    max_speed: float = 0.02
    min_tcp_distance: float = 0.04

    def displacement(self, ws: WorldState) -> tuple[float, float]:
        """(along, across) the offset direction: the object's horizontal displacement since the reset (m)."""
        u = np.asarray(self.offset, float) / np.linalg.norm(self.offset)
        d = ws.pos(self.obj)[:2] - ws.init_pos[self.obj][:2]
        along = float(d @ u)
        return along, float(np.linalg.norm(d - along * u))

    def __call__(self, ws: WorldState) -> bool:
        if self.obj not in ws.init_pos:
            return False
        along, across = self.displacement(ws)
        if abs(along - float(np.linalg.norm(self.offset))) > self.along_tol or across > self.lateral_tol:
            return False
        if ws.lifted(self.obj, self.max_lift) or not ws.upright(self.obj, self.max_tilt_deg) or not OnTable(self.obj)(ws):
            return False
        return _released(ws, self.obj, self.max_speed, self.min_tcp_distance)

    def describe(self) -> str:
        return f"PushedBy({self.obj}, {self.offset[0]:+.3f}, {self.offset[1]:+.3f})"


@dataclass(frozen=True)
class TurnedOnSide:
    """``obj`` (a box; its body z face starts on top) was picked up and now lies
    on one of its side faces: that face's normal is horizontal (within ``max_tilt_deg``), the box rests flat (one of its
    faces within ``max_tilt_deg`` of straight down) directly on the table within ``tol`` (horizontal) of where it
    started, and it is released, at rest and clear of the gripper. The environment tracks the body orientation, so this
    holds only after a real 90 degree turn."""

    obj: str
    tol: float = 0.02
    max_tilt_deg: float = 15.0
    min_lift: float = 0.02
    max_speed: float = 0.02
    min_tcp_distance: float = 0.04

    def __call__(self, ws: WorldState) -> bool:
        if self.obj not in ws.init_pos:
            return False
        rot = ws.rot(self.obj)
        if abs(rot[2, 2]) > math.sin(math.radians(self.max_tilt_deg)):  # the start-up face must now face sideways
            return False
        if float(np.max(np.abs(rot[2, :]))) < math.cos(math.radians(self.max_tilt_deg)):  # resting flat on a face
            return False
        p = ws.pos(self.obj)
        if float(np.linalg.norm(p[:2] - ws.init_pos[self.obj][:2])) > self.tol or not OnTable(self.obj)(ws):
            return False
        return _released(ws, self.obj, self.max_speed, self.min_tcp_distance) and ws.lifted(self.obj, self.min_lift)

    def describe(self) -> str:
        return f"TurnedOnSide({self.obj})"


@dataclass(frozen=True)
class Undisturbed:
    """The listed objects stayed roughly where they started: moved <= ``max_shift`` horizontally, tilted <=
    ``max_tilt_deg`` from their start orientation, still on the table and untouched by the robot."""

    objs: tuple[str, ...]
    max_shift: float = 0.02
    max_tilt_deg: float = 20.0

    def __call__(self, ws: WorldState) -> bool:
        for o in self.objs:
            if ws.shift_xy(o) > self.max_shift or ws.tilt_from_start(o) > math.radians(self.max_tilt_deg):
                return False
            if inside_any_container(ws, o) is not None or ws.touching_robot(o):
                return False
        return True

    def describe(self) -> str:
        return f"Undisturbed({', '.join(self.objs)})"


@dataclass(frozen=True)
class FlippedInPlace:
    """``obj`` (a box) was picked up, turned over and put back where it started. Its orientation differs from the start
    orientation by a rotation of at least ``min_rotation_deg`` (the flip is a 180 degree turn), and the face that started
    on top no longer is (turned by at least ``min_top_turn_deg``: a 180 degree spin about the vertical axis is no flip).
    The box rests flat on one of its faces (``max_flat_deg``) directly on the table within ``tol`` (horizontal) of its
    start, was carried at least ``min_lift`` high, and is released: no robot contact, at rest, the TCP at least
    ``min_tcp_distance`` from its centre. The environment tracks the body orientation, so this is the hidden ground
    truth for a plain cube that looks the same before and after the flip; no policy ever sees it."""

    obj: str
    tol: float = 0.02
    min_rotation_deg: float = 150.0
    min_top_turn_deg: float = 60.0
    max_flat_deg: float = 15.0
    min_lift: float = 0.02
    max_speed: float = 0.02
    min_tcp_distance: float = 0.04

    def __call__(self, ws: WorldState) -> bool:
        if self.obj not in ws.init_pos:
            return False
        if ws.rotation_from_start(self.obj) < math.radians(self.min_rotation_deg):
            return False
        if ws.start_top_turn(self.obj) < math.radians(self.min_top_turn_deg) or not ws.flat(self.obj, self.max_flat_deg):
            return False
        if ws.shift_xy(self.obj) > self.tol or not OnTable(self.obj)(ws):
            return False
        return _released(ws, self.obj, self.max_speed, self.min_tcp_distance) and ws.lifted(self.obj, self.min_lift)

    def describe(self) -> str:
        return f"FlippedInPlace({self.obj})"


def row_targets(ws: WorldState, objs: tuple[str, ...], targets, slot_centroid) -> np.ndarray:
    """Target centres (k, 2) of a row goal in the current episode: the nominal ``targets`` shifted by the layout offset,
    i.e. by how far the objects' common start centroid lies from the nominal ``slot_centroid``. The start slots are the
    same set of points in every permutation, so the centroid moves only with the layout."""
    start = np.mean([ws.init_pos[o][:2] for o in objs], axis=0)
    return np.asarray(targets, dtype=float) + (start - np.asarray(slot_centroid, dtype=float))


@dataclass(frozen=True)
class SortedRow:
    """``objs`` stand in a row at their targets (``objs[k]`` at target k; robot frame, ``targets`` listed left to right):
    each one's centre within ``tol`` (horizontal) of its target, resting flat (``max_tilt_deg``) directly on the table
    (not on another object), released (no robot contact), at rest, the TCP at least ``min_tcp_distance`` from every one.
    Targets move with the layout (``row_targets``)."""

    objs: tuple[str, ...]
    targets: tuple[tuple[float, float], ...]  # nominal target centres (layout 0), same order as objs
    slot_centroid: tuple[float, float]  # nominal centroid of the objects' start slots (layout 0)
    tol: float = 0.02
    max_tilt_deg: float = 20.0
    max_speed: float = 0.02
    min_tcp_distance: float = 0.04

    def target_xy(self, ws: WorldState) -> np.ndarray:
        return row_targets(ws, self.objs, self.targets, self.slot_centroid)

    def errors(self, ws: WorldState) -> np.ndarray:
        """Horizontal distance (m) of each object from its target."""
        return np.array([float(np.linalg.norm(ws.pos(o)[:2] - t)) for o, t in zip(self.objs, self.target_xy(ws))])

    def __call__(self, ws: WorldState) -> bool:
        if any(o not in ws.init_pos for o in self.objs):
            return False
        if float(self.errors(ws).max()) > self.tol:
            return False
        for o in self.objs:
            if not OnTable(o)(ws) or not ws.flat(o, self.max_tilt_deg):
                return False
            if not _released(ws, o, self.max_speed, self.min_tcp_distance):
                return False
        return True

    def describe(self) -> str:
        return f"SortedRow({', '.join(self.objs)})"


@dataclass(frozen=True)
class OnTopOfPair:
    """``obj`` rests centred on top of two boxes standing side by side (``bases``): its centre within ``tol``
    (horizontal) of the midpoint of their centres, its bottom within ``max_gap`` of their tops, touching both of them and
    not the table, flat (``max_tilt_deg``). The bases still rest flat directly on the table, each at most
    ``max_base_shift`` from its start. Everything is released and at rest, the TCP at least ``min_tcp_distance`` from
    ``obj``."""

    obj: str
    bases: tuple[str, str]
    tol: float = 0.01
    max_gap: float = 0.004
    max_tilt_deg: float = 15.0
    max_base_shift: float = 0.01
    max_speed: float = 0.02
    min_tcp_distance: float = 0.04

    def midpoint(self, ws: WorldState) -> np.ndarray:
        return (ws.pos(self.bases[0]) + ws.pos(self.bases[1])) / 2.0

    def __call__(self, ws: WorldState) -> bool:
        po, mid = ws.pos(self.obj), self.midpoint(ws)
        if float(np.linalg.norm(po[:2] - mid[:2])) > self.tol:
            return False
        tops = [ws.pos(b)[2] + ws.h.objects[b].half_height for b in self.bases]
        if abs((po[2] - ws.h.objects[self.obj].half_height) - max(tops)) > self.max_gap:
            return False
        if not all(ws.touching(self.obj, b) for b in self.bases) or ws.touching(self.obj, "table"):
            return False
        if not ws.flat(self.obj, self.max_tilt_deg):
            return False
        for b in self.bases:
            if not OnTable(b)(ws) or not ws.flat(b, self.max_tilt_deg) or ws.shift_xy(b) > self.max_base_shift:
                return False
            if ws.touching_robot(b) or not ws.at_rest(b, self.max_speed):
                return False
        return _released(ws, self.obj, self.max_speed, self.min_tcp_distance)

    def describe(self) -> str:
        return f"OnTopOf({self.obj}, {self.bases[0]}|{self.bases[1]})"


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
class PlaceAt:
    """Expert plan: pick ``obj`` and put it on the table at a random spot inside ``region`` shrunk by ``margin``
    (object half-size + placement jitter) on every side."""

    obj: str
    region: Region
    margin: float = 0.025


@dataclass(frozen=True)
class StackOn:
    """Expert plan: pick ``obj`` and put it on top of ``base`` (another object), centred on base's top face."""

    obj: str
    base: str


@dataclass(frozen=True)
class PlaceUprightAt:
    """Expert plan for a tall object (the 6 cm cylinder): a side grasp at its middle (the wrist turned so the wrist
    camera is on top, the approach tilted ``tilt_deg`` from vertical, the jaws closing across it horizontally), then
    stand it at a free reachable spot inside ``region`` (shrunk by ``margin``), the gripper keeping one orientation
    while it holds the object, so the object stays upright."""

    obj: str
    region: Region
    margin: float = 0.02
    tilt_deg: float = 45.0


@dataclass(frozen=True)
class PushBy:
    """Expert plan: push ``obj`` by ``offset`` (world xy, m) along the table with the closed gripper pointing down, both
    closed fingers side by side across the push direction, their edges against the object's face that looks away from
    the push."""

    obj: str
    offset: tuple[float, float]


@dataclass(frozen=True)
class TipOver:
    """Expert plan: turn a box onto a neighbouring face in place. Grasp it across two vertical faces with the approach
    tilted ``tilt_deg`` (45) outward, lift, roll the wrist 180 degrees about the approach axis (the box turns 90 degrees:
    its top face ends up facing the robot) and set it back down where it was."""

    obj: str
    tilt_deg: float = 45.0


@dataclass(frozen=True)
class FlipInPlace:
    """Expert plan: ``TipOver`` with ONE fixed motion for every demonstration (consistent data): the jaws always close
    from the same side (``psi_deg``: closing direction relative to the arm's azimuth, +-90 = tangential) and the wrist
    always rolls the same way (``roll_dir``). The rolled box turns 180 degrees about the tilted approach axis and is set
    down where it started; the demonstration ends with a short retreat and hold (no return home). The defaults are the
    combination that is feasible on the SO-101: closing direction at azimuth - 90 deg keeps the wrist-camera mount above
    the jaws (at +90 deg it hangs below them and hits the table), rolling in the negative direction."""

    obj: str
    tilt_deg: float = 45.0
    psi_deg: float = -90.0
    roll_dir: int = -1


@dataclass(frozen=True)
class SortIntoRow:
    """Expert plan: pick the objects one by one in row order (left to right) and put each on its row target (the rule
    of ``SortedRow``, which this plan mirrors: ``targets`` and ``slot_centroid``)."""

    objs: tuple[str, ...]
    targets: tuple[tuple[float, float], ...]
    slot_centroid: tuple[float, float]

    @property
    def obj(self) -> str:  # the first object handled (single-object diagnostics)
        return self.objs[0]


@dataclass(frozen=True)
class StackOnPair:
    """Expert plan: pick ``obj`` and put it on top of two boxes standing side by side, centred on their midpoint."""

    obj: str
    bases: tuple[str, str]


@dataclass(frozen=True)
class Task:
    task_id: str
    language: str
    layout: Layout
    goal: tuple  # predicates; all must hold
    plan: (PickPlace | PlaceRelative | PlaceAt | StackOn | PlaceUprightAt | PushBy | TipOver | FlipInPlace | SortIntoRow
           | StackOnPair | None) = None
    max_steps: int = 450  # 15 s at 30 Hz
    success_hold_steps: int = 5  # goal must hold this many consecutive control steps
    tags: tuple[str, ...] = field(default_factory=tuple)
    # Arm pose at reset in LeRobot units (None: the home pose). When set, the expert first moves to the home pose.
    start_deg: tuple[float, ...] | None = None
    # Demonstrations end by moving back to the start pose (via home, gripper closing) and holding there, so a policy
    # learns to park after the task instead of hovering over the target.
    return_to_start: bool = False

    def check(self, ws: WorldState) -> bool:
        return all(p(ws) for p in self.goal)

    def describe_goal(self) -> str:
        return " and ".join(p.describe() for p in self.goal)
