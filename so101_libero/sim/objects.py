"""Graspable objects and containers: specifications plus MjSpec builders.

Every item is a free rigid body, so the robot interacts with it only through contacts. Objects are single
primitive geoms; containers are a base plate plus box walls (a rectangular tray or a ring of segments).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    """A graspable object with a free joint, resting on its flat face."""

    name: str  # unique id used by tasks, e.g. "red_cylinder"
    shape: str  # "cylinder" | "box"
    size: tuple[float, ...]  # cylinder: (radius, half_height); box: (hx, hy, hz)
    rgba: tuple[float, float, float, float]
    description: str  # how instructions refer to it, e.g. "red cylinder"
    mass: float = 0.02
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    # Per-episode size randomization (None = fixed size): every dimension is multiplied by one factor drawn
    # uniformly from this range at reset (``Layout.sample_sizes``). The model is compiled at ``size``; the episode's
    # size is written into it by ``SceneHandles.apply_sizes`` and recorded in the initial state ("sizes").
    scale_range: tuple[float, float] | None = None

    def with_size(self, size: tuple[float, ...]) -> "ObjectSpec":
        return self if tuple(size) == tuple(self.size) else replace(self, size=tuple(float(v) for v in size))

    @property
    def half_height(self) -> float:
        return self.size[1] if self.shape == "cylinder" else self.size[2]

    @property
    def footprint_radius(self) -> float:
        """Radius of a circle containing the object's footprint, for collision-free placement."""
        return self.size[0] if self.shape == "cylinder" else math.hypot(self.size[0], self.size[1])

    @property
    def grasp_width(self) -> float:
        """Width between the jaws for a top-down grasp across the object (box: across x faces)."""
        return 2 * self.size[0]

    @property
    def yaw_symmetry(self) -> float:
        """Rotation (rad) after which a top-down grasp looks the same: 0 = any yaw (cylinder)."""
        if self.shape == "cylinder":
            return 0.0
        return math.pi / 2 if abs(self.size[0] - self.size[1]) < 1e-6 else math.pi


@dataclass(frozen=True)
class ContainerSpec:
    """An open container: a rectangular tray or a round bowl-like container with straight walls."""

    name: str
    shape: str  # "tray" | "round"
    inner_size: tuple[float, ...]  # tray: (inner_hx, inner_hy); round: (inner_radius,)
    rgba: tuple[float, float, float, float]
    description: str
    wall_height: float = 0.025  # above the table (includes the base plate)
    wall_thickness: float = 0.006
    base_thickness: float = 0.005
    mass: float = 0.15
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    n_segments: int = 24  # round walls

    @property
    def footprint_radius(self) -> float:
        if self.shape == "tray":
            return math.hypot(self.inner_size[0] + self.wall_thickness, self.inner_size[1] + self.wall_thickness)
        return self.inner_size[0] + self.wall_thickness

    def inside_xy(self, local_xy: np.ndarray, margin: float = 0.0) -> bool:
        """Is a point (container frame, metres) inside the inner area shrunk by ``margin``?"""
        x, y = float(local_xy[0]), float(local_xy[1])
        if self.shape == "tray":
            return abs(x) <= self.inner_size[0] - margin and abs(y) <= self.inner_size[1] - margin
        return math.hypot(x, y) <= self.inner_size[0] - margin


def _common_geom(geom, rgba, friction, mass=None) -> None:
    geom.rgba = list(rgba)
    geom.friction = list(friction)
    geom.condim = 4  # torsional friction: a grasped object does not spin about the contact normal
    geom.group = 2
    if mass is not None:
        geom.mass = mass


def add_object(parent: mujoco.MjsBody, spec: ObjectSpec) -> mujoco.MjsBody:
    """Add ``spec`` as a free body (pose is set later through qpos)."""
    body = parent.add_body(name=spec.name, pos=[0.0, 0.0, spec.half_height])
    body.add_freejoint(name=f"{spec.name}_joint")
    if spec.shape == "cylinder":
        geom = body.add_geom(name=f"{spec.name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[*spec.size, 0])
    elif spec.shape == "box":
        geom = body.add_geom(name=f"{spec.name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(spec.size))
    else:
        raise ValueError(f"unknown object shape {spec.shape!r}")
    _common_geom(geom, spec.rgba, spec.friction, spec.mass)
    return body


def add_container(parent: mujoco.MjsBody, spec: ContainerSpec) -> mujoco.MjsBody:
    """Add ``spec`` as a free body whose frame origin is the centre of its bottom face."""
    body = parent.add_body(name=spec.name, pos=[0.0, 0.0, 0.0])
    body.add_freejoint(name=f"{spec.name}_joint")
    t, bt, h = spec.wall_thickness, spec.base_thickness, spec.wall_height
    wall_h = (h - bt) / 2  # half height of the walls standing on the base plate
    parts = []  # (name, geom type, size, pos, quat, volume)
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    if spec.shape == "tray":
        hx, hy = spec.inner_size
        parts.append(("base", box, [hx + t, hy + t, bt / 2], [0, 0, bt / 2], [1, 0, 0, 0]))
        for sx in (-1, 1):
            parts.append((f"wall_x{sx:+d}", box, [t / 2, hy + t, wall_h], [sx * (hx + t / 2), 0, bt + wall_h], [1, 0, 0, 0]))
        for sy in (-1, 1):
            parts.append((f"wall_y{sy:+d}", box, [hx, t / 2, wall_h], [0, sy * (hy + t / 2), bt + wall_h], [1, 0, 0, 0]))
    elif spec.shape == "round":
        r_in = spec.inner_size[0]
        parts.append(("base", cyl, [r_in + t, bt / 2, 0], [0, 0, bt / 2], [1, 0, 0, 0]))
        n = spec.n_segments
        r_mid = r_in + t / 2
        half_chord = (r_in + t) * math.tan(math.pi / n)  # segments overlap slightly at the outer edge: no gaps
        for i in range(n):
            a = 2 * math.pi * i / n
            parts.append((f"wall{i}", box, [t / 2, half_chord, wall_h], [r_mid * math.cos(a), r_mid * math.sin(a), bt + wall_h],
                          [math.cos(a / 2), 0, 0, math.sin(a / 2)]))
    else:
        raise ValueError(f"unknown container shape {spec.shape!r}")

    def volume(gtype, size):
        return 8 * size[0] * size[1] * size[2] if gtype == box else 2 * math.pi * size[0] ** 2 * size[1]

    total = sum(volume(g, s) for _, g, s, _, _ in parts)
    for name, gtype, size, pos, quat in parts:
        geom = body.add_geom(name=f"{spec.name}_{name}", type=gtype, size=size, pos=pos, quat=quat)
        # Split the container's mass over its geoms by volume (inertia then follows from the geoms).
        _common_geom(geom, spec.rgba, spec.friction, mass=spec.mass * volume(gtype, size) / total)
    return body


# The default catalogue. Sizes follow the real objects: 3 cm cube, cylinder 3 cm diameter x 3 cm tall.
RED_CYLINDER = ObjectSpec("red_cylinder", "cylinder", (0.015, 0.015), (0.85, 0.12, 0.10, 1.0), "red cylinder")
BLUE_CUBE = ObjectSpec("blue_cube", "box", (0.015, 0.015, 0.015), (0.10, 0.30, 0.85, 1.0), "blue cube")
RED_CUBE = ObjectSpec("red_cube", "box", (0.015, 0.015, 0.015), (0.85, 0.12, 0.10, 1.0), "red cube")
# The same red cube with a per-episode edge length of 2.7-4.0 cm (scale 0.9-4/3 of the 3 cm cube; mass kept at 20 g).
RED_CUBE_VAR = replace(RED_CUBE, scale_range=(0.027 / 0.03, 0.04 / 0.03))
BLACK_TRAY = ContainerSpec("black_tray", "tray", (0.060, 0.045), (0.06, 0.06, 0.06, 1.0), "black tray", mass=0.2)
ROUND_CONTAINER = ContainerSpec(
    "round_container", "round", (0.045,), (0.92, 0.92, 0.88, 1.0), "circular container", wall_height=0.03, mass=0.15
)
# The real white tray of red_cube_to_white_tray: inner 14 x 9 cm, 6 cm walls. It is fixed on the real table; the
# simulated one stays a free body, made heavy so the arm cannot shove it around.
WHITE_TRAY = ContainerSpec("white_tray", "tray", (0.070, 0.045), (0.93, 0.93, 0.91, 1.0), "white tray",
                           wall_height=0.06, mass=1.0)

OBJECTS = {o.name: o for o in (RED_CYLINDER, BLUE_CUBE, RED_CUBE)}
CONTAINERS = {c.name: c for c in (BLACK_TRAY, ROUND_CONTAINER, WHITE_TRAY)}
