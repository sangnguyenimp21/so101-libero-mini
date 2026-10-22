"""Suite ``so101_tri``: three language-conditioned tasks in ONE tabletop scene (a fixed white tray on the robot's right,
an all-red cube, a green half-cylinder, a light gray cylinder) with exactly five predefined initial
placements.

Directions. Every placement and goal below is built from ``Directions`` (robot frame, never from the table edges):
  FRONT = +x  away from the robot base, towards the front camera (lower in the front image)
  LEFT  = +y  the robot's OWN left; the front camera faces the robot, so this is the RIGHT side of the front image
  RIGHT = -y, BACK = -x
Set ``TRI_SCENE.directions`` (e.g. ``Directions(left=(0.0, -1.0))``) to mirror the whole scene and its goals.

Default scene (metres, world frame; the base origin is 3.9 cm behind the shoulder's pan axis):
  white tray   fixed, open top, 15 cm long (FRONT-BACK) x 9 cm wide x 5 cm tall outside, 1 mm walls and bottom (inside
               14.8 x 8.8 x 4.9 cm); centre 7.5 cm FRONT and 12.5 cm RIGHT: beside the robot base on its right
  gray cyl.    3 cm diameter x 6 cm tall, upright; centre 20 cm FRONT, 13 cm LEFT (left-front)
  green block  half-cylinder, 3.5 cm diameter x 3 cm long, flat face down, extrusion axis along FRONT; centre 22.5 cm
               FRONT, 3 cm LEFT (centre-front, in front of the folded arm)
  red cube     3 cm, its top face white; centre 23.2 cm FRONT, 7.2 cm RIGHT (right-front), turned to face the shoulder's
               pan axis (yaw -20.4 deg: faces tangential/radial there, so the 45 degree flip grasp closes across two
               faces)
  aside zone   5-12 cm FRONT, 12-19 cm LEFT: the left-back corner of the workspace, opposite the tray
Over the five placements the objects stand ~10 cm apart (centre to centre, >= 5.8 cm between footprints), 17.9-21.6 cm
from the pan axis, >= 3.4 cm clear of the folded arm at rest and while it unfolds, and >= 5.1 cm clear of the tray
(``python -m so101_libero.cli.render_placements --task cylinder_aside`` measures all of this).
Placements: the cylinder, green block and cube move together by 0 or 1 cm (default, left, right, forward, backward);
the tray, the aside zone, the table, the robot base and the cameras never move. Orientations never change. Nothing else
is random at reset (the arm starts at the fixed, folded rest pose, as in stack_red_on_blue); the demonstrations differ
only through the expert's own seeded randomness (timing, heights, release points).

Physical assumptions (not measured on the real objects): masses 20 g (cube, ~0.74 g/cm^3), 15 g (green block,
~1.04 g/cm^3, a printed part), 30 g (cylinder, ~0.71 g/cm^3); sliding friction 1.0 with torsional/rolling friction
0.005/0.0001 like every other item in this project (the jaw pads' friction is the robot model's); the tray is welded to
the table (fix the real one), so its mass does not matter. Collision geometry: the cube and cylinder are MuJoCo
primitives; the green block is a convex mesh of
the exact half-cylinder (32 segments on the arc) with stiffer contacts (solref 0.005 s, priority 1: the default 0.02 s
lets the light spinning block sink through a floor when it lands on an edge); the tray is five boxes (1 mm walls and
bottom, open top) whose bottom slab also reaches 1 cm into the table (invisible, keeps a dropped object from tunnelling
through the 1 mm bottom).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from so101_libero.sim.objects import ContainerSpec, ObjectSpec
from so101_libero.sim.robot import REST_DEG
from so101_libero.tasks.base import (FixedPlacementLayout, OnTableInRegion, PlaceInBox, PlaceUprightAt, Region,
                                     RestingInContainer, Task, TipOver, TurnedOnSide, Undisturbed)

PAN_AXIS_FRONT = 0.0388  # the shoulder's pan axis, along FRONT from the base origin (so101.xml)


@dataclass(frozen=True)
class Directions:
    """Named horizontal directions of the scene as world-frame unit vectors (x, y)."""

    front: tuple[float, float] = (1.0, 0.0)  # away from the robot base, towards the front camera
    left: tuple[float, float] = (0.0, 1.0)  # the robot's own left

    def __post_init__(self):
        f, l = np.asarray(self.front, float), np.asarray(self.left, float)
        if abs(np.linalg.norm(f) - 1) > 1e-9 or abs(np.linalg.norm(l) - 1) > 1e-9 or abs(f @ l) > 1e-9:
            raise ValueError(f"front {self.front} and left {self.left} must be orthogonal unit vectors")

    def vec(self, name: str) -> np.ndarray:
        f, l = np.asarray(self.front, float), np.asarray(self.left, float)
        return {"front": f, "back": -f, "left": l, "right": -l}[name]

    def xy(self, front: float = 0.0, left: float = 0.0) -> np.ndarray:
        """World xy of a point ``front`` metres along FRONT and ``left`` metres along LEFT from the base origin."""
        return front * self.vec("front") + left * self.vec("left")

    @property
    def yaw(self) -> float:
        """World yaw of the FRONT axis: items built along FRONT/LEFT are placed with this yaw."""
        return math.atan2(self.front[1], self.front[0])


@dataclass(frozen=True)
class TriTolerances:
    """Success tolerances of the three tasks."""

    hold_steps: int = 10  # the goal must hold this many consecutive control steps (0.33 s at 30 Hz): settled
    max_speed: float = 0.02  # m/s: "at rest"
    min_tcp_distance: float = 0.04  # released: no robot contact and the TCP this far from the object's centre
    min_lift: float = 0.02  # picked up: carried in the jaws at least this high above the table at some point
    aside_max_tilt_deg: float = 15.0  # the cylinder stands upright in the aside zone
    aside_min_shift: float = 0.06  # ... at least 6 cm from where it started
    aside_max_release_offset: float = 0.03  # ... released within 3 cm of where it rests (carried there, not shoved)
    flip_tol: float = 0.02  # cube back within 2 cm (horizontal) of its start
    flip_max_tilt_deg: float = 15.0  # start-top face within 15 deg of sideways; resting flat within 15 deg
    tray_margin: float = 0.003  # block centre this far inside the inner walls
    tray_max_float: float = 0.005  # block centre at most max half extent + this above the tray floor (resting on it)
    others_max_shift: float = 0.02  # the other objects stay within 2 cm of their start ...
    others_max_tilt_deg: float = 20.0  # ... and within 20 deg of their start orientation


@dataclass(frozen=True)
class TriSceneConfig:
    """Geometry of the scene and its five placements (metres, along FRONT / LEFT from the base origin)."""

    directions: Directions = field(default_factory=Directions)
    tray_front: float = 0.075  # tray centre
    tray_left: float = -0.125  # (negative: the robot's right)
    tray_length: float = 0.15  # outer, along FRONT
    tray_width: float = 0.09  # outer, along LEFT/RIGHT
    tray_height: float = 0.05  # outer
    tray_wall: float = 0.001
    tray_bottom: float = 0.001
    cube_size: float = 0.03
    green_diameter: float = 0.035
    green_length: float = 0.03
    green_axis: str = "front"  # extrusion axis of the half-cylinder: "front" or "left"
    cylinder_diameter: float = 0.03
    cylinder_height: float = 0.06
    cylinder_at: tuple[float, float] = (0.20, 0.13)  # default placement centres (FRONT, LEFT)
    green_at: tuple[float, float] = (0.225, 0.03)
    cube_at: tuple[float, float] = (0.232, -0.072)
    aside_front: tuple[float, float] = (0.05, 0.12)  # aside zone (the cylinder's centre must end inside it)
    aside_left: tuple[float, float] = (0.12, 0.19)
    placement_step: float = 0.01
    tolerances: TriTolerances = field(default_factory=TriTolerances)

    @property
    def placement_names(self) -> tuple[str, ...]:
        return ("default", "left", "right", "forward", "backward")

    def placement_offset(self, k: int) -> np.ndarray:
        name = self.placement_names[k]
        return np.zeros(2) if name == "default" else self.placement_step * self.directions.vec(
            {"forward": "front", "backward": "back"}.get(name, name))

    def tray_center(self) -> np.ndarray:
        return self.directions.xy(self.tray_front, self.tray_left)

    def default_positions(self) -> dict[str, np.ndarray]:
        d = self.directions
        return {"gray_cylinder": d.xy(*self.cylinder_at), "green_block": d.xy(*self.green_at),
                "red_cube": d.xy(*self.cube_at)}

    def cube_yaw(self) -> float:
        """The cube faces the pan axis in the default placement (faces tangential/radial there)."""
        c = self.default_positions()["red_cube"] - self.directions.xy(PAN_AXIS_FRONT, 0.0)
        return math.atan2(c[1], c[0])

    def placements(self) -> tuple[dict[str, tuple[float, float, float]], ...]:
        yaw = self.directions.yaw
        yaws = {"gray_cylinder": yaw, "green_block": yaw if self.green_axis == "front" else yaw + math.pi / 2,
                "red_cube": self.cube_yaw()}
        out = []
        for k in range(len(self.placement_names)):
            off = self.placement_offset(k)
            out.append({n: (float(p[0] + off[0]), float(p[1] + off[1]), float(yaws[n]))
                        for n, p in self.default_positions().items()})
        return tuple(out)

    def aside_zone(self) -> Region:
        """The aside zone as an axis-aligned world region (default directions; mirrored ones mirror it)."""
        corners = np.array([self.directions.xy(f, l) for f in self.aside_front for l in self.aside_left])
        return Region(x=(float(corners[:, 0].min()), float(corners[:, 0].max())),
                      y=(float(corners[:, 1].min()), float(corners[:, 1].max())))


TRI_SCENE = TriSceneConfig()
_c, _tol = TRI_SCENE, TRI_SCENE.tolerances

# The tray's own x axis runs along FRONT: inner half sizes (along FRONT, along LEFT).
WHITE_TRAY = ContainerSpec(
    "white_tray", "tray",
    inner_size=(_c.tray_length / 2 - _c.tray_wall, _c.tray_width / 2 - _c.tray_wall),
    rgba=(0.95, 0.95, 0.93, 1.0), description="white tray", wall_height=_c.tray_height, wall_thickness=_c.tray_wall,
    base_thickness=_c.tray_bottom, mass=0.2, base_below_table=0.01,
    fixed_pose=(float(_c.tray_center()[0]), float(_c.tray_center()[1]), _c.directions.yaw))
TRI_RED_CUBE = ObjectSpec("red_cube", "box", (_c.cube_size / 2,) * 3, (0.85, 0.12, 0.10, 1.0), "red cube", mass=0.02)
GREEN_BLOCK = ObjectSpec("green_block", "half_cylinder", (_c.green_diameter / 2, _c.green_length / 2),
                         (0.20, 0.88, 0.18, 1.0), "green semicircular block", mass=0.015, solref=(0.005, 1.0))
GRAY_CYLINDER = ObjectSpec("gray_cylinder", "cylinder", (_c.cylinder_diameter / 2, _c.cylinder_height / 2),
                           (0.80, 0.80, 0.80, 1.0), "light gray cylinder", mass=0.03)
ASIDE_ZONE = _c.aside_zone()

TRI_LAYOUT = FixedPlacementLayout(
    name="tri_tabletop_v2",
    objects=(GRAY_CYLINDER, TRI_RED_CUBE, GREEN_BLOCK),
    containers=(WHITE_TRAY,),
    regions={},
    placements=_c.placements(),
    placement_names=_c.placement_names,
)
# Like stack_red_on_blue: every episode starts in the folded rest pose (how the real arm starts) and the demonstrations
# park there again and hold 2 s, so the policy stops after the task.
REST_START = tuple(float(v) for v in REST_DEG)
_others = {"gray_cylinder": ("red_cube", "green_block"), "red_cube": ("gray_cylinder", "green_block"),
           "green_block": ("gray_cylinder", "red_cube")}


def _undisturbed(obj: str) -> Undisturbed:
    return Undisturbed(_others[obj], _tol.others_max_shift, _tol.others_max_tilt_deg)


CYLINDER_ASIDE = Task(
    task_id="cylinder_aside",
    language="Pick up the light gray cylinder and put it aside.",
    layout=TRI_LAYOUT,
    goal=(OnTableInRegion("gray_cylinder", ASIDE_ZONE, region_name="aside_zone", min_lift=_tol.min_lift,
                          min_shift=_tol.aside_min_shift, max_speed=_tol.max_speed,
                          max_tilt_deg=_tol.aside_max_tilt_deg, max_release_offset=_tol.aside_max_release_offset,
                          min_tcp_distance=_tol.min_tcp_distance),
          _undisturbed("gray_cylinder")),
    plan=PlaceUprightAt("gray_cylinder", ASIDE_ZONE),
    max_steps=450,
    success_hold_steps=_tol.hold_steps,
    tags=("so101_tri",),
    start_deg=REST_START,
    return_to_start=True,
)

CUBE_FLIP = Task(
    task_id="cube_flip",
    language="Pick up the red cube and flip it onto its side.",
    layout=TRI_LAYOUT,
    goal=(TurnedOnSide("red_cube", tol=_tol.flip_tol, max_tilt_deg=_tol.flip_max_tilt_deg, min_lift=_tol.min_lift,
                       max_speed=_tol.max_speed, min_tcp_distance=_tol.min_tcp_distance),
          _undisturbed("red_cube")),
    plan=TipOver("red_cube"),
    max_steps=450,
    success_hold_steps=_tol.hold_steps,
    tags=("so101_tri",),
    start_deg=REST_START,
    return_to_start=True,
)

GREEN_INTO_TRAY = Task(
    task_id="green_into_tray",
    language="Pick up the green semicircular block and put it into the white tray.",
    layout=TRI_LAYOUT,
    goal=(RestingInContainer("green_block", "white_tray", margin=_tol.tray_margin, max_float=_tol.tray_max_float,
                             min_lift=_tol.min_lift, max_speed=_tol.max_speed,
                             min_tcp_distance=_tol.min_tcp_distance),
          _undisturbed("green_block")),
    plan=PlaceInBox("green_block", "white_tray"),
    max_steps=450,
    success_hold_steps=_tol.hold_steps,
    tags=("so101_tri",),
    start_deg=REST_START,
    return_to_start=True,
)

TRI_TASKS = (CYLINDER_ASIDE, CUBE_FLIP, GREEN_INTO_TRAY)
