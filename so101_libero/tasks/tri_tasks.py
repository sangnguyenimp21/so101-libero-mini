"""Suite ``so101_tri``: three language-conditioned tasks in ONE tabletop scene (white box, red cube, green
half-cylinder, light gray cylinder) with exactly five predefined initial placements.

Directions. Every placement and goal below is built from ``Directions`` (robot frame, never from the table edges):
  FRONT = +x  away from the robot base, towards the front camera (lower in the front image)
  LEFT  = +y  the robot's OWN left; the front camera faces the robot, so this is the RIGHT side of the front image
  RIGHT = -y, BACK = -x
Set ``TRI_SCENE.directions`` (e.g. ``Directions(left=(0.0, -1.0))``) to mirror the whole scene and its goals.

Default scene (metres, world frame; the base origin is 3.9 cm behind the shoulder's pan axis):
  white box   fixed, open top, 15 cm wide (LEFT-RIGHT) x 9 cm deep (FRONT-BACK) x 5 cm tall outside, 1 mm walls and
              bottom (inside 14.8 x 8.8 x 4.9 cm), centred 8.5 cm FRONT and 14 cm LEFT of the base origin: beside the
              robot base on its left, 1 cm clear of the base's side
  red cube    3 cm, directly FRONT of the box, centred on it, 1.5 cm gap to the box's front wall
  green block half-cylinder, 3.5 cm diameter x 3 cm long, flat face down, extrusion axis along FRONT (the front
              camera sees its semicircular end); centre 6 cm FRONT of the cube's centre (height 1.75 cm)
  gray cyl.   3 cm diameter x 6 cm tall, upright; centre 6 cm RIGHT of the green block's centre
Placements: the cube, green block and cylinder move together by 0 or 1 cm (default, left, right, forward, backward);
the box, the table, the robot base and the cameras never move. Orientations never change. Nothing else is random at
reset (the arm starts at the fixed, folded rest pose, as in stack_red_on_blue); the demonstrations differ only through
the expert's own seeded randomness (timing, heights, release points).

Physical assumptions (not measured on the real objects): masses 20 g (cube, ~0.74 g/cm^3), 15 g (green block,
~1.04 g/cm^3, a printed part), 30 g (cylinder, ~0.71 g/cm^3); sliding friction 1.0 with torsional/rolling friction
0.005/0.0001 like every other item in this project (the jaw pads' friction is the robot model's); the box is welded to
the table (it is fixed in the real setup), so its mass does not matter. Collision geometry: the cube and cylinder are
MuJoCo primitives; the green block is a convex mesh of the exact half-cylinder (32 segments on the arc) with stiffer
contacts (solref 0.005 s, priority 1: the default 0.02 s lets the light spinning block sink through a floor when it
lands on an edge); the box is five boxes (1 mm walls and bottom, open top) whose bottom slab also reaches 1 cm into the
table (invisible, keeps a dropped object from tunnelling through the 1 mm bottom).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from so101_libero.sim.objects import ContainerSpec, ObjectSpec
from so101_libero.sim.robot import REST_DEG
from so101_libero.tasks.base import (FixedPlacementLayout, FlippedInPlace, FlipViaStation, MoveBy, MovedBy,
                                     PlaceInBox, RestingInContainer, Task, Undisturbed)


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
    """Success tolerances of the three tasks (also used by the expert's own checks)."""

    hold_steps: int = 10  # the goal must hold this many consecutive control steps (0.33 s at 30 Hz): settled
    max_speed: float = 0.02  # m/s: "at rest"
    min_tcp_distance: float = 0.04  # released: no robot contact and the TCP this far from the object's centre
    min_lift: float = 0.02  # picked up: carried in the jaws at least this high above the table at some point
    cube_pos_tol: float = 0.01  # cube: +- along the 5 cm move, and +- front/back (its original front-back position)
    cube_max_tilt_deg: float = 15.0  # cube keeps its original top face up
    cube_max_yaw_deg: float = 15.0  # cube keeps its yaw (modulo 90 deg)
    cube_max_release_offset: float = 0.03  # cube released within 3 cm of where it rests (carried, not shoved)
    box_margin: float = 0.003  # block centre this far inside the inner walls
    box_max_float: float = 0.005  # block centre at most max half extent + this above the box floor (resting on it)
    flip_pos_tol: float = 0.015  # cylinder back within 1.5 cm (horizontal) of its start
    flip_max_tilt_deg: float = 15.0  # cylinder axis within 15 deg of straight DOWN (upside down)
    flip_height_tol: float = 0.004  # standing on an end: centre at half height +- 4 mm
    flip_min_lift: float = 0.03  # lifted clear of the table at least once
    others_max_shift: float = 0.02  # the other objects stay within 2 cm of their start ...
    others_max_tilt_deg: float = 20.0  # ... and within 20 deg of their start orientation


@dataclass(frozen=True)
class TriSceneConfig:
    """Geometry of the scene and its five placements (metres)."""

    directions: Directions = field(default_factory=Directions)
    box_front: float = 0.08  # box centre: along FRONT from the base origin
    box_left: float = 0.14  # box centre: along LEFT from the base origin
    box_width: float = 0.15  # outer, along LEFT/RIGHT
    box_depth: float = 0.09  # outer, along FRONT/BACK
    box_height: float = 0.05  # outer
    box_wall: float = 0.001
    box_bottom: float = 0.001
    cube_size: float = 0.03
    cube_box_gap: float = 0.02  # box front wall (outer face) to the cube's back face
    green_diameter: float = 0.035
    green_length: float = 0.03
    green_axis: str = "front"  # extrusion axis of the half-cylinder: "front" or "left"
    cylinder_diameter: float = 0.03
    cylinder_height: float = 0.06
    green_from_cube: float = 0.06  # along FRONT, centre to centre
    cylinder_from_green: float = 0.06  # along RIGHT, centre to centre
    placement_step: float = 0.01
    cube_move: float = 0.05  # along LEFT, relative to the cube's position at the start of the episode
    # Where the flip task stands the cylinder for its side grasp: 33.9 cm FRONT of the base origin (30 cm from the pan
    # axis). A horizontal side grasp at the cylinder's middle (3 cm up) needs >= ~29 cm from the pan axis with the
    # expert's 4 deg joint margins (wrist_flex +-95 deg); the front camera sees the table only to ~26 cm from the base
    # origin, so the station is just beyond the front view (the wrist camera sees it).
    station_front: float = 0.339
    station_left: float = 0.0
    tolerances: TriTolerances = field(default_factory=TriTolerances)

    @property
    def placement_names(self) -> tuple[str, ...]:
        return ("default", "left", "right", "forward", "backward")

    def placement_offset(self, k: int) -> np.ndarray:
        name = self.placement_names[k]
        return np.zeros(2) if name == "default" else self.placement_step * self.directions.vec(
            {"forward": "front", "backward": "back"}.get(name, name))

    def box_center(self) -> np.ndarray:
        return self.directions.xy(self.box_front, self.box_left)

    def default_positions(self) -> dict[str, np.ndarray]:
        """Centres (world xy) of the three small objects in the default placement."""
        d = self.directions
        cube = self.box_center() + d.vec("front") * (self.box_depth / 2 + self.cube_box_gap + self.cube_size / 2)
        green = cube + d.vec("front") * self.green_from_cube
        cylinder = green + d.vec("right") * self.cylinder_from_green
        return {"red_cube": cube, "green_block": green, "gray_cylinder": cylinder}

    def placements(self) -> tuple[dict[str, tuple[float, float, float]], ...]:
        yaw = self.directions.yaw
        green_yaw = yaw if self.green_axis == "front" else yaw + math.pi / 2
        yaws = {"red_cube": yaw, "green_block": green_yaw, "gray_cylinder": yaw}
        out = []
        for k in range(len(self.placement_names)):
            off = self.placement_offset(k)
            out.append({n: (float(p[0] + off[0]), float(p[1] + off[1]), float(yaws[n]))
                        for n, p in self.default_positions().items()})
        return tuple(out)

    def station(self) -> np.ndarray:
        return self.directions.xy(self.station_front, self.station_left)


TRI_SCENE = TriSceneConfig()
_c, _tol = TRI_SCENE, TRI_SCENE.tolerances

WHITE_BOX = ContainerSpec(
    "white_box", "tray",
    # tray inner half sizes along the box's own x (= FRONT, depth) and y (= LEFT, width)
    inner_size=(_c.box_depth / 2 - _c.box_wall, _c.box_width / 2 - _c.box_wall),
    rgba=(0.95, 0.95, 0.93, 1.0), description="white box", wall_height=_c.box_height, wall_thickness=_c.box_wall,
    base_thickness=_c.box_bottom, mass=0.2, base_below_table=0.01,
    fixed_pose=(float(_c.box_center()[0]), float(_c.box_center()[1]), _c.directions.yaw))
TRI_RED_CUBE = ObjectSpec("red_cube", "box", (_c.cube_size / 2,) * 3, (0.85, 0.12, 0.10, 1.0), "red cube", mass=0.02)
GREEN_BLOCK = ObjectSpec("green_block", "half_cylinder", (_c.green_diameter / 2, _c.green_length / 2),
                         (0.20, 0.88, 0.18, 1.0), "green semicircular block", mass=0.015, solref=(0.005, 1.0))
GRAY_CYLINDER = ObjectSpec("gray_cylinder", "cylinder", (_c.cylinder_diameter / 2, _c.cylinder_height / 2),
                           (0.80, 0.80, 0.80, 1.0), "light gray cylinder", mass=0.03)

TRI_LAYOUT = FixedPlacementLayout(
    name="tri_tabletop_v1",
    objects=(TRI_RED_CUBE, GREEN_BLOCK, GRAY_CYLINDER),
    containers=(WHITE_BOX,),
    regions={},
    placements=_c.placements(),
    placement_names=_c.placement_names,
)

_cube_offset = tuple(float(v) for v in _c.cube_move * _c.directions.vec("left"))
# Like stack_red_on_blue: every episode starts in the folded rest pose (how the real arm starts); the cube and box
# demonstrations park there again and hold 2 s. The flip demonstration ends hovering above the cylinder instead: its
# final frame must not look like its first one (the cylinder looks the same either way up).
REST_START = tuple(float(v) for v in REST_DEG)

CUBE_LEFT = Task(
    task_id="cube_left",
    language="Pick up the red cube and place it 5 cm to the left.",
    layout=TRI_LAYOUT,
    goal=(MovedBy("red_cube", _cube_offset, tol=_tol.cube_pos_tol, max_tilt_deg=_tol.cube_max_tilt_deg,
                  max_yaw_deg=_tol.cube_max_yaw_deg, min_lift=_tol.min_lift, max_speed=_tol.max_speed,
                  max_release_offset=_tol.cube_max_release_offset, min_tcp_distance=_tol.min_tcp_distance),
          Undisturbed(("green_block", "gray_cylinder"), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=MoveBy("red_cube", _cube_offset),
    max_steps=450,
    start_deg=REST_START,
    return_to_start=True,
    success_hold_steps=_tol.hold_steps,
    tags=("so101_tri",),
)

GREEN_INTO_BOX = Task(
    task_id="green_into_box",
    language="Pick up the green semicircular block and put it into the white box.",
    layout=TRI_LAYOUT,
    goal=(RestingInContainer("green_block", "white_box", margin=_tol.box_margin, max_float=_tol.box_max_float,
                             min_lift=_tol.min_lift, max_speed=_tol.max_speed,
                             min_tcp_distance=_tol.min_tcp_distance),
          Undisturbed(("red_cube", "gray_cylinder"), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=PlaceInBox("green_block", "white_box"),
    max_steps=450,
    start_deg=REST_START,
    return_to_start=True,
    success_hold_steps=_tol.hold_steps,
    tags=("so101_tri",),
)

CYLINDER_FLIP = Task(
    task_id="cylinder_flip",
    language="Pick up the light gray cylinder at its middle, flip it upside down, and place it back in its original "
             "position.",
    layout=TRI_LAYOUT,
    goal=(FlippedInPlace("gray_cylinder", tol=_tol.flip_pos_tol, max_tilt_deg=_tol.flip_max_tilt_deg,
                         height_tol=_tol.flip_height_tol, min_lift=_tol.flip_min_lift, max_speed=_tol.max_speed,
                         min_tcp_distance=_tol.min_tcp_distance),
          Undisturbed(("red_cube", "green_block"), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=FlipViaStation("gray_cylinder", tuple(float(v) for v in _c.station())),
    max_steps=1200,  # 40 s: three grasps (to the station, the flip, back)
    start_deg=REST_START,
    success_hold_steps=_tol.hold_steps,
    tags=("so101_tri",),
)

TRI_TASKS = (CUBE_LEFT, GREEN_INTO_BOX, CYLINDER_FLIP)
