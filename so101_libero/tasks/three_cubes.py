"""Suite ``so101_three_cubes``: five tasks in ONE scene with three plain 3 cm cubes (red, yellow, green) and exactly five
global workspace layouts. Every task shows all three cubes; the tasks differ only in where the cubes start, the
instruction and the success rule (docs/THREE_CUBES.md).

  flip_red_cube          "Flip the red cube and place it back at its original position."
  sort_ryg               "Arrange the cubes from left to right as red, yellow, green."
  build_pyramid          "Place the green cube centered on top of the red and yellow cubes."
  stack_yellow_on_red    "Put the yellow cube on top of the red cube."
  green_right_of_red     "Move the green cube to the right of the red cube."

The last two start from the same row (green, yellow, red from the robot's left: the sort's "GYR" start), so only the
instruction tells them apart.

Shared by all the tasks: the robot, the table, the cameras (front = the measured real camera, wrist), the 30 Hz
joint-position action space, the observations, the three cube models and the HOME pose the episodes start from.

Frame: robot frame of the world, metres. +x = FRONT (away from the robot base, towards the front camera), +y = the
robot's own LEFT (the front camera faces the robot, so the robot's left is the right side of the front image). "Left to
right" in the sort instruction is the robot's left to right: red at the largest y, green at the smallest. The base
origin lies 3.9 cm behind the shoulder's pan axis.

The cubes are plain solid colours, exactly like the physical ones: no markers, no textures, nothing that exists only in
simulation. In particular the red cube looks the same before and after a flip; the simulator judges the flip from the
hidden body orientation it tracks (``FlippedInPlace``), which no policy ever sees.

Layouts: five predefined global offsets (``ThreeCubeScene.layout_offsets``) applied to everything a task places (and to
its targets); nothing else is random at reset, the arm always starts at HOME. The sort task combines each layout with
five starting permutations (25 placements, each permutation in every layout); the other tasks have one placement per
layout. ``python -m so101_libero.cli.debug_three_cubes --task <task> layouts`` renders and checks them.

Flip. A 5-DoF SO-101 cannot turn a cube upside down with one grasp at table height (its approach can only lean in the
vertical plane through the pan axis, and both grasp and release need >= ~40 degrees of downward tilt). The flip is the
one-grasp motion that is physically possible and consistent: grasp the red cube across two side faces with the
approach tilted 45 degrees outward, lift, roll the wrist 180 degrees about that approach axis, set the cube back down at
its start. The cube turns by a 180 degree rotation (about the tilted axis): its top face ends up facing the robot.
Every demonstration uses the same grasp side and roll direction (``FlipInPlace``). The cube faces the pan axis at the
start (in layout "center" it is square to the table; the others turn it by <= 7 degrees) so the tilted jaws close
flat on two faces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from so101_libero.sim.objects import ObjectSpec
from so101_libero.sim.robot import HOME_DEG as ROBOT_HOME_DEG
from so101_libero.tasks.base import (FixedPlacementLayout, FlipInPlace, FlippedInPlace, OnTopOfPair, PlaceRelative,
                                     RelativeTo, Released, SortedRow, SortIntoRow, StackedOn, StackOn, StackOnPair,
                                     Task, Undisturbed)

PAN_AXIS_FRONT = 0.0388  # the shoulder's pan axis, along FRONT from the base origin (so101.xml)
SUITE = "so101_three_cubes"
CUBES = ("red_cube", "yellow_cube", "green_cube")
LETTER = {"R": "red_cube", "Y": "yellow_cube", "G": "green_cube"}

FLIP_INSTRUCTION = "Flip the red cube and place it back at its original position."
SORT_INSTRUCTION = "Arrange the cubes from left to right as red, yellow, green."
PYRAMID_INSTRUCTION = "Place the green cube centered on top of the red and yellow cubes."
STACK_INSTRUCTION = "Put the yellow cube on top of the red cube."
RIGHT_INSTRUCTION = "Move the green cube to the right of the red cube."


@dataclass(frozen=True)
class ThreeCubeTolerances:
    """Success tolerances (simulation ground truth)."""

    hold_steps: int = 10  # the goal must hold this many consecutive control steps (0.33 s at 30 Hz): settled
    max_speed: float = 0.02  # m/s: "at rest"
    min_tcp_distance: float = 0.04  # released: no robot contact and the TCP this far from the cube's centre
    min_lift: float = 0.02  # flip: the red cube was carried at least this high
    flip_tol: float = 0.02  # flip: back within 2 cm (horizontal) of its start
    flip_min_rotation_deg: float = 150.0  # flip: turned by >= 150 deg (the motion turns it by 180)
    flip_min_top_turn_deg: float = 60.0  # flip: the start-top face turned >= 60 deg away from up (not a yaw spin)
    flat_max_tilt_deg: float = 15.0  # a cube lies flat on a face
    sort_tol: float = 0.02  # sort: every cube within 2 cm of its target (targets are 5 cm apart: the order is implied)
    pyramid_tol: float = 0.01  # pyramid: green's centre within 1 cm of the red/yellow midpoint
    pyramid_max_gap: float = 0.004  # pyramid: green's bottom within 4 mm of the base tops
    pyramid_base_max_shift: float = 0.01  # pyramid: red and yellow each moved <= 1 cm
    stack_tol: float = 0.012  # stack: yellow's centre over red's top face (within 1.2 cm of red's centre) ...
    stack_max_gap: float = 0.004  # ... its bottom within 4 mm of red's top; red moved <= pyramid_base_max_shift
    right_min_dist: float = 0.04  # green_right: green's centre >= 4 cm right of red's (1 cm between their faces) ...
    right_max_lateral: float = 0.025  # ... and <= 2.5 cm in front of or behind it; red moved <= 1 cm
    others_max_shift: float = 0.02  # flip: yellow and green stay within 2 cm of their start ...
    others_max_tilt_deg: float = 20.0  # ... and within 20 deg of their start orientation


@dataclass(frozen=True)
class ThreeCubeScene:
    """Geometry of the scene, its five layouts and the three tasks' start configurations (metres, robot frame)."""

    cube_size: float = 0.03
    cube_mass: float = 0.02  # assumption (~0.74 g/cm^3, a wooden cube), not measured
    red_rgba: tuple[float, float, float, float] = (0.85, 0.12, 0.10, 1.0)
    yellow_rgba: tuple[float, float, float, float] = (0.95, 0.78, 0.10, 1.0)
    green_rgba: tuple[float, float, float, float] = (0.13, 0.62, 0.22, 1.0)
    # The five global layouts: (FRONT, LEFT) offsets of everything a task places. Small enough that every cube stays
    # reachable, inside the front image and clear of the arm's HOME pose.
    layout_names: tuple[str, ...] = ("center", "left", "right", "forward", "back")
    layout_offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.02), (0.0, -0.02), (0.008, 0.0),
                                                        (-0.008, 0.0))
    # flip_red_cube (layout "center"): red straight in front of the pan axis, yellow and green as distractors to its
    # sides. The 45 deg tilted flip grasp reaches the cube only >= ~18.5 cm from the pan axis (x >= ~0.224 on the centre
    # line) and the front image ends at x ~0.26 (a cube centre at x <= 0.240 keeps >= 6 px to the image's bottom edge):
    # the red cube spans x 0.224-0.240 over the layouts, which is why FRONT/BACK offsets are 8 mm.
    flip_red_at: tuple[float, float] = (0.232, 0.0)
    flip_yellow_at: tuple[float, float] = (0.175, 0.105)
    flip_green_at: tuple[float, float] = (0.175, -0.105)
    # sort_ryg: the cubes start in three slots (left, centre, right) in one of the permutations below (letters for the
    # left, centre and right slot) and end in the target row (red, yellow, green from the robot's left), 5 cm apart.
    # The start row is the one nearer the robot (picked with the jaws closing along the row, 9 cm apart), the target row
    # 6 cm further out (placed with the jaws closing front-back, so the open jaws never reach between row neighbours).
    # At HOME the arm hides the table from the front camera only near y = 0 for x <= 0.15.
    sort_slots: tuple[tuple[float, float], ...] = ((0.172, 0.09), (0.172, 0.0), (0.172, -0.09))
    sort_targets: tuple[tuple[float, float], ...] = ((0.232, 0.05), (0.232, 0.0), (0.232, -0.05))
    sort_permutations: tuple[str, ...] = ("GRY", "YGR", "RGY", "GYR", "YRG")
    # build_pyramid: red and yellow side by side (red on the robot's left), 1 mm apart; green starts on the left
    pyramid_base_at: tuple[float, float] = (0.215, -0.04)  # midpoint between red and yellow
    pyramid_base_gap: float = 0.001
    pyramid_green_at: tuple[float, float] = (0.19, 0.09)
    # stack_yellow_on_red and green_right_of_red: one start for both (only the instruction differs), the cubes in the
    # sort's start slots in this order (robot's left, centre, right): green, yellow, red, i.e. red on the robot's right.
    row_order: str = "GYR"
    right_distance: float = 0.06  # green's expert target: 6 cm to the robot's right of red (3 cm between their faces)
    # HOME, in LeRobot so101_follower units (degrees x5, gripper 0..100): the pose every episode starts in and the
    # deterministic finisher returns to (so101_libero/control/home.py). The arm angles are the project's HOME_DEG
    # (sim/robot.py, the real runner's home): gripper 13 cm above the table over the workspace centre, tilted 30 deg
    # forward, clear of every cube; the gripper at 10/100.
    home_deg: tuple[float, ...] = (*(float(v) for v in ROBOT_HOME_DEG[:5]), 10.0)
    flip_max_steps: int = 600  # policy horizons (control steps at 30 Hz)
    sort_max_steps: int = 1050
    pyramid_max_steps: int = 450
    stack_max_steps: int = 450
    right_max_steps: int = 450
    tolerances: ThreeCubeTolerances = field(default_factory=ThreeCubeTolerances)

    # ---------------------------------------------------------------------------------------------- helpers
    @property
    def num_layouts(self) -> int:
        return len(self.layout_offsets)

    def offset(self, layout_id: int) -> np.ndarray:
        return np.asarray(self.layout_offsets[layout_id], dtype=float)

    def facing_pan_axis(self, xy) -> float:
        """Yaw that turns a cube at ``xy`` so that one face points at the shoulder's pan axis."""
        return math.atan2(float(xy[1]), float(xy[0]) - PAN_AXIS_FRONT)

    def flip_placements(self) -> tuple[dict, ...]:
        out = []
        for k in range(self.num_layouts):
            o = self.offset(k)
            red = np.asarray(self.flip_red_at) + o
            out.append({"red_cube": (*map(float, red), self.facing_pan_axis(red)),
                        "yellow_cube": (*map(float, np.asarray(self.flip_yellow_at) + o), 0.0),
                        "green_cube": (*map(float, np.asarray(self.flip_green_at) + o), 0.0)})
        return tuple(out)

    def sort_placements(self) -> tuple[list, list, list, list]:
        """25 placements: placement 5 * layout + p = layout ``layout`` with permutation ``p``."""
        poses, names, layouts, perms = [], [], [], []
        for k in range(self.num_layouts):
            o = self.offset(k)
            for perm in self.sort_permutations:
                poses.append({LETTER[c]: (*map(float, np.asarray(s) + o), 0.0) for c, s in zip(perm, self.sort_slots)})
                names.append(f"{self.layout_names[k]}/{perm}")
                layouts.append(k)
                perms.append(perm)
        return poses, names, layouts, perms

    @property
    def sort_slot_centroid(self) -> tuple[float, float]:
        return tuple(float(v) for v in np.mean(np.asarray(self.sort_slots), axis=0))

    def pyramid_placements(self) -> tuple[dict, ...]:
        half = (self.cube_size + self.pyramid_base_gap) / 2
        out = []
        for k in range(self.num_layouts):
            o = self.offset(k)
            base = np.asarray(self.pyramid_base_at) + o
            out.append({"red_cube": (float(base[0]), float(base[1] + half), 0.0),  # red on the robot's left
                        "yellow_cube": (float(base[0]), float(base[1] - half), 0.0),
                        "green_cube": (*map(float, np.asarray(self.pyramid_green_at) + o), 0.0)})
        return tuple(out)

    def row_placements(self) -> tuple[dict, ...]:
        """The shared start of stack_yellow_on_red and green_right_of_red: ``row_order`` in the sort slots."""
        return tuple({LETTER[c]: (*map(float, np.asarray(s) + self.offset(k)), 0.0)
                      for c, s in zip(self.row_order, self.sort_slots)} for k in range(self.num_layouts))


@dataclass(frozen=True)
class CubeLayout(FixedPlacementLayout):
    """``FixedPlacementLayout`` whose placements are (global layout, starting permutation) pairs."""

    placement_layout_ids: tuple[int, ...] = ()
    placement_permutations: tuple[str | None, ...] = ()
    layout_names: tuple[str, ...] = ()
    permutations: tuple[str, ...] = ()  # the permutation ids index this tuple (empty: no permutations)

    def placement_info(self, placement_id: int) -> dict:
        """Layout id/name and starting permutation of a placement (episode metadata, evaluation tables)."""
        k = int(placement_id)
        perm = self.placement_permutations[k]
        return {"placement_id": k, "placement_name": self.placement_names[k],
                "layout_id": int(self.placement_layout_ids[k]),
                "layout_name": self.layout_names[self.placement_layout_ids[k]],
                "permutation": perm, "permutation_id": self.permutations.index(perm) if perm else -1}


SCENE = ThreeCubeScene()
_tol = SCENE.tolerances
_size = (SCENE.cube_size / 2,) * 3
RED = ObjectSpec("red_cube", "box", _size, SCENE.red_rgba, "red cube", mass=SCENE.cube_mass)
YELLOW = ObjectSpec("yellow_cube", "box", _size, SCENE.yellow_rgba, "yellow cube", mass=SCENE.cube_mass)
GREEN = ObjectSpec("green_cube", "box", _size, SCENE.green_rgba, "green cube", mass=SCENE.cube_mass)
OBJECTS = (RED, YELLOW, GREEN)  # the same three cubes, in this order, in every task: one compiled scene
HOME_DEG = tuple(float(v) for v in SCENE.home_deg)
_layout_ids = tuple(range(SCENE.num_layouts))

FLIP_LAYOUT = CubeLayout(
    name="three_cubes_v1_flip", objects=OBJECTS, containers=(), regions={},
    placements=SCENE.flip_placements(), placement_names=SCENE.layout_names,
    placement_layout_ids=_layout_ids, placement_permutations=(None,) * SCENE.num_layouts,
    layout_names=SCENE.layout_names)
_sp, _sn, _sl, _sperm = SCENE.sort_placements()
SORT_LAYOUT = CubeLayout(
    name="three_cubes_v1_sort", objects=OBJECTS, containers=(), regions={},
    placements=tuple(_sp), placement_names=tuple(_sn), placement_layout_ids=tuple(_sl),
    placement_permutations=tuple(_sperm), layout_names=SCENE.layout_names, permutations=SCENE.sort_permutations)
PYRAMID_LAYOUT = CubeLayout(
    name="three_cubes_v1_pyramid", objects=OBJECTS, containers=(), regions={},
    placements=SCENE.pyramid_placements(), placement_names=SCENE.layout_names,
    placement_layout_ids=_layout_ids, placement_permutations=(None,) * SCENE.num_layouts,
    layout_names=SCENE.layout_names)
ROW_LAYOUT = CubeLayout(
    name="three_cubes_v1_row", objects=OBJECTS, containers=(), regions={},
    placements=SCENE.row_placements(), placement_names=SCENE.layout_names,
    placement_layout_ids=_layout_ids, placement_permutations=(None,) * SCENE.num_layouts,
    layout_names=SCENE.layout_names)

FLIP_RED_CUBE = Task(
    task_id="flip_red_cube",
    language=FLIP_INSTRUCTION,
    layout=FLIP_LAYOUT,
    goal=(FlippedInPlace("red_cube", tol=_tol.flip_tol, min_rotation_deg=_tol.flip_min_rotation_deg,
                         min_top_turn_deg=_tol.flip_min_top_turn_deg, max_flat_deg=_tol.flat_max_tilt_deg,
                         min_lift=_tol.min_lift, max_speed=_tol.max_speed, min_tcp_distance=_tol.min_tcp_distance),
          Undisturbed(("yellow_cube", "green_cube"), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=FlipInPlace("red_cube", tilt_deg=45.0, psi_deg=-90.0, roll_dir=-1),  # one motion for every demonstration
    max_steps=SCENE.flip_max_steps,
    success_hold_steps=_tol.hold_steps,
    tags=(SUITE,),
    start_deg=HOME_DEG,
)

SORT_ROW = SortedRow(CUBES, SCENE.sort_targets, SCENE.sort_slot_centroid, tol=_tol.sort_tol,
                     max_tilt_deg=_tol.flat_max_tilt_deg, max_speed=_tol.max_speed,
                     min_tcp_distance=_tol.min_tcp_distance)
SORT_RYG = Task(
    task_id="sort_ryg",
    language=SORT_INSTRUCTION,
    layout=SORT_LAYOUT,
    goal=(SORT_ROW,),
    plan=SortIntoRow(CUBES, SCENE.sort_targets, SCENE.sort_slot_centroid),
    max_steps=SCENE.sort_max_steps,
    success_hold_steps=_tol.hold_steps,
    tags=(SUITE,),
    start_deg=HOME_DEG,
)

BUILD_PYRAMID = Task(
    task_id="build_pyramid",
    language=PYRAMID_INSTRUCTION,
    layout=PYRAMID_LAYOUT,
    goal=(OnTopOfPair("green_cube", ("red_cube", "yellow_cube"), tol=_tol.pyramid_tol, max_gap=_tol.pyramid_max_gap,
                      max_tilt_deg=_tol.flat_max_tilt_deg, max_base_shift=_tol.pyramid_base_max_shift,
                      max_speed=_tol.max_speed, min_tcp_distance=_tol.min_tcp_distance),),
    plan=StackOnPair("green_cube", ("red_cube", "yellow_cube")),
    max_steps=SCENE.pyramid_max_steps,
    success_hold_steps=_tol.hold_steps,
    tags=(SUITE,),
    start_deg=HOME_DEG,
)

STACK_YELLOW_ON_RED = Task(
    task_id="stack_yellow_on_red",
    language=STACK_INSTRUCTION,
    layout=ROW_LAYOUT,
    goal=(StackedOn("yellow_cube", "red_cube", max_offset=_tol.stack_tol, max_gap=_tol.stack_max_gap,
                    max_tilt_deg=_tol.flat_max_tilt_deg, max_speed=_tol.max_speed,
                    min_tcp_distance=_tol.min_tcp_distance),
          Undisturbed(("red_cube",), _tol.pyramid_base_max_shift, _tol.others_max_tilt_deg),
          Undisturbed(("green_cube",), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=StackOn("yellow_cube", "red_cube"),
    max_steps=SCENE.stack_max_steps,
    success_hold_steps=_tol.hold_steps,
    tags=(SUITE,),
    start_deg=HOME_DEG,
)

GREEN_RIGHT_OF_RED = Task(
    task_id="green_right_of_red",
    language=RIGHT_INSTRUCTION,
    layout=ROW_LAYOUT,
    goal=(RelativeTo("green_cube", "red_cube", "right", min_dist=_tol.right_min_dist,
                     max_lateral=_tol.right_max_lateral, max_speed=_tol.max_speed,
                     max_ref_shift=_tol.pyramid_base_max_shift),
          Released("green_cube", _tol.max_speed, _tol.min_tcp_distance),
          Undisturbed(("yellow_cube",), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=PlaceRelative("green_cube", "red_cube", "right", distance=SCENE.right_distance),
    max_steps=SCENE.right_max_steps,
    success_hold_steps=_tol.hold_steps,
    tags=(SUITE,),
    start_deg=HOME_DEG,
)

THREE_CUBE_TASKS = (FLIP_RED_CUBE, SORT_RYG, BUILD_PYRAMID, STACK_YELLOW_ON_RED, GREEN_RIGHT_OF_RED)
# Short names used by the scripts (bash scripts/train_flip_smolvla.sh, --task flip, ...)
TASK_ALIASES = {"flip": "flip_red_cube", "sort": "sort_ryg", "pyramid": "build_pyramid",
                "stack": "stack_yellow_on_red", "green_right": "green_right_of_red"}


def resolve_task_id(name: str) -> str:
    """'flip' / 'sort' / 'pyramid' / 'stack' / 'green_right' or a task id -> task id."""
    return TASK_ALIASES.get(name, name)
