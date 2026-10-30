"""Task suites. Add a task by appending a ``Task`` here (or registering one at runtime with ``register``).

Suite ``so101_mini`` (LIBERO-Goal style): one shared layout with all four items; the instruction decides
which object goes where.
"""

from __future__ import annotations

import math

from so101_libero.sim.objects import (BLACK_BOX, BLACK_TRAY, BLUE_CUBE, RED_CUBE, RED_CUBE_VAR, RED_CYLINDER,
                                      ROUND_CONTAINER)
from so101_libero.sim.robot import REST_DEG
from so101_libero.tasks.base import (GridRegion, InContainer, Layout, OnTableInRegion, PickPlace, PlaceAt, PlaceRelative,
                                     Region, RelativeRegion, RelativeTo, StackedOn, StackOn, Task)

DEG = math.pi / 180

# Placement regions (world frame; robot base at the origin facing +x, +y = robot's left). All grasp and
# release poses stay inside the arm's top-down workspace (radius ~0.10-0.25 m around the pan axis).
TABLETOP = Layout(
    name="tabletop_v1",
    objects=(RED_CYLINDER, BLUE_CUBE),
    containers=(BLACK_TRAY, ROUND_CONTAINER),
    regions={
        "black_tray": Region(x=(0.15, 0.21), y=(-0.19, -0.13), yaw=(-15 * DEG, 15 * DEG)),
        "round_container": Region(x=(0.15, 0.21), y=(0.13, 0.19), yaw=(-math.pi, math.pi)),
        "red_cylinder": Region(x=(0.16, 0.26), y=(-0.07, 0.07)),
        "blue_cube": Region(x=(0.16, 0.26), y=(-0.07, 0.07), yaw=(-45 * DEG, 45 * DEG)),
    },
)

TASKS: dict[str, Task] = {}
SUITES: dict[str, list[str]] = {}
# Seed slots: the task registered k-th (retired tasks included) draws its training seeds from k * 100,000 and its
# evaluation seeds from 10,000,000 + k * 100,000. A removed task keeps its slot (``retire``), so the seed ranges of the
# tasks after it, and the datasets and evaluation states made from them, never move.
SEED_SLOTS: list[str] = []


def register(task: Task, suite: str | None = None) -> Task:
    if task.task_id in TASKS or task.task_id in SEED_SLOTS:
        raise ValueError(f"duplicate task id {task.task_id}")
    TASKS[task.task_id] = task
    SEED_SLOTS.append(task.task_id)
    if suite:
        SUITES.setdefault(suite, []).append(task.task_id)
    return task


def retire(task_id: str) -> None:
    """Keep the seed slot of a task that was removed from the registry at this point."""
    SEED_SLOTS.append(task_id)


def seed_slot(task_id: str) -> int:
    """Seed slot of a registered task (see SEED_SLOTS)."""
    if task_id not in TASKS:
        raise KeyError(f"unknown task {task_id!r}")
    return SEED_SLOTS.index(task_id)


def pick_place(task_id: str, language: str, obj: str, container: str, layout: Layout = TABLETOP, **kw) -> Task:
    return Task(task_id=task_id, language=language, layout=layout, goal=(InContainer(obj, container),),
                plan=PickPlace(obj, container), **kw)


register(pick_place("cylinder_to_tray", "Pick up the red cylinder and place it inside the black tray.",
                    "red_cylinder", "black_tray"), suite="so101_mini")
register(pick_place("cylinder_to_container", "Pick up the red cylinder and place it inside the circular container.",
                    "red_cylinder", "round_container"), suite="so101_mini")
register(pick_place("cube_to_tray", "Pick up the blue object and place it inside the black tray.",
                    "blue_cube", "black_tray"), suite="so101_mini")


# Suite so101_extra: other goal types on the same scene. "left" is the robot's left (+y; the right side of the front
# camera image, which faces the robot).
# Same items, other regions: the cube stays right of centre so the spot on its left is free of the round container.
TABLETOP_SPATIAL = Layout(
    name="tabletop_v1_spatial",
    objects=TABLETOP.objects,
    containers=TABLETOP.containers,
    regions={**TABLETOP.regions, "blue_cube": Region(x=(0.17, 0.25), y=(-0.08, -0.02), yaw=(-45 * DEG, 45 * DEG))},
)
register(Task(task_id="cylinder_left_of_cube", language="Move the red object to the left of the blue object.",
              layout=TABLETOP_SPATIAL, goal=(RelativeTo("red_cylinder", "blue_cube", "left"),),
              plan=PlaceRelative("red_cylinder", "blue_cube", "left")), suite="so101_extra")


# Suite so101_real: the physical setup (docs/real_setup/PROMPT.md): a single 3 cm red cube on an otherwise empty
# table. "Aside" is a fixed zone on the robot's LEFT (+y; the right-hand side of the front camera image). Both regions
# lie inside the real front camera's view (table x <= ~0.26 m is visible) and inside the arm's top-down workspace
# (0.13-0.22 m from the pan axis for the release spot; grasps also use tilted approaches).
ASIDE_ZONE = Region(x=(0.08, 0.18), y=(0.13, 0.20))
REAL_TABLETOP = Layout(
    name="real_tabletop_v1",
    objects=(RED_CUBE,),
    containers=(),
    regions={"red_cube": Region(x=(0.14, 0.24), y=(-0.08, 0.08), yaw=(-45 * DEG, 45 * DEG))},
)
register(Task(task_id="red_cube_aside", language="Pick up the red cube and place it aside.", layout=REAL_TABLETOP,
              goal=(OnTableInRegion("red_cube", ASIDE_ZONE, region_name="aside_zone"),),
              plan=PlaceAt("red_cube", ASIDE_ZONE)), suite="so101_real")


# Red cube into the black box: the same red cube, now 2.7-4.0 cm wide (drawn per episode, stored in the initial state),
# and the black tray (inner 12 x 9 cm, 2.5 cm walls) on the robot's LEFT, centred at (0.15, 0.15) +- 2 cm, yaw +-10 deg
# (long side along x; with its centre beyond y = 0.17 the outer wall leaves the front camera image). The cube's start
# region is the red_cube_aside one; starts that overlap the tray's footprint are resampled by the layout, which pushes
# the cube towards the robot's right.
REAL_BOX_TABLETOP = Layout(
    name="real_box_tabletop_v1",
    objects=(RED_CUBE_VAR,),
    containers=(BLACK_TRAY,),
    regions={"red_cube": Region(x=(0.14, 0.24), y=(-0.08, 0.08), yaw=(-45 * DEG, 45 * DEG)),
             "black_tray": Region(x=(0.13, 0.17), y=(0.13, 0.17), yaw=(-10 * DEG, 10 * DEG))},
)
register(pick_place("red_cube_to_box", "Pick up the red cube and put it in the black box.", "red_cube", "black_tray",
                    layout=REAL_BOX_TABLETOP), suite="so101_real")


# Retired: red_cube_to_white_tray and red_cube_far_to_white_tray (a fixed white tray on the robot's left), removed
# 2026-10-21. Their seed slots (6, 7) stay reserved.
retire("red_cube_to_white_tray")
retire("red_cube_far_to_white_tray")


# The next two tasks keep everything inside the front camera's policy crop (the table up to x ~0.24 m, |y| <~ 0.19 m at
# x 0.15-0.20; the 6 cm rims reach the image edge sooner) and start in the folded rest pose (as real rollouts do) and
# park there again after the task (via home, then a 2 s hold), so the policy stops after the task. At rest the jaws lie just above the table, x 0.12-0.20 m: a cube there
# needs |y| >= 0.06 (>= 3.5 cm clearance, also with the pan +-7 deg off, as the real arm rests at ~-6.6), and a
# two-cube stack y >= 0.12 or y <= -0.06 (the camera mount and the moving jaw stick out to the left at 3-6 cm height).
# On the real table, measure x from the shoulder's pan axis (the vertical axis of the base motor, at x = 0.039 m in this
# frame) and y sideways from the robot's centre line.
REST_START = tuple(float(v) for v in REST_DEG)

# Red cube into the black box (svla_so100_pickplace style): the 3 cm red cube and the black box (10 x 10 cm outside,
# 6 cm walls) on the robot's RIGHT, centred at x 0.14, y -0.13 (10 cm ahead of the pan axis, 13 cm right of the centre
# line; sides along x), +-1 cm and +-5 deg for placing it by hand (< 1 cm further right, its rim leaves the camera
# image; forward there are ~3 cm). The cube starts on the robot's LEFT, beside the folded arm, in one of 6 cells
# (2 rows along x x 3 columns along y, cell = seed % 6) of a small area, x 0.16-0.22, y 0.06-0.12 (12-18 cm ahead of the
# pan axis, 6-12 cm left; the expert's high carry over the 6 cm walls needs x >= ~0.155), and
# is carried across to the box.
REAL_BLACK_BOX_TABLETOP = Layout(
    name="real_black_box_tabletop_v1",
    objects=(RED_CUBE,),
    containers=(BLACK_BOX,),
    regions={"red_cube": GridRegion(x=(0.16, 0.22), y=(0.06, 0.12), nx=2, ny=3, yaw=(-45 * DEG, 45 * DEG)),
             "black_box": Region(x=(0.13, 0.15), y=(-0.14, -0.12), yaw=(-5 * DEG, 5 * DEG))},
    clearance=0.015,
    tray_rectangles=True,
)
register(pick_place("red_cube_to_black_box", "Pick up the red cube and put it in the black box.", "red_cube",
                    "black_box", layout=REAL_BLACK_BOX_TABLETOP, start_deg=REST_START, return_to_start=True),
         suite="so101_real")

# Red cube onto the blue cube (svla_so100_stacking style): two 3 cm cubes side by side on the robot's LEFT, the blue
# one always 10 cm (centre to centre) further left than the red one (+y; the right-hand side of the front image), so
# the finished stack stands clear of the folded arm. The pair moves as one: the red cube starts in one of 4 cells
# (2 x 2 grid over x 0.15-0.21, y 0.06-0.09: 11-17 cm ahead of the pan axis, 6-9 cm left; cell = seed % 4), the blue
# cube at the same x, y 0.16-0.19; each cube's yaw is +-45 deg (any orientation of a cube).
REAL_STACK_TABLETOP = Layout(
    name="real_stack_tabletop_v1",
    objects=(RED_CUBE, BLUE_CUBE),  # red first: the blue cube is placed relative to it
    containers=(),
    regions={"red_cube": GridRegion(x=(0.15, 0.21), y=(0.06, 0.09), nx=2, ny=2, yaw=(-45 * DEG, 45 * DEG)),
             "blue_cube": RelativeRegion("red_cube", dx=0.0, dy=0.10, yaw=(-45 * DEG, 45 * DEG))},
)
register(Task(task_id="stack_red_on_blue", language="Put the red cube on top of the blue cube.",
              layout=REAL_STACK_TABLETOP, goal=(StackedOn("red_cube", "blue_cube"),),
              plan=StackOn("red_cube", "blue_cube"), start_deg=REST_START, return_to_start=True), suite="so101_real")


def get_task(task_id: str) -> Task:
    try:
        return TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(TASKS)}") from None


def get_suite(name: str) -> list[Task]:
    if name in SUITES:
        return [TASKS[t] for t in SUITES[name]]
    if name in TASKS:
        return [TASKS[name]]
    raise KeyError(f"unknown suite or task {name!r}; suites: {sorted(SUITES)}, tasks: {sorted(TASKS)}")


# Suite so101_tri: one scene (a fixed white tray on the robot's right, an all-red cube, a green
# half-cylinder, a light gray cylinder), exactly five predefined placements of the three small objects, three tasks
# (so101_libero/tasks/tri_tasks.py documents the geometry, directions and tolerances). The first version of this suite
# (cube_left, green_into_box, cylinder_flip: white box on the robot's left) was retired on 2026-10-21; its seed slots
# 10-12 stay reserved. green_into_tray (slot 15) was replaced by green_push_forward on 2026-10-23. Slots 13, 14, 16:
# training seeds 1,300,000+ (cylinder_aside), 1,400,000+ (cube_flip), 1,600,000+ (green_push_forward).
from so101_libero.tasks.tri_tasks import CUBE_FLIP, CYLINDER_ASIDE, GREEN_PUSH_FORWARD  # noqa: E402

for _retired in ("cube_left", "green_into_box", "cylinder_flip"):
    retire(_retired)
register(CYLINDER_ASIDE, suite="so101_tri")
register(CUBE_FLIP, suite="so101_tri")
retire("green_into_tray")
register(GREEN_PUSH_FORWARD, suite="so101_tri")

# Suite so101_three_cubes: ONE scene with three plain 3 cm cubes (red, yellow, green), five global layouts, three tasks
# (so101_libero/tasks/three_cubes.py, docs/THREE_CUBES.md). Every episode starts at HOME. Slots: flip_red_cube 17,
# sort_ryg 18, build_pyramid 21 (training seeds 1,700,000+, 1,800,000+, 2,100,000+). sort_ryg has 25 placements (5 layouts
# x 5 starting permutations) and --demos-per-placement draws placement p's seeds from 1,800,000 + p * 10,000, i.e. up to
# 2,049,999: slots 19 and 20 are reserved for that range, so no two tasks ever share a seed (merged datasets need unique
# episode seeds).
from so101_libero.tasks.three_cubes import (BUILD_PYRAMID, FLIP_RED_CUBE, GREEN_RIGHT_OF_RED,  # noqa: E402
                                            SORT_RYG, STACK_YELLOW_ON_RED)

register(FLIP_RED_CUBE, suite="so101_three_cubes")
register(SORT_RYG, suite="so101_three_cubes")
retire("sort_ryg_placements_10_19")
retire("sort_ryg_placements_20_24")
register(BUILD_PYRAMID, suite="so101_three_cubes")
# Added 2026-10-31, both starting from the same row: stack_yellow_on_red 22, green_right_of_red 23.
register(STACK_YELLOW_ON_RED, suite="so101_three_cubes")
register(GREEN_RIGHT_OF_RED, suite="so101_three_cubes")
