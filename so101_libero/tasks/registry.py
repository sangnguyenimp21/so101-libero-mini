"""Task suites. Add a task by appending a ``Task`` here (or registering one at runtime with ``register``).

Suite ``so101_mini`` (LIBERO-Goal style): one shared layout with all four items; the instruction decides
which object goes where.
"""

from __future__ import annotations

import math

from so101_libero.sim.objects import (BLACK_TRAY, BLUE_CUBE, RED_CUBE, RED_CUBE_VAR, RED_CYLINDER, ROUND_CONTAINER,
                                      WHITE_TRAY)
from so101_libero.tasks.base import (GridRegion, InContainer, Layout, OnTableInRegion, PickPlace, PlaceAt, PlaceRelative,
                                     Region, RelativeTo, Task)

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


def register(task: Task, suite: str | None = None) -> Task:
    if task.task_id in TASKS:
        raise ValueError(f"duplicate task id {task.task_id}")
    TASKS[task.task_id] = task
    if suite:
        SUITES.setdefault(suite, []).append(task.task_id)
    return task


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


# Red cube into the white tray: the 3 cm red cube and a FIXED white tray (inner 14 x 9 cm, 6 cm walls) centred 15 cm in
# front of the base and 15 cm to its left, long side along x. The cube starts in one of 10 cells (2 rows along x x
# 5 columns along y, cell = seed % 10, so demonstrations and evaluation states cover the cells equally) of
# x 0.16-0.24, y -0.10..+0.06: the cube area of the other real tasks moved 2 cm to the right, as the tray's near wall
# (y = 0.099) would cut through its left edge at y = 0.08, and starting 2 cm further out, as the expert's higher carry
# over the 6 cm walls (TCP ~10-11.5 cm) is out of reach straight above cubes closer than ~15.5 cm. The tray counts by
# its outer rectangle, so cubes may start 1.5 cm from its wall.
REAL_WHITE_TRAY_TABLETOP = Layout(
    name="real_white_tray_tabletop_v1",
    objects=(RED_CUBE,),
    containers=(WHITE_TRAY,),
    regions={"red_cube": GridRegion(x=(0.16, 0.24), y=(-0.10, 0.06), nx=2, ny=5, yaw=(-45 * DEG, 45 * DEG)),
             "white_tray": Region(x=(0.15, 0.15), y=(0.15, 0.15), yaw=(0.0, 0.0))},
    clearance=0.015,
    tray_rectangles=True,
)
register(pick_place("red_cube_to_white_tray", "Pick up the red cube and put it in the white tray.", "red_cube",
                    "white_tray", layout=REAL_WHITE_TRAY_TABLETOP), suite="so101_real")


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
