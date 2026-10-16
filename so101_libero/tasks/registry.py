"""Task suites. Add a task by appending a ``Task`` here (or registering one at runtime with ``register``).

Suite ``so101_mini`` (LIBERO-Goal style): one shared layout with all four items; the instruction decides
which object goes where.
"""

from __future__ import annotations

import math

from so101_libero.sim.objects import BLACK_TRAY, BLUE_CUBE, RED_CYLINDER, ROUND_CONTAINER
from so101_libero.tasks.base import InContainer, Layout, PickPlace, PlaceRelative, Region, RelativeTo, Task

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
