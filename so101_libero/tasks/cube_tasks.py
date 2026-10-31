"""Suite ``so101_cubes``: two language-conditioned tasks in ONE tabletop scene (three plain 3 cm cubes: red, yellow,
green) with exactly five predefined initial placements, built with the so101_tri recipe (tasks/tri_tasks.py,
docs/TRI_TASKS.md; the recipe of smolvla_tri_cube_flip_100, which works on the real arm):

  stack_yellow_on_red    "Put the yellow cube on top of the red cube."
  green_right_of_red     "Move the green cube to the right of the red cube."

Both start from the same row, so only the instruction tells them apart. Every episode starts in the folded rest pose
(how the real arm starts) and every demonstration parks there again and holds 2 s (``return_to_start``), so the policy
stops after the task. The scripted expert is the pick-and-place expert (expert/pick_place.py: ``StackOn`` as in
stack_red_on_blue, ``PlaceRelative``).

Directions (robot frame, ``Directions`` of tri_tasks.py): FRONT = +x away from the robot base, towards the front camera;
LEFT = +y the robot's OWN left (the right side of the front image, which faces the robot).

Default scene (metres, world frame; the base origin is 3.9 cm behind the shoulder's pan axis):
  one row across the robot's front, 22.5 cm FRONT of the base origin (18.6 cm in front of the pan axis, like the
  so101_tri green block), cubes 10 cm apart centre to centre (5.8 cm between footprints), square to the table: green
  10 cm LEFT, yellow on the centre line, red 10 cm RIGHT. Distances from the pan axis: yellow 18.6 cm, red and green
  21.1 cm (so101_tri: 17.9-21.6 cm); >= 2 cm clear of the folded arm at rest and while it unfolds.
  green_right_of_red: the expert puts the green cube 6.5 cm to the robot's RIGHT of red (3.5 cm between their faces;
  25 cm from the pan axis); success needs >= 4 cm.
Placements: all three cubes move together by 0 or 1 cm (default, left, right, forward, backward); nothing else is
random at reset; the demonstrations differ only through the expert's own seeded randomness (timing, heights, release
points). ``python -m so101_libero.cli.render_placements --task stack_yellow_on_red`` renders and checks them.

Physical assumptions (not measured on the real cubes): 20 g each, sliding friction 1.0 like every other item.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from so101_libero.sim.objects import ObjectSpec
from so101_libero.tasks.base import (FixedPlacementLayout, PlaceRelative, RelativeTo, Released, StackedOn, StackOn,
                                     Task, Undisturbed)
from so101_libero.tasks.tri_tasks import REST_START, Directions

SUITE = "so101_cubes"


@dataclass(frozen=True)
class CubeTolerances:
    """Success tolerances of the two tasks."""

    hold_steps: int = 10  # the goal must hold this many consecutive control steps (0.33 s at 30 Hz): settled
    max_speed: float = 0.02  # m/s: "at rest"
    min_tcp_distance: float = 0.04  # released: no robot contact and the TCP this far from the cube's centre
    flat_max_tilt_deg: float = 15.0  # a cube lies flat on a face
    stack_tol: float = 0.012  # stack: yellow's centre over red's top face (within 1.2 cm of red's centre) ...
    stack_max_gap: float = 0.004  # ... its bottom within 4 mm of red's top
    base_max_shift: float = 0.01  # red (the stack's base / green's reference) moved <= 1 cm
    right_min_dist: float = 0.04  # green_right: green's centre >= 4 cm right of red's (1 cm between their faces) ...
    right_max_lateral: float = 0.025  # ... and <= 2.5 cm in front of or behind it
    others_max_shift: float = 0.02  # the uninvolved cube stays within 2 cm of its start ...
    others_max_tilt_deg: float = 20.0  # ... and within 20 deg of its start orientation


@dataclass(frozen=True)
class CubeSceneConfig:
    """Geometry of the scene and its five placements (metres, along FRONT / LEFT from the base origin)."""

    directions: Directions = field(default_factory=Directions)
    cube_size: float = 0.03
    cube_mass: float = 0.02
    red_rgba: tuple[float, float, float, float] = (0.85, 0.12, 0.10, 1.0)
    yellow_rgba: tuple[float, float, float, float] = (0.95, 0.78, 0.10, 1.0)
    green_rgba: tuple[float, float, float, float] = (0.13, 0.62, 0.22, 1.0)
    row_front: float = 0.225  # default placement: the row's distance along FRONT
    row_spacing: float = 0.10  # centre to centre (>= 5.8 cm between footprints)
    right_distance: float = 0.065  # green_right: the expert's target, RIGHT of red (PlaceRelative's default)
    placement_step: float = 0.01
    tolerances: CubeTolerances = field(default_factory=CubeTolerances)

    @property
    def placement_names(self) -> tuple[str, ...]:
        return ("default", "left", "right", "forward", "backward")

    def placement_offset(self, k: int) -> np.ndarray:
        name = self.placement_names[k]
        return np.zeros(2) if name == "default" else self.placement_step * self.directions.vec(
            {"forward": "front", "backward": "back"}.get(name, name))

    def default_positions(self) -> dict[str, np.ndarray]:
        d, f, s = self.directions, self.row_front, self.row_spacing
        return {"red_cube": d.xy(f, -s), "yellow_cube": d.xy(f, 0.0), "green_cube": d.xy(f, s)}

    def placements(self) -> tuple[dict[str, tuple[float, float, float]], ...]:
        yaw = self.directions.yaw  # square to the table
        out = []
        for k in range(len(self.placement_names)):
            off = self.placement_offset(k)
            out.append({n: (float(p[0] + off[0]), float(p[1] + off[1]), float(yaw))
                        for n, p in self.default_positions().items()})
        return tuple(out)


CUBE_SCENE = CubeSceneConfig()
_c, _tol = CUBE_SCENE, CUBE_SCENE.tolerances
_half = (_c.cube_size / 2,) * 3
RED_CUBE = ObjectSpec("red_cube", "box", _half, _c.red_rgba, "red cube", mass=_c.cube_mass)
YELLOW_CUBE = ObjectSpec("yellow_cube", "box", _half, _c.yellow_rgba, "yellow cube", mass=_c.cube_mass)
GREEN_CUBE = ObjectSpec("green_cube", "box", _half, _c.green_rgba, "green cube", mass=_c.cube_mass)

CUBES_LAYOUT = FixedPlacementLayout(
    name="cubes_row_v1",
    objects=(RED_CUBE, YELLOW_CUBE, GREEN_CUBE),
    containers=(),
    regions={},
    placements=_c.placements(),
    placement_names=_c.placement_names,
)

# PlaceRelative / RelativeTo name their side in the world frame (+y = left): only the default directions match it.
assert math.isclose(_c.directions.yaw, 0.0), "PlaceRelative and RelativeTo assume the default directions"

STACK_YELLOW_ON_RED = Task(
    task_id="stack_yellow_on_red",
    language="Put the yellow cube on top of the red cube.",
    layout=CUBES_LAYOUT,
    goal=(StackedOn("yellow_cube", "red_cube", max_offset=_tol.stack_tol, max_gap=_tol.stack_max_gap,
                    max_tilt_deg=_tol.flat_max_tilt_deg, max_speed=_tol.max_speed,
                    min_tcp_distance=_tol.min_tcp_distance),
          Undisturbed(("red_cube",), _tol.base_max_shift, _tol.others_max_tilt_deg),
          Undisturbed(("green_cube",), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=StackOn("yellow_cube", "red_cube"),
    max_steps=450,
    success_hold_steps=_tol.hold_steps,
    tags=(SUITE,),
    start_deg=REST_START,
    return_to_start=True,
)

GREEN_RIGHT_OF_RED = Task(
    task_id="green_right_of_red",
    language="Move the green cube to the right of the red cube.",
    layout=CUBES_LAYOUT,
    goal=(RelativeTo("green_cube", "red_cube", "right", min_dist=_tol.right_min_dist,
                     max_lateral=_tol.right_max_lateral, max_speed=_tol.max_speed, max_ref_shift=_tol.base_max_shift),
          Released("green_cube", _tol.max_speed, _tol.min_tcp_distance),
          Undisturbed(("yellow_cube",), _tol.others_max_shift, _tol.others_max_tilt_deg)),
    plan=PlaceRelative("green_cube", "red_cube", "right", distance=_c.right_distance),
    max_steps=450,
    success_hold_steps=_tol.hold_steps,
    tags=(SUITE,),
    start_deg=REST_START,
    return_to_start=True,
)

CUBE_TASKS = (STACK_YELLOW_ON_RED, GREEN_RIGHT_OF_RED)
