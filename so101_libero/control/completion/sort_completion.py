"""Completion of "Arrange the cubes from left to right as red, yellow, green." (left/right = the ROBOT's).

simulation ("sim"): the hidden ground truth (``SortedRow``): red, yellow and green each within 2 cm of their row
    targets (5 cm apart; the row moves with the layout), flat on the table, released, at rest, the TCP >= 4 cm from
    every cube; held 10 steps.

real robot ("behavior", default): three grasp -> release cycles, the jaws open, the TCP lifted >= 7 cm, the arm still
    for 0.4 s (the learned sorting behaviour has finished; the arrangement itself is not checked).
real robot ("vision"): the behaviour signals stop the policy; after the robot is back at HOME the front camera must see
    all three cubes (colour blobs back-projected onto the table with the calibrated camera) in the order red, yellow,
    green from the robot's left (+y), neighbours ``spacing`` apart (3.5-6.5 cm) and within ``max_dx`` of each other
    front-back, on every one of a few consecutive frames: a stable row in the right order. Logged as the vision
    verification of the episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_libero.control.completion.base import SimGoalDetector, VisionVerifier, make_behavior


@dataclass
class SortSimDetector(SimGoalDetector):
    name: str = "sort_sim_goal"

    def diagnostics(self, ws) -> dict:
        row = self.task.goal[0]
        return {"target_errors_cm": [round(100 * float(e), 2) for e in row.errors(ws)],
                "touching_robot": [bool(ws.touching_robot(o)) for o in row.objs]}


@dataclass
class RowInOrder(VisionVerifier):
    """Vision verification: red, yellow, green in a row from the robot's left, neighbours 3.5-6.5 cm apart."""

    spacing: tuple[float, float] = (0.035, 0.065)
    max_dx: float = 0.02
    name: str = "row_in_order"

    def check(self, obs: dict) -> tuple[bool, dict]:
        found = self.locator.locate(obs["observation.images.front"])
        missing = [c for c in ("red_cube", "yellow_cube", "green_cube") if c not in found]
        if missing:
            return False, {"missing": missing}
        r, y, g = (found[c]["xy"] for c in ("red_cube", "yellow_cube", "green_cube"))
        gaps = [float(r[1] - y[1]), float(y[1] - g[1])]  # left (+y) to right: both positive in the right order
        dxs = [abs(float(r[0] - y[0])), abs(float(y[0] - g[0]))]
        ok = all(self.spacing[0] <= gp <= self.spacing[1] for gp in gaps) and max(dxs) <= self.max_dx
        return ok, {"gaps_cm": [round(100 * v, 1) for v in gaps], "dx_cm": [round(100 * v, 1) for v in dxs],
                    "xy": {c: np.round(found[c]["xy"], 3).tolist() for c in found}}


def make(mode: str, task, confirm_steps: int, **kw):
    if mode == "sim":
        return SortSimDetector(confirm_steps=confirm_steps, task=task)
    if mode in ("behavior", "vision"):
        det = make_behavior(3, f"sort_{mode}", mode, confirm_steps, **kw)
        if mode == "vision":
            from so101_libero.control.completion.vision import ColorCubeLocator

            det.verifier = RowInOrder(locator=ColorCubeLocator())
        return det
    raise ValueError(f"sort: unknown detector mode {mode!r} (sim | behavior | vision)")
