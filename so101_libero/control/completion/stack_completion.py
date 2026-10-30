"""Completion of "Put the yellow cube on top of the red cube."

simulation ("sim"): the hidden ground truth (``StackedOn`` + ``Undisturbed``): yellow's centre within 1.2 cm of red's
    (horizontal), its bottom within 4 mm of red's top, both flat; red moved <= 1 cm and still on the table, green
    undisturbed; everything released and at rest, the TCP >= 4 cm from yellow; held 10 steps.

real robot ("behavior", default): one grasp -> release cycle, the jaws open, the TCP lifted >= 7 cm, the arm still for
    0.4 s (the learned stacking behaviour has finished; the stack itself is not checked).
real robot ("vision"): the behaviour signals stop the policy; after the robot is back at HOME, on a few consecutive
    front images: yellow's blob where a cube on top of red projects (back-projected at stack height: within
    ``max_offset`` of red's blob) with at least ``min_area_ratio`` of an unhidden cube's area. As for the pyramid, one
    camera cannot measure height: a yellow cube on the table right behind red projects to similar pixels, but is then
    mostly hidden behind red, which the area check rejects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_libero.control.completion.base import SimGoalDetector, VisionVerifier, make_behavior


@dataclass
class StackSimDetector(SimGoalDetector):
    name: str = "stack_sim_goal"

    def diagnostics(self, ws) -> dict:
        y, r = ws.pos("yellow_cube"), ws.pos("red_cube")
        return {"yellow_offset_cm": round(100 * float(np.linalg.norm(y[:2] - r[:2])), 2),
                "yellow_z_cm": round(100 * float(y[2]), 2),
                "touch_red_table": [bool(ws.touching("yellow_cube", "red_cube")),
                                    bool(ws.touching("yellow_cube", "table"))],
                "red_shift_cm": round(100 * ws.shift_xy("red_cube"), 2),
                "green_shift_cm": round(100 * ws.shift_xy("green_cube"), 2)}


@dataclass
class YellowOnRed(VisionVerifier):
    """Vision verification of the stack (see the module docstring)."""

    # Yellow on top hides red's top face, so red's blob is its front face only and its back-projected centroid moves
    # towards the camera: at HOME after the expert's 5 layouts the measured offset is 1.0-1.2 cm (area ratio 1.16-1.17);
    # the untouched start row gives ~9 cm.
    max_offset: float = 0.02
    min_area_ratio: float = 0.6
    name: str = "yellow_on_red"

    def check(self, obs: dict) -> tuple[bool, dict]:
        img = obs["observation.images.front"]
        red = self.locator.locate(img).get("red_cube")
        if red is None:
            return False, {"missing": ["red_cube"]}
        stacked = self.locator.locate(img, height=self.locator.stack_height).get("yellow_cube")
        if stacked is None:
            return False, {"missing": ["yellow_cube"]}
        off = float(np.linalg.norm(stacked["xy"] - red["xy"]))
        area_ratio = stacked["area"] / max(1.0, self.locator.expected_area(stacked["xy"], self.locator.stack_height))
        ok = off <= self.max_offset and area_ratio >= self.min_area_ratio
        return ok, {"yellow_offset_cm": round(100 * off, 1), "yellow_area_ratio": round(float(area_ratio), 2)}


def make(mode: str, task, confirm_steps: int, **kw):
    if mode == "sim":
        return StackSimDetector(confirm_steps=confirm_steps, task=task)
    if mode in ("behavior", "vision"):
        det = make_behavior(1, f"stack_{mode}", mode, confirm_steps, **kw)
        if mode == "vision":
            from so101_libero.control.completion.vision import ColorCubeLocator

            det.verifier = YellowOnRed(locator=ColorCubeLocator())
        return det
    raise ValueError(f"stack: unknown detector mode {mode!r} (sim | behavior | vision)")
