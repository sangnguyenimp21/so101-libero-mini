"""Completion of "Place the green cube centered on top of the red and yellow cubes."

simulation ("sim"): the hidden ground truth (``OnTopOfPair``): green's centre within 1 cm of the red/yellow midpoint,
    its bottom within 4 mm of their tops, touching both and not the table, flat; red and yellow each moved <= 1 cm and
    still on the table; everything released and at rest, the TCP >= 4 cm from green; held 10 steps.

real robot ("behavior", default): one grasp -> release cycle, the jaws open, the TCP lifted >= 7 cm, the arm still for
    0.4 s (the learned stacking behaviour has finished; the stack itself is not checked).
real robot ("vision"): the behaviour signals stop the policy; after the robot is back at HOME, on a few consecutive
    front images: red and yellow side by side (blob positions <= 5.5 cm apart) and green's blob where a cube centred on
    top of them projects (back-projected at stack height: within ``max_offset`` of the red/yellow midpoint) with at
    least ``min_area_ratio`` of an unhidden cube's area. One camera cannot measure height: a green cube standing on the
    table ~2 cm behind the pair projects to almost the same pixels, but is then mostly hidden behind the pair, which the
    area check rejects. Logged as the vision verification of the episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_libero.control.completion.base import SimGoalDetector, VisionVerifier, make_behavior


@dataclass
class PyramidSimDetector(SimGoalDetector):
    name: str = "pyramid_sim_goal"

    def diagnostics(self, ws) -> dict:
        pred = self.task.goal[0]
        mid, g = pred.midpoint(ws), ws.pos("green_cube")
        return {"green_offset_cm": round(100 * float(np.linalg.norm(g[:2] - mid[:2])), 2),
                "green_z_cm": round(100 * float(g[2]), 2),
                "touch_red_yellow_table": [bool(ws.touching("green_cube", "red_cube")),
                                           bool(ws.touching("green_cube", "yellow_cube")),
                                           bool(ws.touching("green_cube", "table"))],
                "base_shift_cm": [round(100 * ws.shift_xy(b), 2) for b in pred.bases]}


@dataclass
class GreenOnPair(VisionVerifier):
    """Vision verification of the pyramid (see the module docstring)."""

    max_offset: float = 0.015
    # Red and yellow touch (3.1 cm apart), but green on top hides their inner halves from the front camera: their blob
    # centroids move outwards and the measured distance is 4.3-4.5 cm in simulation (the midpoint does not move).
    max_pair_dist: float = 0.055
    min_area_ratio: float = 0.6
    name: str = "green_on_pair"

    def check(self, obs: dict) -> tuple[bool, dict]:
        img = obs["observation.images.front"]
        found = self.locator.locate(img)
        if "red_cube" not in found or "yellow_cube" not in found:
            return False, {"missing": [c for c in ("red_cube", "yellow_cube") if c not in found]}
        stacked = self.locator.locate(img, height=self.locator.stack_height).get("green_cube")
        if stacked is None:
            return False, {"missing": ["green_cube"]}
        r, y = found["red_cube"]["xy"], found["yellow_cube"]["xy"]
        mid = (r + y) / 2
        pair = float(np.linalg.norm(r - y))
        off = float(np.linalg.norm(stacked["xy"] - mid))
        area_ratio = stacked["area"] / max(1.0, self.locator.expected_area(stacked["xy"], self.locator.stack_height))
        ok = pair <= self.max_pair_dist and off <= self.max_offset and area_ratio >= self.min_area_ratio
        return ok, {"pair_cm": round(100 * pair, 1), "green_offset_cm": round(100 * off, 1),
                    "green_area_ratio": round(float(area_ratio), 2)}


def make(mode: str, task, confirm_steps: int, **kw):
    if mode == "sim":
        return PyramidSimDetector(confirm_steps=confirm_steps, task=task)
    if mode in ("behavior", "vision"):
        det = make_behavior(1, f"pyramid_{mode}", mode, confirm_steps, **kw)
        if mode == "vision":
            from so101_libero.control.completion.vision import ColorCubeLocator

            det.verifier = GreenOnPair(locator=ColorCubeLocator())
        return det
    raise ValueError(f"pyramid: unknown detector mode {mode!r} (sim | behavior | vision)")
