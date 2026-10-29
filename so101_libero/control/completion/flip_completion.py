"""Completion of "Flip the red cube and place it back at its original position."

simulation ("sim"): the hidden ground truth (``FlippedInPlace`` + ``Undisturbed``): the red cube's tracked body
    orientation turned by >= 150 deg (the flip turns it 180 deg about the tilted grasp axis), its start-top face no
    longer on top, flat, back within 2 cm, released, the TCP >= 4 cm away, yellow and green undisturbed; held 10 steps.

real robot ("behavior", default; "vision" adds a position check): the red cube is a plain red cube. Before and after
    a flip it looks the same, so NO camera-based check can tell whether it was turned, and none is attempted here. The
    detector confirms that the learned flip behaviour has run to its end: one grasp -> release cycle of the gripper
    (an object blocked the closing jaws, then they opened freely), the jaws open, the TCP lifted >= 7 cm, the arm still
    for 0.4 s. "vision" adds a verification after the robot is back at HOME: the red cube (colour blob in the front
    image, back-projected onto the table) back within ``max_shift`` of where it was when the policy started. Position
    only, never orientation: it is logged next to the stop decision, it does not make the flip verifiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from so101_libero.control.completion.base import SimGoalDetector, VisionVerifier, make_behavior


@dataclass
class FlipSimDetector(SimGoalDetector):
    name: str = "flip_sim_goal"

    def diagnostics(self, ws) -> dict:
        return {"rotation_deg": round(math.degrees(ws.rotation_from_start("red_cube")), 1),
                "top_turn_deg": round(math.degrees(ws.start_top_turn("red_cube")), 1),
                "shift_cm": round(100 * ws.shift_xy("red_cube"), 2),
                "lifted_cm": round(100 * ws.max_lift.get("red_cube", 0.0), 2),
                "touching_robot": bool(ws.touching_robot("red_cube"))}


@dataclass
class RedBackInPlace(VisionVerifier):
    """Vision verification of the flip, position only: the red cube's table position at HOME within ``max_shift`` of its
    position in the first frame of the episode."""

    max_shift: float = 0.025
    name: str = "red_back_in_place"
    _ref: object = field(default=None, repr=False)

    def capture_start(self, obs: dict) -> None:
        found = self.locator.locate(obs["observation.images.front"]).get("red_cube")
        self._ref = None if found is None else found["xy"]

    def check(self, obs: dict) -> tuple[bool, dict]:
        import numpy as np

        found = self.locator.locate(obs["observation.images.front"]).get("red_cube")
        if found is None or self._ref is None:
            return False, {"red": "not found" if found is None else "not found in the first frame"}
        shift = float(np.linalg.norm(found["xy"] - self._ref))
        return shift <= self.max_shift, {"red_shift_cm": round(100 * shift, 2)}


def make(mode: str, task, confirm_steps: int, **kw):
    if mode == "sim":
        return FlipSimDetector(confirm_steps=confirm_steps, task=task)
    if mode in ("behavior", "vision"):
        det = make_behavior(1, f"flip_{mode}", mode, confirm_steps, **kw)
        if mode == "vision":
            from so101_libero.control.completion.vision import ColorCubeLocator

            det.verifier = RedBackInPlace(locator=ColorCubeLocator())
        return det
    raise ValueError(f"flip: unknown detector mode {mode!r} (sim | behavior | vision)")
