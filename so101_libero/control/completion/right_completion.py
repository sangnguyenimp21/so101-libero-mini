"""Completion of "Move the green cube to the right of the red cube."

simulation ("sim"): the hidden ground truth (``RelativeTo`` + ``Released`` + ``Undisturbed``): green's centre >= 4 cm to
    the robot's right of red's and <= 2.5 cm in front of or behind it, both directly on the table; red moved <= 1 cm,
    yellow undisturbed; green released and at rest, the TCP >= 4 cm from it; held 10 steps.

real robot ("behavior", default): one grasp -> release cycle, the jaws open, the TCP lifted >= 7 cm, the arm still for
    0.4 s (the learned placing behaviour has finished; where green ended up is not checked).
real robot ("vision"): the behaviour signals stop the policy; after the robot is back at HOME, on a few consecutive
    front images: the green and red blobs (back-projected onto the table) with green at least ``min_beside`` to the
    robot's right of red (the left of the front image) and at most ``max_offset`` in front of or behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

from so101_libero.control.completion.base import SimGoalDetector, VisionVerifier, make_behavior


@dataclass
class RightSimDetector(SimGoalDetector):
    name: str = "green_right_sim_goal"

    def diagnostics(self, ws) -> dict:
        d = ws.pos("green_cube") - ws.pos("red_cube")
        return {"green_beside_cm": round(-100 * float(d[1]), 2), "green_offset_cm": round(100 * float(d[0]), 2),
                "green_z_cm": round(100 * float(ws.pos("green_cube")[2]), 2),
                "lifted_cm": round(100 * ws.max_lift.get("green_cube", 0.0), 2),
                "red_shift_cm": round(100 * ws.shift_xy("red_cube"), 2),
                "yellow_shift_cm": round(100 * ws.shift_xy("yellow_cube"), 2)}


@dataclass
class GreenRightOfRed(VisionVerifier):
    """Vision verification of the placement (see the module docstring); 5 mm looser than the simulator's rule for the
    locator's error."""

    min_beside: float = 0.035
    max_offset: float = 0.03
    name: str = "green_right_of_red"

    def check(self, obs: dict) -> tuple[bool, dict]:
        found = self.locator.locate(obs["observation.images.front"])
        missing = [c for c in ("green_cube", "red_cube") if c not in found]
        if missing:
            return False, {"missing": missing}
        d = found["green_cube"]["xy"] - found["red_cube"]["xy"]
        ok = -float(d[1]) >= self.min_beside and abs(float(d[0])) <= self.max_offset
        return ok, {"green_beside_cm": round(-100 * float(d[1]), 1), "green_offset_cm": round(100 * float(d[0]), 1)}


def make(mode: str, task, confirm_steps: int, **kw):
    if mode == "sim":
        return RightSimDetector(confirm_steps=confirm_steps, task=task)
    if mode in ("behavior", "vision"):
        det = make_behavior(1, f"green_right_{mode}", mode, confirm_steps, **kw)
        if mode == "vision":
            from so101_libero.control.completion.vision import ColorCubeLocator

            det.verifier = GreenRightOfRed(locator=ColorCubeLocator())
        return det
    raise ValueError(f"green_right: unknown detector mode {mode!r} (sim | behavior | vision)")
