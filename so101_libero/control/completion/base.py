"""Task-completion detectors: decide, after every executed control step, whether the task is done.

The episode controller (so101_libero/control/episode_controller.py) stops the policy the moment a detector CONFIRMS
completion: the raw condition must hold ``confirm_steps`` consecutive control steps (one noisy frame never stops an
episode). Three kinds, one interface:

  SimGoalDetector       simulation ground truth: the task's goal predicates (so101_libero/tasks/base.py) on the
                        hidden simulator state. Used by simulated evaluation and inference only.
  BehaviorDetector      what a real robot can measure: the gripper went through the task's number of grasp -> release
                        cycles (an object blocked the closing jaws, then they opened freely), the jaws are open, the arm
                        has lifted the TCP above ``min_tcp_z`` (numpy FK of the measured joints) and stands still. It
                        detects that the LEARNED BEHAVIOUR finished, not that the result is right.
  Vision verifiers      (mode "vision") colour segmentation of the front image, back-projected onto the table with
                        the calibrated front camera (completion/vision.py). They do not decide when to stop: while the
                        policy holds still after the task, its open jaws hang over the result and hide part of it from
                        the front camera. The controller runs them after RETURN_HOME, when nothing in the workspace is
                        hidden, on several consecutive frames (``verify``), and logs the result next to the stop decision.

Nothing here ever feeds the policy: detectors only read the step signal after the action was executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepSignal:
    """Everything a detector may look at after one executed control step."""

    step: int  # policy steps executed so far (1 after the first action)
    t: float  # seconds since the policy started
    joints: np.ndarray  # measured joints after the step, policy units (deg x5, gripper 0..100)
    command: np.ndarray  # the command sent in this step, policy units
    obs: dict | None = None  # the latest policy observation (camera images), for vision checks
    world: object | None = None  # simulator WorldState (ground truth); None on the real robot


@dataclass
class CompletionDetector:
    """Base class: ``raw(signal) -> (bool, details)``; ``update`` counts consecutive raw successes."""

    confirm_steps: int = 10
    name: str = "detector"
    mode: str = "sim"
    streak: int = 0
    confirmed_step: int | None = None
    last: dict = field(default_factory=dict)

    def reset(self, context: dict | None = None) -> None:
        self.streak, self.confirmed_step, self.last = 0, None, {}

    def raw(self, s: StepSignal) -> tuple[bool, dict]:
        raise NotImplementedError

    def update(self, s: StepSignal) -> bool:
        """True from the step at which the raw condition has held ``confirm_steps`` consecutive steps."""
        ok, details = self.raw(s)
        self.streak = self.streak + 1 if ok else 0
        self.last = {"raw": bool(ok), "streak": self.streak, **details}
        if self.streak >= self.confirm_steps and self.confirmed_step is None:
            self.confirmed_step = s.step
        return self.confirmed_step is not None

    def status(self) -> dict:
        return {"detector": self.name, "mode": self.mode, "confirm_steps": self.confirm_steps,
                "confirmed_step": self.confirmed_step, **self.last}


@dataclass
class SimGoalDetector(CompletionDetector):
    """The task's goal predicates on the simulator's hidden state (privileged; simulation only)."""

    task: object = None
    name: str = "sim_goal"
    mode: str = "sim"

    def raw(self, s: StepSignal) -> tuple[bool, dict]:
        if s.world is None:
            raise RuntimeError("the simulator ground-truth detector needs the simulator state (use mode 'behavior' on a "
                               "real robot)")
        return bool(self.task.check(s.world)), self.diagnostics(s.world)

    def diagnostics(self, ws) -> dict:  # task-specific detectors report the predicate's ingredients
        return {}


# ------------------------------------------------------------------------------------------- robot-state signals
@dataclass
class GripperCycleTracker:
    """Counts grasp -> release cycles from the gripper alone (policy units 0 = closed .. 100 = open).

    holding:  the command is closed (<= ``closed_cmd``) but the measured opening stays >= ``blocked_margin`` above it for
              ``min_hold_steps`` steps: an object is between the jaws (in simulation a 3 cm cube holds the measured
              value at ~18 while the command is 0; an empty close follows the command down to ~0).
    released: after holding, the command opens (>= ``open_cmd``) and the measured value follows it within
              ``follow_tol``: nothing blocks the jaws any more. A cycle counts only when both happened, so a missed grasp
              (empty close) or a dropped object (measured falls to the command) does not count."""

    closed_cmd: float = 10.0
    blocked_margin: float = 8.0
    min_hold_steps: int = 5
    open_cmd: float = 20.0
    follow_tol: float = 5.0
    cycles: int = 0
    state: str = "free"  # free | holding | released
    _blocked: int = 0

    def reset(self) -> None:
        self.cycles, self.state, self._blocked = 0, "free", 0

    def update(self, command_grip: float, measured_grip: float) -> None:
        blocked = command_grip <= self.closed_cmd and measured_grip - command_grip >= self.blocked_margin
        self._blocked = self._blocked + 1 if blocked else 0
        if self._blocked >= self.min_hold_steps:
            self.state = "holding"
        elif self.state == "holding" and command_grip >= self.open_cmd and abs(measured_grip - command_grip) <= self.follow_tol:
            self.state = "released"
            self.cycles += 1
        elif self.state == "holding" and not blocked and command_grip <= self.closed_cmd:
            self.state = "free"  # the object slipped out of closed jaws: no release, no cycle

    @property
    def open_and_free(self) -> bool:
        return self.state in ("free", "released")


@dataclass
class QuiescenceTracker:
    """The commanded arm stands still: every arm joint's command changes by < ``max_step_deg`` per step (and the gripper
    command by < ``max_grip_step``) for ``steps`` consecutive steps."""

    max_step_deg: float = 0.5
    max_grip_step: float = 1.0
    steps: int = 10
    count: int = 0
    _prev: np.ndarray | None = None

    def reset(self) -> None:
        self.count, self._prev = 0, None

    def update(self, command: np.ndarray) -> bool:
        cmd = np.asarray(command, dtype=float)
        if self._prev is not None:
            d = np.abs(cmd - self._prev)
            still = float(d[:5].max()) < self.max_step_deg and float(d[5]) < self.max_grip_step
            self.count = self.count + 1 if still else 0
        self._prev = cmd
        return self.count >= self.steps


@dataclass
class BehaviorDetector(CompletionDetector):
    """Completion of the learned behaviour from robot-state signals only (works on the real robot): at least
    ``required_cycles`` grasp -> release cycles, the jaws open and free, the TCP lifted to >= ``min_tcp_z`` above the
    table and the arm still (``QuiescenceTracker``). The demonstrations end exactly like that (release, retreat up,
    1 s hold). It cannot tell a correct result from a wrong one: it says the policy has finished acting. ``verifier``
    (mode "vision") checks the result afterwards from the HOME pose (see the module docstring)."""

    required_cycles: int = 1
    min_tcp_z: float = 0.07
    gripper: GripperCycleTracker = field(default_factory=GripperCycleTracker)
    quiet: QuiescenceTracker = field(default_factory=QuiescenceTracker)
    verifier: object | None = None  # VisionVerifier (mode "vision") or None
    name: str = "behavior"
    mode: str = "behavior"

    def reset(self, context: dict | None = None) -> None:
        super().reset(context)
        self.gripper.reset()
        self.quiet.reset()

    def raw(self, s: StepSignal) -> tuple[bool, dict]:
        from so101_libero.control.kinematics import fk

        self.gripper.update(float(s.command[5]), float(s.joints[5]))
        still = self.quiet.update(s.command)
        tcp_z = float(fk(s.joints[:5])[0][2])
        ok = (self.gripper.cycles >= self.required_cycles and self.gripper.open_and_free and still
              and tcp_z >= self.min_tcp_z)
        details = {"cycles": self.gripper.cycles, "gripper_state": self.gripper.state, "still_steps": self.quiet.count,
                   "tcp_z": round(tcp_z, 4)}
        return ok, details


@dataclass
class VisionVerifier:
    """Checks the final arrangement in front-camera images (completion/vision.py): ``capture_start`` sees the first
    frame of the episode, ``verify`` a few consecutive frames taken at HOME; every frame must pass (stable result)."""

    locator: object = None
    name: str = "vision"

    def capture_start(self, obs: dict) -> None:
        pass

    def check(self, obs: dict) -> tuple[bool, dict]:
        raise NotImplementedError

    def verify(self, observations: list[dict]) -> tuple[bool, dict]:
        results = [self.check(o) for o in observations]
        return all(ok for ok, _ in results), {"frames": len(results), "passed": sum(ok for ok, _ in results),
                                              **(results[-1][1] if results else {})}


def make_behavior(required_cycles: int, name: str, mode: str, confirm_steps: int, min_tcp_z: float = 0.07,
                  closed_cmd: float = 10.0, blocked_margin: float = 8.0, open_cmd: float = 20.0,
                  still_steps: int = 10, **_) -> BehaviorDetector:
    """A ``BehaviorDetector`` with its thresholds (the real-robot command-line flags of run_episodes map onto these)."""
    return BehaviorDetector(confirm_steps=confirm_steps, required_cycles=required_cycles, name=name, mode=mode,
                            min_tcp_z=min_tcp_z,
                            gripper=GripperCycleTracker(closed_cmd=closed_cmd, blocked_margin=blocked_margin,
                                                        open_cmd=open_cmd),
                            quiet=QuiescenceTracker(steps=still_steps))
