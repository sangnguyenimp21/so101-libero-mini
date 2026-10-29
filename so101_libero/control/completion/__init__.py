"""Task-completion detectors for the three-cube tasks (see base.py): ``make_detector(task_id, mode)``.

Modes: "sim" (simulator ground truth; simulated evaluation and inference), "behavior" (robot-state signals only; the
real-robot default), "vision" (behavior + a front-camera verification of the result after returning HOME; sort and
pyramid check the arrangement, flip only the red cube's position).
"""

from __future__ import annotations

from so101_libero.control.completion.base import (BehaviorDetector, CompletionDetector, SimGoalDetector, StepSignal,
                                                  VisionVerifier)

MODES = ("sim", "behavior", "vision")


BEHAVIOR_CONFIRM_STEPS = 5  # on top of the 10 still steps the behaviour signals already require (0.5 s in all)


def make_detector(task, mode: str = "sim", confirm_steps: int | None = None, **kw) -> CompletionDetector:
    """Detector of ``task`` (a Task or task id) in ``mode``. ``confirm_steps`` (consecutive steps the raw condition must
    hold) defaults to the task's success hold (10) for "sim" and to BEHAVIOR_CONFIRM_STEPS otherwise."""
    from so101_libero.control.completion import flip_completion, pyramid_completion, sort_completion
    from so101_libero.tasks.registry import get_task

    task = get_task(task) if isinstance(task, str) else task
    default = task.success_hold_steps if mode == "sim" else BEHAVIOR_CONFIRM_STEPS
    n = int(confirm_steps if confirm_steps is not None else default)
    module = {"flip_red_cube": flip_completion, "sort_ryg": sort_completion,
              "build_pyramid": pyramid_completion}.get(task.task_id)
    if module is not None:
        return module.make(mode, task, n, **kw)
    if mode == "sim":  # any other task: its goal predicates
        return SimGoalDetector(confirm_steps=n, task=task)
    raise ValueError(f"{task.task_id}: no real-robot completion detector (only the three-cube tasks have one)")


__all__ = ["MODES", "BehaviorDetector", "CompletionDetector", "SimGoalDetector", "StepSignal", "VisionVerifier",
           "make_detector"]
