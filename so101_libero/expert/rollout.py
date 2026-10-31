"""Run one scripted-expert episode and judge it by the task's goal predicates (actual physics)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from so101_libero.env import SO101TabletopEnv
from so101_libero.expert.pick_place import ExpertConfig, PickPlaceExpert


@dataclass
class ExpertEpisode:
    seed: int
    task_id: str
    success: bool
    failure: str | None
    steps: int
    success_step: int | None
    init_state: dict
    phases: list[tuple[int, str]]
    max_ik_err: float
    wall_time: float
    grasp: dict = field(default_factory=dict)  # physical grasp-quality metrics (GraspMonitor)
    deliberate_miss: bool = False  # recovery data: the first grasp attempt was offset on purpose
    frames: list[tuple[dict, np.ndarray]] = field(default_factory=list)  # (observation, action) if recorded

    def summary(self) -> dict:
        return {"seed": self.seed, "task_id": self.task_id, "success": self.success, "failure": self.failure,
                "placement_id": self.init_state.get("placement_id"),
                "regrasps": sum(1 for _, p in self.phases if p.startswith("regrasp")),
                "deliberate_miss": self.deliberate_miss,
                "regrasp_frames": [t for t, p in self.phases if p.startswith("regrasp")],
                "steps": self.steps, "success_step": self.success_step, "max_ik_err_mm": round(self.max_ik_err * 1e3, 3),
                "wall_time_s": round(self.wall_time, 2), "grasp": self.grasp, "phases": self.phases,
                "init_state": self.init_state}


def make_expert(env: SO101TabletopEnv, plan, rng: np.random.Generator, cfg: ExpertConfig | None = None):
    """The scripted expert for a task's plan: the so101_tri experts for their plans, else the pick-and-place expert."""
    from so101_libero.expert.tri_tasks import make_tri_expert

    return make_tri_expert(env, plan, rng, cfg) or PickPlaceExpert(env, plan, rng, cfg)


class GraspMonitor:
    """While the object is carried (lift/transport/lower, and the wrist roll of cube_flip): its slip relative to the
    TCP frame, the deepest object-jaw penetration and the pinch force. Evidence that the grasp is physical and stable.
    With several grasps per episode the slip is measured per grasp (maximum over grasps)."""

    CARRY = ("lift", "transport", "lower")
    CARRY_EXTRA = ("roll",)  # so101_tri cube_flip: the 180 degree wrist roll while holding the cube

    def __init__(self, env: SO101TabletopEnv, obj: str):
        self.env, self.obj = env, obj
        self.ref = None  # object position in the TCP frame when the carry starts
        self.max_slip = 0.0
        self.max_pen = 0.0
        self.forces: list[float] = []
        self._f6 = np.zeros(6)

    def update(self, phase: str) -> None:
        if phase == "regrasp":  # a new grasp follows: measure the carry of the grasp that actually holds the object
            self.ref, self.max_slip, self.max_pen, self.forces = None, 0.0, 0.0, []
            return
        if phase not in self.CARRY and phase not in self.CARRY_EXTRA:
            if phase == "release":
                self.ref = None  # a later carry would be a new grasp: new reference
            return
        d, h, ws = self.env.data, self.env.h, self.env.world
        rot = d.site_xmat[h.tcp_site].reshape(3, 3)
        local = rot.T @ (ws.pos(self.obj) - ws.tcp_pos())
        if self.ref is None:
            self.ref = local
        self.max_slip = max(self.max_slip, float(np.linalg.norm(local - self.ref)))
        geoms = h.item_geoms[self.obj]
        jaw = h.fixed_pad_geoms | h.moving_jaw_geoms
        force = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in geoms and c.geom2 in jaw) or (c.geom2 in geoms and c.geom1 in jaw):
                self.max_pen = max(self.max_pen, -float(c.dist))
                mujoco.mj_contactForce(self.env.model, d, i, self._f6)
                force += abs(float(self._f6[0]))
        self.forces.append(force)

    def summary(self) -> dict:
        f = np.array(self.forces) if self.forces else np.zeros(1)
        return {"carry_steps": len(self.forces), "max_slip_mm": round(self.max_slip * 1e3, 2),
                "max_penetration_mm": round(self.max_pen * 1e3, 2), "mean_normal_force_n": round(float(f.mean()), 2),
                "min_normal_force_n": round(float(f.min()), 2)}


def expert_rng(seed: int) -> np.random.Generator:
    """The expert's own random stream (timing, heights, release point), independent of the layout stream."""
    return np.random.default_rng([seed, 7919])


def run_expert_episode(env: SO101TabletopEnv, seed: int, cfg: ExpertConfig | None = None, record: bool = False,
                       init_state: dict | None = None, placement_id: int | None = None) -> ExpertEpisode:
    """Reset with ``seed`` (or ``init_state``; ``placement_id`` selects a predefined placement), let the expert act
    until it finishes, then check the goal.

    ``env`` should have ``terminate_on_success=False`` so the expert completes its retreat. Recorded frames
    pair each observation with the action taken from it (LeRobot convention: obs_t, action_t).
    """
    t0 = time.time()
    task = env.task
    if task.plan is None:
        raise ValueError(f"task {task.task_id} has no expert plan")
    options = {"init_state": init_state} if init_state else {}
    if placement_id is not None:
        options["placement_id"] = placement_id
    obs, info = env.reset(seed=seed, options=options or None)
    init = env.init_state
    expert = make_expert(env, task.plan, expert_rng(seed), cfg)
    monitor = GraspMonitor(env, task.plan.obj)
    frames = []
    success_step = None
    failure = None
    steps = 0
    while True:
        action = expert.act()
        if action is None:
            failure = expert.failure
            break
        if record:
            frames.append((obs, action))
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        monitor.update(expert.phase)
        if info["is_success"] and success_step is None:
            success_step = steps
        if info.get("failure"):
            failure = info["failure"]
            break
        if terminated or truncated:
            failure = None if info["is_success"] else "timeout"
            break
    success = failure is None and info["is_success"]
    if failure is None and not success:
        failure = "goal_not_reached"
    return ExpertEpisode(seed=seed, task_id=task.task_id, success=bool(success), failure=failure, steps=steps,
                         success_step=success_step, init_state=init, phases=expert.log, max_ik_err=expert.max_ik_err,
                         wall_time=time.time() - t0, grasp=monitor.summary(), deliberate_miss=expert.deliberate_miss,
                         frames=frames)
