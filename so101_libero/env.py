"""Gymnasium environment: SO-101 on a tabletop, one task (instruction + goal) per environment instance.

Policy-facing interface (identical for data collection, training and evaluation, all in LeRobot units):
  observation["observation.state"]           float32 (6,)  measured joint positions (deg x5, gripper 0..100)
  observation["observation.images.front"]    uint8 (H, W, 3)  third-person camera
  observation["observation.images.wrist"]    uint8 (H, W, 3)  wrist camera
  action                                     float32 (6,)  absolute joint position targets, same units
Privileged simulator state (object poses, contacts) is available only through ``env.world`` and ``info``
for the scripted expert and success checks; it is never part of the observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from so101_libero.sim.robot import HOME_QPOS, lerobot_to_sim, sim_to_lerobot
from so101_libero.sim.scene import SceneHandles, build_scene, yaw_quat
from so101_libero.tasks.base import Task, WorldState
from so101_libero.tasks.registry import get_task

CAMERAS = {"front": "front", "wrist": "wrist_cam"}  # observation name -> MuJoCo camera


@dataclass
class EnvConfig:
    image_size: tuple[int, int] = (256, 256)  # (height, width)
    cameras: tuple[str, ...] = ("front", "wrist")
    render: bool = True  # False: physics only, observations have no images (expert debugging)
    control_hz: float = 30.0
    substeps: int = 16
    settle_time: float = 0.3  # s of physics after placing items, before the first observation
    home_noise: float = 0.0  # uniform noise (rad) added to the home pose of the 5 arm joints at reset
    terminate_on_success: bool = True
    max_steps: int | None = None  # default: the task's horizon
    # Appearance-only domain randomization (lights, colours, camera jitter, image noise; so101_libero/sim/randomize.py)
    # resampled at every reset from the reset seed (or options["visual_seed"]). Physics is untouched.
    visual_rand: bool = False


class SO101TabletopEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task: str | Task, config: EnvConfig | None = None, handles: SceneHandles | None = None):
        self.task = get_task(task) if isinstance(task, str) else task
        self.cfg = config or EnvConfig()
        self.metadata = {**self.metadata, "render_fps": self.cfg.control_hz}
        self.h = handles or build_scene(self.task.layout.scene_config(control_hz=self.cfg.control_hz,
                                                                      substeps=self.cfg.substeps))
        self.model = self.h.model
        self.data = mujoco.MjData(self.model)
        self.world = WorldState(self.h, self.data)
        self.max_steps = self.cfg.max_steps or self.task.max_steps
        self.dt = 1.0 / self.cfg.control_hz
        assert abs(self.model.opt.timestep * self.cfg.substeps - self.dt) < 1e-9, "timestep/substeps mismatch"

        ctrl_range = self.model.actuator_ctrlrange[self.h.actuator_ids]
        self._ctrl_lo, self._ctrl_hi = ctrl_range[:, 0], ctrl_range[:, 1]
        lo, hi = sim_to_lerobot(ctrl_range[:, 0]), sim_to_lerobot(ctrl_range[:, 1])
        lo[5], hi[5] = 0.0, 100.0  # nominal gripper range (the actuator ctrlrange is rounded; ctrl is clipped anyway)
        self.action_space = gym.spaces.Box(np.minimum(lo, hi), np.maximum(lo, hi), dtype=np.float32)
        H, W = self.cfg.image_size
        spaces = {"observation.state": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)}
        if self.cfg.render:
            for cam in self.cfg.cameras:
                spaces[f"observation.images.{cam}"] = gym.spaces.Box(0, 255, (H, W, 3), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)
        self._renderer = mujoco.Renderer(self.model, H, W) if self.cfg.render else None
        self.randomizer = None
        self.visual_sample: dict | None = None  # the appearance sample of the current episode (side-file metadata)
        self._aug_rng = np.random.default_rng(0)
        if self.cfg.visual_rand and self.cfg.render:
            from so101_libero.sim.randomize import VisualRandomizer

            self.randomizer = VisualRandomizer(self.model, self.h.cfg.cameras)

        self._step = 0
        self._streak = 0
        self.init_state: dict | None = None

    # ------------------------------------------------------------------------------------------- core
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = np.random.default_rng(seed)
        if "init_state" in options:  # fixed (evaluation) state: used as is
            self._apply_init(options["init_state"])
        else:
            # Resample (same deterministic stream) while the goal already holds at the start (e.g. the object
            # happens to lie left of the reference): such episodes would be trivial.
            for _ in range(100):
                q = HOME_QPOS.copy()
                q[:5] += rng.uniform(-self.cfg.home_noise, self.cfg.home_noise, 5)
                self._apply_init({"robot_qpos": q.tolist(), "items": self.task.layout.sample(rng)})
                if not self.task.check(self.world):
                    break
            else:
                raise RuntimeError(f"{self.task.task_id}: goal holds in 100 sampled initial states")
        if self.randomizer is not None:
            vseed = seed if seed is not None else options.get("visual_seed", 0)
            # data= : camera/light placement is cached in MjData (mj_camlight) and must be refreshed before frame 0
            self.visual_sample = self.randomizer.randomize(np.random.default_rng([int(vseed), 4242]), data=self.data)
            self._aug_rng = np.random.default_rng([int(vseed), 4243])
        self._step = 0
        self._streak = 0
        return self._obs(), self._info(False)

    def _apply_init(self, init: dict) -> None:
        """Set the robot pose and item poses, then let the items settle (physics) before the first observation."""
        self.init_state = {"robot_qpos": list(map(float, init["robot_qpos"])),
                           "items": {k: list(map(float, v)) for k, v in init["items"].items()}}
        d, h = self.data, self.h
        mujoco.mj_resetData(self.model, d)
        q = np.asarray(self.init_state["robot_qpos"])
        d.qpos[h.joint_qpos_adr] = q
        d.ctrl[h.actuator_ids] = q
        for name, (x, y, yaw) in self.init_state["items"].items():
            z = h.objects[name].half_height if name in h.objects else 0.0
            adr = h.item_qpos_adr[name]
            d.qpos[adr:adr + 3] = (x, y, z + 0.001)
            d.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
        mujoco.mj_forward(self.model, d)
        for _ in range(int(round(self.cfg.settle_time / self.model.opt.timestep))):
            mujoco.mj_step(self.model, d)
        if self._diverged():
            raise RuntimeError(f"simulation diverged while settling the initial state {self.init_state}")
        self.world.init_pos = {n: self.world.pos(n) for n in self.h.item_body}
        self.world.max_lift, self.world.held_xy = {}, {}
        self.world.track()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite action {a}")
        self.data.ctrl[self.h.actuator_ids] = np.clip(lerobot_to_sim(a), self._ctrl_lo, self._ctrl_hi)
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)
        # Note: after mj_step, body poses (images, success check) are one 2 ms substep behind qpos (observation.state).
        # Kept as is on purpose: recomputing them would change what the expert reads and so the generated datasets
        # (which are reproducible bit for bit from their seeds); 2 ms is < 0.3 deg of arm motion.
        self._step += 1
        self.world.track()
        failure = self._failure()
        if failure == "simulation diverged":
            self._streak = 0  # an auto-reset puts every item at the origin: never a success
        else:
            self._streak = self._streak + 1 if self.task.check(self.world) else 0
        success = self._streak >= self.task.success_hold_steps
        terminated = (success and self.cfg.terminate_on_success) or failure is not None
        truncated = self._step >= self.max_steps and not terminated
        info = self._info(success)
        info["failure"] = failure
        return self._obs(), float(success), terminated, truncated, info

    # ---------------------------------------------------------------------------------------- helpers
    def joint_state(self) -> np.ndarray:
        """Measured joint positions in LeRobot units (6,)."""
        return sim_to_lerobot(self.data.qpos[self.h.joint_qpos_adr])

    def render_camera(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("environment created with render=False")
        self._renderer.update_scene(self.data, camera=CAMERAS.get(camera, camera))
        img = self._renderer.render().copy()
        if self.randomizer is not None:
            img = self.randomizer.augment(img, self._aug_rng)
        return img

    def render(self):
        """Front and wrist views side by side (for videos)."""
        return np.concatenate([self.render_camera(c) for c in self.cfg.cameras], axis=1)

    def _obs(self) -> dict:
        obs = {"observation.state": self.joint_state()}
        if self._renderer is not None:
            for cam in self.cfg.cameras:
                obs[f"observation.images.{cam}"] = self.render_camera(cam)
        return obs

    def _info(self, success: bool) -> dict:
        return {"is_success": success, "task_id": self.task.task_id, "task": self.task.language,
                "step": self._step, "success_streak": self._streak}

    def _diverged(self) -> bool:
        """MuJoCo auto-resets the state when it becomes unstable (and counts a warning); catch both cases."""
        w = mujoco.mjtWarning
        if any(self.data.warning[k].number > 0 for k in (w.mjWARN_BADQPOS, w.mjWARN_BADQVEL, w.mjWARN_BADQACC)):
            return True
        return not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

    def _failure(self) -> str | None:
        if self._diverged():
            return "simulation diverged"
        for name in self.h.objects:
            p = self.world.pos(name)
            if p[2] < -0.05 or not (-0.22 < p[0] < 0.62 and -0.5 < p[1] < 0.5):
                return f"{name} fell off the table"
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
