"""Robot backends of the episode controller: the MuJoCo simulation and the real SO-101 follower behind one interface.

All joint values are policy units (LeRobot so101_follower, use_degrees=True: degrees for the 5 arm joints, gripper
0 = closed .. 100 = open). One ``send`` = one control step: the target is checked, clamped and commanded, and the call
returns when the step is over (simulation: 16 physics substeps; real arm: commands at least one 30 Hz period apart).

  read_joints()      measured joints
  observe()          the policy observation: observation.state + two 256x256 RGB images (front, wrist)
  send(target)       command one absolute joint target; returns what was actually sent (after clamping)
  last_sent          the last command (the finisher continues from it)
  holding_object()   True / False / None (unknown) whether an object is between the jaws
  prepare_scene(ep)  simulation: reset to the episode's start state (the arm at HOME); real: ask the operator
  world              simulation ground truth (WorldState) or None
"""

from __future__ import annotations

import sys
import time

import numpy as np


# The real-robot runner's joint limits (policy units: the simulator's action space) and its per-step clamp; one
# definition for both runners.
from scripts.real.run_policy import JOINT_LIMITS, clamp_step  # noqa: E402


class RobotError(RuntimeError):
    """A hardware or observation problem that ends the policy run (camera, communication, invalid data)."""


def clamp(prev: np.ndarray, target: np.ndarray, max_step_deg: float | None, max_gripper_step: float | None,
          limits: np.ndarray = JOINT_LIMITS) -> np.ndarray:
    """``target`` clipped to the joint limits, then to +-``max_step_deg`` per arm joint (and +-``max_gripper_step`` for
    the gripper) relative to ``prev`` (None / <= 0: no step cap)."""
    return clamp_step(prev, target, max_step_deg, limits, max_gripper_step)


class SimRobot:
    """The so101_libero MuJoCo environment, 256x256 images rendered directly (the policy's view, as in the datasets)."""

    name = "sim"
    is_sim = True

    def __init__(self, task, image_size: tuple[int, int] = (256, 256), max_step_deg: float | None = None,
                 max_gripper_step: float | None = None):
        from so101_libero.env import EnvConfig, SO101TabletopEnv

        self.env = SO101TabletopEnv(task, EnvConfig(image_size=image_size, terminate_on_success=False,
                                                    max_steps=10 ** 9))
        self.task = self.env.task
        self.fps = self.env.cfg.control_hz
        self.max_step_deg, self.max_gripper_step = max_step_deg, max_gripper_step
        lo, hi = self.env.action_space.low.astype(float), self.env.action_space.high.astype(float)
        self.limits = np.stack([lo, hi], axis=1)
        self._obs: dict | None = None
        self.info: dict = {}
        self.failure: str | None = None
        self.last_sent = None

    @property
    def world(self):
        return self.env.world

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        self.env.close()

    def prepare_scene(self, episode: dict) -> dict:
        """Reset to the episode's start: ``init_state`` (frozen evaluation state) or ``placement_id``; the arm starts at
        the task's start pose (HOME)."""
        options = {"init_state": episode["init_state"]} if episode.get("init_state") else {}
        if episode.get("placement_id") is not None and not options:
            options["placement_id"] = int(episode["placement_id"])
        self.env.render_observations = True
        self._obs, self.info = self.env.reset(seed=episode.get("seed"), options=options or None)
        self.failure = None
        self.last_sent = self.read_joints().astype(np.float32)
        return dict(self.info)

    def set_rendering(self, on: bool) -> None:
        """Camera images on/off for the following steps (off while the finisher moves the arm and no video records)."""
        self.env.render_observations = bool(on)

    def read_joints(self) -> np.ndarray:
        return self.env.joint_state().astype(np.float64)

    def observe(self) -> dict:
        if self._obs is None or "observation.images.front" not in self._obs:
            self.env.render_observations = True
            self._obs = self.env._obs()
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._obs.items()}

    def send(self, target) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(target)):
            raise RobotError(f"non-finite command {target}")
        prev = self.last_sent if self.last_sent is not None else self.read_joints()
        cmd = clamp(prev, target, self.max_step_deg, self.max_gripper_step, self.limits)
        self._obs, _, _, _, self.info = self.env.step(cmd)
        self.failure = self.info.get("failure")
        self.last_sent = cmd
        return cmd

    def held_objects(self) -> list[str]:
        return [o for o in self.env.h.objects if self.env.world.grasped(o)]

    def holding_object(self) -> bool | None:
        """True if a cube is pinched between both jaws, None if a single jaw touches one (the jaws may be closing on
        it), False otherwise."""
        if self.held_objects():
            return True
        ws = self.env.world
        return None if any(ws.contacts(o)[0] or ws.contacts(o)[1] for o in self.env.h.objects) else False

    def frame(self) -> np.ndarray:
        """Front | wrist (for videos)."""
        return self.env.render()


class RealRobot:
    """The real SO-101 follower through scripts/real/run_policy.py's I/O (LeRobot SO101Follower, use_degrees=True, two
    OpenCV cameras; the policy sees the centre square of each frame resized to 256x256, exactly as in simulation).

    Safety on every command: clipped to the joint limits and to ``max_step_deg`` per arm joint (``max_gripper_step``
    gripper units) relative to the previous command; LeRobot's ``max_relative_target`` additionally caps every goal
    relative to the MEASURED position on the hardware side. ``joint_offsets``: policy = real - offset (calibration)."""

    name = "real"
    is_sim = False
    world = None
    failure = None

    def __init__(self, io, fps: float = 30.0, offsets=None, max_step_deg: float = 6.0,
                 max_gripper_step: float | None = 10.0, send: bool = True, pace: bool = True,
                 interactive: bool | None = None, holding_margin: float = 8.0, closed_cmd: float = 10.0,
                 open_cmd: float = 20.0):
        self.io, self.fps = io, float(fps)
        self.offsets = np.zeros(6) if offsets is None else np.asarray(offsets, dtype=np.float64)
        self.max_step_deg, self.max_gripper_step = max_step_deg, max_gripper_step
        self.do_send, self.pace = bool(send), bool(pace)
        self.interactive = sys.stdin.isatty() if interactive is None else bool(interactive)
        self.holding_margin, self.closed_cmd, self.open_cmd = holding_margin, closed_cmd, open_cmd
        self.dry_run = not self.do_send  # nothing is sent: the controller observes and infers, never moves
        self._last_cmd_t: float | None = None  # when the last command went out (pacing)
        self.last_sent = None
        self._last_obs = None
        self._last_joints = None
        self.limits = JOINT_LIMITS

    def connect(self) -> None:
        self.io.connect()
        self.last_sent = self.read_joints().astype(np.float32)

    def disconnect(self) -> None:
        self.io.disconnect()

    def _raw(self):
        try:
            return self.io.observe()
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001  (camera timeout, serial error, ...)
            raise RobotError(f"observation failed ({type(e).__name__}: {e})") from e

    def read_joints(self) -> np.ndarray:
        """Motor positions only (no cameras: the finisher can still park the arm after a camera failure)."""
        from scripts.real.run_policy import robot_to_policy
        from so101_libero.sim.robot import JOINT_NAMES

        bus = getattr(getattr(self.io, "robot", None), "bus", None)
        try:
            if bus is not None:
                pos = bus.sync_read("Present_Position")
                state = np.array([pos[j] for j in JOINT_NAMES], dtype=np.float64)
            else:  # a robot I/O without a motor bus (tests): its full observation
                o = self._raw()
                self._last_obs = o
                state = np.asarray(o.state, dtype=np.float64)
        except KeyboardInterrupt:
            raise
        except RobotError:
            raise
        except Exception as e:  # noqa: BLE001  (serial error)
            raise RobotError(f"joint read failed ({type(e).__name__}: {e})") from e
        q = robot_to_policy(state, self.offsets).astype(np.float64)
        if not np.all(np.isfinite(q)):
            raise RobotError(f"invalid joint reading {q}")
        self._last_joints = q
        return q

    def observe(self) -> dict:
        from scripts.real.run_policy import center_square_resize, robot_to_policy

        o = self._raw()
        self._last_obs = o
        state = robot_to_policy(o.state, self.offsets)
        if not np.all(np.isfinite(state)):
            raise RobotError(f"invalid joint reading {state}")
        self._last_joints = np.asarray(state, dtype=np.float64)
        obs = {"observation.state": state.astype(np.float32)}
        for cam in ("front", "wrist"):
            img = o.images.get(cam)
            if img is None:
                raise RobotError(f"camera '{cam}' returned no frame")
            img = center_square_resize(img)
            if float(np.asarray(img, dtype=np.float32).std()) < 1.0:  # black / frozen-constant frame
                raise RobotError(f"camera '{cam}' frame is blank (std < 1): camera failure?")
            obs[f"observation.images.{cam}"] = img
        return obs

    def send(self, target) -> np.ndarray:
        from scripts.real.run_policy import policy_to_robot

        target = np.asarray(target, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(target)):
            raise RobotError(f"non-finite command {target}")
        prev = self.last_sent if self.last_sent is not None else self.read_joints()
        cmd = clamp(prev, target, self.max_step_deg, self.max_gripper_step, self.limits)
        if self.pace and self._last_cmd_t is not None:
            # At least one control period between two commands, never a burst: after a slow step (a chunk inference)
            # the next target still waits a full period, so the arm is never driven faster than in the data.
            from scripts.real.run_policy import sleep_until

            sleep_until(self._last_cmd_t + 1.0 / self.fps)
        if self.do_send:
            try:
                self.io.command(policy_to_robot(cmd, self.offsets))
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                raise RobotError(f"command failed ({type(e).__name__}: {e})") from e
        self._last_cmd_t = time.perf_counter()
        self.last_sent = cmd
        return cmd

    def holding_object(self) -> bool | None:
        """From the gripper alone: closed command but the measured opening blocked well above it = holding; the jaws
        following an open command = not holding; anything else unknown."""
        if self.last_sent is None or self._last_joints is None:
            return None
        cmd = float(self.last_sent[5])
        meas = float(self._last_joints[5])
        if cmd <= self.closed_cmd and meas - cmd >= self.holding_margin:
            return True
        if cmd >= self.open_cmd and abs(meas - cmd) < self.holding_margin / 2:
            return False  # commanded open and the jaws follow freely
        return None

    def prepare_scene(self, episode: dict) -> dict:
        """The operator sets the cubes up (the arm stands at HOME); ENTER starts the policy."""
        setup = episode.get("setup_text")
        if setup:
            print(setup, flush=True)
        if self.interactive:
            input(f"episode {episode.get('index', 0)}: set up the scene, then press ENTER to start the policy "
                  "(Ctrl-C aborts) ")
        self._last_cmd_t = None  # fresh schedule after the pause
        return {}

    def set_rendering(self, on: bool) -> None:
        pass

    def frame(self) -> np.ndarray | None:
        if self._last_obs is None:
            return None
        from scripts.real.run_policy import center_square_resize

        return np.concatenate([center_square_resize(self._last_obs.images[c]) for c in ("front", "wrist")], axis=1)
