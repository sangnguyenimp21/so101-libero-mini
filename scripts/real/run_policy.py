#!/usr/bin/env python
"""Run a trained LeRobot checkpoint on the real SO-101 follower arm, or in the simulator through the same code path.

Policy interface (the dataset's, identical in simulation and on the real arm):
  observation.state          float32 (6,)  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in
                                           degrees (LeRobot so101_follower, use_degrees=True) + gripper 0..100
  observation.images.front   uint8 (256, 256, 3) RGB: the CENTRE SQUARE of the 4:3 camera frame, resized to 256x256
  observation.images.wrist   uint8 (256, 256, 3) RGB, same crop
  action                     float32 (6,)  absolute joint targets, same units

Backends: --robot real (LeRobot SO101Follower + two OpenCV cameras) or --robot sim (so101_libero's MuJoCo env,
rendered at 480x640 and cropped exactly like the real frames). One control loop (run_episode) serves both.

Safety (both backends): every command is clipped to the simulator's joint limits and to --max-step-deg per arm joint
relative to the previous command (the first one relative to the measured pose); NaN outputs hold the last command;
LeRobot's max_relative_target caps the goal relative to the MEASURED position on the hardware side; Ctrl-C stops the
loop and leaves the arm where it is (torque stays on; the arm holds, nothing is released).

Examples (defaults = the user's rig as in scripts/real_act.sh: port /dev/tty.usbmodem5B610366661, front camera 3 =
iPhone Continuity Camera at 1280x720, wrist camera 0 at 640x480, calibration id my_follower)
  # dry run with the hardware connected: observe + infer, never send
  python scripts/real/run_policy.py --policy outputs/train/act_t1_200/checkpoints/last/pretrained_model \\
      --no-send --log-dir outputs/real/dry
  # the real thing, 1 episode of 20 s
  python scripts/real/run_policy.py --policy <ckpt> --task "Pick up the red cube and place it aside." \\
      --log-dir outputs/real/run1
  # the same loop in simulation (no hardware; renders 480x640 and crops like the real cameras)
  SO101_RENDER=osmesa bash scripts/env.sh python scripts/real/run_policy.py --robot sim --task-id red_cube_aside \\
      --policy <ckpt> --seconds 15 --log-dir outputs/real/sim_check
  # read the arm's pose (nothing is sent, torque untouched): measure --joint-offset values
  python scripts/real/run_policy.py --robot real --read-pose
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import so101_libero  # noqa: E402,F401  (offscreen-GL setup; must precede any mujoco import, sim backend only)
from so101_libero.sim.robot import HOME_DEG, JOINT_NAMES, MOTOR_NAMES  # noqa: E402  (numpy only, no mujoco)

IMAGE_SIZE = 256
CAMERAS = ("front", "wrist")
HOME = np.asarray(HOME_DEG, dtype=np.float64)  # [0, -60, 35, 85, 0, 0]: the dataset's start pose (policy units)

# Joint limits in policy units (degrees x5, gripper 0..100). Source: the simulator's action space,
#   SO101TabletopEnv("cylinder_to_tray", EnvConfig(render=False)).action_space   (computed 2026-10-17)
#   low  = [-109.9999, -100.0000, -96.8299, -94.9998, -157.2110, 0.0]
#   high = [ 109.9999,  100.0000,  90.0000,  94.9998,  162.7893, 100.0]
# i.e. the Menagerie so101.xml joint ranges, the elbow capped at 90 deg by so101_libero/sim/scene.py (beyond it the
# forearm hits the shoulder) and the gripper's -10..100 deg mapped onto 0..100. Rounded inwards to 0.1 deg.
JOINT_LIMITS = np.array([
    [-110.0, 110.0],   # shoulder_pan
    [-100.0, 100.0],   # shoulder_lift
    [-96.8, 90.0],     # elbow_flex
    [-95.0, 95.0],     # wrist_flex
    [-157.2, 162.7],   # wrist_roll
    [0.0, 100.0],      # gripper
], dtype=np.float64)


# ------------------------------------------------------------------------------------------ pure helpers
def center_square_resize(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """The policy image of a camera frame: the centre square of the frame resized to ``size`` x ``size``
    (cv2.INTER_AREA). A 480x640 frame -> rows 0:480, columns 80:560 -> 256x256. The simulator's front camera renders
    this square directly (its fovy is the vertical field of view), so a 480x640 sim render goes through the same
    function. A square frame of the right size is returned as a contiguous copy."""
    import cv2

    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 image, got shape {img.shape}")
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    crop = np.ascontiguousarray(img[top:top + s, left:left + s])
    if s == size:
        return crop
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def clamp_step(prev_cmd, target, max_step_deg: float | None, limits: np.ndarray = JOINT_LIMITS,
               max_gripper_step: float | None = None) -> np.ndarray:
    """The next command: ``target`` clipped to ``limits``, then limited to +-``max_step_deg`` per ARM joint relative
    to ``prev_cmd`` (and +-``max_gripper_step`` gripper units, if given; None / <= 0 = no cap). A ``prev_cmd`` outside
    the limits (e.g. the arm's folded rest pose) is brought back inside at that rate. Raises on non-finite input."""
    prev = np.asarray(prev_cmd, dtype=np.float64).reshape(6)
    tgt = np.asarray(target, dtype=np.float64).reshape(6)
    if not (np.all(np.isfinite(prev)) and np.all(np.isfinite(tgt))):
        raise ValueError(f"non-finite command: prev={prev}, target={tgt}")
    clipped = np.clip(tgt, limits[:, 0], limits[:, 1])
    cap = np.full(6, np.inf)
    if max_step_deg is not None and max_step_deg > 0:
        cap[:5] = max_step_deg
    if max_gripper_step is not None and max_gripper_step > 0:
        cap[5] = max_gripper_step
    delta = np.clip(clipped - prev, -cap, cap)
    return (prev + delta).astype(np.float32)


def parse_joint_offsets(specs: list[str] | None) -> np.ndarray:
    """``["shoulder_lift=3.5", "wrist_roll.pos=-2"]`` -> offsets (6,) in joint order (missing joints: 0)."""
    off = np.zeros(6)
    for spec in specs or []:
        name, sep, val = spec.partition("=")
        name = name.strip().removesuffix(".pos")
        if not sep or name not in JOINT_NAMES:
            raise ValueError(f"--joint-offset expects NAME=DEG with NAME in {JOINT_NAMES}, got {spec!r}")
        off[JOINT_NAMES.index(name)] = float(val)
    return off


def robot_to_policy(q, offsets) -> np.ndarray:
    """Measured robot values -> policy (dataset / simulator) frame: policy = real - offset."""
    return (np.asarray(q, dtype=np.float64) - offsets).astype(np.float32)


def policy_to_robot(q, offsets) -> np.ndarray:
    """Policy-frame command -> robot command: real = policy + offset."""
    return (np.asarray(q, dtype=np.float64) + offsets).astype(np.float32)


def min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def sleep_until(deadline: float) -> None:
    """Sleep (coarsely, then spin the last ~2 ms) until ``time.perf_counter() >= deadline``."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.003:
            time.sleep(remaining - 0.002)


class Pacer:
    """Fixed-rate schedule for a control loop: ``wait()`` sleeps until the next deadline (deadlines are spaced by
    ``period`` from the start, so sleep jitter does not accumulate). A step that ends past its deadline counts as an
    overrun and restarts the schedule from now: the loop never bursts to catch up (absolute joint targets sent
    faster than the training cadence would move the arm faster than it was trained)."""

    def __init__(self, period: float, enabled: bool = True):
        self.period, self.enabled, self.overruns = period, enabled, 0
        self.deadline = time.perf_counter() + period

    def wait(self) -> None:
        now = time.perf_counter()
        if now > self.deadline:
            self.overruns += 1
            self.deadline = now + self.period
            return
        if self.enabled:
            sleep_until(self.deadline)
        self.deadline += self.period


def camera_ref(s: str):
    """OpenCV index ("0") or device path ("/dev/video2") as OpenCVCameraConfig.index_or_path (int | Path)."""
    return int(s) if s.strip().lstrip("-").isdigit() else Path(s)


def parse_size(s: str) -> tuple[int, int]:
    """'1280x720' -> (1280, 720) = (width, height)."""
    try:
        w, h = (int(v) for v in s.lower().strip().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must be WIDTHxHEIGHT, e.g. 640x480, got {s!r}") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive, got {s!r}")
    return w, h


def default_device() -> str:
    """mps (Apple) if available, else cuda if available, else cpu."""
    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class CameraSetting:
    """One real camera: OpenCV index or device path, and the frame size it is opened with (width, height)."""

    index_or_path: int | Path
    width: int
    height: int


# ------------------------------------------------------------------------------------------ robot backends
@dataclass
class Observation:
    state: np.ndarray  # (6,) measured joint values in ROBOT units (degrees x5, gripper 0..100)
    images: dict[str, np.ndarray]  # camera name -> raw HxWx3 uint8 RGB frame (before crop/resize)


class RobotIO:
    """What the control loop needs from a robot. ``done`` / ``result()`` only mean something in simulation."""

    name = "robot"
    done = False

    def connect(self) -> None:
        pass

    def reset(self, episode: int) -> None:  # sim: new layout; real: the user resets the scene by hand
        pass

    def observe(self) -> Observation:
        raise NotImplementedError

    def command(self, q: np.ndarray) -> None:  # q (6,) absolute targets in ROBOT units
        raise NotImplementedError

    def result(self) -> dict:
        return {}

    def disconnect(self) -> None:
        pass


class SimRobotIO(RobotIO):
    """so101_libero's MuJoCo environment, rendered at 480x640 so the frames go through the real cameras' crop."""

    name = "sim"

    def __init__(self, env, seed: int = 0):
        self.env, self.seed = env, seed
        self._obs, self._info, self.done = None, {}, False

    def reset(self, episode: int = 0) -> None:
        self._obs, self._info = self.env.reset(seed=self.seed + episode)
        self.done = False

    def observe(self) -> Observation:
        if self._obs is None:
            self.reset(0)
        images = {}
        for cam in CAMERAS:
            img = self._obs.get(f"observation.images.{cam}")
            images[cam] = self.env.render_camera(cam) if img is None else img
        return Observation(np.asarray(self._obs["observation.state"], dtype=np.float32), images)

    def command(self, q: np.ndarray) -> None:
        self._obs, _, terminated, truncated, self._info = self.env.step(np.asarray(q, dtype=np.float32))
        self.done = bool(terminated or truncated)

    def result(self) -> dict:
        return {"success": bool(self._info.get("is_success", False)), "failure": self._info.get("failure"),
                "sim_steps": int(self._info.get("step", 0))}

    def disconnect(self) -> None:
        self.env.close()


class RealRobotIO(RobotIO):
    """LeRobot ``so101_follower`` (use_degrees=True) with two OpenCV cameras. LeRobot is imported lazily."""

    name = "real"

    def __init__(self, port: str, robot_id: str, cameras: dict[str, CameraSetting], fps: int,
                 max_relative_target: float | None, torque_off_at_exit: bool = False,
                 calibration_dir: str | Path | None = None):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        try:  # lerobot 0.4.x (verified against 0.4.2): module and calibration dir "so101_follower"
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # newer lerobot (the user's Mac): module and calibration dir "so_follower"
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cams = {name: OpenCVCameraConfig(index_or_path=c.index_or_path, fps=fps, width=c.width, height=c.height)
                for name, c in cameras.items()}  # color_mode defaults to RGB
        # ensure_safe_goal_position (lerobot/robots/utils.py) accepts a float or a dict, not an int.
        mrt = float(max_relative_target) if max_relative_target is not None and max_relative_target > 0 else None
        self.config = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True, max_relative_target=mrt,
                                          cameras=cams, disable_torque_on_disconnect=torque_off_at_exit,
                                          calibration_dir=Path(calibration_dir) if calibration_dir else None)
        self.robot = SO101Follower(self.config)  # loads <calibration_dir>/<id>.json if it exists
        self.cameras = list(cameras)
        self.calibration_file = Path(self.robot.calibration_fpath)

    def _require_calibration_file(self) -> None:
        """Without <calibration_dir>/<id>.json, connect(calibrate=True) would start an interactive calibration and
        a bus read could not be normalized: refuse early with the path LeRobot looked at."""
        fpath = self.calibration_file
        if not fpath.is_file():
            have = sorted(p.stem for p in fpath.parent.glob("*.json"))
            raise SystemExit(f"no calibration file for robot id {self.robot.id!r}: {fpath}\n"
                             f"  ids with a calibration file there: {have or 'none'}\n"
                             f"  calibrate with: lerobot-calibrate --robot.type={self.robot.name} "
                             f"--robot.port={self.config.port} --robot.id={self.robot.id}")

    def connect(self) -> None:
        self._require_calibration_file()
        print(f"calibration file: {self.calibration_file}")
        # Uses the saved calibration (asks for ENTER only if the motors hold different values), connects the
        # cameras and configures the motors (torque is OFF for a moment during configure: have the arm resting).
        self.robot.connect(calibrate=True)

    def read_pose_only(self) -> np.ndarray:
        """Measured joint values without configuring the motors or opening the cameras (torque untouched):
        opens the bus (ping only), one normalized read with the saved calibration, closes the port."""
        self._require_calibration_file()
        bus = self.robot.bus
        bus.connect()
        try:
            pos = bus.sync_read("Present_Position")
        finally:
            bus.disconnect(disable_torque=False)
        return np.array([pos[name] for name in JOINT_NAMES], dtype=np.float32)

    def observe(self) -> Observation:
        o = self.robot.get_observation()  # {'<motor>.pos': float, '<camera>': HxWx3 uint8 RGB}
        state = np.array([o[m] for m in MOTOR_NAMES], dtype=np.float32)
        return Observation(state, {cam: o[cam] for cam in self.cameras})

    def command(self, q: np.ndarray) -> None:
        self.robot.send_action({m: float(v) for m, v in zip(MOTOR_NAMES, np.asarray(q, dtype=np.float64))})

    def disconnect(self) -> None:
        try:
            self.robot.disconnect()
        except Exception as e:  # noqa: BLE001  (e.g. a camera that never connected)
            print(f"disconnect: {type(e).__name__}: {e}", file=sys.stderr)


# ------------------------------------------------------------------------------------------ control loops
@dataclass
class EpisodeResult:
    index: int
    steps: int
    seconds: float
    achieved_hz: float
    target_hz: float
    inference_ms_mean: float
    inference_ms_max: float
    overruns: int  # steps that ended after their deadline (took longer than the control period)
    clamped_steps: int  # steps where the sent command differs from the raw policy output (> 0.01)
    max_arm_step_deg: float  # largest change of an arm joint command between consecutive steps
    nonfinite_actions: int
    interrupted: bool
    sent: bool
    success: bool | None = None  # simulation only
    failure: str | None = None
    csv: str | None = None
    video: str | None = None


def go_home(io: RobotIO, *, fps: float, seconds: float, max_step_deg: float, offsets: np.ndarray,
            limits: np.ndarray = JOINT_LIMITS, home: np.ndarray = HOME, send: bool = True, pace: bool = True,
            settle_s: float = 0.5, max_gripper_step: float | None = None) -> np.ndarray:
    """Move smoothly (min-jerk, ``seconds`` long, at the control rate, every step clamped) from the measured pose to
    ``home`` (policy frame). Returns the list of commands sent (policy frame)."""
    period = 1.0 / fps
    start = robot_to_policy(io.observe().state, offsets).astype(np.float64)
    n = max(1, int(round(seconds * fps)))
    print(f"go-home: {np.round(start, 1).tolist()} -> {home.tolist()} in {seconds:.1f} s ({n} steps)"
          + ("" if send else "  [not sent: --no-send]"))
    prev, sent_cmds = start.copy(), []
    t0 = time.perf_counter()
    pacer = Pacer(period, pace)
    for k in range(1, n + 1):
        target = start + (home - start) * min_jerk(k / n)
        cmd = clamp_step(prev, target, max_step_deg, limits, max_gripper_step)
        if send:
            io.command(policy_to_robot(cmd, offsets))
        sent_cmds.append(cmd)
        prev = cmd.astype(np.float64)
        if io.done:
            break
        pacer.wait()
    if send and pace and settle_s > 0:
        time.sleep(settle_s)
    if send:
        reached = robot_to_policy(io.observe().state, offsets)
        err = np.abs(reached[:5] - home[:5])
        print(f"go-home done in {time.perf_counter() - t0:.1f} s: measured {np.round(reached, 1).tolist()}, "
              f"max arm error {err.max():.1f} deg")
        if err.max() > 5.0:
            print("WARNING: the arm did not reach the home pose (blocked joint, wrong calibration, or missing "
                  "--joint-offset); compare with the simulator's home pose before running the policy")
    return np.array(sent_cmds, dtype=np.float32)


def run_episode(io: RobotIO, runner, task: str, *, index: int = 0, fps: float = 30.0, seconds: float = 20.0,
                max_step_deg: float = 6.0, max_gripper_step: float | None = 10.0, limits: np.ndarray = JOINT_LIMITS,
                offsets: np.ndarray | None = None, send: bool = True, log_dir: str | Path | None = None,
                pace: bool = True, verbose: bool = True) -> EpisodeResult:
    """One episode of the policy loop (shared by the real and the simulated backend).

    Per step: observe -> crop/resize -> runner.act -> clamp (limits + max step relative to the last command) ->
    command -> sleep until the period ends. ``seconds <= 0`` runs until Ctrl-C (simulation: until the env ends).
    With ``log_dir``, writes ``<log_dir>/episode_<index>/states_actions.csv`` (t, step, 6 states as the policy saw
    them, 6 raw policy outputs, 6 commands in the policy frame = sent - offset, inference ms), ``policy_view.mp4``
    (front | wrist, 256x512, what the policy saw) and ``summary.json``.
    """
    offsets = np.zeros(6) if offsets is None else np.asarray(offsets, dtype=np.float64)
    period = 1.0 / fps
    n_steps = int(round(seconds * fps)) if seconds and seconds > 0 else None
    runner.reset()
    obs = io.observe()
    prev_cmd = robot_to_policy(obs.state, offsets).astype(np.float64)  # the first command is relative to the pose
    rows, frames, inference_ms = [], [], []
    clamped = nonfinite = 0
    max_arm_step = 0.0
    interrupted = False
    step = 0
    t_start = time.perf_counter()
    pacer = Pacer(period, pace)
    try:
        while n_steps is None or step < n_steps:
            t_loop = time.perf_counter()
            if step > 0:
                obs = io.observe()
            state = robot_to_policy(obs.state, offsets)
            front = center_square_resize(obs.images["front"])
            wrist = center_square_resize(obs.images["wrist"])
            policy_obs = {"observation.state": state, "observation.images.front": front,
                          "observation.images.wrist": wrist}
            t_inf = time.perf_counter()
            raw = np.asarray(runner.act(policy_obs, task), dtype=np.float32).reshape(6)
            dt_inf = (time.perf_counter() - t_inf) * 1e3
            inference_ms.append(dt_inf)
            if np.all(np.isfinite(raw)):
                cmd = clamp_step(prev_cmd, raw, max_step_deg, limits, max_gripper_step)
            else:
                nonfinite += 1
                cmd = prev_cmd.astype(np.float32)  # hold
                if verbose:
                    print(f"step {step}: non-finite policy output {raw}, holding the last command")
            if step == 0 and verbose and np.max(np.abs(raw[:5] - state[:5])) > 15.0:
                print(f"WARNING: first policy output {np.round(raw, 1).tolist()} is far from the measured pose "
                      f"{np.round(state, 1).tolist()}: check units / calibration / --joint-offset")
            if np.max(np.abs(cmd - raw)) > 0.01:
                clamped += 1
            max_arm_step = max(max_arm_step, float(np.max(np.abs(cmd[:5] - prev_cmd[:5]))))
            if send:
                io.command(policy_to_robot(cmd, offsets))
            prev_cmd = cmd.astype(np.float64)
            rows.append([round(t_loop - t_start, 4), step, *state.tolist(), *raw.tolist(), *cmd.tolist(),
                         round(dt_inf, 2)])
            if log_dir is not None:
                from so101_libero.utils.video import label

                frames.append(label(np.concatenate([front, wrist], axis=1), f"ep{index} step {step}"))
            step += 1
            if io.done:
                break
            pacer.wait()
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted at step {step}: the arm holds its last command")
    elapsed = time.perf_counter() - t_start
    sim = io.result()
    res = EpisodeResult(index=index, steps=step, seconds=round(elapsed, 3),
                        achieved_hz=round(step / elapsed, 2) if elapsed > 0 else 0.0, target_hz=fps,
                        inference_ms_mean=round(float(np.mean(inference_ms)), 2) if inference_ms else 0.0,
                        inference_ms_max=round(float(np.max(inference_ms)), 2) if inference_ms else 0.0,
                        overruns=pacer.overruns, clamped_steps=clamped, max_arm_step_deg=round(max_arm_step, 3),
                        nonfinite_actions=nonfinite, interrupted=interrupted, sent=send,
                        success=sim.get("success"), failure=sim.get("failure"))
    if log_dir is not None:
        ep_dir = Path(log_dir) / f"episode_{index:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        header = ["t", "step", *(f"state_{j}" for j in JOINT_NAMES), *(f"policy_{j}" for j in JOINT_NAMES),
                  *(f"cmd_{j}" for j in JOINT_NAMES), "inference_ms"]
        with open(ep_dir / "states_actions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        res.csv = str(ep_dir / "states_actions.csv")
        if frames:
            from so101_libero.utils.video import write_mp4

            res.video = str(write_mp4(frames, ep_dir / "policy_view.mp4", fps=fps))
        (ep_dir / "summary.json").write_text(json.dumps({**asdict(res), "task": task, "offsets": offsets.tolist(),
                                                         "backend": io.name}, indent=2) + "\n")
    if verbose:
        print(summary_line(res))
    return res


def summary_line(r: EpisodeResult) -> str:
    s = (f"episode {r.index}: {r.steps} steps in {r.seconds:.1f} s = {r.achieved_hz:.1f} Hz (target {r.target_hz:g}), "
         f"inference {r.inference_ms_mean:.1f} ms/step (max {r.inference_ms_max:.0f}), overruns {r.overruns}, "
         f"clamped {r.clamped_steps}, max arm step {r.max_arm_step_deg:.2f} deg")
    if r.nonfinite_actions:
        s += f", NON-FINITE outputs {r.nonfinite_actions}"
    if not r.sent:
        s += ", NOT SENT (--no-send)"
    if r.success is not None:
        s += f", success={r.success}" + (f" ({r.failure})" if r.failure else "")
    if r.interrupted:
        s += ", INTERRUPTED"
    return s


# ------------------------------------------------------------------------------------------ CLI
class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


EPILOG = """\
joint offsets (--joint-offset NAME=DEG, repeatable; NAME in shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper): real = policy + offset for commands, policy = real - offset for observations. They absorb the
difference between the real arm's calibration zero and the simulator's (sim zero = middle of each joint's LeRobot
calibration range, so a correct calibration needs offsets of only a few degrees). To measure them: with torque off,
put the real arm by hand in the simulator's home pose [0, -60, 35, 85, 0, 0] (base centred, upper arm leaning back,
gripper pointing down over the table in front of the robot; compare with outputs/sim_front_home.png or the sim
viewer), then run `--robot real --read-pose`: it prints the measured pose and measured - home per joint, which are the
offsets (ignore values below ~1 deg). Verify afterwards with `--no-send` (the first policy output should be close to
the measured pose) and with the real camera aligned (scripts/real/check_cameras.py).

safety: commands are clipped to the simulator's joint limits and limited to --max-step-deg per arm joint relative to
the previous command (the first relative to the measured pose: no jump at start); the expert that generated the
training data never exceeded 5.3 deg/step. LeRobot's --max-relative-target caps every goal relative to the MEASURED
position on top (hardware side). The arm holds its last command at the end of an episode and on Ctrl-C (torque is
not disabled; --torque-off-at-exit releases it when the script ends: support the arm first). Connect with the arm
resting: LeRobot switches torque off for a moment while configuring the motors.

alternative: `RUN=<run> CKPT=last bash scripts/real_act.sh` runs the same checkpoint through lerobot-rollout, with a
deploy copy made by scripts/deploy_policy.py that crops/resizes the frames inside the policy preprocessor (same
centre-square crop as here). The one difference: real_act.sh does not move to the home pose first, so place the arm
near [0, -60, 35, 85, 0] by hand (torque off) before starting it; run_policy.py does that itself with --go-home.
"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG, formatter_class=_Fmt)
    g = p.add_argument_group("policy")
    g.add_argument("--policy", required=True, metavar="DIR", help="checkpoint directory (.../pretrained_model)")
    g.add_argument("--task", default=None, help="language instruction given to the policy (ACT ignores it). "
                   "Default: real = 'Pick up the red cube and place it aside.', sim = the task's instruction")
    g.add_argument("--device", default=None, help="torch device for inference: cpu, cuda, mps (default: mps if "
                   "available, else cuda if available, else cpu)")
    g.add_argument("--n-action-steps", type=int, default=None, help="actions executed per inference (ACT: <= chunk "
                   "size; default: the checkpoint's, 50)")
    g.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEF",
                   help="ACT temporal ensembling coefficient (re-plans every step; e.g. 0.01)")
    g = p.add_argument_group("episode")
    g.add_argument("--fps", type=float, default=30.0, help="control rate (the dataset's)")
    g.add_argument("--seconds", type=float, default=20.0, help="episode length; 0 = until Ctrl-C (sim: until the "
                   "env ends)")
    g.add_argument("--episodes", type=int, default=1)
    g.add_argument("--go-home", action=argparse.BooleanOptionalAction, default=True,
                   help="move smoothly to the home pose [0, -60, 35, 85, 0, 0] before each episode")
    g.add_argument("--home-seconds", type=float, default=3.0, help="duration of the go-home move")
    g.add_argument("--prompt", action=argparse.BooleanOptionalAction, default=True,
                   help="real robot: wait for ENTER before each episode (after go-home)")
    g.add_argument("--log-dir", default=None, metavar="DIR", help="write per-episode states_actions.csv, "
                   "policy_view.mp4 (front|wrist 256x512 as the policy saw them), summary.json and results.json")
    g = p.add_argument_group("safety")
    g.add_argument("--max-step-deg", type=float, default=6.0, help="max change per step of each arm joint command "
                   "relative to the previous command (0 = off)")
    g.add_argument("--max-gripper-step", type=float, default=10.0, help="same for the gripper, in 0..100 units "
                   "(0 = off)")
    g.add_argument("--max-relative-target", type=float, default=5.0, help="LeRobot so101_follower "
                   "max_relative_target: cap of every goal relative to the measured position, all motors (0 = off)")
    g.add_argument("--joint-offset", action="append", default=[], metavar="NAME=DEG", help="see below")
    g = p.add_argument_group("backend")
    g.add_argument("--robot", choices=["real", "sim"], default="real")
    g.add_argument("--port", default="/dev/tty.usbmodem5B610366661", help="real: serial port of the follower arm")
    g.add_argument("--robot-id", default="my_follower", help="real: LeRobot calibration id: <id>.json in "
                   "~/.cache/huggingface/lerobot/calibration/robots/so101_follower/ (so_follower/ on newer LeRobot); "
                   "the file actually used is printed")
    g.add_argument("--calibration-dir", default=None, metavar="DIR", help="real: directory of the calibration file "
                   "(default: LeRobot's, see --robot-id)")
    g.add_argument("--front-camera", default="3", help="real: OpenCV index or /dev/video path (3 = the iPhone "
                   "Continuity Camera; check with `lerobot-find-cameras opencv`, indices move)")
    g.add_argument("--wrist-camera", default="0", help="real: OpenCV index or /dev/video path")
    g.add_argument("--front-size", type=parse_size, default="1280x720", metavar="WxH", help="real: frame size the "
                   "front camera is opened with; the policy sees its centre square")
    g.add_argument("--wrist-size", type=parse_size, default="640x480", metavar="WxH", help="real: same for the "
                   "wrist camera")
    g.add_argument("--no-send", action="store_true", help="real: observe and run the policy, print/log the commands, "
                   "never call send_action (dry run with the hardware connected)")
    g.add_argument("--torque-off-at-exit", action="store_true", help="real: disable torque when disconnecting "
                   "(default: the arm keeps holding its last command)")
    g.add_argument("--read-pose", action="store_true", help="real: print the measured pose and its difference to "
                   "the home pose, then exit (no command, torque untouched, cameras not opened)")
    g.add_argument("--task-id", default="red_cube_aside", help="sim: so101_libero task id")
    g.add_argument("--seed", type=int, default=0, help="sim: layout seed of episode 0 (episode k uses seed + k)")
    args = p.parse_args(argv)  # argparse also converts the string defaults of --front-size/--wrist-size
    if args.fps <= 0:
        p.error("--fps must be positive")
    try:
        args.offsets = parse_joint_offsets(args.joint_offset)
    except ValueError as e:
        p.error(str(e))
    return args


def make_io(args) -> RobotIO:
    if args.robot == "sim":
        from so101_libero.env import EnvConfig, SO101TabletopEnv  # imports mujoco

        n_home = int(round(args.home_seconds * args.fps)) if args.go_home else 0
        n_steps = int(round(args.seconds * args.fps)) if args.seconds > 0 else None
        max_steps = (n_steps + n_home + 10) if n_steps is not None else None  # go-home steps count as env steps
        env = SO101TabletopEnv(args.task_id, EnvConfig(image_size=(480, 640), max_steps=max_steps))
        return SimRobotIO(env, seed=args.seed)
    cams = {"front": CameraSetting(camera_ref(args.front_camera), *args.front_size),
            "wrist": CameraSetting(camera_ref(args.wrist_camera), *args.wrist_size)}
    return RealRobotIO(args.port, args.robot_id, cams, int(round(args.fps)), args.max_relative_target,
                       torque_off_at_exit=args.torque_off_at_exit, calibration_dir=args.calibration_dir)


def main(argv=None) -> int:
    args = parse_args(argv)
    offsets = args.offsets
    send = not (args.robot == "real" and args.no_send)
    log_dir = Path(args.log_dir) if args.log_dir else None
    if args.read_pose:
        if args.robot != "real":
            raise SystemExit("--read-pose is for --robot real")
        io = make_io(args)
        print(f"calibration file: {io.calibration_file}")
        pose = io.read_pose_only()
        print(f"measured pose (robot units): {np.round(pose, 2).tolist()}")
        print(f"home pose (policy units):    {HOME.tolist()}")
        print(f"measured - home:             {np.round(pose - HOME, 2).tolist()}  <- --joint-offset values if the "
              f"arm is physically in the home pose")
        print("  e.g. " + " ".join(f"--joint-offset {j}={v:.1f}" for j, v in zip(JOINT_NAMES, pose - HOME)
                                   if abs(v) >= 1.0 and j != "gripper"))
        return 0

    from so101_libero.policy.runner import PolicyRunner  # torch + lerobot, no mujoco

    device = args.device or default_device()
    runner = PolicyRunner(args.policy, device=device, n_action_steps=args.n_action_steps,
                          temporal_ensemble_coeff=args.temporal_ensemble)
    print("policy:", json.dumps(runner.describe()))
    io = make_io(args)
    task = args.task
    if task is None:
        task = io.env.task.language if args.robot == "sim" else "Pick up the red cube and place it aside."
    cams = ("" if args.robot == "sim" else f", cameras front {args.front_camera} {args.front_size[0]}x"
            f"{args.front_size[1]} / wrist {args.wrist_camera} {args.wrist_size[0]}x{args.wrist_size[1]}")
    print(f"backend: {io.name}{cams}, task: {task!r}, fps {args.fps:g}, {args.seconds:g} s x {args.episodes} "
          f"episode(s), max step {args.max_step_deg:g} deg, max_relative_target {args.max_relative_target:g}, "
          f"offsets {offsets.tolist()}" + ("" if send else ", NO-SEND dry run"))
    io.connect()
    results: list[EpisodeResult] = []
    status = 0
    try:
        for ep in range(args.episodes):
            io.reset(ep)
            if args.go_home:
                go_home(io, fps=args.fps, seconds=args.home_seconds, max_step_deg=args.max_step_deg,
                        offsets=offsets, send=send, max_gripper_step=args.max_gripper_step)
            if args.robot == "real" and args.prompt and sys.stdin.isatty():
                input(f"episode {ep}: set up the scene, then press ENTER to start the policy (Ctrl-C aborts) ")
            res = run_episode(io, runner, task, index=ep, fps=args.fps, seconds=args.seconds,
                              max_step_deg=args.max_step_deg, max_gripper_step=args.max_gripper_step,
                              offsets=offsets, send=send, log_dir=log_dir)
            results.append(res)
            if res.interrupted:
                break
    except KeyboardInterrupt:
        print("\ninterrupted: the arm holds its position")
    except Exception:  # noqa: BLE001  (hardware errors: report, hold, disconnect)
        traceback.print_exc()
        print("ERROR above: stopping; the arm holds its last command")
        status = 1
    finally:
        io.disconnect()
    if log_dir is not None and results:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "results.json").write_text(json.dumps({
            "args": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(args).items()},
            "policy": runner.describe(), "task": task, "episodes": [asdict(r) for r in results]}, indent=2) + "\n")
    if results and args.robot == "sim":
        n_ok = sum(1 for r in results if r.success)
        print(f"sim: {n_ok}/{len(results)} episode(s) successful")
    return status


if __name__ == "__main__":
    sys.exit(main())
