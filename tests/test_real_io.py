"""Tests of the real-robot deployment script (scripts/real/run_policy.py) without hardware:
bash scripts/env.sh python -m pytest -q tests/test_real_io.py -p no:cacheprovider"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

from scripts.real.check_cameras import compose_outputs
from scripts.real.run_policy import (HOME, JOINT_LIMITS, MOTOR_NAMES, CameraSetting, Observation, RealRobotIO, RobotIO,
                                     SimRobotIO, camera_ref, center_square_resize, clamp_step, default_device, go_home,
                                     parse_args, parse_joint_offsets, parse_size, policy_to_robot, robot_to_policy,
                                     run_episode)

cv2 = pytest.importorskip("cv2")


# ------------------------------------------------------------------------------------------- fakes
class FakeRobotIO(RobotIO):
    """Constant state and frames; records every command. Optionally 'follows' the commands (state = last command)."""

    name = "fake"

    def __init__(self, state, frame_hw=(480, 640), follow=False):
        self.state = np.asarray(state, dtype=np.float32)
        self.frames = {cam: np.full((*frame_hw, 3), v, np.uint8) for cam, v in (("front", 90), ("wrist", 160))}
        self.commands = []
        self.follow = follow

    def observe(self):
        return Observation(self.state.copy(), {k: v.copy() for k, v in self.frames.items()})

    def command(self, q):
        q = np.asarray(q, dtype=np.float32)
        assert q.shape == (6,) and np.all(np.isfinite(q))
        self.commands.append(q.copy())
        if self.follow:
            self.state = q.copy()


class FakeRunner:
    def __init__(self, action, interrupt_at=None):
        self.action = np.asarray(action, dtype=np.float32)
        self.interrupt_at = interrupt_at
        self.calls, self.resets = 0, 0

    def reset(self):
        self.resets += 1

    def act(self, obs, task):
        assert set(obs) == {"observation.state", "observation.images.front", "observation.images.wrist"}
        assert obs["observation.state"].dtype == np.float32 and obs["observation.state"].shape == (6,)
        for k in ("observation.images.front", "observation.images.wrist"):
            assert obs[k].shape == (256, 256, 3) and obs[k].dtype == np.uint8
        if self.interrupt_at is not None and self.calls == self.interrupt_at:
            raise KeyboardInterrupt
        self.calls += 1
        return self.action.copy()


# ------------------------------------------------------------------------------------------- tests
def test_center_square_resize_matches_manual_crop():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out = center_square_resize(frame)
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8 and out.flags.c_contiguous
    ref = cv2.resize(frame[:, 80:560], (256, 256), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, ref)
    # other aspect ratios / sizes: the centre square, then the same resize
    tall = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    assert np.array_equal(center_square_resize(tall), cv2.resize(tall[:, 280:1000], (256, 256), interpolation=cv2.INTER_AREA))
    portrait = rng.integers(0, 256, (640, 480, 3), dtype=np.uint8)
    assert np.array_equal(center_square_resize(portrait), cv2.resize(portrait[80:560, :], (256, 256), interpolation=cv2.INTER_AREA))
    square = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert np.array_equal(center_square_resize(square), square)  # already the policy size: unchanged
    with pytest.raises(ValueError):
        center_square_resize(np.zeros((480, 640), np.uint8))


def test_clamp_step_cap_and_limits():
    prev = np.zeros(6)
    # small in-range steps pass unchanged
    small = np.array([1.0, -2.0, 3.0, 0.5, -0.1, 7.0])
    assert np.allclose(clamp_step(prev, small, 6.0), small, atol=1e-6)
    # per-step cap on the arm joints, both directions; the gripper only by its own cap
    big = np.array([50.0, -50.0, 20.0, -6.0, 6.0, 100.0])
    out = clamp_step(prev, big, 6.0, max_gripper_step=None)
    assert np.allclose(out, [6.0, -6.0, 6.0, -6.0, 6.0, 100.0], atol=1e-6)
    out = clamp_step(prev, big, 6.0, max_gripper_step=10.0)
    assert out[5] == pytest.approx(10.0)
    # joint limits win over the raw target even when the step cap would allow more
    near = JOINT_LIMITS[:, 1] - 1.0
    out = clamp_step(near, JOINT_LIMITS[:, 1] + 30.0, 6.0)
    assert np.allclose(out, JOINT_LIMITS[:, 1], atol=1e-5)
    out = clamp_step(JOINT_LIMITS[:, 0] + 1.0, JOINT_LIMITS[:, 0] - 30.0, 6.0)
    assert np.allclose(out, JOINT_LIMITS[:, 0], atol=1e-5)
    # a previous command outside the limits (e.g. the folded rest pose) moves back inside by at most the cap
    outside = np.array([0.0, 0.0, 97.0, 0.0, 0.0, 50.0])
    out = clamp_step(outside, np.array([0.0, 0.0, 97.0, 0.0, 0.0, 50.0]), 6.0)
    assert out[2] == pytest.approx(91.0)
    assert clamp_step(np.array([0, 0, 91.0, 0, 0, 50]), outside, 6.0)[2] == pytest.approx(90.0)
    # no cap
    assert np.allclose(clamp_step(prev, big, 0), np.clip(big, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]))
    assert clamp_step(prev, small, 6.0).dtype == np.float32
    with pytest.raises(ValueError):
        clamp_step(prev, np.array([np.nan, 0, 0, 0, 0, 0]), 6.0)


def test_joint_offsets_roundtrip():
    off = parse_joint_offsets(["shoulder_lift=3.5", "wrist_roll.pos=-2", " elbow_flex = 1.25 "])
    assert np.allclose(off, [0.0, 3.5, 1.25, 0.0, -2.0, 0.0])
    assert np.all(parse_joint_offsets([]) == 0) and np.all(parse_joint_offsets(None) == 0)
    with pytest.raises(ValueError):
        parse_joint_offsets(["elbow=3"])
    with pytest.raises(ValueError):
        parse_joint_offsets(["shoulder_pan"])
    real = np.array([1.0, -56.5, 36.25, 85.0, -2.0, 30.0], np.float32)
    policy = robot_to_policy(real, off)
    assert np.allclose(policy, real - off) and policy.dtype == np.float32
    assert np.allclose(policy_to_robot(policy, off), real, atol=1e-5)
    assert np.allclose(robot_to_policy(policy_to_robot(HOME, off), off), HOME, atol=1e-5)


def test_run_episode_dry_run_with_logging(tmp_path):
    io = FakeRobotIO(HOME)
    action = HOME + np.array([30.0, -12.0, 0.0, 3.0, -20.0, 100.0])
    runner = FakeRunner(action)
    res = run_episode(io, runner, "task", index=2, fps=200.0, seconds=0.1, max_step_deg=6.0, max_gripper_step=10.0,
                      log_dir=tmp_path, verbose=False)
    assert runner.resets == 1 and runner.calls == 20 and res.steps == 20
    cmds = np.array(io.commands)
    assert cmds.shape == (20, 6)
    # first command relative to the measured pose, then at most max_step per arm joint / gripper step per step
    steps = np.abs(np.diff(np.vstack([HOME[None], cmds]), axis=0))
    assert np.all(steps[:, :5] <= 6.0 + 1e-4) and np.all(steps[:, 5] <= 10.0 + 1e-4)
    assert res.max_arm_step_deg == pytest.approx(6.0, abs=1e-3)
    # converges to the action: joint 0 needs 5 steps of 6 deg, the gripper 10 steps of 10 units
    assert np.allclose(cmds[0], HOME + [6, -6, 0, 3, -6, 10], atol=1e-4)
    assert np.allclose(cmds[-1], action, atol=1e-4)
    assert np.allclose(cmds[4], action[:5].tolist() + [50.0], atol=1e-4)
    # every command inside the limits, never NaN
    assert np.all(cmds >= JOINT_LIMITS[:, 0] - 1e-4) and np.all(cmds <= JOINT_LIMITS[:, 1] + 1e-4)
    assert res.clamped_steps == 9 and res.nonfinite_actions == 0 and not res.interrupted and res.sent
    assert res.success is None  # not a simulation
    # the loop was paced at the requested rate (200 Hz over 0.1 s; generous bounds for a loaded machine)
    assert 0.08 <= res.seconds <= 0.5 and res.achieved_hz > 30
    # CSV: header + 20 rows, 21 columns
    ep_dir = tmp_path / "episode_002"
    with open(ep_dir / "states_actions.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0][:3] == ["t", "step", "state_shoulder_pan"] and rows[0][-1] == "inference_ms"
    assert len(rows) == 21 and all(len(r) == 2 + 18 + 1 for r in rows)
    data = np.array(rows[1:], dtype=np.float64)
    assert np.allclose(data[:, 2:8], HOME) and np.allclose(data[:, 8:14], action) and np.allclose(data[:, 14:20], cmds, atol=1e-4)
    assert np.all(np.diff(data[:, 0]) > 0) and data[0, 1] == 0 and data[-1, 1] == 19
    # mp4: 20 frames of 256x512
    import av

    with av.open(str(ep_dir / "policy_view.mp4")) as container:
        frames = [fr for fr in container.decode(video=0)]
    assert len(frames) == 20 and (frames[0].height, frames[0].width) == (256, 512)
    summary = json.loads((ep_dir / "summary.json").read_text())
    assert summary["steps"] == 20 and summary["backend"] == "fake" and summary["task"] == "task"
    assert res.csv == str(ep_dir / "states_actions.csv") and res.video == str(ep_dir / "policy_view.mp4")


def test_run_episode_offsets_nan_and_interrupt(tmp_path):
    off = parse_joint_offsets(["shoulder_lift=5", "gripper=-3"])
    real_state = policy_to_robot(HOME, off)  # the arm is physically at the sim home pose
    io = FakeRobotIO(real_state)
    runner = FakeRunner(HOME + [2, 0, 0, 0, 0, 0])
    res = run_episode(io, runner, "t", fps=1000.0, seconds=0.003, offsets=off, pace=False, verbose=False)
    assert res.steps == 3
    # the policy saw HOME (real - offset), commands were sent as policy + offset
    assert np.allclose(io.commands[0], policy_to_robot(HOME + [2, 0, 0, 0, 0, 0], off), atol=1e-4)
    # NaN output: hold the previous command, count it; no-send: nothing sent
    io = FakeRobotIO(HOME)
    res = run_episode(io, FakeRunner([np.nan] * 6), "t", fps=1000.0, seconds=0.002, pace=False, verbose=False,
                      send=False)
    assert res.nonfinite_actions == 2 and io.commands == [] and not res.sent
    # Ctrl-C inside the policy: the loop stops, logs what it has, reports interrupted
    io = FakeRobotIO(HOME)
    res = run_episode(io, FakeRunner(HOME, interrupt_at=5), "t", fps=1000.0, seconds=1.0, pace=False, verbose=False,
                      log_dir=tmp_path)
    assert res.interrupted and res.steps == 5 and len(io.commands) == 5
    with open(tmp_path / "episode_000" / "states_actions.csv") as f:
        assert len(list(csv.reader(f))) == 6


def test_go_home_is_smooth_and_clamped():
    start = np.array([20.0, -100.0, 97.0, 30.0, 150.0, 80.0])  # a folded rest pose; the elbow beyond the sim limit
    io = FakeRobotIO(start, follow=True)
    cmds = go_home(io, fps=30.0, seconds=3.0, max_step_deg=6.0, offsets=np.zeros(6), pace=False, settle_s=0.0)
    assert cmds.shape == (90, 6) and len(io.commands) == 90
    steps = np.abs(np.diff(np.vstack([start[None], cmds]), axis=0))
    assert np.all(steps[:, :5] <= 6.0 + 1e-4)
    assert np.allclose(cmds[-1], HOME, atol=1e-4)
    assert np.all(cmds[1:, 2] <= 97.0) and cmds[-1, 2] == pytest.approx(35.0)
    # min-jerk: slow at both ends, fastest in the middle
    assert steps[1, 0] < steps[45, 0] and steps[-1, 0] < steps[45, 0]


def test_check_cameras_compose_outputs():
    rng = np.random.default_rng(1)
    front = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)  # iPhone front camera: centre square = cols 280:1000
    wrist = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    sim = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    out = compose_outputs(front, wrist, sim)
    assert set(out) == {"front", "wrist", "front_raw", "wrist_raw", "front_vs_sim", "front_sim_side_by_side"}
    assert out["front"].shape == (256, 256, 3) and np.array_equal(out["front"], center_square_resize(front))
    assert np.array_equal(out["front"], cv2.resize(front[:, 280:1000], (256, 256), interpolation=cv2.INTER_AREA))
    assert out["wrist"].shape == (256, 256, 3) and np.array_equal(out["wrist"], center_square_resize(wrist))
    assert out["front_sim_side_by_side"].shape == (256, 512, 3)
    assert np.array_equal(out["front_sim_side_by_side"][:, 256:], sim)
    expected = ((out["front"].astype(np.float32) + sim.astype(np.float32)) / 2).round().astype(np.uint8)
    assert np.array_equal(out["front_vs_sim"], expected)
    assert out["front_raw"].shape == front.shape and np.all(out["front_raw"][0, 280:1000] == (255, 255, 0))
    assert np.array_equal(out["front_raw"][:, :280], front[:, :280])  # outside the square untouched
    assert out["wrist_raw"].shape == wrist.shape and np.all(out["wrist_raw"][0, 80:560] == (255, 255, 0))
    assert set(compose_outputs(front, wrist)) == {"front", "wrist", "front_raw", "wrist_raw"}


def test_cli_defaults_and_size_parsing():
    assert parse_size("1280x720") == (1280, 720) and parse_size(" 640X480 ") == (640, 480)
    for bad in ("640", "640x", "0x480", "axb"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_size(bad)
    assert camera_ref("3") == 3 and str(camera_ref("/dev/video2")) == "/dev/video2"
    args = parse_args(["--policy", "x"])  # the user's rig (scripts/real_act.sh) is the default
    assert (args.port, args.robot_id, args.front_camera, args.wrist_camera) == ("/dev/tty.usbmodem5B610366661",
                                                                                "my_follower", "3", "0")
    assert args.front_size == (1280, 720) and args.wrist_size == (640, 480)
    assert (args.max_step_deg, args.max_gripper_step, args.max_relative_target) == (6.0, 10.0, 5.0)
    assert args.go_home and args.prompt and args.device is None and args.robot == "real"
    args = parse_args(["--policy", "x", "--front-size", "640x480", "--joint-offset", "wrist_roll=-2", "--no-go-home"])
    assert args.front_size == (640, 480) and np.allclose(args.offsets, [0, 0, 0, 0, -2, 0]) and not args.go_home
    assert default_device() in ("cpu", "cuda", "mps")


def test_real_robot_io_config(tmp_path, monkeypatch):
    """RealRobotIO builds LeRobot's SO101FollowerConfig / SO101Follower with per-camera sizes (no hardware: the
    Feetech SDK is stubbed when absent, the bus is never opened)."""
    pytest.importorskip("lerobot.motors.feetech")
    if importlib.util.find_spec("scservo_sdk") is None:
        scs = types.ModuleType("scservo_sdk")

        class _Handler:
            def __init__(self, *a, **k):
                pass

            def setPacketTimeout(self, *a, **k):  # noqa: N802  (SDK name)
                pass

        scs.PortHandler = scs.PacketHandler = scs.GroupSyncRead = scs.GroupSyncWrite = _Handler
        scs.COMM_SUCCESS = 0
        monkeypatch.setitem(sys.modules, "scservo_sdk", scs)
    cams = {"front": CameraSetting(3, 1280, 720), "wrist": CameraSetting(camera_ref("/dev/video0"), 640, 480)}
    io = RealRobotIO("/dev/tty.usbmodem5B610366661", "my_follower", cams, 30, 5.0, calibration_dir=tmp_path)
    cfg = io.config
    assert cfg.port == "/dev/tty.usbmodem5B610366661" and cfg.id == "my_follower" and cfg.use_degrees is True
    assert cfg.max_relative_target == 5.0 and isinstance(cfg.max_relative_target, float)
    assert cfg.disable_torque_on_disconnect is False  # the arm keeps holding after disconnect
    assert cfg.cameras["front"].index_or_path == 3 and (cfg.cameras["front"].width, cfg.cameras["front"].height) == (1280, 720)
    assert str(cfg.cameras["wrist"].index_or_path) == "/dev/video0"
    assert (cfg.cameras["wrist"].width, cfg.cameras["wrist"].height, cfg.cameras["wrist"].fps) == (640, 480, 30)
    assert all(c.color_mode.value == "rgb" for c in cfg.cameras.values())
    robot = io.robot
    assert list(robot.action_features) == list(MOTOR_NAMES)
    assert robot.observation_features["front"] == (720, 1280, 3) and robot.observation_features["wrist"] == (480, 640, 3)
    assert [m.norm_mode.value for m in robot.bus.motors.values()] == ["degrees"] * 5 + ["range_0_100"]
    # calibration file: <calibration_dir>/<id>.json; refused when missing (LeRobot would start calibrating)
    assert io.calibration_file == tmp_path / "my_follower.json"
    with pytest.raises(SystemExit, match="no calibration file"):
        io._require_calibration_file()
    io.calibration_file.write_text("{}")
    io._require_calibration_file()
    # max_relative_target 0 -> None (LeRobot's cap off)
    assert RealRobotIO("/dev/null", "my_follower", cams, 30, 0, calibration_dir=tmp_path).config.max_relative_target is None


def test_sim_backend_physics_only(tmp_path, monkeypatch):
    """The simulator backend without rendering: fake 480x640 frames, real physics, env flags and success reporting."""
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    env = SO101TabletopEnv("cylinder_to_tray", EnvConfig(render=False, image_size=(480, 640), max_steps=40))
    monkeypatch.setattr(env, "render_camera", lambda cam: np.full((480, 640, 3), 30, np.uint8))
    io = SimRobotIO(env, seed=3)
    io.reset(0)
    obs = io.observe()
    assert obs.state.shape == (6,) and obs.images["front"].shape == (480, 640, 3)
    assert np.allclose(obs.state[:5], HOME[:5], atol=1.0)
    target = HOME + np.array([15.0, 0, 0, 0, 0, 0])
    res = run_episode(io, FakeRunner(target), env.task.language, fps=30.0, seconds=1.0, max_step_deg=6.0,
                      pace=False, verbose=False, log_dir=tmp_path)
    assert res.steps == 30 and res.success is False and res.failure is None and not res.interrupted
    assert abs(io.observe().state[0] - 15.0) < 2.0  # the simulated arm followed the (clamped) commands
    assert (tmp_path / "episode_000" / "policy_view.mp4").exists()
    # the env's own end (max_steps) ends the loop early and is reported through io.done
    io.reset(1)
    res = run_episode(io, FakeRunner(HOME), env.task.language, fps=30.0, seconds=10.0, pace=False, verbose=False)
    assert res.steps == 40 and io.done and res.success is False
    env.close()


@pytest.mark.skipif(os.environ.get("MUJOCO_GL", "disable") in ("disable", ""), reason="needs offscreen rendering")
def test_sim_crop_matches_direct_render():
    """A 480x640 sim render through center_square_resize equals the 256x256 training render (same fovy)."""
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    big = SO101TabletopEnv("cylinder_to_tray", EnvConfig(image_size=(480, 640)))
    small = SO101TabletopEnv("cylinder_to_tray", EnvConfig(image_size=(256, 256)), handles=big.h)
    big.reset(seed=0)
    small.reset(options={"init_state": big.init_state})
    try:
        for cam in ("front", "wrist"):
            a = center_square_resize(big.render_camera(cam)).astype(np.float32)
            b = small.render_camera(cam).astype(np.float32)
            assert np.abs(a - b).mean() < 3.0, cam
    finally:
        big.close()
        small.close()
