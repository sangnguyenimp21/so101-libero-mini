"""Tests of the three-cube suite and the episode controller (physics only, no rendering):
bash scripts/local_env.sh python -m pytest -q tests/test_three_cubes.py -p no:cacheprovider"""

import math

import mujoco
import numpy as np
import pytest

from so101_libero.control.completion import StepSignal, make_detector
from so101_libero.control.completion.base import GripperCycleTracker, QuiescenceTracker
from so101_libero.control.episode_controller import EpisodeController, EpisodeState
from so101_libero.control.episode_finisher import EpisodeFinisher, FinisherConfig
from so101_libero.control.home import min_jerk_path, parse_pose, resolve_home
from so101_libero.control.kinematics import fk, vertical_lift_path
from so101_libero.control.robots import JOINT_LIMITS
from so101_libero.tasks.registry import SEED_SLOTS, get_suite, seed_slot
from so101_libero.tasks.three_cubes import (FLIP_INSTRUCTION, PYRAMID_INSTRUCTION, SCENE, SORT_INSTRUCTION,
                                            THREE_CUBE_TASKS)

TASK_IDS = ("flip_red_cube", "sort_ryg", "build_pyramid")


@pytest.fixture(scope="module")
def envs():
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    out = {t: SO101TabletopEnv(t, EnvConfig(render=False, terminate_on_success=False, max_steps=2000)) for t in TASK_IDS}
    yield out
    for e in out.values():
        e.close()


# ------------------------------------------------------------------------------------------------ suite
def test_suite_tasks_instructions_and_layouts():
    tasks = get_suite("so101_three_cubes")
    assert [t.task_id for t in tasks] == list(TASK_IDS)
    assert [t.language for t in tasks] == [FLIP_INSTRUCTION, SORT_INSTRUCTION, PYRAMID_INSTRUCTION]
    assert FLIP_INSTRUCTION == "Flip the red cube and place it back at its original position."
    assert SORT_INSTRUCTION == "Arrange the cubes from left to right as red, yellow, green."
    assert PYRAMID_INSTRUCTION == "Place the green cube centered on top of the red and yellow cubes."
    flip, sort, pyr = tasks
    assert flip.layout.num_placements == pyr.layout.num_placements == 5 == SCENE.num_layouts
    assert sort.layout.num_placements == 25
    # every permutation appears in every layout (no permutation tied to one layout), exactly once
    combos = {(sort.layout.placement_info(k)["layout_id"], sort.layout.placement_info(k)["permutation"])
              for k in range(25)}
    assert combos == {(lay, p) for lay in range(5) for p in SCENE.sort_permutations}
    assert "RYG" not in SCENE.sort_permutations  # never starts sorted
    for t in tasks:  # all three cubes in every task and every placement; the arm starts at HOME
        assert [o.name for o in t.layout.objects] == ["red_cube", "yellow_cube", "green_cube"]
        assert all(set(p) == {"red_cube", "yellow_cube", "green_cube"} for p in t.layout.placements)
        assert t.start_deg == tuple(SCENE.home_deg) and not t.return_to_start


def test_seed_ranges_never_overlap():
    """sort_ryg's 25 placements use 25 x 10,000 training seeds: the two slots after it are reserved."""
    from so101_libero.cli.generate import PLACEMENT_SEED_STRIDE, TASK_SEED_STRIDE

    ranges = []
    for t in THREE_CUBE_TASKS:
        lo = seed_slot(t.task_id) * TASK_SEED_STRIDE
        ranges.append((lo, lo + t.layout.num_placements * PLACEMENT_SEED_STRIDE))
    ranges.sort()
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    nxt = len(SEED_SLOTS) * TASK_SEED_STRIDE  # the next registered task's first seed
    assert ranges[-1][1] <= nxt


def test_one_shared_scene_with_plain_cubes(envs):
    from so101_libero.cli.make_init_states import scene_hash

    hashes = {scene_hash(t) for t in THREE_CUBE_TASKS}
    assert len(hashes) == 1  # one compiled scene for the three tasks
    m = envs["flip_red_cube"].model
    for name, rgba in (("red_cube", SCENE.red_rgba), ("yellow_cube", SCENE.yellow_rgba),
                       ("green_cube", SCENE.green_rgba)):
        b = m.body(name).id
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        assert len(geoms) == 1  # a single plain box: no marker geoms
        g = geoms[0]
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and np.allclose(m.geom_size[g], 0.015)
        assert m.geom_matid[g] == -1 and np.allclose(m.geom_rgba[g], rgba)  # solid colour, no texture


def test_layouts_settle_in_place_and_flip_cube_faces_pan_axis(envs):
    for tid, env in envs.items():
        for k in range(env.task.layout.num_placements):
            env.reset(options={"placement_id": k})
            for name, (x, y, yaw) in env.task.layout.placements[k].items():
                p = env.world.pos(name)
                assert np.linalg.norm(p[:2] - (x, y)) < 1e-3 and abs(p[2] - 0.015) < 1.5e-3
                assert not env.world.touching_robot(name)
            assert not env.task.check(env.world)  # never solved at the start
    flip = envs["flip_red_cube"]
    for k in range(5):
        x, y, yaw = flip.task.layout.placements[k]["red_cube"]
        assert abs(yaw - math.atan2(y, x - 0.0388)) < 1e-9 and abs(yaw) < math.radians(7)


# ------------------------------------------------------------------------------------------------ predicates
def _place(env, name, xyz, quat=(1, 0, 0, 0)):
    adr = env.h.item_qpos_adr[name]
    env.data.qpos[adr:adr + 3] = xyz
    env.data.qpos[adr + 3:adr + 7] = quat
    env.data.qvel[env.h.item_dof_adr[name]:env.h.item_dof_adr[name] + 6] = 0


def _settle(env, n=300):
    for _ in range(n):
        mujoco.mj_step(env.model, env.data)


def test_flip_predicate_needs_a_real_turn(envs):
    env = envs["flip_red_cube"]
    pred = env.task.goal[0]
    env.reset(options={"placement_id": 0})
    p0 = env.world.pos("red_cube")
    env.world.max_lift["red_cube"] = 0.05  # pretend it was carried
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), math.pi)  # 180 deg spin about vertical: no flip
    _place(env, "red_cube", (p0[0], p0[1], 0.0152), q)
    _settle(env)
    assert math.degrees(env.world.rotation_from_start("red_cube")) > 170 and not pred(env.world)
    axis = np.array([1.0, 0.0, -1.0]) / math.sqrt(2)  # 180 deg about the 45 deg tilted approach axis = the flip
    mujoco.mju_axisAngle2Quat(q, axis, math.pi)
    _place(env, "red_cube", (p0[0], p0[1], 0.0152), q)
    _settle(env)
    assert pred(env.world)
    _place(env, "red_cube", (p0[0] + 0.03, p0[1], 0.0152), q)  # flipped but 3 cm away
    _settle(env)
    assert not pred(env.world)


def test_sort_predicate_targets_follow_the_layout(envs):
    env = envs["sort_ryg"]
    pred = env.task.goal[0]
    for k in (0, 7, 18):
        env.reset(options={"placement_id": k})
        targets = pred.target_xy(env.world)
        off = SCENE.offset(env.task.layout.placement_info(k)["layout_id"])
        assert np.allclose(targets, np.asarray(SCENE.sort_targets) + off, atol=1e-3)
        env.data.qpos[env.h.joint_qpos_adr[1]] = math.radians(-90)  # arm out of the way (TCP far from the cubes)
        for name, t in zip(pred.objs, targets):
            _place(env, name, (t[0], t[1], 0.0152))
        _settle(env)
        assert pred(env.world)
        # yellow and green swapped: wrong order
        _place(env, "yellow_cube", (targets[2][0], targets[2][1], 0.0152))
        _place(env, "green_cube", (targets[1][0], targets[1][1], 0.0152))
        _settle(env)
        assert not pred(env.world)


def test_pyramid_predicate(envs):
    env = envs["build_pyramid"]
    pred = env.task.goal[0]
    env.reset(options={"placement_id": 0})
    env.data.qpos[env.h.joint_qpos_adr[1]] = math.radians(-90)
    mid = pred.midpoint(env.world)
    _place(env, "green_cube", (mid[0], mid[1], 0.0452))
    _settle(env)
    assert pred(env.world)
    _place(env, "green_cube", (mid[0], mid[1] + 0.02, 0.0452))  # resting on red only
    _settle(env)
    assert not pred(env.world)


# ------------------------------------------------------------------------------------------------ geometry
def test_numpy_kinematics_matches_mujoco(envs):
    env = envs["flip_red_cube"]
    m, d = env.model, mujoco.MjData(env.model)
    rng = np.random.default_rng(0)
    h = env.h
    for _ in range(50):
        q = rng.uniform(h.joint_range[:5, 0], h.joint_range[:5, 1])
        d.qpos[h.joint_qpos_adr[:5]] = q
        mujoco.mj_kinematics(m, d)
        p, r = fk(np.degrees(q))
        assert np.linalg.norm(p - d.site_xpos[h.tcp_site]) < 1e-5
        assert np.abs(r - d.site_xmat[h.tcp_site].reshape(3, 3)).max() < 1e-6
    q0 = np.array([0.0, 10.0, 40.0, 60.0, 0.0])  # a low pose over the table
    path, gained = vertical_lift_path(q0, 0.05, 40)
    p0, r0 = fk(q0)
    p1, r1 = fk(path[-1])
    assert gained > 0.03 and np.linalg.norm(p1[:2] - p0[:2]) < 2e-3 and np.abs(r1 - r0).max() < 0.03


def test_numpy_camera_matches_mujoco(envs):
    from so101_libero.cli.render_placements import project as project_mj
    from so101_libero.sim.cameras import FRONT_CAMERA, back_project, project

    env = envs["flip_red_cube"]
    env.reset(options={"placement_id": 0})
    pts = np.array([[0.2, 0.05, 0.0], [0.24, -0.1, 0.03], [0.15, 0.12, 0.015]])
    assert np.abs(project(FRONT_CAMERA, pts, 256) - project_mj(env.model, env.data, "front", pts, 256)).max() < 1e-6
    uv = project(FRONT_CAMERA, pts[1:2], 256)[0]
    assert np.allclose(back_project(FRONT_CAMERA, uv, 256, 0.03), pts[1], atol=1e-9)


def test_home_pose_and_path():
    home = resolve_home()
    assert np.allclose(home, SCENE.home_deg)
    assert np.allclose(parse_pose("0,-60,35,85,0,10"), home)
    with pytest.raises(ValueError):
        parse_pose("1,2,3")
    path = min_jerk_path(np.zeros(6), home, 30.0, max_joint_speed=40.0)
    assert np.allclose(path[-1], home) and np.max(np.abs(np.diff(path[:, :5], axis=0))) < 3.0
    z = fk(home[:5])[0][2]
    assert 0.12 < z < 0.14  # the gripper 13 cm above the table


# ------------------------------------------------------------------------------------------------ detectors
def test_gripper_cycles_count_only_real_grasp_and_release():
    g = GripperCycleTracker()
    def run(seq):
        for cmd, meas in seq:
            g.update(cmd, meas)
    run([(38, 38)] * 5 + [(0, 0)] * 10 + [(38, 38)] * 5)  # empty close: nothing held
    assert g.cycles == 0
    run([(0, 18)] * 10 + [(15, 18)] * 3 + [(27, 27)] * 3)  # grasp a cube, then open freely
    assert g.cycles == 1 and g.state == "released"
    run([(0, 18)] * 10 + [(0, 1)] * 5 + [(27, 27)] * 3)  # grasp, the cube slips out, open: no cycle
    assert g.cycles == 1
    q = QuiescenceTracker(steps=3)
    assert not any(q.update(np.r_[np.zeros(5) + i, 20.0]) for i in range(5))
    assert [q.update(np.r_[np.full(5, 4.0), 20.0]) for _ in range(4)] == [False, False, True, True]


def test_behavior_detector_fires_after_last_release_and_hold(envs):
    """The real-robot rule on a full expert demonstration: fires after the ground truth, during the final hold."""
    from so101_libero.expert.rollout import expert_rng, make_expert

    env = envs["build_pyramid"]
    env.reset(seed=3, options={"placement_id": 2})
    ex = make_expert(env, env.task.plan, expert_rng(3))
    sim, beh = make_detector(env.task, "sim"), make_detector(env.task, "behavior")
    step = 0
    while (a := ex.act()) is not None:
        env.step(a)
        step += 1
        sig = StepSignal(step=step, t=step / 30, joints=env.joint_state().astype(float), command=np.asarray(a, float),
                         world=env.world)
        sim.update(sig)
        beh.update(sig)
    assert sim.confirmed_step is not None and beh.confirmed_step is not None
    assert sim.confirmed_step < beh.confirmed_step <= step


# ------------------------------------------------------------------------------------------------ controller
class FakeRobot:
    """A first-order arm: the measured joints follow the command one step later. No physics, no cameras."""

    name, is_sim, world, failure, fps = "fake", False, None, None, 30.0
    limits = JOINT_LIMITS

    def __init__(self, start):
        self.q = np.asarray(start, float)
        self.last_sent = self.q.astype(np.float32)
        self.sent = []

    def read_joints(self):
        return self.q.copy()

    def observe(self):
        img = np.full((256, 256, 3), 100, np.uint8)
        return {"observation.state": self.q.astype(np.float32), "observation.images.front": img,
                "observation.images.wrist": img}

    def send(self, target):
        self.last_sent = np.asarray(target, np.float32)
        self.sent.append(self.last_sent.copy())
        self.q = self.last_sent.astype(float)
        return self.last_sent

    def holding_object(self):
        return False

    def prepare_scene(self, episode):
        return {}

    def set_rendering(self, on):
        pass


class FakeChunkPolicy:
    """Chunks of 50 actions creeping away from HOME; counts calls; optionally raises / returns NaN at a step."""

    def __init__(self, nan_at=None, raise_at=None, interrupt_at=None):
        self.calls, self.queue, self.chunks = 0, [], 0
        self.nan_at, self.raise_at, self.interrupt_at = nan_at, raise_at, interrupt_at

    def reset(self):
        self.queue, self.calls, self.chunks = [], 0, 0

    def act(self, obs, instruction):
        if self.calls == self.raise_at:
            raise RuntimeError("model exploded")
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt
        if not self.queue:
            self.chunks += 1
            base = np.asarray(obs["observation.state"], float)
            self.queue = [base + np.r_[0.1 * (k + 1), np.zeros(5)] for k in range(50)]
        self.calls += 1
        a = self.queue.pop(0)
        return np.full(6, np.nan) if self.calls - 1 == self.nan_at else a

    def pending(self):
        return len(self.queue)

    def discard_pending(self):
        n, self.queue = len(self.queue), []
        return n


class StepDetector:
    """Confirms success at policy step ``at`` (None: never)."""

    name, mode, confirm_steps = "fake", "fake", 1

    def __init__(self, at):
        self.at, self.confirmed_step = at, None

    def reset(self, ctx=None):
        self.confirmed_step = None

    def update(self, s):
        if self.at is not None and s.step >= self.at and self.confirmed_step is None:
            self.confirmed_step = s.step
        return self.confirmed_step is not None

    def status(self):
        return {"confirmed_step": self.confirmed_step}


def _controller(policy, detector, max_steps=200, confirm=None):
    home = resolve_home()
    robot = FakeRobot(home)
    fin = EpisodeFinisher(robot, home, FinisherConfig(home_min_s=0.2, home_settle_s=0.1), log=lambda *a: None)
    ctl = EpisodeController(robot, policy, detector, fin, "test", max_steps, log=lambda *a: None, confirm_finish=confirm)
    return ctl, robot


def test_stop_immediately_on_success_and_discard_the_chunk():
    policy = FakeChunkPolicy()
    ctl, robot = _controller(policy, StepDetector(at=73))  # success in the middle of the second chunk
    rec = ctl.run({"index": 0})
    assert rec["task_success"] and rec["success_step"] == 73 and rec["termination"] == "success"
    assert policy.calls == 73  # not one policy call after success
    assert rec["discarded_actions"] == 100 - 73 == 27
    states = [s[0] for s in rec["states"]]
    assert states == ["INITIALIZE", "MOVE_TO_HOME", "PREPARE_SCENE", "RUN_POLICY", "SUCCESS", "STOP_POLICY",
                      "SAFE_RELEASE", "SAFE_RETREAT", "RETURN_HOME", "FINISHED"]
    assert rec["returned_home"] and rec["episode_complete"]
    assert np.allclose(robot.q[:5], resolve_home()[:5], atol=0.05)


def test_timeout_and_runtime_failures_end_at_home():
    for policy, kind in ((FakeChunkPolicy(), "timeout"), (FakeChunkPolicy(nan_at=12), "invalid action"),
                         (FakeChunkPolicy(raise_at=5), "policy exception")):
        ctl, robot = _controller(policy, StepDetector(at=None), max_steps=60)
        rec = ctl.run({"index": 0})
        assert not rec["task_success"] and kind in (rec["termination"] + rec["termination_reason"])
        assert rec["returned_home"] and rec["episode_complete"]
        assert all(np.all(np.isfinite(c)) for c in robot.sent)  # a NaN is never sent


def test_operator_interrupt_asks_before_moving():
    ctl, robot = _controller(FakeChunkPolicy(interrupt_at=20), StepDetector(at=None), confirm=lambda reason: False)
    rec = ctl.run({"index": 0})
    assert rec["termination"].startswith("aborted: operator interrupt") and rec.get("finisher_skipped")
    assert not rec["returned_home"] and not rec["episode_complete"] and len(robot.sent) == 20
    assert ctl.state is EpisodeState.FINISHED


def test_controller_with_expert_in_simulation():
    """The whole chain in the simulator: success detected, chunk discarded, nothing released twice, HOME, intact."""
    from so101_libero.cli.run_episodes import episode_specs, run_sim_episode
    from so101_libero.tasks.registry import get_task

    task = get_task("build_pyramid")
    spec = episode_specs(task, 1)[0]
    rec, robot, _ = run_sim_episode(task, "expert", spec, log=lambda *a: None)
    robot.disconnect()
    assert rec["task_success"] and rec["returned_home"] and rec["episode_complete"]
    assert rec["goal_holds_after_home"] and rec["discarded_actions"] > 0


def test_three_cube_env_wrapper():
    from so101_libero.env import EnvConfig
    from so101_libero.three_cube_env import SO101ThreeCubeEnv

    env = SO101ThreeCubeEnv("sort", EnvConfig(render=False))
    obs, _ = env.reset(options={"placement_id": 7})
    info = env.task_info()
    assert set(obs) == {"observation.state"}  # no privileged keys in the observation
    assert info["layout_name"] == "left" and info["permutation"] == "RGY" and len(info["targets"]) == 3
    with pytest.raises(ValueError):
        SO101ThreeCubeEnv("cube_flip")
    env.close()


def test_real_robot_backend_with_a_fake_bus():
    """RealRobot without hardware: offsets, step clamp, blank-camera detection, holding heuristic, motor-only reads."""
    from scripts.real.run_policy import Observation
    from so101_libero.control.robots import RealRobot, RobotError

    class FakeIO:
        def __init__(self):
            self.state = np.array([1.0, -60.0, 35.0, 85.0, 0.0, 30.0], np.float32)  # robot units
            self.frames = {c: np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
                           for c in ("front", "wrist")}
            self.commands = []

        def connect(self):
            pass

        def disconnect(self):
            pass

        def observe(self):
            return Observation(self.state.copy(), {k: v.copy() for k, v in self.frames.items()})

        def command(self, q):
            self.commands.append(np.asarray(q, np.float32))

    io = FakeIO()
    robot = RealRobot(io, offsets=np.array([1.0, 0, 0, 0, 0, 0]), pace=False, interactive=False)
    robot.connect()
    assert np.allclose(robot.read_joints(), [0, -60, 35, 85, 0, 30])  # policy = real - offset
    obs = robot.observe()
    assert obs["observation.images.front"].shape == (256, 256, 3) and obs["observation.state"].dtype == np.float32
    sent = robot.send(np.array([20.0, -60, 35, 85, 0, 30]))  # 20 deg jump: clamped to 6 deg / step
    assert np.isclose(sent[0], 6.0) and np.isclose(io.commands[-1][0], 7.0)  # robot command = policy + offset
    with pytest.raises(RobotError):
        robot.send(np.full(6, np.nan))
    robot.send(np.array([6.0, -60, 35, 85, 0, 20.0]))  # gripper commanded 20, measured stays 30 + offset 0
    io.state[5] = 30.0
    robot.read_joints()
    assert robot.holding_object() is None  # neither blocked-closed nor following
    for g in (10.0, 0.0, 0.0):
        robot.send(np.array([6.0, -60, 35, 85, 0, g]))
    io.state[5] = 18.0  # closed command, jaws stuck at 18: an object between them
    robot.read_joints()
    assert robot.holding_object() is True
    io.frames["wrist"][:] = 0
    with pytest.raises(RobotError):
        robot.observe()  # a blank camera frame is a camera failure


class FakeVerifier:
    name = "fake_vision"

    def __init__(self, ok):
        self.ok, self.started = ok, False

    def capture_start(self, obs):
        self.started = True

    def verify(self, frames):
        return self.ok, {"frames": len(frames)}


def test_vision_mode_success_requires_the_camera_check():
    for ok in (True, False):
        det = StepDetector(at=30)
        det.mode, det.verifier = "vision", FakeVerifier(ok)
        ctl, robot = _controller(FakeChunkPolicy(), det)
        rec = ctl.run({"index": 0})
        assert rec["completion_detected"] and rec["vision_verified"] is ok and det.verifier.started
        assert rec["task_success"] is ok  # a failed camera check is a failed episode
        assert rec["returned_home"] and rec["episode_complete"]


def test_behavior_mode_success_is_marked_unverified():
    det = StepDetector(at=30)
    det.mode = "behavior"
    ctl, _ = _controller(FakeChunkPolicy(), det)
    rec = ctl.run({"index": 0})
    assert rec["task_success"] and "not verified" in rec["success_basis"]


def test_interrupted_finisher_is_recorded_and_ends_the_session():
    """Ctrl-C (or a bus error) while parking: the arm holds, the record says so, nothing is complete."""
    ctl, robot = _controller(FakeChunkPolicy(), StepDetector(at=10))
    send = robot.send
    calls = {"n": 0}

    def send_then_interrupt(target):
        calls["n"] += 1
        if ctl.state is EpisodeState.RETURN_HOME and calls["n"] > 15:
            raise KeyboardInterrupt
        return send(target)

    robot.send = send_then_interrupt
    rec = ctl.run({"index": 0})
    assert rec["task_success"] and rec["finisher_interrupted"].startswith("RETURN_HOME")
    assert not rec["returned_home"] and not rec["episode_complete"]


def test_dry_run_runs_the_policy_and_never_moves():
    policy = FakeChunkPolicy()
    ctl, robot = _controller(policy, StepDetector(at=None), max_steps=40)
    robot.dry_run = True
    robot.q[:] = [0, -100, 88, 68, 0, 3]  # folded at rest, far from HOME: a dry run must still reach the policy
    rec = ctl.run({"index": 0})
    assert policy.calls == 40 and rec["dry_run"] and rec["finisher_skipped"] == "dry run"
    assert rec["move_to_home"]["success"] is None and not rec["episode_complete"]


def test_move_to_home_needs_confirmation_on_the_real_arm():
    home = resolve_home()
    robot = FakeRobot([0, -100, 88, 68, 0, 3])
    fin = EpisodeFinisher(robot, home, FinisherConfig(home_min_s=0.2, home_settle_s=0.1), log=lambda *a: None)
    policy = FakeChunkPolicy()
    ctl = EpisodeController(robot, policy, StepDetector(at=5), fin, "test", 50, log=lambda *a: None,
                            confirm_move=lambda reason: False)
    rec = ctl.run({"index": 0})
    assert rec["termination"].startswith("aborted: the operator declined") and robot.sent == [] and policy.calls == 0
