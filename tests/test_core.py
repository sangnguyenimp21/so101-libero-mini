"""Fast physics-only tests:  bash scripts/env.sh python -m pytest -q tests"""

import math

import mujoco
import numpy as np
import pytest

from so101_libero.expert.ik import IKSolver, top_down_frame
from so101_libero.sim.robot import GRIPPER_RANGE_RAD, lerobot_to_sim, sim_to_lerobot
from so101_libero.tasks.base import InContainer
from so101_libero.tasks.registry import TABLETOP, TASKS, get_suite


def test_unit_conversion_roundtrip():
    q = np.array([0.3, -1.0, 0.5, 1.2, -2.0, 0.4])
    assert np.allclose(lerobot_to_sim(sim_to_lerobot(q)), q, atol=1e-5)
    g = sim_to_lerobot(np.array([0, 0, 0, 0, 0, GRIPPER_RANGE_RAD[0]]))[5], \
        sim_to_lerobot(np.array([0, 0, 0, 0, 0, GRIPPER_RANGE_RAD[1]]))[5]
    assert g[0] == pytest.approx(0.0, abs=1e-4) and g[1] == pytest.approx(100.0, abs=1e-3)
    assert sim_to_lerobot(np.radians([10, 20, 30, 40, 50, 0]))[:5] == pytest.approx([10, 20, 30, 40, 50], abs=1e-4)


def test_scene_invariants(env_t1):
    m = env_t1.model
    assert m.opt.timestep * env_t1.cfg.substeps == pytest.approx(1 / 30)
    assert m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    cyl = m.geom("red_cylinder_geom")
    assert np.allclose(cyl.size[:2], [0.015, 0.015])  # 3 cm diameter x 3 cm tall
    assert np.allclose(m.geom("blue_cube_geom").size, [0.015] * 3)  # 3 cm cube
    assert m.actuator("gripper").forcerange[1] == pytest.approx(1.2)


def test_ik_top_down_accuracy(env_t1):
    ik = IKSolver(env_t1.h)
    rng = np.random.default_rng(1)
    for _ in range(25):
        r, a = rng.uniform(0.12, 0.22), rng.uniform(-1.2, 1.2)
        tgt = np.array([ik.pan_axis_xy[0] + r * math.cos(a), r * math.sin(a), rng.uniform(0.017, 0.07)])
        res = ik.solve_multi(tgt, top_down_frame(a), [np.r_[a, -0.5, 0.7, 1.2, 0.0], np.r_[a, 0, 0, 1.5, 0]])
        assert res.success
        pos, rot = ik.fk(res.q)
        assert np.linalg.norm(pos - tgt) < 1e-3
        assert rot[2, 2] > math.cos(math.radians(1.0))  # approach within 1 deg of vertical


def test_layout_sampling_is_collision_free_and_in_regions():
    rng = np.random.default_rng(0)
    items = {i.name: i for i in (*TABLETOP.objects, *TABLETOP.containers)}
    for _ in range(200):
        poses = TABLETOP.sample(rng)
        for name, (x, y, yaw) in poses.items():
            reg = TABLETOP.regions[name]
            assert reg.x[0] <= x <= reg.x[1] and reg.y[0] <= y <= reg.y[1] and reg.yaw[0] <= yaw <= reg.yaw[1]
        names = list(poses)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = math.hypot(poses[a][0] - poses[b][0], poses[a][1] - poses[b][1])
                assert d >= items[a].footprint_radius + items[b].footprint_radius + TABLETOP.clearance - 1e-9


def _place(env, name, xyz):
    adr = env.h.item_qpos_adr[name]
    env.data.qpos[adr:adr + 3] = xyz
    env.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    env.data.qvel[env.h.item_dof_adr[name]:env.h.item_dof_adr[name] + 6] = 0


def test_in_container_predicate(env_t1):
    env = env_t1
    env.reset(seed=3)
    pred = InContainer("red_cylinder", "black_tray")
    tray = env.world.pos("black_tray")
    assert not pred(env.world)  # starts on the table
    _place(env, "red_cylinder", tray + [0.0, 0.0, 0.005 + 0.015 + 0.001])  # resting on the tray floor
    for _ in range(100):
        mujoco.mj_step(env.model, env.data)
    assert pred(env.world)
    _place(env, "red_cylinder", tray + [0.0, 0.0, 0.10])  # above the tray: not inside
    mujoco.mj_forward(env.model, env.data)
    assert not pred(env.world)
    _place(env, "red_cylinder", tray + [0.10, 0.0, 0.016])  # beside the tray
    mujoco.mj_forward(env.model, env.data)
    assert not pred(env.world)


def test_reset_is_deterministic_and_obs_has_no_privileged_keys(env_t1):
    o1, _ = env_t1.reset(seed=42)
    s1 = env_t1.init_state
    o2, _ = env_t1.reset(seed=42)
    assert s1 == env_t1.init_state
    assert np.array_equal(o1["observation.state"], o2["observation.state"])
    assert set(o1) == {"observation.state"}  # render=False: state only; with cameras only images are added
    o3, _ = env_t1.reset(seed=43)
    assert s1 != env_t1.init_state
    o4, _ = env_t1.reset(options={"init_state": s1})
    assert np.array_equal(o4["observation.state"], o1["observation.state"])


@pytest.mark.parametrize("task_id", list(TASKS))
def test_expert_solves_task(task_id):
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.rollout import run_expert_episode

    env = SO101TabletopEnv(task_id, EnvConfig(render=False, terminate_on_success=False, max_steps=900))
    for seed in range(3):
        ep = run_expert_episode(env, seed)
        assert ep.success, (seed, ep.failure)
        assert ep.grasp["max_slip_mm"] < 5.0
        assert ep.max_ik_err < 1e-3


def test_suite_registry():
    assert [t.task_id for t in get_suite("so101_mini")] == ["cylinder_to_tray", "cylinder_to_container", "cube_to_tray"]


def test_real_setup_scene_and_registry():
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.sim.scene import FRONT_CAMERA

    assert FRONT_CAMERA.roll == pytest.approx(2.3) and FRONT_CAMERA.fovy == pytest.approx(45.0)
    assert [t.task_id for t in get_suite("so101_real")] == ["red_cube_aside", "red_cube_to_box", "red_cube_to_white_tray",
                                                          "red_cube_far_to_white_tray"]
    env = SO101TabletopEnv("red_cube_aside", EnvConfig(render=False))
    m = env.model
    assert np.allclose(m.geom("red_cube_geom").size, [0.015] * 3)
    assert set(env.h.objects) == {"red_cube"} and not env.h.containers
    cam = m.camera("front")
    assert np.allclose(m.cam_pos[cam.id], [0.374, 0.009, 0.557]) and m.cam_fovy[cam.id] == pytest.approx(45.0)
    for i in range(m.nmat):  # every robot material is near black, table/floor keep their own colours
        name = m.material(i).name
        if name not in ("table_mat", "floor_mat"):
            assert np.allclose(m.mat_rgba[i][:3], 0.12, atol=1e-6), name


def test_on_table_in_region_requires_lift_rest_release_and_zone():
    import mujoco as mj

    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.tasks.registry import ASIDE_ZONE

    env = SO101TabletopEnv("red_cube_aside", EnvConfig(render=False, terminate_on_success=False))
    env.reset(seed=1)
    pred = env.task.goal[0]
    ws = env.world
    assert not pred(ws)  # starts outside the zone, never lifted
    cx, cy = (ASIDE_ZONE.x[0] + ASIDE_ZONE.x[1]) / 2, (ASIDE_ZONE.y[0] + ASIDE_ZONE.y[1]) / 2
    _place(env, "red_cube", [cx, cy, 0.0151])
    for _ in range(100):
        mj.mj_step(env.model, env.data)
    assert not pred(ws)  # in the zone, at rest, but it was pushed there, never raised 2 cm
    ws.max_lift["red_cube"] = 0.05  # as if it had been carried ...
    assert not pred(ws)  # ... but never released here (no held position)
    ws.held_xy["red_cube"] = np.array([cx + 0.2, cy])  # released 20 cm away, then shoved into the zone
    assert not pred(ws)
    ws.held_xy["red_cube"] = np.array([cx + 0.02, cy - 0.01])  # released right here
    assert pred(ws)
    assert ws.tcp_pos()[2] > 0.1 or np.linalg.norm(ws.tcp_pos() - ws.pos("red_cube")) > 0.04  # arm is at home: far away
    # the cube lying on another face (rolled 90 deg) is still a placed cube
    adr = env.h.item_qpos_adr["red_cube"]
    _place(env, "red_cube", [cx, cy, 0.0151])
    env.data.qpos[adr + 3:adr + 7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
    for _ in range(100):
        mj.mj_step(env.model, env.data)
    assert ws.upright("red_cube") and pred(ws)
    _place(env, "red_cube", [cx, cy, 0.0151])
    env.data.qvel[env.h.item_dof_adr["red_cube"]] = 0.2  # moving: not at rest
    mj.mj_forward(env.model, env.data)
    assert not pred(ws)
    _place(env, "red_cube", [ASIDE_ZONE.x[1] + 0.03, cy, 0.0151])  # outside the zone
    for _ in range(50):
        mj.mj_step(env.model, env.data)
    assert not pred(ws)
    _place(env, "red_cube", [cx, cy, 0.06])  # hovering above the zone (held / dropped from above)
    mj.mj_forward(env.model, env.data)
    assert not pred(ws)
    # a tipped cube (resting on an edge is unstable; check the orientation test directly): 40 deg roll
    adr = env.h.item_qpos_adr["red_cube"]
    _place(env, "red_cube", [cx, cy, 0.0151])
    env.data.qpos[adr + 3:adr + 7] = [math.cos(math.radians(20)), math.sin(math.radians(20)), 0, 0]
    mj.mj_forward(env.model, env.data)
    assert not ws.upright("red_cube")


def test_env_tracks_lift_and_resets_it():
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.rollout import run_expert_episode

    env = SO101TabletopEnv("red_cube_aside", EnvConfig(render=False, terminate_on_success=False, max_steps=900))
    ep = run_expert_episode(env, 400001)
    assert ep.success and env.world.lifted("red_cube") and env.world.max_lift["red_cube"] > 0.04
    assert env.world.release_offset("red_cube") < 0.02  # released where it now rests
    env.reset(seed=400002)
    assert env.world.max_lift.get("red_cube", 0.0) < 0.002 and "red_cube" not in env.world.held_xy


def test_pushed_or_flicked_cube_is_not_a_success():
    import mujoco as mj

    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.tasks.registry import ASIDE_ZONE

    env = SO101TabletopEnv("red_cube_aside", EnvConfig(render=False, terminate_on_success=False))
    env.reset(seed=4)
    cx, cy = (ASIDE_ZONE.x[0] + ASIDE_ZONE.x[1]) / 2, (ASIDE_ZONE.y[0] + ASIDE_ZONE.y[1]) / 2
    # a ballistic hop: thrown 5 cm up without any grasp, lands in the zone -> max_lift stays 0 (not carried)
    _place(env, "red_cube", [cx, cy, 0.0151])
    env.data.qvel[env.h.item_dof_adr["red_cube"] + 2] = 1.0
    for _ in range(600):
        mj.mj_step(env.model, env.data)
        env.world.track()
    assert env.world.max_lift.get("red_cube", 0.0) == 0.0 and not env.task.goal[0](env.world)


def test_per_episode_cube_size_roundtrip():
    """red_cube_to_box draws a 2.7-4.0 cm cube per episode, records it in the initial state and restores it from there;
    fixed-size tasks never get a "sizes" entry."""
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    env = SO101TabletopEnv("red_cube_to_box", EnvConfig(render=False))
    gid = env.model.geom("red_cube_geom").id
    edges = []
    for seed in range(5):
        env.reset(seed=seed)
        size = env.init_state["sizes"]["red_cube"]
        assert 0.027 - 1e-9 <= 2 * size[0] <= 0.04 + 1e-9 and size[0] == size[1] == size[2]
        assert np.allclose(env.model.geom_size[gid], size) and env.h.objects["red_cube"].half_height == size[2]
        assert env.world.pos("red_cube")[2] == pytest.approx(size[2], abs=2e-3)  # resting on the table at its size
        edges.append(2 * size[0])
    assert len(set(edges)) == 5
    s0 = None
    env.reset(seed=0)
    s0 = env.init_state
    q0 = env.data.qpos.copy()
    env.reset(seed=3)
    env.reset(options={"init_state": s0})
    assert np.allclose(env.model.geom_size[gid], s0["sizes"]["red_cube"]) and np.array_equal(env.data.qpos, q0)
    aside = SO101TabletopEnv("red_cube_aside", EnvConfig(render=False))
    aside.reset(seed=0)
    assert "sizes" not in aside.init_state and np.allclose(aside.model.geom("red_cube_geom").size, [0.015] * 3)


def test_white_tray_task_cells_tray_gap_and_tall_wall_heights():
    """red_cube_to_white_tray: consecutive seeds start the cube in the 10 grid cells in turn, 1.5 cm or more from the
    fixed tray's outer rectangle; the expert carries and releases higher over its 6 cm walls, and exactly as before for
    the low black tray."""
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.pick_place import PickPlaceExpert
    from so101_libero.expert.rollout import expert_rng
    from so101_libero.tasks.base import _gap_to_tray

    task = TASKS["red_cube_to_white_tray"]
    grid, tray = task.layout.regions["red_cube"], task.layout.containers[0]
    env = SO101TabletopEnv(task, EnvConfig(render=False))
    for seed in range(20):
        env.reset(seed=seed)
        items = env.init_state["items"]
        x, y, _ = items["red_cube"]
        cell = grid.cell(seed % 10)
        assert cell.x[0] <= x <= cell.x[1] and cell.y[0] <= y <= cell.y[1]
        assert _gap_to_tray(tray, items["white_tray"], (x, y)) >= 0.0212 + task.layout.clearance - 1e-4
        assert list(items["white_tray"]) == pytest.approx([0.15, 0.15, 0.0])
    for task_id, wall in (("red_cube_to_white_tray", 0.06), ("red_cube_to_box", 0.025)):
        e = SO101TabletopEnv(task_id, EnvConfig(render=False))
        e.reset(seed=0)
        ex = PickPlaceExpert(e, e.task.plan, expert_rng(0))
        c = ex.cfg
        for grasp_z in (c.grasp_height[0], c.grasp_height[1]):
            ex._grasp_z = grasp_z
            carry, release = ex._carry_z(), ex._release_z()
            if wall > 0.04:  # object bottom 2 cm over the rim while carried, <= 2.5 cm under it when released
                assert carry - grasp_z >= wall + c.wall_clearance - 1e-9
                assert release - grasp_z >= wall - c.release_depth - 1e-9
            else:
                assert c.carry_height[0] <= carry <= c.carry_height[1]
                assert c.release_height[0] <= release <= c.release_height[1]


def test_far_cube_task_starts_and_parks_at_rest():
    """red_cube_far_to_white_tray: the cube lies 26 cm ahead of the pan axis, the arm starts in the folded rest pose,
    and the expert's demonstration ends parked at that pose (rest -> home -> pick, place -> home -> rest -> hold)."""
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.rollout import run_expert_episode
    from so101_libero.sim.robot import REST_DEG, sim_to_lerobot

    env = SO101TabletopEnv("red_cube_far_to_white_tray", EnvConfig(render=False, terminate_on_success=False, max_steps=900))
    ep = run_expert_episode(env, 0, record=True)
    assert ep.success, ep.failure
    x, y, _ = ep.init_state["items"]["red_cube"]
    assert 0.289 <= x <= 0.309 and abs(y) <= 0.01
    assert np.allclose(sim_to_lerobot(np.array(ep.init_state["robot_qpos"])), REST_DEG, atol=1e-3)
    phases = [p for _, p in ep.phases]
    assert phases[0] == "unfold" and phases[-2:] == ["return", "hold"]
    assert np.allclose(ep.frames[-1][1], REST_DEG, atol=1e-3)  # last action: the rest pose
    assert ep.success_step < env.task.max_steps
