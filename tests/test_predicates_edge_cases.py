"""Edge cases found by the adversarial review: none of these may count as success."""

import math

import mujoco
import numpy as np

from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.tasks.base import InContainer, OnTable, RelativeTo


def _place(env, name, xyz, quat=(1, 0, 0, 0)):
    adr = env.h.item_qpos_adr[name]
    env.data.qpos[adr:adr + 3] = xyz
    env.data.qpos[adr + 3:adr + 7] = quat
    env.data.qvel[env.h.item_dof_adr[name]:env.h.item_dof_adr[name] + 6] = 0


def _settle(env, n=300):
    for _ in range(n):
        mujoco.mj_step(env.model, env.data)


def test_flipped_tray_over_object_is_not_inside(env_t1):
    env = env_t1
    env.reset(seed=5)
    cyl = env.world.pos("red_cylinder")
    flip = (0.0, 1.0, 0.0, 0.0)  # 180 deg about x: tray upside down over the cylinder
    _place(env, "black_tray", (cyl[0], cyl[1], 0.026), flip)
    _settle(env)
    assert not env.world.upright("black_tray")
    assert not InContainer("red_cylinder", "black_tray")(env.world)


def test_object_on_container_floor_is_not_on_table(env_t1):
    env = env_t1
    env.reset(seed=6)
    tray = env.world.pos("black_tray")
    _place(env, "red_cylinder", tray + [0.0, 0.0, 0.005 + 0.015 + 0.001])
    _settle(env)
    assert InContainer("red_cylinder", "black_tray")(env.world)
    assert not OnTable("red_cylinder")(env.world)


def test_moving_the_reference_does_not_satisfy_relative_goal():
    env = SO101TabletopEnv("cylinder_left_of_cube", EnvConfig(render=False, terminate_on_success=False))
    env.reset(seed=11)
    pred = RelativeTo("red_cylinder", "blue_cube", "left")
    assert not pred(env.world)
    cyl = env.world.pos("red_cylinder")
    # put the cube 7 cm to the right of the (unmoved) cylinder: geometrically "cylinder left of cube"
    _place(env, "blue_cube", (cyl[0], cyl[1] - 0.07, 0.0151))
    _settle(env)
    assert not pred(env.world)  # the reference moved: not a success
    # the honest way: move the cylinder next to the (restored) cube
    cube0 = env.world.init_pos["blue_cube"]
    _place(env, "blue_cube", (cube0[0], cube0[1], 0.0151))
    _place(env, "red_cylinder", (cube0[0], cube0[1] + 0.065, 0.0151))
    _settle(env)
    assert pred(env.world)


def test_simulation_divergence_is_a_failure_not_a_success(env_t1):
    env = SO101TabletopEnv("cylinder_to_tray", EnvConfig(render=False))
    obs, _ = env.reset(seed=0)
    env.data.qvel[env.h.item_dof_adr["red_cylinder"]] = math.nan
    obs, reward, terminated, truncated, info = env.step(obs["observation.state"])
    assert info["failure"] == "simulation diverged" and not info["is_success"] and terminated and reward == 0.0
    assert np.all(np.isfinite(obs["observation.state"]))
