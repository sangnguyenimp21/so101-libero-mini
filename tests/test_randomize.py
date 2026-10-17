"""Appearance-only domain randomization (so101_libero/sim/randomize.py).

Physics-only tests (MUJOCO_GL=disable, see conftest.py) plus one render test that runs only with a GL backend:
    SO101_RENDER=osmesa bash scripts/env.sh python -m pytest -q tests/test_randomize.py -p no:cacheprovider
"""

import json
import math
import os

import mujoco
import numpy as np
import pytest

from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.sim.randomize import DEFAULT, VisualRandConfig, VisualRandomizer, gaussian_blur

VISUAL_FIELDS = ("light_pos", "light_dir", "light_diffuse", "light_specular", "light_ambient", "mat_rgba", "geom_rgba",
                 "cam_pos", "cam_quat", "cam_fovy")
# Everything the dynamics and the renderer's one-time uploads read: must never change.
PHYSICS_FIELDS = ("geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_contype", "geom_conaffinity", "body_mass",
                  "body_inertia", "body_pos", "jnt_range", "actuator_ctrlrange", "actuator_forcerange", "dof_damping",
                  "light_castshadow", "tex_data", "cam_bodyid", "light_bodyid")


def snapshot(model, fields=VISUAL_FIELDS):
    s = {f: np.array(getattr(model, f), copy=True) for f in fields}
    if fields is VISUAL_FIELDS:
        s["headlight_ambient"] = np.array(model.vis.headlight.ambient, copy=True)
        s["headlight_diffuse"] = np.array(model.vis.headlight.diffuse, copy=True)
    return s


def equal(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def env():
    return SO101TabletopEnv("cylinder_to_tray", EnvConfig(render=False, terminate_on_success=False, max_steps=900))


@pytest.fixture
def randomizer(env):
    r = VisualRandomizer(env.model, env.h.cfg.cameras)
    yield r
    r.restore()  # never leave the shared model randomized


def test_restore_is_exact(env, randomizer):
    nominal, physics = snapshot(env.model), snapshot(env.model, PHYSICS_FIELDS)
    for seed in range(6):
        s = randomizer.randomize(np.random.default_rng(seed), data=env.data)
        assert s["enabled"] and randomizer.active and randomizer.sample is s
    changed = {k for k, v in snapshot(env.model).items() if not np.array_equal(v, nominal[k])}
    assert {"light_pos", "light_dir", "light_diffuse", "mat_rgba", "geom_rgba", "cam_pos", "cam_quat", "cam_fovy",
            "headlight_ambient", "headlight_diffuse"} <= changed
    assert equal(snapshot(env.model, PHYSICS_FIELDS), physics)
    randomizer.restore(data=env.data)
    assert equal(snapshot(env.model), nominal)
    assert randomizer.sample is None and not randomizer.active
    randomizer.restore()  # idempotent
    assert equal(snapshot(env.model), nominal)


def test_randomize_is_deterministic(env, randomizer):
    s1 = randomizer.randomize(np.random.default_rng(5))
    a1 = snapshot(env.model)
    randomizer.restore()
    s2 = randomizer.randomize(np.random.default_rng(5))
    a2 = snapshot(env.model)
    assert s1 == s2 and equal(a1, a2)
    assert json.loads(json.dumps(s1)) == s1  # plain JSON for the dataset side files
    assert set(s1) == {"enabled", "lights", "headlight", "materials", "items", "cameras", "post"}
    assert set(s1["items"]) == {"red_cylinder", "blue_cube", "black_tray", "round_container"}
    assert set(s1["cameras"]) == {"front", "wrist_cam"} and set(s1["lights"]) == {"key", "fill"}
    s3 = randomizer.randomize(np.random.default_rng(6))  # another stream: another sample (restore is implicit)
    assert s3 != s1 and not equal(snapshot(env.model), a1)


def _rollout(env, randomizer=None):
    """60 steps of a fixed joint-space action sequence; the randomizer (if any) resamples at reset and twice during."""
    env.reset(seed=0)
    if randomizer is not None:
        randomizer.randomize(np.random.default_rng(1), data=env.data)
    s0 = env.joint_state()
    qpos = []
    for t in range(60):
        a = s0.copy()
        a[:5] += 25.0 * np.sin(np.arange(5) * 0.7 + t / 8.0)
        a[5] = max(0.0, 100.0 - 2.0 * t)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        if randomizer is not None and t in (15, 40):
            randomizer.randomize(np.random.default_rng(100 + t), data=env.data)
        env.step(a)
        qpos.append(env.data.qpos.copy())
    if randomizer is not None:
        randomizer.restore(data=env.data)
    return np.array(qpos)


def test_physics_is_invariant(env, randomizer):
    ref = _rollout(env)
    assert np.abs(ref[-1] - ref[0]).max() > 0.1  # the arm actually moved
    assert np.array_equal(_rollout(env, randomizer), ref)  # bit-identical with randomization (incl. mj_camlight)
    assert np.array_equal(_rollout(env), ref)  # and nominal again after restore


def test_augment(env, randomizer):
    grey = np.full((48, 64, 3), 128, np.uint8)
    yy, xx = np.mgrid[0:48, 0:64]
    ramp = (np.stack([xx * 4, yy * 5, (xx + yy) * 2], axis=-1) % 256).astype(np.uint8)
    rng = np.random.default_rng(0)
    assert randomizer.augment(grey, rng) is grey  # never randomized: pass-through
    for seed in range(8):
        s = randomizer.randomize(np.random.default_rng(seed))
        p = s["post"]
        assert abs(p["brightness"] - 1) <= DEFAULT.brightness and abs(p["contrast"] - 1) <= DEFAULT.contrast
        assert 0 <= p["noise_std"] <= DEFAULT.noise_std and (p["blur_sigma"] is None or 0.3 <= p["blur_sigma"] <= 0.8)
        for img in (grey, ramp):
            out = randomizer.augment(img, np.random.default_rng(seed))
            assert out.dtype == np.uint8 and out.shape == img.shape
            assert np.abs(out.astype(int) - img.astype(int)).mean() < 40
        out = randomizer.augment(ramp, np.random.default_rng(seed))
        assert not np.array_equal(out, ramp)
        randomizer.restore()
        assert randomizer.augment(grey, rng) is grey and np.array_equal(randomizer.augment(ramp, rng), ramp)
    # every post-processing stage on its own
    only = dict(brightness=0.0, contrast=0.0, noise_std=0.0, blur_prob=0.0)
    r = VisualRandomizer(env.model, env.h.cfg.cameras, VisualRandConfig(**{**only, "brightness": 0.2}))
    b = r.randomize(np.random.default_rng(1))["post"]["brightness"]
    assert np.allclose(r.augment(grey, rng).astype(float), np.clip(round(128 * b), 0, 255), atol=1)
    r = VisualRandomizer(env.model, env.h.cfg.cameras, VisualRandConfig(**{**only, "contrast": 0.2}))
    r.randomize(np.random.default_rng(1))
    assert np.array_equal(r.augment(grey, rng), grey)  # a flat image has no contrast to change
    r = VisualRandomizer(env.model, env.h.cfg.cameras, VisualRandConfig(**{**only, "noise_std": 3.0}))
    r.randomize(np.random.default_rng(1))
    n1, n2 = r.augment(grey, np.random.default_rng(7)), r.augment(grey, np.random.default_rng(8))
    assert not np.array_equal(n1, n2) and abs(n1.mean() - 128) < 1 and 0 < n1.astype(float).std() < 4  # fresh noise per frame
    r = VisualRandomizer(env.model, env.h.cfg.cameras, VisualRandConfig(**{**only, "blur_prob": 1.0}))
    sigma = r.randomize(np.random.default_rng(1))["post"]["blur_sigma"]
    noisy = np.random.default_rng(0).integers(0, 256, (32, 40, 3)).astype(np.uint8)
    hf = lambda a: np.abs(np.diff(a.astype(int), axis=1)).mean()  # noqa: E731
    assert hf(r.augment(noisy, rng)) < 0.7 * hf(noisy)  # blurred
    ref = __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter(
        noisy.astype(np.float32), sigma=(sigma, sigma, 0), mode="reflect", truncate=4.0)
    assert np.abs(gaussian_blur(noisy.astype(np.float32), sigma) - ref).max() < 0.05
    r.restore()
    # disabled: nothing happens
    nominal = snapshot(env.model)
    r = VisualRandomizer(env.model, env.h.cfg.cameras, VisualRandConfig(enabled=False))
    assert r.randomize(rng) == {"enabled": False} and r.augment(grey, rng) is grey and equal(snapshot(env.model), nominal)


def _axis_angle_deg(q_a, q_b):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q_a, q_b))))))


def test_colours_and_poses_stay_plausible(env, randomizer):
    m = env.model
    nominal = snapshot(m)
    red, blue = m.geom("red_cylinder_geom").id, m.geom("blue_cube_geom").id
    tray = [g for g in range(m.ngeom) if m.geom(g).name.startswith("black_tray")]
    cont = [g for g in range(m.ngeom) if m.geom(g).name.startswith("round_container")]
    legs = [m.geom(f"table_leg{i}").id for i in range(4)]
    robot_mats = [i for i in range(m.nmat) if m.material(i).name not in ("table_mat", "floor_mat")]
    table_mat, floor_mat = m.material("table_mat").id, m.material("floor_mat").id
    front, wrist = m.camera("front").id, m.camera("wrist_cam").id
    front_spec = env.h.cfg.cameras[0]
    rng = np.random.default_rng(123)
    for _ in range(200):
        randomizer.randomize(rng)
        rc, bc = m.geom_rgba[red], m.geom_rgba[blue]
        assert rc[0] > 1.8 * max(rc[1], rc[2]) and rc[0] > 0.55  # red stays red
        assert bc[2] > 1.8 * max(bc[0], bc[1]) and bc[2] > 0.55  # blue stays blue
        assert all(m.geom_rgba[g, :3].max() < 0.1 for g in tray)  # black stays dark
        for g in cont:  # white stays bright and neutral
            c = m.geom_rgba[g, :3]
            assert c.min() > 0.55 and c.max() <= 1.0 and c.max() / c.min() < 1.3
        assert len({tuple(m.geom_rgba[g]) for g in cont}) == 1 and len({tuple(m.geom_rgba[g]) for g in tray}) == 1
        for i in robot_mats:
            assert 0.05 <= m.mat_rgba[i, :3].min() and m.mat_rgba[i, :3].max() <= 0.25
        assert len({tuple(m.mat_rgba[i]) for i in robot_mats}) == 1  # the arm is one colour
        t, f = m.mat_rgba[table_mat, :3], m.mat_rgba[floor_mat, :3]
        assert 0.7 * 0.92 - 1e-9 <= t.min() and t.max() <= 1.06 * 1.08 + 1e-9 and t.max() / t.min() < 1.08 / 0.92 + 1e-9
        assert 0.6 * 0.92 - 1e-9 <= f.min() and f.max() <= 1.4 * 1.08 + 1e-9
        assert all(np.all(m.geom_rgba[g, :3] <= 1.0) for g in legs)
        assert np.array_equal(m.geom_rgba[:, 3], nominal["geom_rgba"][:, 3])  # alpha untouched
        assert np.array_equal(m.mat_rgba[:, 3], nominal["mat_rgba"][:, 3])
        # lights: moved by at most the jitter, still aimed at the workspace, intensity in range, neutral-ish colour
        for i in range(m.nlight):
            p, d = m.light_pos[i], m.light_dir[i]
            assert np.abs(p - nominal["light_pos"][i]).max() <= DEFAULT.light_pos_jitter + 1e-9
            assert abs(np.linalg.norm(d) - 1) < 1e-9 and d[2] < 0
            hit = p - d * (p[2] / d[2])  # where the light axis crosses the table plane
            assert np.linalg.norm(hit[:2] - [0.15, 0.0]) < 0.25
            gain = m.light_diffuse[i] / nominal["light_diffuse"][i]
            assert 0.6 * 0.95 - 1e-9 <= gain.min() and gain.max() <= 1.2 * 1.05 + 1e-9 and gain.max() / gain.min() <= 1.1 / 0.9 + 1e-9
            assert np.allclose(m.light_specular[i], nominal["light_specular"][i] * gain)
        assert 0.7 - 1e-9 <= m.vis.headlight.ambient[0] / 0.35 <= 1.15 + 1e-9
        assert 0.7 - 1e-9 <= m.vis.headlight.diffuse[0] / 0.35 <= 1.15 + 1e-9
        # front camera: within the jitters and still looking at the (jittered) look-at point
        assert np.abs(m.cam_pos[front] - front_spec.pos).max() <= DEFAULT.cam_pos_jitter + 1e-9
        assert abs(m.cam_fovy[front] - front_spec.fovy) <= DEFAULT.cam_fovy_jitter + 1e-9
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.cam_quat[front])
        forward = -rot.reshape(3, 3)[:, 2]  # a MuJoCo camera looks along its -z axis
        p = m.cam_pos[front]
        hit = p - forward * (p[2] / forward[2])
        assert np.linalg.norm(hit[:2] - front_spec.lookat[:2]) < DEFAULT.cam_lookat_jitter * 2 + 0.02
        assert abs(np.linalg.norm(m.cam_quat[front]) - 1) < 1e-9
        # wrist camera: small local jitters only
        assert np.abs(m.cam_pos[wrist] - nominal["cam_pos"][wrist]).max() <= DEFAULT.wrist_pos_jitter + 1e-9
        assert _axis_angle_deg(m.cam_quat[wrist], nominal["cam_quat"][wrist]) <= DEFAULT.wrist_rot_jitter + 1e-6
        assert abs(m.cam_fovy[wrist] - nominal["cam_fovy"][wrist]) <= DEFAULT.wrist_fovy_jitter + 1e-9


def test_unknown_camera_is_rejected(env):
    from so101_libero.sim.scene import CameraSpec

    with pytest.raises(ValueError, match="nowhere"):
        VisualRandomizer(env.model, (CameraSpec("nowhere", (1, 0, 1), (0, 0, 0)),))


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") in (None, "disable"), reason="needs an offscreen GL backend")
def test_render_changes_and_restores():
    """The in-place edits reach the next render of an existing Renderer; restore() gives the nominal image back."""
    env = SO101TabletopEnv("cylinder_to_tray", EnvConfig(image_size=(128, 128)))
    try:
        env.reset(seed=0)
        nominal = {c: env.render_camera(c) for c in ("front", "wrist")}
        r = VisualRandomizer(env.model, env.h.cfg.cameras)
        r.randomize(np.random.default_rng(3), data=env.data)
        for c, img in nominal.items():
            out = env.render_camera(c)
            assert out.shape == img.shape and not np.array_equal(out, img)
            assert np.abs(out.astype(int) - img.astype(int)).mean() > 2
            aug = r.augment(out, np.random.default_rng(0))
            assert aug.dtype == np.uint8 and aug.shape == img.shape
        r.restore(data=env.data)
        for c, img in nominal.items():
            assert np.array_equal(env.render_camera(c), img)
    finally:
        env.close()
