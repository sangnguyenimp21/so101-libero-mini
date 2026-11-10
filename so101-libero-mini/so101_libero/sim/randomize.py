"""Appearance-only domain randomization: lights, colours, camera jitter and image post-processing.

Everything here changes only what the cameras see. The physics is untouched: the fields that are modified
(``light_*``, ``vis.headlight``, ``mat_rgba``, ``geom_rgba``, ``cam_pos/quat/fovy``) are read by ``mjv_updateScene``
at every render and never enter the dynamics, so object and robot trajectories are bit-identical with or
without randomization. Textures (``tex_*``) are never touched: the renderer uploads them once.

Usage (one sample per episode)::

    rnd = VisualRandomizer(env.model, env.h.cfg.cameras)       # stores the nominal appearance
    sample = rnd.randomize(np.random.default_rng(seed), data=env.data)   # new appearance, in place
    img = rnd.augment(env.render_camera("front"), rng)          # per-frame post-processing
    rnd.restore(data=env.data)                                  # exact nominal appearance again

Camera and light *placement* live in ``mjData`` (``cam_xpos``, ``cam_xmat``, ``light_xpos``, ``light_xdir``) and
are recomputed from the model by ``mj_camlight`` inside every ``mj_forward`` / ``mj_step``. A render issued after
``randomize()`` but before the next physics step would still use the previous placement; passing ``data``
refreshes it immediately with ``mj_camlight``, which writes only those four arrays (no effect on the state, the
solver warm start or anything else the simulation reads).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from so101_libero.sim.scene import CameraSpec, camera_quat

WORKSPACE_CENTRE = (0.15, 0.0, 0.0)  # m, where the lights are aimed (with a small random deviation)
LIGHT_AIM_JITTER = 0.10  # m, per axis, around WORKSPACE_CENTRE
LIGHT_TINT = 0.10  # max relative difference between the red and blue gains of a light (warm / cool)
BLUR_SIGMA = (0.3, 0.8)  # px, when a blur is applied


@dataclass(frozen=True)
class VisualRandConfig:
    """Ranges of the appearance distribution. Every range is centred on the nominal scene and kept moderate: the
    nominal appearance (every multiplier 1, every jitter 0) stays well inside the distribution."""

    # The nominal scene is already bright (table ~206/255 in the front image, 15 % of its pixels saturated), so the
    # intensity ranges reach further down than up: one factor of 1.2 alone already saturates ~30 % of the image.
    light_pos_jitter: float = 0.35  # m, uniform per axis, each light; the light is re-aimed at the workspace
    light_diffuse: tuple[float, float] = (0.6, 1.2)  # multiplier on each light's diffuse and specular
    ambient: tuple[float, float] = (0.7, 1.15)  # multiplier on vis.headlight.ambient and (independently) .diffuse
    table_tint: tuple[float, float] = (0.7, 1.06)  # brightness factor on the table material (+ per channel 0.92-1.08)
    floor_tint: tuple[float, float] = (0.6, 1.4)
    robot_tint: tuple[float, float] = (0.6, 1.8)  # robot materials: near black 0.12 nominal -> 0.07..0.22
    object_tint: tuple[float, float] = (0.75, 1.15)  # objects / containers, per item (+ per channel 0.9-1.1)
    cam_pos_jitter: float = 0.015  # m, fixed world cameras (the "front" camera)
    cam_lookat_jitter: float = 0.015  # m
    cam_roll_jitter: float = 1.5  # deg
    cam_fovy_jitter: float = 2.0  # deg
    wrist_pos_jitter: float = 0.003  # m, every other camera (e.g. wrist_cam on the robot)
    wrist_rot_jitter: float = 2.0  # deg, rotation about a random axis
    wrist_fovy_jitter: float = 5.0  # deg
    brightness: float = 0.10  # post-processing: multiplicative brightness in [1 - b, 1 + b]
    contrast: float = 0.10  # contrast factor in [1 - c, 1 + c] around the image mean
    noise_std: float = 3.0  # gaussian noise std (0..255 units) sampled in [0, noise_std], fresh noise per frame
    blur_prob: float = 0.3  # probability of a gaussian blur with sigma in BLUR_SIGMA
    enabled: bool = True


DEFAULT = VisualRandConfig()


class VisualRandomizer:
    """Samples a new appearance into a model in place and post-processes rendered frames; ``restore`` undoes it."""

    def __init__(self, model: mujoco.MjModel, cameras: tuple[CameraSpec, ...] = (), cfg: VisualRandConfig | None = None):
        self.model = model
        self.cfg = cfg or DEFAULT
        self.sample: dict | None = None  # the current appearance sample (None = nominal)
        self._post: dict | None = None  # the current post-processing parameters (None = pass-through)

        self.fixed_cams: dict[int, CameraSpec] = {}
        for spec in cameras:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec.name)
            if cam_id < 0:
                raise ValueError(f"camera {spec.name!r} is not in the model")
            self.fixed_cams[cam_id] = spec
        self.other_cams = [c for c in range(model.ncam) if c not in self.fixed_cams]

        # Geoms by role. Items are the bodies of the kinematic trees rooted at a free joint (objects, containers);
        # every other non-world body is the robot. Robot geoms are coloured through their materials, world geoms
        # through the table/floor materials or their own rgba (table legs), item geoms through geom_rgba.
        self.item_geoms: dict[str, list[int]] = {}
        robot_mats: set[int] = set()
        self.table_mats: list[int] = []
        self.floor_mats: list[int] = []
        self.table_geoms: list[int] = []  # world geoms without a material (table legs)
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            mat = int(model.geom_matid[g])
            if body == 0:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                if mat >= 0:
                    group = self.floor_mats if name == "floor" else self.table_mats
                    if mat not in group:
                        group.append(mat)
                else:
                    self.table_geoms.append(g)
            elif _is_free_tree(model, body):
                item = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"
                self.item_geoms.setdefault(item, []).append(g)
            elif mat >= 0:
                robot_mats.add(mat)
        self.robot_mats = sorted(robot_mats)

        self.light_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or f"light{i}" for i in range(model.nlight)]
        self.cam_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}" for i in range(model.ncam)]
        self.mat_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}" for i in range(model.nmat)]

        self._fields = ("light_pos", "light_dir", "light_diffuse", "light_specular", "light_ambient", "mat_rgba",
                        "geom_rgba", "cam_pos", "cam_quat", "cam_fovy")
        self._nominal = {f: np.array(getattr(model, f), copy=True) for f in self._fields}
        self._nominal["headlight_ambient"] = np.array(model.vis.headlight.ambient, copy=True)
        self._nominal["headlight_diffuse"] = np.array(model.vis.headlight.diffuse, copy=True)

    # --------------------------------------------------------------------------------------- sampling
    def randomize(self, rng: np.random.Generator, data: mujoco.MjData | None = None) -> dict:
        """Sample a new appearance into the model (in place) and this episode's post-processing parameters.

        Deterministic: the same ``rng`` state gives the same sample. Returns a JSON-serialisable description.
        With ``data``, the camera / light placement in ``data`` is refreshed right away (see the module docstring).
        """
        self.restore()  # every sample starts from the nominal values
        cfg, m = self.cfg, self.model
        if not cfg.enabled:
            self.sample = {"enabled": False}
            return self.sample
        s: dict = {"enabled": True, "lights": {}, "headlight": {}, "materials": {}, "items": {}, "cameras": {}}

        # Lights: jitter the position, re-aim at the workspace centre (+ a small deviation), scale the intensity with
        # a slight warm / cool tint. castshadow and the (zero) ambient term are left as they are.
        for i in range(m.nlight):
            pos = self._nominal["light_pos"][i] + rng.uniform(-cfg.light_pos_jitter, cfg.light_pos_jitter, 3)
            aim = np.asarray(WORKSPACE_CENTRE) + rng.uniform(-LIGHT_AIM_JITTER, LIGHT_AIM_JITTER, 3)
            direction = aim - pos
            direction /= np.linalg.norm(direction)
            warm = rng.uniform(-LIGHT_TINT, LIGHT_TINT)  # warm (+) / cool (-): at most LIGHT_TINT between channels
            gain = rng.uniform(*cfg.light_diffuse) * np.array([1 + warm / 2, 1.0, 1 - warm / 2])
            m.light_pos[i] = pos
            m.light_dir[i] = direction
            m.light_diffuse[i] = self._nominal["light_diffuse"][i] * gain
            m.light_specular[i] = self._nominal["light_specular"][i] * gain
            s["lights"][self.light_names[i]] = {"pos": pos.tolist(), "dir": direction.tolist(), "gain": gain.tolist()}

        amb, dif = rng.uniform(*cfg.ambient), rng.uniform(*cfg.ambient)
        m.vis.headlight.ambient[:] = self._nominal["headlight_ambient"] * amb
        m.vis.headlight.diffuse[:] = self._nominal["headlight_diffuse"] * dif
        s["headlight"] = {"ambient_gain": float(amb), "diffuse_gain": float(dif)}

        # Materials: one brightness factor per surface family plus a small per-channel factor (hue stays plausible).
        table = rng.uniform(*cfg.table_tint) * rng.uniform(0.92, 1.08, 3)
        floor = rng.uniform(*cfg.floor_tint) * rng.uniform(0.92, 1.08, 3)
        robot = rng.uniform(*cfg.robot_tint) * rng.uniform(0.95, 1.05, 3)
        # Material rgba may exceed 1: the table / floor colour lives in their textures (material rgba 1.0 nominal),
        # and the renderer honours values above 1 (it modulates the texture, then clamps the lit colour).
        for mats, gain in ((self.table_mats, table), (self.floor_mats, floor), (self.robot_mats, robot)):
            for mat in mats:
                m.mat_rgba[mat, :3] = self._nominal["mat_rgba"][mat, :3] * gain
        for g in self.table_geoms:
            m.geom_rgba[g, :3] = np.clip(self._nominal["geom_rgba"][g, :3] * table, 0.0, 1.0)
        s["materials"] = {"table_gain": table.tolist(), "floor_gain": floor.tolist(), "robot_gain": robot.tolist(),
                          "robot_rgb": (self._nominal["mat_rgba"][self.robot_mats[0], :3] * robot).tolist()
                          if self.robot_mats else None}

        # Items: per item one brightness factor and a small per-channel factor; red stays red, black stays dark.
        for item, geoms in self.item_geoms.items():
            gain = rng.uniform(*cfg.object_tint) * rng.uniform(0.9, 1.1, 3)
            for g in geoms:
                m.geom_rgba[g, :3] = np.clip(self._nominal["geom_rgba"][g, :3] * gain, 0.0, 1.0)
            s["items"][item] = {"gain": gain.tolist(), "rgb": m.geom_rgba[geoms[0], :3].tolist()}

        # Cameras: fixed world cameras are rebuilt from a jittered CameraSpec; the others get small local jitters.
        for cam_id, spec in self.fixed_cams.items():
            pos = np.asarray(spec.pos, float) + rng.uniform(-cfg.cam_pos_jitter, cfg.cam_pos_jitter, 3)
            lookat = np.asarray(spec.lookat, float) + rng.uniform(-cfg.cam_lookat_jitter, cfg.cam_lookat_jitter, 3)
            roll = spec.roll + rng.uniform(-cfg.cam_roll_jitter, cfg.cam_roll_jitter)
            fovy = spec.fovy + rng.uniform(-cfg.cam_fovy_jitter, cfg.cam_fovy_jitter)
            jittered = CameraSpec(spec.name, tuple(map(float, pos)), tuple(map(float, lookat)), float(fovy), float(roll))
            m.cam_pos[cam_id] = pos
            m.cam_quat[cam_id] = camera_quat(jittered)
            m.cam_fovy[cam_id] = fovy
            s["cameras"][spec.name] = {"pos": pos.tolist(), "lookat": lookat.tolist(), "fovy": float(fovy), "roll": float(roll)}
        for cam_id in self.other_cams:
            pos = self._nominal["cam_pos"][cam_id] + rng.uniform(-cfg.wrist_pos_jitter, cfg.wrist_pos_jitter, 3)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = math.radians(rng.uniform(-cfg.wrist_rot_jitter, cfg.wrist_rot_jitter))
            fovy = self._nominal["cam_fovy"][cam_id] + rng.uniform(-cfg.wrist_fovy_jitter, cfg.wrist_fovy_jitter)
            dq, quat = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, angle)
            mujoco.mju_mulQuat(quat, self._nominal["cam_quat"][cam_id], dq)  # rotation in the camera's own frame
            m.cam_pos[cam_id] = pos
            m.cam_quat[cam_id] = quat
            m.cam_fovy[cam_id] = fovy
            s["cameras"][self.cam_names[cam_id]] = {"pos": pos.tolist(), "quat": quat.tolist(), "fovy": float(fovy),
                                                     "rot_axis": axis.tolist(), "rot_deg": math.degrees(angle)}

        # Per-episode image post-processing parameters (the noise itself is drawn per frame in augment()).
        blur = rng.uniform(*BLUR_SIGMA) if rng.uniform() < cfg.blur_prob else None
        self._post = {"brightness": float(rng.uniform(1 - cfg.brightness, 1 + cfg.brightness)),
                      "contrast": float(rng.uniform(1 - cfg.contrast, 1 + cfg.contrast)),
                      "noise_std": float(rng.uniform(0.0, cfg.noise_std)),
                      "blur_sigma": None if blur is None else float(blur)}
        s["post"] = dict(self._post)
        if data is not None:
            mujoco.mj_camlight(m, data)
        self.sample = s
        return s

    def restore(self, data: mujoco.MjData | None = None) -> None:
        """Put every modified model field back to its nominal value (bit-exact) and disable post-processing."""
        m = self.model
        for f in self._fields:
            getattr(m, f)[:] = self._nominal[f]
        m.vis.headlight.ambient[:] = self._nominal["headlight_ambient"]
        m.vis.headlight.diffuse[:] = self._nominal["headlight_diffuse"]
        self.sample = None
        self._post = None
        if data is not None:
            mujoco.mj_camlight(m, data)

    # ----------------------------------------------------------------------------------- post-processing
    def augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply the current episode's post-processing to one rendered uint8 frame (H, W, C): contrast around the
        image mean, brightness, optional gaussian blur, gaussian noise drawn from ``rng`` (fresh per frame), clipped
        to 0..255. Returns the input itself when no sample is active (after ``restore()`` or before ``randomize()``).
        """
        p = self._post
        if p is None:
            return img
        x = img.astype(np.float32)
        mean = np.float32(x.mean())
        x = (x - mean) * np.float32(p["contrast"]) + mean
        x *= np.float32(p["brightness"])
        if p["blur_sigma"] is not None:
            x = gaussian_blur(x, p["blur_sigma"])
        if p["noise_std"] > 0:
            x += rng.standard_normal(x.shape, dtype=np.float32) * np.float32(p["noise_std"])
        return (np.clip(x, 0.0, 255.0) + 0.5).astype(np.uint8)

    @property
    def active(self) -> bool:
        return self._post is not None

    def describe(self) -> dict:
        """The configuration as a plain dict (for dataset side files)."""
        return asdict(self.cfg)


def gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur over the first two axes of a float image (edge-mirrored padding, like
    ``scipy.ndimage.gaussian_filter(mode="reflect")``), kernel radius 3 sigma."""
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / np.float32(sigma)) ** 2)
    k /= k.sum()
    h, w = x.shape[:2]
    pad = ((r, r),) + ((0, 0),) * (x.ndim - 1)
    p = np.pad(x, pad, mode="symmetric")
    y = np.zeros_like(x)
    for i in range(2 * r + 1):
        y += k[i] * p[i:i + h]
    pad = ((0, 0), (r, r)) + ((0, 0),) * (x.ndim - 2)
    p = np.pad(y, pad, mode="symmetric")
    out = np.zeros_like(x)
    for i in range(2 * r + 1):
        out += k[i] * p[:, i:i + w]
    return out


def _is_free_tree(model: mujoco.MjModel, body: int) -> bool:
    """Is ``body`` part of a kinematic tree whose root hangs on a free joint (an object or container)?"""
    root = int(model.body_rootid[body])
    adr, n = int(model.body_jntadr[root]), int(model.body_jntnum[root])
    return n > 0 and int(model.jnt_type[adr]) == int(mujoco.mjtJoint.mjJNT_FREE)
