"""Build the tabletop scene: vendored SO-101 + table + objects/containers + cameras, composed with MjSpec.

The upstream robot XML is never edited; everything scene-specific is added here. World frame: the robot base
sits at the origin on the table top (z = 0) and faces +x; +y is the robot's left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from so101_libero.sim.objects import ContainerSpec, ObjectSpec, add_container, add_object
from so101_libero.sim.robot import JOINT_NAMES

ASSETS = Path(__file__).resolve().parents[2] / "assets"
ROBOT_XML = ASSETS / "robotstudio_so101" / "so101.xml"
LOWPOLY_DIR = ASSETS / "robotstudio_so101_lowpoly"  # decimated visual meshes (scripts/decimate_meshes.py)

# Tool centre point in the "gripper" (wrist-roll) body frame: between the jaw pads, 12 mm behind the fixed jaw
# tip, centred on a 3 cm object held against the fixed jaw. Body axes: +x = from the fixed jaw towards the
# moving jaw (closing direction), -z = towards the finger tips (approach direction).
TCP_POS = (0.005, 0.0, -0.089)
# Grasp checks: the fixed jaw is every collision geom on the wrist-roll ("gripper") body, the moving jaw every
# collision geom on its own body. The wrist-camera mount is a separate body and does not count.
MOVING_JAW_BODY = "moving_jaw_so101_v1"


@dataclass(frozen=True)
class CameraSpec:
    """A fixed world camera defined by position, look-at target and vertical field of view (degrees)."""

    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy: float = 45.0


# Third-person view facing the robot from the front, elevated ~40 deg (LIBERO "agentview"-like): the whole
# workspace (tray on the image left, round container on the right) and the arm are visible.
FRONT_CAMERA = CameraSpec("front", pos=(0.56, 0.0, 0.46), lookat=(0.15, 0.0, 0.01), fovy=52.0)


@dataclass(frozen=True)
class SceneConfig:
    objects: tuple[ObjectSpec, ...]
    containers: tuple[ContainerSpec, ...]
    cameras: tuple[CameraSpec, ...] = (FRONT_CAMERA,)
    wrist_camera_fovy: float = 90.0  # SO-101 wrist cams are wide-angle (Menagerie uses 61; real fits ~100)
    control_hz: float = 30.0
    substeps: int = 16  # physics steps per control step -> timestep = 1 / (control_hz * substeps)
    table_rgba: tuple[float, float, float, float] = (0.78, 0.72, 0.62, 1.0)
    offscreen_size: int = 640  # max render width/height
    lowpoly_visuals: bool = True  # decimated visual meshes: same physics, ~4x faster CPU rendering
    gripper_force: float = 1.2  # N*m actuator limit of the gripper servo
    elbow_max_deg: float = 90.0


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (w, x, y, z) of a MuJoCo camera at ``pos`` looking at ``target`` (camera looks along -z)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.modelname = "so101_tabletop"
    if cfg.lowpoly_visuals:
        visual = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.contype == 0
                  and g.conaffinity == 0}
        for mesh in spec.meshes:
            if (mesh.name or Path(mesh.file).stem) in visual and (LOWPOLY_DIR / mesh.file).exists():
                mesh.file = str(LOWPOLY_DIR / mesh.file)

    opt = spec.option
    opt.timestep = 1.0 / (cfg.control_hz * cfg.substeps)
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10.0
    opt.iterations = 50
    opt.ls_iterations = 30
    # Flat faces on flat faces (cylinder on table/tray, cube on round base) go through the general convex
    # collider, which returns ONE contact point: objects then rock, spin and sink. multiccd returns a contact
    # patch instead, so resting objects are actually at rest.
    opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    spec.visual.global_.offwidth = cfg.offscreen_size
    spec.visual.global_.offheight = cfg.offscreen_size
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    # The real follower runs the gripper servo at 50 % torque (LeRobot so101_follower.configure); the model's
    # 2.94 N*m gives ~80 N pinch forces and mm-deep interpenetration. 1.2 N*m still holds 30 g objects firmly.
    spec.actuator("gripper").forcerange = [-cfg.gripper_force, cfg.gripper_force]
    # Beyond ~90 deg the forearm collides with the shoulder: keep the elbow (and its commands) below that.
    spec.joint("elbow_flex").range = [-1.69, math.radians(cfg.elbow_max_deg)]
    spec.actuator("elbow_flex").ctrlrange = [-1.69, math.radians(cfg.elbow_max_deg)]

    # The Menagerie base has only visual geoms: give it a low collider (table level up to 3.5 cm, below the shoulder)
    # so items pushed towards the robot stop at the base instead of passing through it.
    spec.body("base").add_geom(name="base_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.043, 0.0555, 0.0175],
                               pos=[0.0211, 0.0, 0.0175], group=3, rgba=[1, 0, 0, 0.3])
    # MuJoCo's parent/child contact filter does not apply when the parent is welded to the world: exclude explicitly.
    spec.add_exclude(bodyname1="base", bodyname2="shoulder")

    gripper = spec.body("gripper")
    gripper.add_site(name="tcp", pos=list(TCP_POS), size=[0.004, 0, 0], rgba=[0, 1, 0, 0.6], group=4)
    spec.camera("wrist_cam").fovy = cfg.wrist_camera_fovy

    world = spec.worldbody
    tex = spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                           rgb1=list(cfg.table_rgba[:3]), rgb2=list(cfg.table_rgba[:3]), width=64, height=64,
                           mark=mujoco.mjtMark.mjMARK_RANDOM, markrgb=[c * 0.9 for c in cfg.table_rgba[:3]], random=0.08)
    mat = spec.add_material(name="table_mat", texrepeat=[6, 6], reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    floor_tex = spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                 builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.32, 0.33, 0.35],
                                 rgb2=[0.27, 0.28, 0.30], width=256, height=256)
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[8, 8])
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = floor_tex.name

    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.76], material="floor_mat")
    # Table top: x in [-0.22, 0.62], y in [-0.50, 0.50], surface at z = 0.
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.42, 0.50, 0.02], pos=[0.20, 0.0, -0.02],
                   material="table_mat", friction=[1.0, 0.005, 0.0001])
    for i, (lx, ly) in enumerate([(-0.18, -0.46), (-0.18, 0.46), (0.58, -0.46), (0.58, 0.46)]):
        world.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.37],
                       pos=[lx, ly, -0.41], rgba=[0.45, 0.40, 0.33, 1], contype=0, conaffinity=0)

    world.add_light(name="key", pos=[0.3, -0.3, 1.2], dir=[-0.1, 0.25, -1.0], diffuse=[0.55, 0.55, 0.55],
                    specular=[0.1, 0.1, 0.1], castshadow=True)
    world.add_light(name="fill", pos=[-0.4, 0.4, 1.0], dir=[0.5, -0.4, -1.0], diffuse=[0.2, 0.2, 0.2],
                    specular=[0, 0, 0], castshadow=False)

    for cam in cfg.cameras:
        world.add_camera(name=cam.name, pos=list(cam.pos), quat=list(look_at_quat(cam.pos, cam.lookat)), fovy=cam.fovy)

    for c in cfg.containers:
        add_container(world, c)
    for o in cfg.objects:
        add_object(world, o)
    return spec


@dataclass
class SceneHandles:
    """Model plus the ids/addresses the environment, expert and success checks need."""

    model: mujoco.MjModel
    cfg: SceneConfig
    joint_qpos_adr: np.ndarray  # (6,) qpos address of each robot joint, in JOINT_NAMES order
    joint_dof_adr: np.ndarray  # (6,)
    actuator_ids: np.ndarray  # (6,)
    joint_range: np.ndarray  # (6, 2) rad
    tcp_site: int
    gripper_body: int
    moving_jaw_body: int
    robot_bodies: frozenset[int]
    fixed_pad_geoms: frozenset[int]
    moving_jaw_geoms: frozenset[int]
    table_geom: int
    item_body: dict[str, int] = field(default_factory=dict)  # objects and containers
    item_qpos_adr: dict[str, int] = field(default_factory=dict)  # start of the 7-d free-joint qpos
    item_dof_adr: dict[str, int] = field(default_factory=dict)
    item_geoms: dict[str, frozenset[int]] = field(default_factory=dict)

    @property
    def objects(self) -> dict[str, ObjectSpec]:
        return {o.name: o for o in self.cfg.objects}

    @property
    def containers(self) -> dict[str, ContainerSpec]:
        return {c.name: c for c in self.cfg.containers}


def build_scene(cfg: SceneConfig) -> SceneHandles:
    model = build_spec(cfg).compile()
    jid = [model.joint(n).id for n in JOINT_NAMES]
    robot_root = model.body("base").id
    robot_bodies = frozenset(b for b in range(model.nbody) if _is_descendant(model, b, robot_root))
    moving_jaw = model.body(MOVING_JAW_BODY).id
    h = SceneHandles(
        model=model,
        cfg=cfg,
        joint_qpos_adr=np.array([model.jnt_qposadr[j] for j in jid]),
        joint_dof_adr=np.array([model.jnt_dofadr[j] for j in jid]),
        actuator_ids=np.array([model.actuator(n).id for n in JOINT_NAMES]),
        joint_range=np.array([model.jnt_range[j] for j in jid]),
        tcp_site=model.site("tcp").id,
        gripper_body=model.body("gripper").id,
        moving_jaw_body=moving_jaw,
        robot_bodies=robot_bodies,
        fixed_pad_geoms=frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("gripper").id
                                  and model.geom_contype[g] | model.geom_conaffinity[g]),
        moving_jaw_geoms=frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] == moving_jaw
                                   and model.geom_contype[g] | model.geom_conaffinity[g]),
        table_geom=model.geom("table").id,
    )
    for item in (*cfg.objects, *cfg.containers):
        b = model.body(item.name).id
        j = model.joint(f"{item.name}_joint").id
        h.item_body[item.name] = b
        h.item_qpos_adr[item.name] = int(model.jnt_qposadr[j])
        h.item_dof_adr[item.name] = int(model.jnt_dofadr[j])
        h.item_geoms[item.name] = frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] == b)
    return h


def _is_descendant(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = model.body_parentid[body]
    return body == root


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
