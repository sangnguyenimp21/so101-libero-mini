"""Camera specifications and the pinhole model of the fixed cameras, without MuJoCo (also used on the real-robot
machine: so101_libero/control/completion/vision.py projects table points into the front image with it).

``sim/scene.py`` builds the cameras from these specs; ``camera_rotation`` reproduces MuJoCo's camera frame for them
(x right, y up, looking along -z), checked against the compiled model in tests/test_three_cubes.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    """A fixed world camera defined by position, look-at target and vertical field of view (degrees)."""

    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy: float = 45.0
    roll: float = 0.0  # degrees about the viewing axis, applied after the look-at orientation


# The real front camera (docs/real_setup/PROMPT.md, measured on the physical setup): 67 cm from the base, looking
# down ~56 deg at the base from the front, so the robot stands in the upper middle of the square image and the
# workspace fills its lower part. The real policy image is the centre square of the camera frame resized to
# 256x256, which this camera renders directly (fovy = vertical field of view of that square).
FRONT_CAMERA = CameraSpec("front", pos=(0.374, 0.009, 0.557), lookat=(0.005, 0.003, 0.0), fovy=45.0, roll=2.3)


def camera_rotation(cam: CameraSpec, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Columns = the camera's x (right), y (up) and z (backward) axes in the world: look-at, then ``roll``."""
    f = np.asarray(cam.lookat, float) - np.asarray(cam.pos, float)
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z], axis=1)
    if cam.roll:
        c, s = math.cos(math.radians(cam.roll)), math.sin(math.radians(cam.roll))
        rot = rot @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot


def project(cam: CameraSpec, pts: np.ndarray, size: int) -> np.ndarray:
    """Pixel coordinates (u right, v down) of world points (n, 3) in a square ``size`` image of ``cam``."""
    rot = camera_rotation(cam)
    f = (size / 2) / math.tan(math.radians(cam.fovy) / 2)
    pc = (np.atleast_2d(pts) - np.asarray(cam.pos)) @ rot
    return np.stack([size / 2 + f * pc[:, 0] / -pc[:, 2], size / 2 - f * pc[:, 1] / -pc[:, 2]], axis=1)


def back_project(cam: CameraSpec, uv, size: int, height: float) -> np.ndarray:
    """World point (3,) on the horizontal plane z = ``height`` seen at pixel ``uv`` of a square ``size`` image."""
    rot = camera_rotation(cam)
    f = (size / 2) / math.tan(math.radians(cam.fovy) / 2)
    d = rot @ np.array([(uv[0] - size / 2) / f, -(uv[1] - size / 2) / f, -1.0])
    t = (height - cam.pos[2]) / d[2]
    return np.asarray(cam.pos) + t * d
