"""Locate the three coloured cubes on the table in the FRONT camera image (real-robot completion checks).

Colour segmentation in HSV (one hue band per cube colour, high saturation, so the beige table, the dark arm and grey
shadows are rejected), the largest connected blob per colour, and its centroid back-projected onto a horizontal plane
with the calibrated front camera (sim/cameras.py FRONT_CAMERA = the measured real camera; the policy image is the centre
square of the camera frame, which this camera model describes at any square size). A blob's centroid is the centre of
the visible top and front faces, which lies ``height`` above the table on average: back-projecting at that height
gives the cube's centre with ~1 cm error (measured in simulation by ``python -m so101_libero.control.completion.vision``).

Limits (document them on the real setup): thresholds are tuned for the simulated colours under its lighting and must be
checked on real frames (``--image`` below prints what is found); a cube hidden behind the arm is not found; one camera
cannot measure height (``height`` is assumed: table level, or ``stack_height`` for a cube on top of two others).

    python -m so101_libero.control.completion.vision --image frame.png       # what is found in a saved front image
    python -m so101_libero.control.completion.vision --sim-check             # accuracy against simulator ground truth
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from so101_libero.sim.cameras import FRONT_CAMERA, CameraSpec, back_project, project

# HSV bands (hue in degrees 0-360, saturation and value in 0-1) per cube.
DEFAULT_BANDS = {
    "red_cube": {"hue": [(0.0, 15.0), (340.0, 360.0)], "s": 0.45, "v": 0.20},
    "yellow_cube": {"hue": [(35.0, 70.0)], "s": 0.45, "v": 0.30},
    "green_cube": {"hue": [(85.0, 165.0)], "s": 0.35, "v": 0.12},
}


def rgb_to_hsv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 RGB (H, W, 3) -> hue in degrees, saturation and value in [0, 1]."""
    x = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(-1)
    c = v - x.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0.0)
    cz = np.maximum(c, 1e-6)
    h = np.where(v == r, ((g - b) / cz) % 6.0, np.where(v == g, (b - r) / cz + 2.0, (r - g) / cz + 4.0)) * 60.0
    return np.where(c > 0, h, 0.0), s, v


def largest_blob(mask: np.ndarray) -> np.ndarray | None:
    """Boolean mask of the largest 8-connected component (None if the mask is empty)."""
    if not mask.any():
        return None
    try:
        import cv2

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == k
    except ImportError:  # small images: a plain flood fill is fast enough
        seen = np.zeros_like(mask, dtype=bool)
        best = None
        for start in zip(*np.nonzero(mask)):
            if seen[start]:
                continue
            stack, comp = [start], []
            seen[start] = True
            while stack:
                i, j = stack.pop()
                comp.append((i, j))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not seen[a, b]:
                            seen[a, b] = True
                            stack.append((a, b))
            if best is None or len(comp) > len(best):
                best = comp
        out = np.zeros_like(mask, dtype=bool)
        out[tuple(np.array(best).T)] = True
        return out


@dataclass
class ColorCubeLocator:
    camera: CameraSpec = FRONT_CAMERA
    cube_size: float = 0.03
    bands: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_BANDS.items()})
    height: float = 0.0156  # effective height of a table cube's blob centroid (simulation: 1.56 +- 0.16 cm)
    min_area_frac: float = 2.5e-4  # smallest blob, as a fraction of the image area (16 px at 256 x 256)

    @property
    def stack_height(self) -> float:
        """Effective blob height of a cube resting on top of two others."""
        return self.height + self.cube_size

    def masks(self, image: np.ndarray) -> dict[str, np.ndarray]:
        h, s, v = rgb_to_hsv(image)
        out = {}
        for name, band in self.bands.items():
            hue = np.zeros(h.shape, dtype=bool)
            for lo, hi in band["hue"]:
                hue |= (h >= lo) & (h <= hi)
            out[name] = hue & (s >= band["s"]) & (v >= band["v"])
        return out

    def locate(self, image: np.ndarray, height: float | None = None) -> dict[str, dict]:
        """{cube: {"uv": centroid pixel, "xy": table position (m), "area": pixels}} for every cube found."""
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[0] != img.shape[1]:
            raise ValueError(f"expected the square policy view of the front camera, got {img.shape}")
        size = img.shape[0]
        min_area = self.min_area_frac * size * size
        found = {}
        for name, mask in self.masks(img).items():
            blob = largest_blob(mask)
            if blob is None or blob.sum() < min_area:
                continue
            vv, uu = np.nonzero(blob)
            uv = np.array([uu.mean() + 0.5, vv.mean() + 0.5])
            p = back_project(self.camera, uv, size, self.height if height is None else height)
            found[name] = {"uv": uv, "xy": p[:2], "area": float(blob.sum()) * (256.0 / size) ** 2}
        return found

    def expected_area(self, xy, height: float, size: int = 256) -> float:
        """Pixel area (at 256 x 256) of an unhidden cube whose blob centroid lies at ``xy`` / ``height``: the convex hull
        of its projected corners."""
        z0 = height - self.height  # the cube's bottom
        half = self.cube_size / 2
        corners = np.array([[xy[0] + sx * half, xy[1] + sy * half, z0 + sz * self.cube_size]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
        return _hull_area(project(self.camera, corners, size)) * 0.85  # edges and shading lose a few pixels


def _hull_area(pts: np.ndarray) -> float:
    pts = sorted(map(tuple, np.asarray(pts, float)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _sim_check() -> None:
    """Accuracy of the locator on rendered start states of every three-cube placement (ground truth = simulator)."""
    import so101_libero  # noqa: F401
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.tasks.three_cubes import THREE_CUBE_TASKS

    loc = ColorCubeLocator()
    errs, missing = [], 0
    for task in THREE_CUBE_TASKS:
        env = SO101TabletopEnv(task, EnvConfig(image_size=(256, 256)))
        for p in range(task.layout.num_placements):
            obs, _ = env.reset(options={"placement_id": p})
            found = loc.locate(obs["observation.images.front"])
            for name in ("red_cube", "yellow_cube", "green_cube"):
                if name not in found:
                    missing += 1
                    continue
                errs.append(float(np.linalg.norm(found[name]["xy"] - env.world.pos(name)[:2])))
        env.close()
    e = np.array(errs)
    print(f"located {len(e)} cubes ({missing} missed): position error mean {100 * e.mean():.2f} cm, "
          f"max {100 * e.max():.2f} cm")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="a saved front-camera image (square policy view, or a frame: its centre square)")
    ap.add_argument("--sim-check", action="store_true")
    args = ap.parse_args()
    if args.sim_check:
        _sim_check()
    if args.image:
        from PIL import Image

        img = np.asarray(Image.open(args.image).convert("RGB"))
        s = min(img.shape[:2])
        top, left = (img.shape[0] - s) // 2, (img.shape[1] - s) // 2
        for name, f in ColorCubeLocator().locate(img[top:top + s, left:left + s]).items():
            print(f"{name}: pixel {np.round(f['uv'], 1).tolist()}, table xy {np.round(f['xy'], 3).tolist()} m, "
                  f"area {f['area']:.0f} px")


if __name__ == "__main__":
    main()
