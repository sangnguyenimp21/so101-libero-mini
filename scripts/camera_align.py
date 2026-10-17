"""Camera alignment: move the real front camera until it sees the robot like the sim front camera does.

The window shows what the policy sees (the same crop + resize to 256x256 as scripts/real_*.sh) with the sim picture
faded on top. The CYAN line is the sim robot's outline: where the robot has to be in the image. The sim robot copies
your arm's pose (joint angles are read from the motors; torque is never touched, the arm does not move). The ORANGE
line is where the tool finds your real robot. Move/turn the camera until orange sits on cyan; the score and the
instructions at the top say how close you are and which way to go (100 % = same place and size).

Optional, more precise: print the marker sheet (`sheet` below; A4 at 100 %, its bar must measure 10 cm) and lay it
in front of the robot as printed on it. While 3+ of its squares are visible the score uses them instead.

    python scripts/camera_align.py live                  # front camera index 1 (FRONT_CAM in scripts/real_*.sh)
    python scripts/camera_align.py live --cam 1 --crop 0,80,480,480 --no-arm
    python scripts/camera_align.py sheet                 # -> outputs/camera_align/marker_sheet.pdf

Keys: q quit, + / - more / less sim picture, s save a snapshot to outputs/camera_align/.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from so101_libero.sim.scene import FRONT_CAMERA  # noqa: E402

OUT = ROOT / "outputs" / "camera_align"
SIZE = 256  # policy image size
ZOOM = 2.5  # on-screen scale of the policy view
FOCAL = (SIZE / 2) / math.tan(math.radians(FRONT_CAMERA.fovy) / 2)  # sim focal length in policy pixels
CAM_DIST = float(np.linalg.norm(np.subtract(FRONT_CAMERA.pos, FRONT_CAMERA.lookat)))  # m
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CYAN, ORANGE, GREEN, RED = (255, 255, 0), (0, 160, 255), (0, 220, 0), (60, 60, 255)

# Marker sheet, A4 landscape, page coordinates in mm (u right, v down as printed). It lies on the table readable
# from the camera: the top edge faces the robot, page right = image right = the robot's left (+y).
PAGE_W, PAGE_H = 297.0, 210.0
MARKER = 60.0
MARKER_CENTERS = {0: (48.5, 45.0), 1: (248.5, 45.0), 2: (48.5, 165.0), 3: (248.5, 165.0)}
PAN_AXIS_X = 0.0388353  # shoulder_pan axis in the world frame (so101.xml, body "shoulder")
CROSS_FROM_PAN = 0.175  # sheet centre (crosshair) in front of the pan axis, m
SHEET_X = PAN_AXIS_X + CROSS_FROM_PAN
DPI = 300


def score_from_px(px: float) -> float:
    return 100 * math.exp(-px / 20)  # 2 px -> 90 %, 10 px -> 61 %


# ------------------------------------------------------------------------------------------- sim side
class SimView:
    """Sim front-camera picture (BGR) and robot mask for joint angles in LeRobot units."""

    def __init__(self):
        import mujoco

        from so101_libero.sim.robot import HOME_QPOS, JOINT_NAMES, lerobot_to_sim, sim_to_lerobot
        from so101_libero.sim.scene import SceneConfig, build_spec

        if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):  # removed after MuJoCo 3.3 (collision only)
            mujoco.mjtEnableBit.mjENBL_MULTICCD = 0
        self.mj, self.to_sim = mujoco, lerobot_to_sim
        self.model = build_spec(SceneConfig(objects=(), containers=())).compile()
        self.data = mujoco.MjData(self.model)
        self.adr = [self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINT_NAMES]
        base = self.model.body("base").id
        self.robot_geoms = np.array([g for g in range(self.model.ngeom)
                                     if self.model.body_rootid[self.model.geom_bodyid[g]] == base])
        self.renderer = mujoco.Renderer(self.model, SIZE, SIZE)
        self.home = sim_to_lerobot(HOME_QPOS)
        self._key, self._out = None, None

    def render(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(np.round(joints).astype(int))
        if key != self._key:
            self.data.qpos[self.adr] = self.to_sim(np.asarray(joints, float))
            self.mj.mj_forward(self.model, self.data)
            r = self.renderer
            r.update_scene(self.data, camera="front")
            rgb = r.render().copy()
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera="front")
            seg = r.render().copy()
            r.disable_segmentation_rendering()
            mask = np.isin(seg[..., 0], self.robot_geoms) & (seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM))
            self._key, self._out = key, (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask.astype(np.uint8))
        return self._out


class ArmReader:
    """Present joint positions of the real follower in LeRobot units. Only reads; torque is never changed."""

    def __init__(self, port: str, robot_id: str):
        self.robot = self.bus = None
        self.warned = False
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

            # keep the robot object alive (its destructor disconnects) and never let a disconnect touch torque
            self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, disable_torque_on_disconnect=False))
            self.robot.bus.connect()
            self.bus = self.robot.bus
        except Exception as e:  # noqa: BLE001
            print(f"arm not read ({e}); the sim robot stays in its home pose: put your arm in the same pose")

    def read(self) -> np.ndarray | None:
        if self.bus is None:
            return None
        try:
            pos = self.bus.sync_read("Present_Position")
            return np.array([pos[m] for m in self.bus.motors], float)
        except Exception as e:  # noqa: BLE001
            if not self.warned:
                print(f"reading the arm failed ({e}); keeping the last pose")
                self.warned = True
            return None

    def close(self) -> None:
        if self.bus is not None:
            self.bus.disconnect(disable_torque=False)


# --------------------------------------------------------------------------------- robot outline match
# The sim robot's outline is slid/scaled over the real image to where the colour differs most between just
# outside and just inside the outline: robot on one side, background on the other. (Matching to edges instead
# lets the outline shrink into the robot's own internal edges.)
SIDE_PX = 2.5  # how far outside/inside the outline the colours are compared
FOUND = 4.0  # best contrast / typical contrast over the search: below this the robot is not found


def outline_samples(mask: np.ndarray, n: int = 200) -> tuple[np.ndarray, np.ndarray] | None:
    """Points on the robot's outline and their outward normals."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts, normals = [], []
    for c in contours:
        c = c.reshape(-1, 2).astype(np.float64)
        if len(c) < 20:
            continue
        t = np.roll(c, -2, 0) - np.roll(c, 2, 0)
        nrm = np.stack([t[:, 1], -t[:, 0]], 1) / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        probe = np.clip(np.rint(c + SIDE_PX * nrm).astype(int), 0, SIZE - 1)
        nrm[mask[probe[:, 1], probe[:, 0]] > 0] *= -1  # point the normals out of the robot
        pts.append(c)
        normals.append(nrm)
    if not pts:
        return None
    pts, normals = np.concatenate(pts), np.concatenate(normals)
    keep = slice(None, None, max(1, len(pts) // n))
    return pts[keep], normals[keep]


def contrast(lab: np.ndarray, pts: np.ndarray, normals: np.ndarray, side: float) -> np.ndarray:
    """Mean colour difference between `side` px outside and inside the outline, for point sets (..., N, 2)."""
    def sample(p):
        x, y = np.rint(p[..., 0]).astype(int), np.rint(p[..., 1]).astype(int)
        ok = (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
        return lab[np.clip(y, 0, SIZE - 1), np.clip(x, 0, SIZE - 1)], ok

    out, ok_out = sample(pts + side * normals)
    inn, ok_in = sample(pts - side * normals)
    return (np.linalg.norm(out - inn, axis=-1) * (ok_out & ok_in)).mean(-1)


MAX_SHIFT, MAX_SCALE = 40, 1.25  # px, x: the 2D fit only holds near the right place (perspective changes beyond)


def match_outline(samples: tuple[np.ndarray, np.ndarray], view: np.ndarray) -> dict:
    """Shift and scale (about the outline centre) that best put the sim outline on the real robot's outline.

    Only searched near the target: farther away the real silhouette's shape changes with perspective and a 2D fit
    is meaningless. `close` is False when the best fit is weak or at the edge of the search."""
    pts, normals = samples
    c = pts.mean(0)
    grid = lambda r, step: np.stack(np.meshgrid(*[np.arange(-r, r + 1e-9, step)] * 2), -1).reshape(-1, 2)  # noqa: E731
    lab = lambda k: cv2.cvtColor(cv2.GaussianBlur(view, (k, k), 0), cv2.COLOR_BGR2LAB).astype(np.float32)  # noqa: E731
    best = [0.0, 0.0, 0.0, 1.0]

    def search(img, side, scales, shifts):
        values, top = [], [-1.0, 0.0, 0.0, 1.0]
        for s in scales:
            v = contrast(img, c + (pts - c) * s + shifts[:, None], normals, side)
            values.append(v)
            k = int(v.argmax())
            if v[k] > top[0]:
                top = [float(v[k]), *map(float, shifts[k]), float(s)]
        best[:] = top
        return np.concatenate(values)

    coarse = search(lab(7), 4.0, np.exp(np.linspace(-math.log(MAX_SCALE), math.log(MAX_SCALE), 15)),
                    grid(MAX_SHIFT, 4))
    fit = best[0] / max(float(np.median(coarse)), 1e-6)
    search(lab(5), SIDE_PX, best[3] * np.exp(np.linspace(-0.02, 0.02, 5)), grid(3, 1) + best[1:3])
    _, dx, dy, s = best
    radius = float(np.linalg.norm(pts - c, axis=1).mean())
    edge = max(abs(dx), abs(dy)) >= MAX_SHIFT - 4 or abs(math.log(s)) >= math.log(MAX_SCALE) - 0.02
    return {"fit": fit, "close": fit >= FOUND and not edge, "dx": dx, "dy": dy, "scale": s, "centre": c,
            "px": math.hypot(dx, dy) + abs(s - 1) * radius}


def placement(vfov: float) -> list[str]:
    """Where to put a camera with this vertical field of view (of the cropped square) to frame like the sim camera."""
    pos, look = np.asarray(FRONT_CAMERA.pos, float), np.asarray(FRONT_CAMERA.lookat, float)
    k = math.tan(math.radians(FRONT_CAMERA.fovy) / 2) / math.tan(math.radians(vfov) / 2)
    cam = look + k * (pos - look)
    tilt = math.degrees(math.atan2(pos[2] - look[2], pos[0] - look[0]))
    return [f"1. camera ~{(cam[0] - PAN_AXIS_X) * 100:.0f} cm in front of the base axis, "
            f"~{cam[2] * 100:.0f} cm above the table",
            f"2. centred, tilted ~{tilt:.0f} deg down at the table ~{(look[0] - PAN_AXIS_X) * 100:.0f} cm "
            "in front of the base axis",
            "3. then move it until your robot sits on the cyan outline"]


def outline_hints(m: dict) -> list[tuple[float, str]]:
    """(size in px, instruction) for the camera, standing behind it facing the robot."""
    out = []
    radius = m["px"] - math.hypot(m["dx"], m["dy"])
    if abs(m["scale"] - 1) > 0.03:
        word = "BACK (farther)" if m["scale"] > 1 else "CLOSER"
        out.append((radius, f"move the camera {word} ~{abs(m['scale'] - 1) * CAM_DIST * 100:.0f} cm"))
    if abs(m["dx"]) > 2:
        out.append((abs(m["dx"]), f"turn the camera {'RIGHT' if m['dx'] > 0 else 'LEFT'} "
                                  f"~{math.degrees(math.atan(abs(m['dx']) / FOCAL)):.0f} deg"))
    if abs(m["dy"]) > 2:
        out.append((abs(m["dy"]), f"tilt the camera {'DOWN' if m['dy'] > 0 else 'UP'} "
                                  f"~{math.degrees(math.atan(abs(m['dy']) / FOCAL)):.0f} deg"))
    return out


# -------------------------------------------------------------------------------- optional marker sheet
def page_to_world(u: float, v: float) -> tuple[float, float, float]:
    return SHEET_X + (v - PAGE_H / 2) / 1000, (u - PAGE_W / 2) / 1000, 0.0


def marker_corners_world() -> dict[int, np.ndarray]:
    """(4, 3) world corners of each marker, in ArUco order (TL, TR, BR, BL of the printed marker)."""
    h = MARKER / 2
    return {i: np.array([page_to_world(u + du, v + dv) for du, dv in ((-h, -h), (h, -h), (h, h), (-h, h))])
            for i, (u, v) in MARKER_CENTERS.items()}


def sim_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV intrinsics in the policy image, and the world->camera rotation and translation of the sim camera."""
    K = np.array([[FOCAL, 0, (SIZE - 1) / 2], [0, FOCAL, (SIZE - 1) / 2], [0, 0, 1]])
    pos = np.asarray(FRONT_CAMERA.pos, float)
    fwd = np.asarray(FRONT_CAMERA.lookat, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])  # same frame as scene.look_at_quat, with OpenCV's y down / z forward
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    return K, R, -R @ pos


def make_sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    px = lambda mm: round(mm * DPI / 25.4)  # noqa: E731
    page = Image.new("RGB", (px(PAGE_W), px(PAGE_H)), "white")
    for i, (u, v) in MARKER_CENTERS.items():
        marker = cv2.aruco.generateImageMarker(ARUCO, i, px(MARKER))
        page.paste(Image.fromarray(marker).convert("RGB"), (px(u - MARKER / 2), px(v - MARKER / 2)))
    d = ImageDraw.Draw(page)
    font, small = ImageFont.load_default(size=px(4.2)), ImageFont.load_default(size=px(3.2))
    cu, cv_ = PAGE_W / 2, PAGE_H / 2
    w = px(0.4)
    d.line([(px(cu), px(4)), (px(cu), px(cv_ + 12))], fill="black", width=w)  # centre line, towards the robot
    d.line([(px(cu - 12), px(cv_)), (px(cu + 12), px(cv_))], fill="black", width=w)
    d.polygon([(px(cu), px(4)), (px(cu - 3), px(10)), (px(cu + 3), px(10))], fill="black")
    d.text((px(cu + 3), px(14)), "ROBOT SIDE", font=font, fill="black")
    d.text((px(cu + 3), px(21)), f"base rotation axis {CROSS_FROM_PAN * 100:.1f} cm", font=small, fill="black")
    d.text((px(cu + 3), px(26)), "from the crosshair, on this line", font=small, fill="black")
    d.line([(px(cu - 50), px(150)), (px(cu + 50), px(150))], fill="black", width=px(0.8))
    for x in (cu - 50, cu + 50):
        d.line([(px(x), px(146)), (px(x), px(154))], fill="black", width=w)
    d.text((px(cu - 50), px(140)), "this bar must measure 10.0 cm", font=small, fill="black")
    d.text((px(cu - 20), px(192)), "CAMERA SIDE", font=font, fill="black")
    OUT.mkdir(parents=True, exist_ok=True)
    page.save(OUT / "marker_sheet.png")
    page.save(OUT / "marker_sheet.pdf", resolution=DPI)
    return OUT / "marker_sheet.pdf"


class Markers:
    """Marker-sheet corners in the policy image vs where the sim camera sees them, plus a camera pose estimate."""

    def __init__(self, to_policy):
        self.to_policy = to_policy
        self.K, self.R, self.t = sim_camera()
        self.pos = -self.R.T @ self.t
        self.rvec = cv2.Rodrigues(self.R)[0]
        self.world = marker_corners_world()
        self.target = {i: cv2.projectPoints(p, self.rvec, self.t, self.K, None)[0].reshape(4, 2)
                       for i, p in self.world.items()}
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(ARUCO, params)

    def measure(self, frame: np.ndarray) -> dict | None:
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = {int(i): self.to_policy(c.reshape(4, 2).astype(np.float64))
                 for c, i in zip(corners, ids.ravel() if ids is not None else []) if int(i) in self.target}
        if len(found) < 3:
            return None
        err = float(np.concatenate([np.linalg.norm(p - self.target[i], axis=1) for i, p in found.items()]).mean())
        obj, img = np.concatenate([self.world[i] for i in found]), np.concatenate(list(found.values()))
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, None, self.rvec.copy(), self.t.reshape(3, 1).copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        hints = self.hints(cv2.Rodrigues(rvec)[0], tvec.ravel()) if ok else []
        return {"found": found, "px": err, "score": score_from_px(err) * len(found) / 4, "hints": hints}

    def hints(self, R: np.ndarray, t: np.ndarray) -> list[tuple[float, str]]:
        move = (self.pos - (-R.T @ t)) * 100  # cm; world +x = away from the robot, +y = right seen from the camera
        azimuth = lambda f: math.degrees(math.atan2(f[1], f[0]))  # noqa: E731
        roll_of = lambda R: math.degrees(math.atan2(R[0][2], -R[1][2]))  # noqa: E731
        turns = ((azimuth(R[2]) - azimuth(self.R[2]) + 180) % 360 - 180,  # > 0: turn right
                 math.degrees(math.asin(R[2][2]) - math.asin(self.R[2][2])),  # > 0: tilt down
                 roll_of(R) - roll_of(self.R))  # > 0: roll clockwise (seen from behind the camera)
        out = []
        for v, pos_word, neg_word in zip(move, ("BACK", "RIGHT", "UP"), ("CLOSER", "LEFT", "DOWN")):
            if abs(v) >= 0.5:
                out.append((abs(v) * 4, f"move the camera {pos_word if v > 0 else neg_word} {abs(v):.1f} cm"))
        for v, pos_word, neg_word in zip(turns, ("turn RIGHT", "tilt DOWN", "roll CLOCKWISE"),
                                         ("turn LEFT", "tilt UP", "roll ANTI-CLOCKWISE")):
            if abs(v) >= 0.5:
                out.append((abs(v) * 4.6, f"{pos_word if v > 0 else neg_word} {abs(v):.1f} deg"))
        return out


# ------------------------------------------------------------------------------------------------ live
def draw_outline(img: np.ndarray, pts: np.ndarray, color, thickness: int) -> None:
    p = np.round((pts + 0.5) * ZOOM - 0.5).astype(np.int32)
    cv2.polylines(img, [p.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def compose(view, sim, mask, alpha, m, markers, score, instructions, mode) -> np.ndarray:
    big = int(SIZE * ZOOM)
    img = cv2.resize(cv2.addWeighted(view, 1 - alpha, sim, alpha, 0), (big, big), interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        draw_outline(img, np.vstack([c.reshape(-1, 2), c[:1].reshape(-1, 2)]).astype(float), CYAN, 2)
    if m is not None and m["close"]:
        for c in contours:
            p = c.reshape(-1, 2).astype(float)
            draw_outline(img, m["centre"] + (p - m["centre"]) * m["scale"] + (m["dx"], m["dy"]), ORANGE, 1)
    if markers is not None:
        for target, found in ((markers["target"], RED), (markers["found"], GREEN)):
            for q in target.values():
                draw_outline(img, np.vstack([q, q[:1]]), found, 2)
    def text(canvas, s, org, scale, color, thick=1):  # shrinks the text to fit the width
        width = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        scale *= min(1.0, (big - org[0] - 10) / max(width, 1))
        cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    head = np.full((175, big, 3), 30, np.uint8)
    color = (170, 170, 170) if score is None else GREEN if score >= 90 else (0, 220, 255) if score >= 70 else RED
    text(head, "--%" if score is None else f"{score:3.0f}%", (10, 62), 2.0, color, 4)
    text(head, mode, (190, 40), 0.6, (170, 170, 170))
    for k, line in enumerate(instructions[:3]):
        text(head, line, (12, 100 + 30 * k), 0.7, (255, 255, 255), 2)
    foot = np.full((30, big, 3), 30, np.uint8)
    text(foot, "cyan = where the robot must be   orange = where your robot is   +/- sim picture   s save   q quit",
         (8, 20), 0.45, (200, 200, 200))
    return np.vstack([head, img, foot])


def live(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read camera {args.cam} (check the index with `lerobot-find-cameras opencv`)")
    if np.abs(np.diff(frame.astype(np.int16), axis=2)).mean() < 1:
        print("WARNING: this camera gives grey images (e.g. the D435 infrared stream); use its colour camera")
    H, W = frame.shape[:2]
    side = min(H, W)
    top, left, h, w = args.crop or ((H - side) // 2, (W - side) // 2, side, side)
    print(f"frame {W}x{H}, crop top,left,h,w = {top},{left},{h},{w} -> {SIZE}x{SIZE}")

    def to_policy(pts):  # full-frame pixels -> policy image (crop, resize; pixel centres at integers)
        return np.stack([(pts[:, 0] - left + 0.5) * SIZE / w - 0.5, (pts[:, 1] - top + 0.5) * SIZE / h - 0.5], 1)

    guide = placement(args.vfov)
    print("rough placement (camera with a {:.0f} deg vertical view):\n  ".format(args.vfov) + "\n  ".join(guide))
    sim = SimView()
    arm = None if args.no_arm else ArmReader(args.port, args.robot_id)
    markers = Markers(to_policy)
    joints, alpha, score, last_arm, last_print = sim.home, 0.35, None, 0.0, 0.0
    try:
        while True:
            view = cv2.resize(frame[top:top + h, left:left + w], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            if arm is not None and time.time() - last_arm > 0.2:
                q = arm.read()
                joints = joints if q is None else q
                last_arm = time.time()
            sim_img, mask = sim.render(joints)
            mk = markers.measure(frame)
            m = None
            if mk is not None:
                mode, cur, hints = "matching: marker sheet (precise)", mk["score"], mk["hints"]
                mk["target"] = markers.target
            else:
                samples = outline_samples(mask)
                m = match_outline(samples, view) if samples is not None else None
                mode = "matching: robot outline"
                if m is None or not m["close"]:
                    mode, cur, hints = "not close yet: place the camera roughly", None, [(3 - k, t) for k, t in
                                                                                        enumerate(guide)]
                else:
                    cur, hints = score_from_px(m["px"]), outline_hints(m)
            score = cur if cur is None or score is None else 0.7 * score + 0.3 * cur
            instructions = [t for _, t in sorted(hints, reverse=True)] or ["aligned: keep the camera here"]
            if time.time() - last_print > 1.0:
                fit = "" if m is None else f"  (fit {m['fit']:.1f})"
                print(("--" if score is None else f"{score:3.0f}") + "%  " + " | ".join(instructions[:3]) + fit,
                      flush=True)
                last_print = time.time()
            shown = compose(view, sim_img, mask, alpha, m, mk, score, instructions, mode)
            cv2.imshow("camera alignment", shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("+"), ord("=")):
                alpha = min(0.9, alpha + 0.1)
            if key in (ord("-"), ord("_")):
                alpha = max(0.0, alpha - 0.1)
            if key == ord("s"):
                OUT.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(OUT / f"frame_{stamp}.png"), frame)
                cv2.imwrite(str(OUT / f"view_{stamp}.png"), shown)
                print(f"saved {OUT}/frame_{stamp}.png and view_{stamp}.png")
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("camera stopped")
    finally:
        cap.release()
        if arm is not None:
            arm.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sheet", help="write the printable marker sheet (optional, for a precise score)")
    p = sub.add_parser("live", help="live alignment view")
    p.add_argument("--cam", type=int, default=1, help="OpenCV camera index (FRONT_CAM in scripts/real_*.sh)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--crop", type=lambda s: [int(v) for v in s.split(",")], metavar="TOP,LEFT,H,W",
                   help="same as FRONT_CROP in scripts/real_*.sh (default: centre square)")
    p.add_argument("--port", default="/dev/tty.usbmodem5B610366661", help="follower port (to read the arm pose)")
    p.add_argument("--robot-id", default="my_follower")
    p.add_argument("--no-arm", action="store_true", help="do not read the arm; the sim robot stays at home pose")
    p.add_argument("--vfov", type=float, default=42.0,
                   help="vertical field of view of your camera in degrees, for the rough placement (D435 colour: 42)")
    args = parser.parse_args()
    if args.cmd == "sheet":
        print(f"print at 100 % (no fit-to-page): {make_sheet()}")
    else:
        live(args)


if __name__ == "__main__":
    main()
