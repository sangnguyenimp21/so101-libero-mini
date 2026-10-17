#!/usr/bin/env python
"""Check the two real cameras and what the policy will see from them.

Opens the front and the wrist camera with the same OpenCVCameraConfig settings as scripts/real/run_policy.py, grabs
N frames from each, applies the same centre-square crop + resize (run_policy.center_square_resize) and writes
  <out>/front.png, <out>/wrist.png          the 256x256 policy images (RGB)
  <out>/front_raw.png, <out>/wrist_raw.png  the full frames with the crop square drawn
and, with --sim-reference <256x256 sim render of the home pose> (e.g. outputs/sim_front_home.png),
  <out>/front_vs_sim.png                    50/50 blend of the real policy image and the sim image
  <out>/front_sim_side_by_side.png          real | sim (256x512)
so the real front camera can be moved until the robot and the table edge coincide with the simulator's view. Also
prints each camera's frame size, the measured capture rate and the mean brightness of the policy image.

    python scripts/real/check_cameras.py --front-camera 3 --front-size 1280x720 --wrist-camera 0 --wrist-size 640x480 \\
        --out outputs/real_check --sim-reference outputs/sim_front_home.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_policy import IMAGE_SIZE, camera_ref, center_square_resize, parse_size  # noqa: E402  (same crop as the policy loop)


def compose_outputs(front: np.ndarray, wrist: np.ndarray, sim_reference: np.ndarray | None = None,
                    size: int = IMAGE_SIZE) -> dict[str, np.ndarray]:
    """All images to save, from the raw camera frames (and the optional sim reference image)."""
    out = {"front": center_square_resize(front, size), "wrist": center_square_resize(wrist, size),
           "front_raw": draw_crop_square(front), "wrist_raw": draw_crop_square(wrist)}
    if sim_reference is not None:
        sim = center_square_resize(sim_reference, size)
        out["front_vs_sim"] = ((out["front"].astype(np.float32) + sim.astype(np.float32)) / 2).round().astype(np.uint8)
        out["front_sim_side_by_side"] = np.concatenate([out["front"], sim], axis=1)
    return out


def draw_crop_square(frame: np.ndarray, color=(255, 255, 0), thickness: int = 2) -> np.ndarray:
    """A copy of the full frame with the centre square (what the policy sees) outlined."""
    img = np.array(frame, copy=True)
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    t = thickness
    img[top:top + t, left:left + s] = color
    img[top + s - t:top + s, left:left + s] = color
    img[top:top + s, left:left + t] = color
    img[top:top + s, left + s - t:left + s] = color
    return img


def grab(index_or_path, width: int, height: int, fps: int, n_frames: int) -> tuple[np.ndarray, float]:
    """Open an OpenCV camera like the policy loop does, read ``n_frames`` and return (last RGB frame, measured fps)."""
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index_or_path, fps=fps, width=width, height=height))
    cam.connect(warmup=True)  # sets and verifies width/height/fps, reads frames for warmup_s = 1 s
    try:
        t0 = time.perf_counter()
        for _ in range(n_frames):
            frame = cam.read()  # HxWx3 uint8, RGB (OpenCVCameraConfig.color_mode defaults to RGB)
        measured = n_frames / max(time.perf_counter() - t0, 1e-9)
    finally:
        cam.disconnect()
    return frame, measured


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--front-camera", default="3", help="OpenCV index or /dev/video path (default 3: the iPhone "
                   "Continuity Camera; check with `lerobot-find-cameras opencv`)")
    p.add_argument("--wrist-camera", default="0", help="OpenCV index or /dev/video path (default 0)")
    p.add_argument("--front-size", type=parse_size, default="1280x720", metavar="WxH",
                   help="frame size the front camera is opened with (default 1280x720)")
    p.add_argument("--wrist-size", type=parse_size, default="640x480", metavar="WxH",
                   help="frame size the wrist camera is opened with (default 640x480)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=10, help="frames to grab per camera (the last one is saved)")
    p.add_argument("--out", default="outputs/real_check", help="output directory")
    p.add_argument("--sim-reference", default=None, metavar="PNG",
                   help="256x256 sim render of the home pose (e.g. outputs/sim_front_home.png)")
    args = p.parse_args(argv)

    from PIL import Image

    sim = np.asarray(Image.open(args.sim_reference).convert("RGB")) if args.sim_reference else None
    frames = {}
    for name, ref, (w, h) in (("front", args.front_camera, args.front_size), ("wrist", args.wrist_camera, args.wrist_size)):
        frame, measured = grab(camera_ref(ref), w, h, args.fps, args.frames)
        frames[name] = frame
        print(f"{name} camera {ref}: frame {frame.shape[1]}x{frame.shape[0]} (w x h), {measured:.1f} fps measured "
              f"over {args.frames} frames, policy image mean brightness {center_square_resize(frame).mean():.0f}/255")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in compose_outputs(frames["front"], frames["wrist"], sim).items():
        Image.fromarray(img).save(out / f"{name}.png")
        print(f"wrote {out / f'{name}.png'} ({img.shape[1]}x{img.shape[0]})")
    if sim is None:
        print("(no --sim-reference: the blend / side-by-side images were not written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
