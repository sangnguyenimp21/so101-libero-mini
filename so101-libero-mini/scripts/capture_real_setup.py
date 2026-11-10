"""Capture reference images of the real setup (front iPhone + wrist camera) together with the arm's joint angles.

They are the input for rebuilding the sim to match the real scene (front camera pose and field of view, black
robot, table, objects): see docs/real_setup/PROMPT.md. The arm is only read, torque is never touched; with torque
off, move the arm by hand between captures. The front preview draws the centre square the policy sees (cropped and
resized to 256x256, as in scripts/real_*.sh): keep the whole workspace inside it.

    python scripts/capture_real_setup.py                  # SPACE capture, q quit
    python scripts/capture_real_setup.py --front 3 --wrist 0 --out docs/real_setup
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from camera_align import ArmReader

ROOT = Path(__file__).resolve().parents[1]
SIZE = 256
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def open_camera(index: int, w: int, h: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    t = time.time()
    while time.time() - t < 1.5:  # let exposure settle
        ok, frame = cap.read()
    if not ok or frame.shape[:2] != (h, w):
        raise SystemExit(f"camera {index}: no {w}x{h} frames (got {None if not ok else frame.shape[1::-1]}); "
                         "check the index with `lerobot-find-cameras opencv`")
    return cap


def centre_square(w: int, h: int) -> tuple[int, int, int, int]:
    s = min(w, h)
    return (h - s) // 2, (w - s) // 2, s, s  # top, left, height, width


def policy_view(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    top, left, h, w = box
    return cv2.resize(frame[top:top + h, left:left + w], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--front", type=int, default=3, help="front camera index (iPhone)")
    parser.add_argument("--wrist", type=int, default=0, help="wrist camera index")
    parser.add_argument("--front-size", default="1280x720")
    parser.add_argument("--wrist-size", default="640x480")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "real_setup")
    parser.add_argument("--port", default="/dev/tty.usbmodem5B610366661")
    parser.add_argument("--robot-id", default="my_follower")
    args = parser.parse_args()
    fw, fh = map(int, args.front_size.split("x"))
    ww, wh = map(int, args.wrist_size.split("x"))
    front, wrist = open_camera(args.front, fw, fh), open_camera(args.wrist, ww, wh)
    fbox, wbox = centre_square(fw, fh), centre_square(ww, wh)
    arm = ArmReader(args.port, args.robot_id)
    args.out.mkdir(parents=True, exist_ok=True)
    calib = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{args.robot_id}.json"
    (args.out / "setup.json").write_text(json.dumps({
        "front": {"camera": "iPhone (Continuity Camera)", "index": args.front, "size": [fw, fh],
                  "crop_top_left_h_w": fbox, "policy_size": [SIZE, SIZE]},
        "wrist": {"camera": "USB2.0_CAM1 on the gripper", "index": args.wrist, "size": [ww, wh],
                  "crop_top_left_h_w": wbox, "policy_size": [SIZE, SIZE]},
        "joint_units": "LeRobot so101_follower, use_degrees=True: degrees for the 5 arm joints, gripper 0..100",
        "joint_order": JOINTS,
        "robot_calibration": json.loads(calib.read_text()) if calib.exists() else None,
    }, indent=2) + "\n")
    poses_file = args.out / "poses.json"
    poses = json.loads(poses_file.read_text()) if poses_file.exists() else []
    print(f"saving to {args.out} ({len(poses)} captures already there); SPACE capture, q quit")
    try:
        while True:
            ok1, f = front.read()
            ok2, w = wrist.read()
            if not (ok1 and ok2):
                raise SystemExit("a camera stopped")
            preview = f.copy()
            top, left, h, wd = fbox
            cv2.rectangle(preview, (left, top), (left + wd - 1, top + h - 1), (0, 255, 255), 3)
            preview = cv2.resize(preview, (960, 540))
            side = cv2.resize(w, (360, 270))
            side = np.vstack([side, np.full((270, 360, 3), 30, np.uint8)])
            cv2.putText(side, f"captures: {len(poses)}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(side, "SPACE capture   q quit", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(side, "yellow box = what the", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(side, "policy sees (front)", (10, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("capture real setup", np.hstack([preview, side]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                q = arm.read()
                k = len(poses)
                name = f"pose_{k:02d}"
                cv2.imwrite(str(args.out / f"{name}_front.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(args.out / f"{name}_front_256.png"), policy_view(f, fbox))
                cv2.imwrite(str(args.out / f"{name}_wrist.jpg"), w, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(args.out / f"{name}_wrist_256.png"), policy_view(w, wbox))
                poses.append({"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "joints": None if q is None else dict(zip(JOINTS, np.round(q, 2).tolist()))})
                poses_file.write_text(json.dumps(poses, indent=2) + "\n")
                print(f"saved {name}: joints {poses[-1]['joints']}")
    finally:
        front.release()
        wrist.release()
        arm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
