#!/usr/bin/env python
"""Height check: does the real arm put its jaws where the simulator says it does?

Reads the real follower's joints (nothing is sent; torque is left as it is) and prints, live, how far above the table
the simulator puts the lowest point of the jaws at those joint angles. Move the arm by hand (torque off), measure the
real height of the lowest jaw point with a ruler, and record it:

  ENTER            the jaw tips touch the table (real height 0)
  <number> ENTER   the real lowest jaw point is that many mm above the table (e.g. 17)
  q ENTER          finish (Ctrl-C too)

Record a varied set: the rest pose, 4-6 touches with the gripper closed and pointing down across the cube's start region
(14-24 cm in front of the base, |y| <= 8 cm), and 1-2 poses a few cm above the table. Every record goes to
outputs/real/height_check/<time>.csv. sim - real < 0 means the REAL jaws are higher than in the simulator for the same
joint angles (the policy's grasp, 1-5 mm above the table in the simulator, then ends up above the cube or misses).

The summary fits joint offsets that explain the differences, in scripts/real/run_policy.py's --joint-offset convention
(policy = real - offset): shoulder_lift alone, elbow_flex alone, and both together (from 3 records on).

  python scripts/real/check_height.py                              # defaults: port and robot id as run_policy.py
  python scripts/real/check_height.py --robot-id home_follower2 --port /dev/tty.usbmodem5C821064861
  python scripts/real/check_height.py --record 17 --out outputs/real/height_check/s1.csv   # one pose, no terminal
  python scripts/real/check_height.py --fit --out outputs/real/height_check/s1.csv         # fit only, no hardware
"""

from __future__ import annotations

import argparse
import csv
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import so101_libero  # noqa: E402,F401  (offscreen-GL setup before mujoco)
import mujoco  # noqa: E402

# MuJoCo >= 3.4 (unpinned Mac envs) dropped this flag: multi-CCD is on by default there. Kinematics only here.
if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    setattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD", 0)

from so101_libero.sim.robot import JOINT_NAMES, lerobot_to_sim  # noqa: E402
from so101_libero.sim.scene import build_scene  # noqa: E402
from so101_libero.tasks.registry import get_task  # noqa: E402

FIT_JOINTS = ("shoulder_lift", "elbow_flex")


class JawHeight:
    """Lowest point of the jaws' collision geoms above the table (z = 0), for joint angles in LeRobot units."""

    def __init__(self):
        self.h = build_scene(get_task("red_cube_aside").layout.scene_config())  # same robot in every layout
        self.m, self.d = self.h.model, mujoco.MjData(self.h.model)
        jaw = (self.h.gripper_body, self.h.moving_jaw_body)
        self.geoms = [g for g in range(self.m.ngeom)
                      if self.m.geom_bodyid[g] in jaw and self.m.geom_contype[g] | self.m.geom_conaffinity[g]]

    def __call__(self, q: np.ndarray) -> tuple[float, float, float, float]:
        """-> (lowest jaw point z, TCP x, y, z), all in mm."""
        m, d = self.m, self.d
        d.qpos[self.h.joint_qpos_adr] = lerobot_to_sim(np.asarray(q, dtype=np.float64))
        mujoco.mj_kinematics(m, d)
        z = np.inf
        for g in self.geoms:
            p, R, s, t = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g], m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                z = min(z, p[2] - s[0])
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = min(z, p[2] - abs(R[2, 2]) * s[1] - s[0])
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * s
                z = min(z, (p + corners @ R.T)[:, 2].min())
            elif t == mujoco.mjtGeom.mjGEOM_MESH:
                i = m.geom_dataid[g]
                verts = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]
                z = min(z, (p + verts @ R.T)[:, 2].min())
        x, y, tz = d.site_xpos[self.h.tcp_site] * 1e3
        return float(z) * 1e3, float(x), float(y), float(tz)

    def slope(self, q: np.ndarray, joint: str) -> float:
        """d(jaw height)/d(joint) in mm per degree (central difference)."""
        e = np.zeros(6)
        e[JOINT_NAMES.index(joint)] = 0.5
        return self(q + e)[0] - self(q - e)[0]


def fit_offsets(jaw: JawHeight, records: list[dict]) -> None:
    """Print the joint offsets that best explain the records: least squares on FK(q_real - offset) = z_real, solved by
    Gauss-Newton (offsets of ~10 deg are too large for a single linearization)."""
    Q = np.array([r["q"] for r in records])
    real = np.array([r["real_mm"] for r in records])
    candidates = [(j,) for j in FIT_JOINTS] + ([FIT_JOINTS] if len(records) >= 3 else [])
    for joints in candidates:
        idx = [JOINT_NAMES.index(j) for j in joints]
        off = np.zeros(len(joints))
        for _ in range(20):
            shift = np.zeros(6)
            shift[idx] = off
            resid = np.array([jaw(q - shift)[0] for q in Q]) - real
            A = np.array([[jaw.slope(q - shift, j) for j in joints] for q in Q])  # d resid / d offset = slope
            step, *_ = np.linalg.lstsq(A, resid, rcond=None)
            off += step
            if np.abs(step).max() < 0.01:
                break
        shift = np.zeros(6)
        shift[idx] = off
        resid = np.array([jaw(q - shift)[0] for q in Q]) - real
        flags = " ".join(f"--joint-offset {j}={o:+.1f}" for j, o in zip(joints, off))
        cond = f", condition {np.linalg.cond(A):.0f}" if len(joints) > 1 else ""
        print(f"  {' + '.join(joints):27s} {flags:62s} residual rms {np.sqrt(np.mean(resid ** 2)):4.1f} mm "
              f"(max {np.abs(resid).max():.1f}{cond})")


CSV_HEADER = ["real_mm", "sim_mm", "sim_minus_real_mm", "tcp_x_mm", "tcp_y_mm", "tcp_z_mm",
              *[f"{n}.pos" for n in JOINT_NAMES]]


def csv_row(q: np.ndarray, real: float, sim: tuple[float, float, float, float]) -> list[str]:
    z, x, y, tz = sim
    return [f"{real:.1f}", f"{z:.1f}", f"{z - real:.1f}", f"{x:.1f}", f"{y:.1f}", f"{tz:.1f}", *[f"{v:.2f}" for v in q]]


def load_records(jaw: JawHeight, path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    records = []
    for r in rows:
        q = np.array([float(r[f"{n}.pos"]) for n in JOINT_NAMES])
        records.append({"q": q, "sim_mm": jaw(q)[0], "real_mm": float(r["real_mm"])})
    return records


def summarize(jaw: JawHeight, records: list[dict], out: Path) -> None:
    err = np.array([r["sim_mm"] - r["real_mm"] for r in records])
    side = "HIGHER" if err.mean() < 0 else "LOWER"
    print(f"{len(records)} records in {out}\nsim - real: mean {err.mean():+.1f} mm (min {err.min():+.1f}, max "
          f"{err.max():+.1f}): for the same joint angles the real jaws are ~{abs(err.mean()):.0f} mm {side} than in the "
          "simulator\njoint offsets that explain it (policy = real - offset):")
    fit_offsets(jaw, records)


def auto_touches(jaw: JawHeight, args: argparse.Namespace, out: Path) -> None:
    """Record ``args.auto`` touches: wait until the arm moves (> 3 deg on any joint from the last capture), then until
    it stays within 0.3 deg for ``args.still_sec``; record it with real height 0 and play a sound."""
    import subprocess

    from scripts.real.run_policy import RealRobotIO

    io = RealRobotIO(args.port, args.robot_id, cameras={}, fps=30, max_relative_target=None,
                     calibration_dir=args.calibration_dir)
    io._require_calibration_file()
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    bus = io.robot.bus
    bus.connect()
    print(f"waiting for {args.auto} touches (move the arm, then hold the jaw tips on the table still for "
          f"{args.still_sec:.0f} s) -> {out}", flush=True)
    t_end = time.time() + args.timeout
    try:
        with out.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_HEADER)
            def read() -> np.ndarray:
                pos = bus.sync_read("Present_Position")
                return np.array([pos[j] for j in JOINT_NAMES], dtype=np.float64)

            last, n = read(), 0
            while n < args.auto and time.time() < t_end:
                q = read()
                if np.abs(q[:5] - last[:5]).max() < 3.0:  # not moved away from the last capture yet
                    time.sleep(0.1)
                    continue
                anchor, t0 = q, time.time()
                while time.time() - t0 < args.still_sec and time.time() < t_end:
                    time.sleep(0.1)
                    q = read()
                    if np.abs(q[:5] - anchor[:5]).max() > 0.3:
                        anchor, t0 = q, time.time()
                if time.time() >= t_end:
                    break
                sim = jaw(q)
                w.writerow(csv_row(q, 0.0, sim))
                f.flush()
                n, last = n + 1, q
                print(f"touch {n}: sim {sim[0]:+.1f} mm (real 0) at TCP x {sim[1]:.0f} y {sim[2]:+.0f} mm, "
                      f"pose {np.round(q, 1).tolist()}", flush=True)
                try:
                    subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"], stderr=subprocess.DEVNULL)
                except OSError:
                    pass
    finally:
        bus.disconnect(disable_torque=False)
    if n < args.auto:
        print(f"timed out after {n} of {args.auto} touches")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/tty.usbmodem5B610366661")
    p.add_argument("--robot-id", default="my_follower", help="LeRobot calibration id (<id>.json)")
    p.add_argument("--calibration-dir", default=None, metavar="DIR")
    p.add_argument("--out", default=None, metavar="CSV", help="interactive: outputs/real/height_check/<time>.csv; "
                   "--record/--fit: outputs/real/height_check/records.csv")
    p.add_argument("--record", type=float, default=None, metavar="MM", help="no terminal needed: read the pose once, "
                   "append it to --out with this real height (0 = touching) and print the fit over the whole file")
    p.add_argument("--fit", action="store_true", help="only print the fit of the records in --out (no hardware)")
    p.add_argument("--auto", type=int, default=None, metavar="N", help="no terminal needed: record N table touches "
                   "(real height 0), each once the arm has moved and is then held still for --still-sec (a sound plays "
                   "on macOS), append them to --out and print the fit over the whole file")
    p.add_argument("--still-sec", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=300.0, help="--auto: give up after this many seconds")
    args = p.parse_args(argv)

    jaw = JawHeight()
    if args.auto is not None:
        out = Path(args.out) if args.out else ROOT / "outputs/real/height_check/records.csv"
        auto_touches(jaw, args, out)
        summarize(jaw, load_records(jaw, out), out)
        return 0
    if args.record is not None or args.fit:
        out = Path(args.out) if args.out else ROOT / "outputs/real/height_check/records.csv"
        if args.record is not None:
            from scripts.real.run_policy import RealRobotIO

            io = RealRobotIO(args.port, args.robot_id, cameras={}, fps=30, max_relative_target=None,
                             calibration_dir=args.calibration_dir)
            q = io.read_pose_only().astype(np.float64)
            sim = jaw(q)
            out.parent.mkdir(parents=True, exist_ok=True)
            new = not out.exists()
            with out.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(CSV_HEADER)
                w.writerow(csv_row(q, args.record, sim))
            print(f"pose {np.round(q, 1).tolist()}\nrecord: real {args.record:.1f} mm, sim {sim[0]:+.1f} mm -> sim - real "
                  f"{sim[0] - args.record:+.1f} mm at TCP x {sim[1]:.0f} y {sim[2]:+.0f} mm")
        if not out.exists():
            raise SystemExit(f"no records in {out}")
        summarize(jaw, load_records(jaw, out), out)
        return 0

    from scripts.real.run_policy import RealRobotIO

    io = RealRobotIO(args.port, args.robot_id, cameras={}, fps=30, max_relative_target=None,
                     calibration_dir=args.calibration_dir)
    io._require_calibration_file()
    out = Path(args.out) if args.out else ROOT / "outputs/real/height_check" / f"{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"calibration file: {io.calibration_file}\nrecords -> {out}\n"
          "ENTER = jaws touch the table | <mm> ENTER = real height of the lowest jaw point | q ENTER = finish\n")
    bus = io.robot.bus
    bus.connect()
    records, typed, done = [], "", False
    fd = sys.stdin.fileno()
    saved_tty = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # read keys one at a time, so the typed height shows on the live line
    try:
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            while True:
                pos = bus.sync_read("Present_Position")
                q = np.array([pos[n] for n in JOINT_NAMES], dtype=np.float64)
                z, x, y, tz = jaw(q)
                sys.stdout.write(f"\r\033[Ksim jaw height {z:+6.1f} mm | TCP x {x:4.0f} y {y:+4.0f} mm | lift {q[1]:+6.1f}"
                                 f" elbow {q[2]:+6.1f} wrist_flex {q[3]:+6.1f} | real mm> {typed}")
                sys.stdout.flush()
                if not select.select([fd], [], [], 0.1)[0]:
                    continue
                for ch in os.read(fd, 1024).decode(errors="ignore"):  # every key pressed since the last read
                    if ch in "\r\n":
                        entry, typed = typed.strip(), ""
                        if entry.lower() == "q":
                            done = True
                            break
                        try:
                            real = float(entry) if entry else 0.0
                        except ValueError:
                            print(f"\n  not a number: {entry!r} (ENTER alone = touching, e.g. 17 ENTER = 17 mm up)")
                            continue
                        records.append({"q": q, "sim_mm": z, "real_mm": real})
                        w.writerow(csv_row(q, real, (z, x, y, tz)))
                        f.flush()
                        print(f"\n  record {len(records)}: real {real:.1f} mm, sim {z:+.1f} mm -> sim - real "
                              f"{z - real:+.1f} mm at TCP x {x:.0f} y {y:+.0f} mm")
                    elif ch in ("\x7f", "\b"):
                        typed = typed[:-1]
                    elif ch.isprintable():
                        typed += ch
                if done:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved_tty)
        bus.disconnect(disable_torque=False)
    print()
    if not records:
        out.unlink(missing_ok=True)
        return 0
    summarize(jaw, records, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
