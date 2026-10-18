#!/usr/bin/env python
"""Automatic table-touch test: the arm lowers its closed gripper until the jaws meet the table, at a few spots.

The same measurement as scripts/real/check_height.py, without a person moving the arm: for each spot the arm goes to a
pose the simulator puts 6 cm above the table (gripper pointing down), then descends slowly along the simulator's
vertical until the jaws stop following the command (they rest on the table). The measured joint angles at that moment
are recorded with real height 0, and the arm rises again. The simulator places the jaws LOWER than the real arm does
(the reason for this test), so every simulator-planned pose is at least as high in reality as planned.

Safety: the table must be clear in front of the robot (no cube, no hands). Every command is capped by LeRobot's
max_relative_target (--max-rel deg from the measured position), which also bounds how hard the jaws can press on the
table; moves run at <= 30 deg/s, the descent at 15 mm/s. A descent that finds no table within --max-depth mm below the
simulator's table aborts the spot. At the end (and on Ctrl-C / errors) the arm returns to its start pose and torque is
switched off, as after a lerobot-rollout.

  python scripts/real/probe_table.py                          # 5 spots, records -> outputs/real/height_check/
  python scripts/real/probe_table.py --dry-run                # simulated arm with a known offset, nothing connected
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real.check_height import CSV_HEADER, JawHeight, csv_row, load_records, summarize  # noqa: E402
from so101_libero.expert.ik import IKSolver, top_down_frame  # noqa: E402
from so101_libero.sim.robot import HOME_DEG, JOINT_NAMES, lerobot_to_sim, sim_to_lerobot  # noqa: E402

FPS = 30.0
GRIPPER_CLOSED = 3.0  # LeRobot 0..100
SPOTS = ((0.14, 0.0), (0.19, 0.0), (0.24, 0.0), (0.19, 0.08), (0.19, -0.08))  # sim TCP x, y (m)
LIMITS = np.array([[-110.0, 110.0], [-100.0, 100.0], [-96.8, 90.0], [-95.0, 95.0], [-157.2, 162.7], [0.0, 100.0]])


class RealArm:
    def __init__(self, port: str, robot_id: str, max_rel: float):
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # lerobot 0.4.x
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True,
                                                       max_relative_target=max_rel, cameras={}))
        if not Path(self.robot.calibration_fpath).is_file():
            raise SystemExit(f"no calibration file {self.robot.calibration_fpath}")

    def connect(self) -> None:
        bus = self.robot.bus
        bus.connect()
        # Hold where the arm is when torque comes on (configure() re-enables it), whatever goal the servos kept.
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
        self.robot.configure()

    def read(self) -> np.ndarray:
        pos = self.robot.bus.sync_read("Present_Position")
        return np.array([pos[n] for n in JOINT_NAMES], dtype=np.float64)

    def command(self, q: np.ndarray) -> None:
        self.robot.send_action({f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, q)})

    def disconnect(self) -> None:
        self.robot.disconnect()  # torque off


class FakeArm:
    """Simulated real arm for --dry-run: it reads true + offset (deg), tracks commands with lag, and cannot push its
    jaws below the table (true simulator kinematics)."""

    def __init__(self, jaw: JawHeight, offset: np.ndarray, max_rel: float, start: np.ndarray):
        self.jaw, self.offset, self.max_rel = jaw, offset, max_rel
        self.true = start - offset

    def connect(self) -> None:
        pass

    def read(self) -> np.ndarray:
        return self.true + self.offset + np.random.default_rng().normal(0, 0.1, 6)

    def command(self, q: np.ndarray) -> None:
        cur = self.true + self.offset
        goal = cur + np.clip(q - cur, -self.max_rel, self.max_rel)
        nxt = self.true + 0.5 * (goal - self.offset - self.true)
        if self.jaw(nxt)[0] >= 0 or self.jaw(nxt)[0] > self.jaw(self.true)[0]:  # blocked by the table
            self.true = nxt
        time.sleep(0.002)

    def disconnect(self) -> None:
        pass


class Planner:
    """Simulator IK for gripper-down poses (LeRobot units in, LeRobot units out)."""

    def __init__(self, jaw: JawHeight):
        self.jaw, self.ik = jaw, IKSolver(jaw.h)

    def pose(self, x: float, y: float, z_tcp: float, seed: np.ndarray) -> np.ndarray:
        # Jaws closing along the radius (wrist_roll ~ 0) at every height: the yaw never changes along a path, so the
        # wrist cannot flip by 90 deg between two ticks. Only the outward tilt adapts (least tilt that reaches).
        yaw = self.ik.azimuth(np.array([x, y]))
        for tilt in np.radians(np.arange(0.0, 31.0, 2.5)):
            rot = self.ik.tilted_frame(np.array([x, y, z_tcp]), float(tilt), yaw) if tilt else top_down_frame(yaw)
            r = self.ik.solve(np.array([x, y, z_tcp]), rot, lerobot_to_sim(seed)[:5])
            if r.success:
                q = sim_to_lerobot(np.r_[r.q, 0.0]).astype(np.float64)
                q[5] = GRIPPER_CLOSED
                return q
        raise RuntimeError(f"no gripper-down IK solution at ({x:.3f}, {y:.3f}, {z_tcp:.3f})")


def move(arm, q_from: np.ndarray, q_to: np.ndarray, max_speed: float = 30.0) -> None:
    """Min-jerk joint move at <= max_speed deg/s, then a short settle."""
    q_to = np.clip(q_to, LIMITS[:, 0], LIMITS[:, 1])
    dur = max(1.0, float(np.abs(q_to[:5] - q_from[:5]).max()) / max_speed * 1.875)  # min-jerk peak = 1.875 x mean
    n = int(dur * FPS)
    for k in range(1, n + 1):
        t0 = time.perf_counter()
        u = k / n
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        arm.command(q_from + s * (q_to - q_from))
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    for _ in range(int(0.4 * FPS)):
        arm.command(q_to)
        time.sleep(1 / FPS)


def descend(arm, jaw: JawHeight, path: list[np.ndarray], z_cmd: np.ndarray, contact_mm: float) -> np.ndarray | None:
    """Follow ``path`` (one pose per tick) until the measured jaw height stops following the command: the lag
    (measured - commanded jaw height, simulator) exceeds its free-motion baseline by ``contact_mm`` for 3 ticks.
    Returns the measured pose at contact, or None if the path ends first."""
    lags, hits = [], 0
    for k, q in enumerate(path):
        t0 = time.perf_counter()
        arm.command(q)
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
        meas = arm.read()
        lags.append(jaw(meas)[0] - z_cmd[k])
        if k < 15:  # baseline from the first 15 ticks (well above any table)
            continue
        base = float(np.median(lags[5:15]))
        hits = hits + 1 if lags[-1] - base > contact_mm else 0
        if hits >= 3:
            time.sleep(0.3)  # let it settle on the table
            return arm.read()
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/tty.usbmodem5B610366661")
    p.add_argument("--robot-id", default="my_follower")
    p.add_argument("--out", default=None, metavar="CSV", help="default: outputs/real/height_check/probe_<time>.csv")
    p.add_argument("--max-rel", type=float, default=4.0, help="LeRobot max_relative_target (deg)")
    p.add_argument("--start-mm", type=float, default=60.0, help="simulator jaw height where each descent starts")
    p.add_argument("--max-depth", type=float, default=70.0, help="give up this many mm below the simulator's table")
    p.add_argument("--speed", type=float, default=15.0, help="descent speed, mm/s")
    p.add_argument("--contact-mm", type=float, default=4.0, help="lag beyond the baseline that counts as contact")
    p.add_argument("--dry-run", action="store_true", help="simulated arm (offsets below), nothing connected")
    p.add_argument("--fake-offset", type=float, nargs=2, default=(3.0, 10.0), metavar=("LIFT", "ELBOW"),
                   help="--dry-run: how much the fake arm's shoulder_lift / elbow_flex readings exceed the truth")
    args = p.parse_args(argv)

    jaw = JawHeight()
    plan = Planner(jaw)
    out = Path(args.out) if args.out else ROOT / "outputs/real/height_check" / f"probe_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rest = np.array([-3.0, -100.0, 96.0, 73.0, -3.0, 3.0])
    if args.dry_run:
        off = np.zeros(6)
        off[1], off[2] = args.fake_offset
        arm = FakeArm(jaw, off, args.max_rel, rest)
    else:
        arm = RealArm(args.port, args.robot_id, args.max_rel)

    home = np.asarray(HOME_DEG, dtype=np.float64)
    home[5] = GRIPPER_CLOSED
    # Plan every spot before anything moves: pre-touch pose and the descent (one pose per tick).
    step = args.speed / 1000.0 / FPS
    plans = []
    for x, y in SPOTS:
        q = plan.pose(x, y, 0.05, home)
        tcp_minus_jaw = (jaw(q)[3] - jaw(q)[0]) / 1000.0  # TCP height above the lowest jaw point (m)
        zs = np.arange(args.start_mm / 1000.0, -args.max_depth / 1000.0, -step) + tcp_minus_jaw
        path, seed = [], q
        for z in zs:
            try:
                seed = plan.pose(x, y, float(z), seed)
            except RuntimeError:  # out of reach further down: the descent ends here
                break
            path.append(seed)
        if len(path) < 2 * FPS:
            raise SystemExit(f"spot ({x}, {y}): descent path too short ({len(path)} poses)")
        z_cmd = np.array([jaw(qq)[0] for qq in path])
        plans.append((x, y, path, z_cmd))
        print(f"spot x {x * 100:.0f} cm y {y * 100:+.0f} cm: descent from sim jaw height {z_cmd[0]:+.0f} to "
              f"{z_cmd[-1]:+.0f} mm in {len(path) / FPS:.1f} s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    arm.connect()
    start = arm.read()
    print(f"start pose {np.round(start, 1).tolist()}", flush=True)
    cur = start
    recorded = 0
    try:
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            move(arm, cur, home)
            cur = home
            for x, y, path, z_cmd in plans:
                move(arm, cur, path[0])
                cur = path[0]
                touch = descend(arm, jaw, path, z_cmd, args.contact_mm)
                if touch is None:
                    print(f"  x {x * 100:.0f} y {y * 100:+.0f} cm: no table within {args.max_depth:.0f} mm below the "
                          "simulator's table -> skipped", flush=True)
                    cur = path[-1]
                else:
                    sim = jaw(touch)
                    w.writerow(csv_row(touch, 0.0, sim))
                    f.flush()
                    recorded += 1
                    print(f"  x {x * 100:.0f} y {y * 100:+.0f} cm: table touched, sim jaw height {sim[0]:+.1f} mm "
                          f"-> real jaws {-sim[0]:+.1f} mm vs sim | pose {np.round(touch, 1).tolist()}", flush=True)
                    cur = touch
                move(arm, cur, path[0])  # back up
                cur = path[0]
            move(arm, cur, home)
            cur = home
    except KeyboardInterrupt:
        print("\ninterrupted: returning to the start pose", flush=True)
    finally:
        try:
            move(arm, arm.read(), start)
        finally:
            arm.disconnect()
    if recorded:
        summarize(jaw, load_records(jaw, out), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
