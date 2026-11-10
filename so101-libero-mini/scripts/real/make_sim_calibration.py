#!/usr/bin/env python
"""Write a copy of a LeRobot follower calibration whose joint readings match the simulator (policy) angles.

With use_degrees, LeRobot reads a joint as (raw - (range_min + range_max) / 2) * 360 / 4095: its zero is the middle of
the range recorded during calibration, which need not be the simulator's zero. scripts/real/check_height.py and
scripts/real/probe_table.py measure the difference as joint offsets (policy = real - offset). This script moves each
joint's mid-point by the offset, so that the NEW file reads policy angles directly and any LeRobot tool
(lerobot-rollout in scripts/real_smolvla.sh / real_act.sh) uses them with ROBOT_ID=<new id>.

The mid-point moves by changing ONE limit by twice the offset: the limit on the side away from the rest pose (the
rest-side limit, where the folded arm lies against its stop, stays as recorded, so the arm does not move when torque
comes on at rest). The homing offsets are unchanged. On the first connect with the new id LeRobot reports a mismatch
with the motors and asks: press ENTER to use the file (it writes the new position limits to the motors); switching back
to the original id asks the same and restores them.

  python scripts/real/make_sim_calibration.py --offset shoulder_lift=2.1 --offset elbow_flex=8.9
  python scripts/real/make_sim_calibration.py --robot-id home_follower2 --new-id home_follower2_sim --offset elbow_flex=10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RES = 4095  # STS3215 ticks per turn (LeRobot's max_res for the degree conversion)
# A folded SO-101 at rest lies against these limits: shoulder_lift fully back (min), elbow fully folded (max).
REST_SIDE = {"shoulder_pan": None, "shoulder_lift": "min", "elbow_flex": "max", "wrist_flex": None, "wrist_roll": None}


def deg(raw: float, c: dict) -> float:
    return (raw - (c["range_min"] + c["range_max"]) / 2) * 360 / RES


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-id", default="my_follower")
    p.add_argument("--new-id", default=None, help="default: <robot-id>_sim")
    p.add_argument("--calibration-dir", default=None, metavar="DIR",
                   help="default: ~/.cache/huggingface/lerobot/calibration/robots/so_follower (or so101_follower)")
    p.add_argument("--offset", action="append", required=True, metavar="NAME=DEG",
                   help="policy = real - offset, as printed by check_height.py / probe_table.py (repeatable)")
    p.add_argument("--force", action="store_true", help="overwrite an existing new file")
    args = p.parse_args(argv)

    if args.calibration_dir:
        cdir = Path(args.calibration_dir).expanduser()
    else:
        base = Path.home() / ".cache/huggingface/lerobot/calibration/robots"
        cdir = next((base / d for d in ("so_follower", "so101_follower") if (base / d / f"{args.robot_id}.json").exists()),
                    base / "so_follower")
    src = cdir / f"{args.robot_id}.json"
    dst = cdir / f"{args.new_id or args.robot_id + '_sim'}.json"
    if not src.exists():
        raise SystemExit(f"no calibration file {src}")
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} exists (--force to overwrite)")
    cal = json.loads(src.read_text())

    for spec in args.offset:
        name, _, val = spec.partition("=")
        name = name.strip().removesuffix(".pos")
        if name not in REST_SIDE:
            raise SystemExit(f"--offset NAME must be one of {list(REST_SIDE)}, got {name!r}")
        c = cal[name]
        if name == "wrist_roll" or c["range_max"] - c["range_min"] >= RES:
            raise SystemExit(f"{name} is calibrated over a full turn: shift its homing_offset instead")
        shift = round(float(val) * RES / 360)  # mid-point shift in ticks: new reading = old reading - offset
        before = dict(c)
        # mid += shift  <=>  one limit += 2 * shift; keep the rest-side limit (or, for the others, widen the range)
        side = {"min": "range_max", "max": "range_min"}.get(REST_SIDE[name]) or ("range_max" if shift > 0 else "range_min")
        c[side] += 2 * shift
        if not 0 <= c[side] <= RES:
            raise SystemExit(f"{name}: {side} would be {c[side]}, outside 0..{RES}")
        probe = (before["range_min"] + before["range_max"]) / 2 + 300  # any raw position
        print(f"{name:13s} offset {float(val):+.1f} deg = {shift:+d} ticks: {side} {before[side]} -> {c[side]}; "
              f"limits {deg(before['range_min'], before):+.1f}..{deg(before['range_max'], before):+.1f} deg -> "
              f"{deg(c['range_min'], c):+.1f}..{deg(c['range_max'], c):+.1f} deg (reading change "
              f"{deg(probe, c) - deg(probe, before):+.2f} deg)")
    dst.write_text(json.dumps(cal, indent=4) + "\n")
    print(f"wrote {dst}\nuse it with ROBOT_ID={dst.stem} (first connect: press ENTER to use the file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
