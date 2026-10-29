"""Run three-cube episodes with the full execution logic, in the simulator or on the real SO-101:

    HOME -> scene -> SmolVLA -> success detected -> policy stopped, queued actions discarded -> safe release (if needed)
    -> safe retreat -> HOME -> finished

    # simulation: a trained checkpoint on the frozen evaluation states (episode k = state k), videos of the first 3
    python -m so101_libero.cli.run_episodes --robot sim --task flip --checkpoint outputs/flip_red/checkpoints/060000/pretrained_model \\
        --episodes 5 --video 3 --out outputs/run_sim/flip
    # the controller itself, with the scripted expert as a (chunked) stand-in policy
    python -m so101_libero.cli.run_episodes --robot sim --task sort --policy expert --episodes 2 --video 2
    # the real arm (lerobot so101_follower + two cameras); success = the behaviour detector (see completion/)
    python -m so101_libero.cli.run_episodes --robot real --task flip --checkpoint <ckpt> --port /dev/ttyACM0 \\
        --front-camera 2 --wrist-camera 0 --episodes 3 --out outputs/run_real/flip

Success detection: --detector sim (simulator ground truth; default in simulation), behavior (robot-state signals; the
only option on the real arm besides vision), vision (behavior + a front-camera check of the result back at HOME).
Every episode writes a JSON record (states, success, timings, discarded actions, release/retreat/home results) and
one CSV row; ``--out`` collects them. Helpers here (episode specs, record rows) are shared by the evaluation CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

CSV_FIELDS = ["checkpoint", "task", "task_id", "index", "layout", "layout_name", "seed", "placement_id",
              "initial_permutation", "initial_configuration", "success", "success_basis", "completion_detected",
              "task_completion_time_s", "policy_steps",
              "returned_home", "episode_complete", "termination", "discarded_actions", "goal_holds_after_home",
              "behavior_detector_step", "vision_verified", "home_error_deg", "wall_time_s", "video"]


# ------------------------------------------------------------------------------------------------ episodes
def task_of(name: str):
    from so101_libero.tasks.registry import get_task
    from so101_libero.tasks.three_cubes import resolve_task_id

    return get_task(resolve_task_id(name))


def episode_specs(task, n: int | None = None, layouts: list[int] | None = None) -> list[dict]:
    """The task's frozen evaluation states (init_states/<task>.json, expert-verified) as episode specs: index, seed,
    init_state and the layout / permutation / initial cube configuration of each (25 per three-cube task: 5 layouts x 5
    trials; sort: every layout x permutation once). ``layouts`` keeps only those layouts."""
    from so101_libero.cli.make_init_states import load_init_states

    specs = []
    for s in load_init_states(task.task_id)["states"]:
        info = placement_info(task, s["init_state"].get("placement_id"))
        if layouts is not None and info.get("layout_id") not in layouts:
            continue
        specs.append({"index": s["index"], "seed": s["seed"], "init_state": s["init_state"], **info,
                      "initial_configuration": {k: [round(float(v), 4) for v in pose]
                                                for k, pose in s["init_state"]["items"].items()}})
    return specs[:n] if n is not None else specs


def placement_info(task, placement_id) -> dict:
    lay = task.layout
    if placement_id is None or not hasattr(lay, "placement_info"):
        return {"placement_id": placement_id}
    return lay.placement_info(placement_id)


def setup_text(task, placement_id: int) -> str:
    """How to set up a placement on the real table (pan-axis frame: forward from the shoulder's pan axis, left of the
    robot's centre line), for the operator."""
    from so101_libero.tasks.three_cubes import PAN_AXIS_FRONT

    info = placement_info(task, placement_id)
    lines = [f"scene for '{task.language}' -- layout {info.get('layout_name')}"
             + (f", start order (robot's left, centre, right) {info['permutation']}" if info.get("permutation") else "")]
    for name, (x, y, yaw) in task.layout.placements[placement_id].items():
        lines.append(f"  {name.replace('_', ' '):12s} {100 * (x - PAN_AXIS_FRONT):5.1f} cm in front of the pan axis, "
                     f"{100 * abs(y):4.1f} cm {'left' if y >= 0 else 'right'} of the centre line, "
                     f"turned {math.degrees(yaw):+.0f} deg")
    return "\n".join(lines)


def make_policy(spec: str, robot, device: str | None = None, seed: int = 0):
    from so101_libero.control.policies import ExpertPolicy, LeRobotPolicy

    if spec == "expert":
        if not robot.is_sim:
            raise SystemExit("--policy expert needs the simulator (it reads the hidden simulator state)")
        return ExpertPolicy(robot.env, seed=seed)
    return LeRobotPolicy(spec, device=device)


def record_row(rec: dict, spec: dict, task, checkpoint: str) -> dict:
    beh = next((d for d in rec.get("diagnostics", []) if d.get("mode") == "behavior"), None)
    return {"checkpoint": checkpoint, "task": task.language, "task_id": task.task_id, "index": spec.get("index"),
            "layout": spec.get("layout_id"), "layout_name": spec.get("layout_name"), "seed": spec.get("seed"),
            "placement_id": spec.get("placement_id"), "initial_permutation": spec.get("permutation") or "",
            "initial_configuration": json.dumps(spec.get("initial_configuration", {}), separators=(",", ":")),
            "success": rec["task_success"], "success_basis": rec.get("success_basis", ""),
            "completion_detected": rec.get("completion_detected", ""),
            "task_completion_time_s": rec["task_completion_time_s"],
            "policy_steps": rec["policy_steps"], "returned_home": rec["returned_home"],
            "episode_complete": rec["episode_complete"], "termination": rec["termination"],
            "discarded_actions": rec["discarded_actions"], "goal_holds_after_home": rec.get("goal_holds_after_home", ""),
            "behavior_detector_step": "" if beh is None else beh.get("confirmed_step"),
            "vision_verified": rec.get("vision_verified", ""),
            "home_error_deg": rec.get("return_home", {}).get("max_error_deg", ""), "wall_time_s": rec["wall_time_s"],
            "video": rec.get("video", "")}


def behavior_kwargs(args) -> dict:
    """Behaviour-detector thresholds from the command line."""
    return {"closed_cmd": args.closed_cmd, "blocked_margin": args.holding_margin, "open_cmd": args.open_cmd,
            "min_tcp_z": args.min_tcp_z, "still_steps": args.still_steps}


def run_sim_episode(task, policy_spec: str, spec: dict, *, robot=None, policy=None, detector: str = "sim",
                    confirm_steps: int | None = None, max_steps: int | None = None, home=None, device=None,
                    video_path: Path | None = None, behavior_diagnostic: bool = True, log=print,
                    detector_kwargs: dict | None = None) -> tuple[dict, object, object]:
    """One simulated episode through the controller. ``robot`` / ``policy`` are reused when given (one model load per
    worker). Returns (record, robot, policy)."""
    from so101_libero.control.completion import make_detector
    from so101_libero.control.episode_controller import EpisodeController
    from so101_libero.control.episode_finisher import EpisodeFinisher
    from so101_libero.control.home import resolve_home
    from so101_libero.control.robots import SimRobot

    robot = robot or SimRobot(task)
    policy = policy or make_policy(policy_spec, robot, device=device, seed=int(spec.get("seed") or 0))
    if policy_spec == "expert":
        policy.seed = int(spec.get("seed") or 0)
    if policy_spec != "expert":
        import torch

        torch.manual_seed(int(spec.get("seed") or 0))  # SmolVLA samples noise: same episode -> same result
    frames = []
    on_frame = None
    if video_path is not None:
        from so101_libero.utils.video import label

        def on_frame(state, step):
            frames.append(label(robot.frame(), f"{task.task_id} #{spec.get('index')} {state} t={step}"))
    kw = detector_kwargs or {}
    diags = [make_detector(task, "behavior", **kw)] if behavior_diagnostic and detector == "sim" else []
    ctl = EpisodeController(robot, policy, make_detector(task, detector, confirm_steps, **kw),
                            EpisodeFinisher(robot, resolve_home(home), log=log), task.language,
                            max_steps or task.max_steps, diagnostics=diags, on_frame=on_frame, log=log)
    rec = ctl.run(spec)
    if video_path is not None and frames:
        from so101_libero.utils.video import write_mp4

        write_mp4(frames, video_path, fps=robot.fps)
        rec["video"] = str(video_path)
    return rec, robot, policy


def write_outputs(out: Path, rows: list[dict], records: list[dict], extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in CSV_FIELDS} for r in rows)
    (out / "episodes.json").write_text(json.dumps({**(extra or {}), "episodes": records}, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=("sim", "real"), default="sim")
    p.add_argument("--task", required=True, help="flip | sort | pyramid (or a task id)")
    p.add_argument("--checkpoint", default=None, help="a .../pretrained_model directory")
    p.add_argument("--policy", default=None, help="'expert' (simulation only): the scripted expert as the policy")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--layout", type=int, nargs="*", default=None, help="simulation: only these layouts (0-4)")
    p.add_argument("--detector", choices=("sim", "behavior", "vision"), default=None,
                   help="success detection (default: sim in simulation, behavior on the real arm)")
    p.add_argument("--confirm-steps", type=int, default=None, help="consecutive steps the success condition must hold")
    b = p.add_argument_group("behaviour detector (real robot; policy units, see so101_libero/control/completion/base.py)")
    b.add_argument("--closed-cmd", type=float, default=10.0, help="gripper commands <= this count as 'closing'")
    b.add_argument("--holding-margin", type=float, default=8.0,
                   help="measured gripper >= command + this while closing = an object in the jaws (sim: ~18 vs 0)")
    b.add_argument("--open-cmd", type=float, default=20.0, help="gripper commands >= this count as 'released'")
    b.add_argument("--min-tcp-z", type=float, default=0.07, help="the TCP must be this high (m) after the task")
    b.add_argument("--still-steps", type=int, default=10, help="control steps the arm must stand still after the task")
    p.add_argument("--max-steps", type=int, default=None, help="policy timeout in control steps (default: the task's)")
    p.add_argument("--max-seconds", type=float, default=None, help="real arm: policy timeout in seconds")
    p.add_argument("--home-pose", default=None, help="HOME override, '0,-60,35,85,0,10' (default: the suite's)")
    p.add_argument("--device", default=None)
    p.add_argument("--video", type=int, default=0, help="record videos of the first N episodes")
    p.add_argument("--out", type=Path, default=None)
    g = p.add_argument_group("real robot (see scripts/real/run_policy.py --help for details)")
    g.add_argument("--port", default="/dev/ttyACM0")
    g.add_argument("--robot-id", default="my_follower")
    g.add_argument("--calibration-dir", default=None)
    g.add_argument("--front-camera", default="2")
    g.add_argument("--wrist-camera", default="0")
    g.add_argument("--front-size", default="1280x720")
    g.add_argument("--wrist-size", default="640x480")
    g.add_argument("--fps", type=float, default=30.0)
    g.add_argument("--max-step-deg", type=float, default=6.0, help="max arm joint change per step (deg)")
    g.add_argument("--max-gripper-step", type=float, default=10.0, help="max gripper change per step (0..100 units)")
    g.add_argument("--max-relative-target", type=float, default=5.0, help="LeRobot cap vs the measured position")
    g.add_argument("--joint-offset", action="append", default=[], metavar="NAME=DEG")
    g.add_argument("--no-send", action="store_true", help="dry run: observe and infer, never command the arm")
    g.add_argument("--seed", type=int, default=0, help="real: episode k seeds the policy's noise with seed + k")
    g.add_argument("--unattended", action="store_true",
                   help="real: allow running without a terminal (no ENTER before episodes, no confirmation before "
                        "moves). Without it, a real run refuses to start when stdin is not a terminal")
    g.add_argument("--layout-for-setup", type=int, default=0, help="real: which layout the operator sets up (0-4)")
    g.add_argument("--permutation", default=None, help="real, sort: the starting order to set up, e.g. GRY")
    args = p.parse_args(argv)
    if bool(args.checkpoint) == bool(args.policy):
        p.error("give exactly one of --checkpoint and --policy expert")
    if args.policy not in (None, "expert"):
        p.error("--policy only accepts 'expert'; trained models go in --checkpoint")
    args.detector = args.detector or ("sim" if args.robot == "sim" else "behavior")
    if args.robot == "real" and args.detector == "sim":
        p.error("the simulator ground-truth detector does not exist on the real arm: --detector behavior | vision")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    import so101_libero  # noqa: F401

    task = task_of(args.task)
    policy_spec = args.policy or args.checkpoint
    out = args.out
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    extra = {"task_id": task.task_id, "policy": policy_spec, "robot": args.robot, "detector": args.detector}
    if args.robot == "sim":
        specs = episode_specs(task, args.episodes, args.layout)
        robot = policy = None
        for i, spec in enumerate(specs):
            video = (out or Path("outputs/run_sim")) / "videos" / f"{task.task_id}_ep{spec['index']:03d}.mp4" \
                if i < args.video else None
            print(f"==== episode {spec['index']}: layout {spec.get('layout_name')} {spec.get('permutation') or ''} "
                  f"seed {spec['seed']}", flush=True)
            rec, robot, policy = run_sim_episode(task, policy_spec, spec, robot=robot, policy=policy,
                                                 detector=args.detector, confirm_steps=args.confirm_steps,
                                                 max_steps=args.max_steps, home=args.home_pose, device=args.device,
                                                 video_path=video, detector_kwargs=behavior_kwargs(args))
            rows.append(record_row(rec, spec, task, policy_spec))
            records.append({**{k: v for k, v in spec.items() if k != "init_state"}, **rec})
            if out is not None:  # after every episode: an interrupted run keeps what it did
                write_outputs(out, rows, records, extra)
            if rec["termination"].startswith("aborted: operator"):
                print("operator interrupt: stopping the run", flush=True)
                break
        if robot is not None:
            robot.disconnect()
    else:
        rows, records = run_real(args, task, policy_spec, out, extra)
    n_ok = sum(1 for r in rows if r["success"])
    basis = rows[0]["success_basis"] if rows else ""
    print(f"\n{task.task_id}: task success {n_ok}/{len(rows)} ({basis}), returned home "
          f"{sum(1 for r in rows if r['returned_home'])}/{len(rows)}, episode complete "
          f"{sum(1 for r in rows if r['episode_complete'])}/{len(rows)}")
    if out is not None:
        write_outputs(out, rows, records, extra)
        print(f"records: {out / 'episodes.json'}, {out / 'episodes.csv'}")
    return 0


def run_real(args, task, policy_spec: str, out: Path | None = None,
             extra: dict | None = None) -> tuple[list[dict], list[dict]]:
    from scripts.real.run_policy import (CameraSetting, RealRobotIO, camera_ref, parse_joint_offsets, parse_size)
    from so101_libero.control.completion import make_detector
    from so101_libero.control.episode_controller import EpisodeController
    from so101_libero.control.episode_finisher import EpisodeFinisher
    from so101_libero.control.home import resolve_home
    from so101_libero.control.policies import LeRobotPolicy
    from so101_libero.control.robots import RealRobot

    offsets = parse_joint_offsets(args.joint_offset)
    cams = {"front": CameraSetting(camera_ref(args.front_camera), *parse_size(args.front_size)),
            "wrist": CameraSetting(camera_ref(args.wrist_camera), *parse_size(args.wrist_size))}
    try:
        io = RealRobotIO(args.port, args.robot_id, cams, int(round(args.fps)), args.max_relative_target,
                         calibration_dir=args.calibration_dir)
    except ModuleNotFoundError as e:
        raise SystemExit(f"{e}: the SO-101 motor driver is not installed in this environment. On the robot machine: "
                         'pip install "lerobot[feetech]==0.4.2" (the Feetech servo SDK)') from e
    interactive = sys.stdin.isatty()
    if not interactive and not args.unattended:
        raise SystemExit("stdin is not a terminal: on the real arm every episode waits for ENTER and every automatic "
                         "move to HOME is confirmed. Run it from a terminal (ssh -t), or pass --unattended to accept "
                         "running without those confirmations.")
    robot = RealRobot(io, fps=args.fps, offsets=offsets, max_step_deg=args.max_step_deg,
                      max_gripper_step=args.max_gripper_step, send=not args.no_send, interactive=interactive,
                      holding_margin=args.holding_margin, closed_cmd=args.closed_cmd, open_cmd=args.open_cmd)
    from scripts.real.run_policy import default_device

    policy = LeRobotPolicy(policy_spec, device=args.device or default_device())
    print("policy:", json.dumps(policy.describe()))
    home = resolve_home(args.home_pose)

    def ask(question: str, default_yes: bool) -> bool:
        if not robot.interactive:
            return bool(args.unattended) and default_yes
        try:
            ans = input(question)
        except (KeyboardInterrupt, EOFError):
            return False
        return ans.strip().lower() not in ("n", "no")

    def confirm_finish(reason: str) -> bool:  # after Ctrl-C: never without a human (unattended runs hold)
        return robot.interactive and ask(f"{reason}: the policy is stopped and the arm holds. ENTER = safe release, "
                                         "retreat and return HOME; 'n' + ENTER = leave it holding: ", True)

    def confirm_move(reason: str) -> bool:
        return ask(f"{reason}: ENTER = move it to HOME {np.round(home, 1).tolist()} now (keep the path clear); "
                   "'n' + ENTER = stop: ", True)

    max_steps = args.max_steps or task.max_steps
    ctl = EpisodeController(robot, policy, make_detector(task, args.detector, args.confirm_steps, **behavior_kwargs(args)),
                            EpisodeFinisher(robot, home), task.language, max_steps, max_seconds=args.max_seconds,
                            confirm_finish=confirm_finish, confirm_move=confirm_move)
    pid = int(args.layout_for_setup)
    if task.layout.num_placements > 5:  # sort: placement = 5 * layout + permutation
        perms = list(task.layout.permutations)
        pid = 5 * pid + (perms.index(args.permutation) if args.permutation else 0)
    rows, records = [], []
    import torch

    robot.connect()
    try:
        for ep in range(args.episodes):
            seed = int(args.seed) + ep
            torch.manual_seed(seed)  # SmolVLA samples noise: the recorded seed reproduces the policy's sampling
            spec = {"index": ep, "seed": seed, "setup_text": setup_text(task, pid), **placement_info(task, pid)}
            rec = ctl.run(spec)
            rows.append(record_row(rec, spec, task, policy_spec))
            records.append({**spec, **rec})
            if out is not None:
                write_outputs(out, rows, records, extra)
            if rec.get("dry_run"):
                continue  # a dry run moves nothing: the next episode is safe
            # Anything abnormal ends the session: the arm holds where it is and nothing moves on its own.
            stop = (rec.get("finisher_skipped") or rec.get("finisher_interrupted") or not rec["returned_home"]
                    or rec.get("holding_at_home") or rec["termination"].startswith("aborted: operator"))
            if stop and ep + 1 < args.episodes:
                print("stopping the session: the episode did not end cleanly at HOME (see the record); the arm "
                      "holds its last command", flush=True)
                break
    finally:
        robot.disconnect()
    return rows, records


if __name__ == "__main__":
    sys.exit(main())
