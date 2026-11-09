#!/usr/bin/env python
"""Real-robot evaluation of the so101_tri SmolVLA checkpoints: N operator-judged trials per task, recorded on disk.

Every run in outputs/train/ with a checkpoint at --ckpt [080000] is evaluated on its task (smolvla_tri_cube_flip_100:
cube_flip, smolvla_tri_green_push_100: green_push_forward, smolvla_tri_cylinder_right_100: cylinder_right_of_cube).
One trial = one launch of scripts/real_smolvla.sh (one attempt at the task), which ends at the first of:
  - back at start: an arm joint moved > 40 deg from the start pose, then every arm joint is back within 20 deg of it
    (the demonstrations park at rest; left running, the policy starts the task again). One SIGINT ends lerobot's
    control loop like its duration limit: the arm returns to its initial position and the process exits 0;
  - time limit: the task's max_seconds (lerobot-rollout --duration): cube_flip 30 s, green_push_forward 25 s,
    cylinder_right_of_cube 35 s;
  - watchdog: no control tick --startup-timeout [180] s after the launch, or still running 30 s after the time limit:
    SIGINT (graceful), then SIGTERM (forced) and SIGKILL;
  - Ctrl+C by the operator: once = graceful stop (the arm returns), twice = forced (it does not).
The time limits and the back-at-start rule are those of voice-robot-controller (policy_tasks.py, robot_runner.py).
After each attempt the operator judges it: success / failure / redo (discarded, e.g. a setup or hardware problem; kept
under discarded/ with the reason). Results are written after every trial; a rerun resumes where it stopped.

Results per run, in outputs/train/<run>/eval_real/<ckpt>/ (same keys as the simulator's eval/<step>/results.json, so
`python -m so101_libero.cli.summarize outputs/train/*/eval*/*/results.json` lists both):
  results.json   summary + every trial (ended_by, rollout time, ticks, arm excursion, operator note, ...)
  episodes.csv   one row per kept trial
  trials/ep000/  rollout.log (launcher output), actions.csv (state + commanded action every tick),
                 policy_view.mp4 (what the policy saw: front | wrist, every DEBUG_EVERY ticks), frames/ (its JPEGs)

    conda activate lerobot
    python scripts/eval_real.py run                          # every task, 10 trials each, checkpoint 080000
    python scripts/eval_real.py run --tasks cube_flip        # one task (resumes if trials are already recorded)
    PORT=/dev/tty.usbmodemXXXX FRONT_CAM=1 python scripts/eval_real.py run   # robot/camera: real_smolvla.sh's env
    python scripts/eval_real.py summary                      # Markdown table of the recorded results
    python scripts/eval_real.py push --dry-run               # list what would be uploaded
    python scripts/eval_real.py push                         # eval_real/<ckpt>/ -> <hf user>/<run> (model repo)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "outputs" / "train"
LAUNCHER = PROJECT_ROOT / "scripts" / "real_smolvla.sh"
OUT_NAME = "eval_real"

# Arm-return detection on the per-tick action log of scripts/rollout_debug.py (voice-robot-controller's values: in
# real rollouts the arm comes back to within ~5-20 deg of its start and stays > ~28 deg away mid-task).
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LEFT_START_DEG = 40.0
BACK_AT_START_DEG = 20.0
DEBUG_DIR_PREFIX = "debug frames and action log -> "  # printed by real_smolvla.sh with DEBUG=1
WATCH_INTERVAL_S = 0.2
STATUS_EVERY_S = 5.0
RUN_GRACE_S = 30.0  # after the time limit, before the watchdog stops a rollout that has not ended by itself
SHUTDOWN_GRACE_S = 30.0  # after SIGINT: return to the initial position, disconnect (~4 s), before forcing
FORCE_GRACE_S = 5.0  # after SIGTERM, before SIGKILL

REQUIRED_FILES = (
    "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
# real_smolvla.sh settings recorded with the results (unset = the script's defaults).
SETTING_KEYS = ("PORT", "ROBOT_ID", "FRONT_CAM", "WRIST_CAM", "FRONT_W", "FRONT_H", "WRIST_W", "WRIST_H", "FPS",
                "MAX_REL", "DEVICE", "FRONT_CROP", "WRIST_CROP", "CROP_RESIZE", "DEBUG_EVERY", "CAMERA_STALL_MS",
                "CAMERA_RECONNECTS")
CSV_FIELDS = ("task_id", "index", "success", "outcome", "ended_by", "rollout_s", "returned_at_s", "ticks",
              "max_arm_dev_deg", "final_arm_dev_deg", "returncode", "camera_stalls", "max_seconds", "started_at",
              "note", "video")


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    run: str  # training run folder in outputs/train/
    instruction: str  # the training instruction (so101_libero/tasks/registry.py)
    max_seconds: int  # time limit of one attempt (voice-robot-controller/policy_tasks.py)
    setup: str
    success: str  # what the operator checks: the simulator's goal predicates (docs/TRI_TASKS.md) by eye


_SCENE = ("arm folded at rest; tray centre 3.6 cm in front of the pan axis, 12.5 cm right; red cube (3 cm) 19.3 cm "
          "in front, 7.2 cm right, turned 20 deg so one face points at the pan axis; green block 18.6 cm in front, "
          "3 cm left, flat face down, long axis forward")
TASKS = (
    EvalTask("cube_flip", "smolvla_tri_cube_flip_100", "Pick up the red cube and flip it onto its side.", 30,
             f"{_SCENE}; gray cylinder standing 16.1 cm in front, 13 cm left (or the group of three shifted by 1 cm)",
             "the face that was on top now faces sideways (upside down does not count); the cube lies flat on the "
             "table within ~2 cm of its start, released; cylinder and green block not moved (<= 2 cm) or tipped"),
    EvalTask("green_push_forward", "smolvla_tri_green_push_100", "Push the green semicircular block forward 5 cm.", 25,
             f"{_SCENE}; gray cylinder standing 16.1 cm in front, 13 cm left (or the group of three shifted by 1 cm); "
             "~6 cm of table clear in front of the green block",
             "the green block moved 4-6 cm forward and <= 1.5 cm sideways, still flat face down, pushed (never "
             "lifted); cylinder and cube not moved (<= 2 cm) or tipped"),
    EvalTask("cylinder_right_of_cube", "smolvla_tri_cylinder_right_100",
             "Put the light gray cylinder to the right of the red cube.", 35,
             f"{_SCENE}; gray cylinder standing anywhere 19-23 cm in front, 11-17 cm left",
             "the cylinder stands upright on the table >= 4 cm to the robot's right of the cube and within 3 cm "
             "in front of / behind it, carried there (not shoved) and released; the cube moved <= 1 cm; the green "
             "block not moved (<= 2 cm) or tipped"),
)
TASKS_BY_ID = {t.task_id: t for t in TASKS}


# ----------------------------------------------------------------------------------------------- small helpers
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (as so101_libero/cli/evaluate.py, without importing the simulator)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def verify_checkpoint(policy: Path) -> None:
    for name in REQUIRED_FILES:
        f = policy / name
        if not f.is_file() or f.stat().st_size == 0:
            raise RuntimeError(f"missing or empty {name} in {policy}")
    try:
        policy_type = json.loads((policy / "config.json").read_text()).get("type")
    except (OSError, ValueError, AttributeError) as e:
        raise RuntimeError(f"cannot read {policy / 'config.json'}: {e}") from e
    if policy_type != "smolvla":
        raise RuntimeError(f"expected a SmolVLA checkpoint, got type={policy_type!r}: {policy}")


def discover(train_dir: Path, ckpt: str, task_ids: list[str] | None) -> list[EvalTask]:
    """The registered tasks whose run has a valid checkpoint `ckpt` in train_dir (all requested ones must)."""
    found, skipped = [], []
    for run_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()) if train_dir.is_dir() else []:
        if (run_dir / "checkpoints" / ckpt / "pretrained_model").is_dir() and run_dir.name not in {t.run for t in TASKS}:
            skipped.append(run_dir.name)
    for task in TASKS:
        if task_ids and task.task_id not in task_ids:
            continue
        policy = train_dir / task.run / "checkpoints" / ckpt / "pretrained_model"
        if not policy.is_dir():
            if task_ids:
                raise RuntimeError(f"{task.task_id}: no checkpoint at {policy}")
            continue
        verify_checkpoint(policy)
        found.append(task)
    for run in skipped:
        print(f"note: {run} has a checkpoint {ckpt} but no task in scripts/eval_real.py TASKS: skipped")
    if not found:
        raise RuntimeError(f"no registered run with a checkpoint {ckpt} in {train_dir}")
    return found


def git_revision() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001  not installed in this interpreter
        return None


# ------------------------------------------------------------------------------------------ arm-return detection
class ReturnToStart:
    """Fed one arm state per control tick; reports when the arm is back at its start pose after leaving it."""

    def __init__(self, left_deg: float = LEFT_START_DEG, back_deg: float = BACK_AT_START_DEG):
        self.left_deg, self.back_deg = left_deg, back_deg
        self.start: list[float] | None = None
        self.left = False
        self.distance = 0.0
        self.max_distance = 0.0

    def update(self, arm: list[float]) -> bool:
        if self.start is None:
            self.start = list(arm)
        self.distance = max(abs(a - s) for a, s in zip(arm, self.start))
        self.max_distance = max(self.max_distance, self.distance)
        self.left = self.left or self.distance > self.left_deg
        return self.left and self.distance < self.back_deg


def parse_arm(values: list[str], time_col: int, arm_cols: list[int]) -> tuple[float, list[float]] | None:
    try:
        t = float(values[time_col])
        arm = [float(values[i]) for i in arm_cols]
    except (ValueError, IndexError):
        return None
    return None if any(math.isnan(v) for v in arm) else (t, arm)  # nan: no state observed yet


class ActionLogTail:
    """Reads the rows appended to rollout_debug.py's log.csv since the last call: [(time_s, arm state)]."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.offset = 0
        self.pending = ""
        self.cols: tuple[int, list[int]] | None = None
        self.bad_header = False

    def read(self) -> list[tuple[float, list[float]]]:
        if self.path is None or self.bad_header:
            return []
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                chunk = f.read()
                self.offset = f.tell()
        except OSError:
            return []  # not written yet
        *lines, self.pending = (self.pending + chunk.decode("utf-8", "replace")).split("\n")  # last: partial line
        rows = []
        for line in lines:
            values = line.strip().split(",")
            if self.cols is None:
                try:
                    self.cols = (values.index("time_s"), [values.index(f"state_{j}") for j in ARM_JOINTS])
                except ValueError:
                    self.bad_header = True  # unexpected format: the attempt runs to its time limit
                    print(f"  warning: unexpected action log header in {self.path}; back-at-start detection is off")
                    return []
                continue
            row = parse_arm(values, *self.cols)
            if row is not None:
                rows.append(row)
        return rows


def action_stats(path: Path) -> dict:
    """Ticks, rollout time and arm excursion from a finished actions.csv."""
    stats = {"ticks": 0, "rollout_s": None, "max_arm_dev_deg": None, "final_arm_dev_deg": None}
    if not path.is_file():
        return stats
    det = ReturnToStart()
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        try:
            cols = (header.index("time_s"), [header.index(f"state_{j}") for j in ARM_JOINTS])
        except (AttributeError, ValueError):
            return stats
        for values in reader:
            stats["ticks"] += 1
            row = parse_arm(values, *cols)
            if row is not None:
                stats["rollout_s"] = round(row[0], 2)
                det.update(row[1])
    if det.start is not None:
        stats["max_arm_dev_deg"] = round(det.max_distance, 1)
        stats["final_arm_dev_deg"] = round(det.distance, 1)
    return stats


# ----------------------------------------------------------------------------------------------- one attempt
class OutputWatcher(threading.Thread):
    """Copies the launcher's output to rollout.log, echoes the important lines, picks out the debug dir."""

    ECHO = re.compile(r"^task: |^crop |Policy loaded|Robot connected|Rollout setup complete|Duration limit reached|"
                      r"Returning robot|Rollout finished|Shutdown signal|Force shutdown|no frame for|reopen|"
                      r"ERROR|Error|Traceback|no checkpoint")

    def __init__(self, stream, log, verbose: bool):
        super().__init__(daemon=True)
        self.stream, self.log, self.verbose = stream, log, verbose
        self.debug_dir: Path | None = None
        self.crop: str | None = None
        self.duration_limit = False
        self.camera_stalls = 0
        self.slow_loop_warnings = 0
        self.tail: list[str] = []

    def run(self) -> None:
        for line in self.stream:
            self.log.write(line)
            self.log.flush()
            s = line.rstrip("\n")
            self.tail = (self.tail + [s])[-20:]
            if s.startswith(DEBUG_DIR_PREFIX):
                self.debug_dir = Path(s[len(DEBUG_DIR_PREFIX):].strip())
            elif s.startswith("crop "):
                self.crop = s[len("crop "):].strip()
            self.duration_limit |= "Duration limit reached" in s
            self.camera_stalls += "no frame for" in s
            self.slow_loop_warnings += "running slower" in s
            if self.verbose or self.ECHO.search(s):
                print(f"    | {s[:220]}", flush=True)


class Supervisor:
    """Signals the launcher's process group: one SIGINT (graceful), then SIGTERM and SIGKILL if it hangs."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.sent: list[str] = []
        self.t_int: float | None = None
        self.t_term: float | None = None
        self.killed = False

    def send(self, sig: signal.Signals) -> None:
        if self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            return
        self.sent.append(sig.name)
        if sig == signal.SIGINT and self.t_int is None:
            self.t_int = time.monotonic()
        elif sig == signal.SIGTERM and self.t_term is None:
            self.t_term = time.monotonic()
        elif sig == signal.SIGKILL:
            self.killed = True

    def stop(self) -> None:
        if self.t_int is None:
            self.send(signal.SIGINT)

    def force(self) -> None:
        if self.t_term is None:
            print("  forcing the rollout to exit: the arm does NOT return to its start pose", flush=True)
            self.send(signal.SIGTERM)
        elif not self.killed:
            self.send(signal.SIGKILL)

    def escalate(self) -> None:
        t = time.monotonic()
        if self.t_int is not None and self.t_term is None and t - self.t_int > SHUTDOWN_GRACE_S:
            print(f"  no exit {SHUTDOWN_GRACE_S:.0f} s after SIGINT", flush=True)
            self.force()
        elif self.t_term is not None and not self.killed and t - self.t_term > FORCE_GRACE_S:
            self.send(signal.SIGKILL)

    def shutdown(self) -> None:
        """Make sure the process is gone (used when the watch loop itself fails)."""
        for action, grace in ((self.stop, SHUTDOWN_GRACE_S), (self.force, FORCE_GRACE_S), (self.force, None)):
            if self.proc.poll() is not None:
                return
            action()
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                continue


def run_attempt(task: EvalTask, ckpt: str, trial_dir: Path, *, max_seconds: float, early_stop: bool,
                startup_timeout: float, launcher: Path, verbose: bool = False) -> dict:
    """Run one attempt through the launcher and collect its files in trial_dir. Returns the measurements."""
    trial_dir.mkdir(parents=True)
    env = dict(os.environ)
    for key in ("POLICY", "STATS_FROM"):  # would make real_smolvla.sh run another policy than RUN/CKPT
        env.pop(key, None)
    env.update(RUN=task.run, CKPT=ckpt, TASK=task.instruction, DURATION=f"{max_seconds:g}", DEBUG="1",
               PYTHONUNBUFFERED="1")
    started_at, t_launch = now(), time.monotonic()
    detector, tail = ReturnToStart(), ActionLogTail()
    ended_by = returned_at = t_first = None
    last_t, last_status, interrupts = 0.0, t_launch, 0
    with (trial_dir / "rollout.log").open("w") as log:
        proc = subprocess.Popen(["bash", str(launcher)], cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, start_new_session=True)
        out = OutputWatcher(proc.stdout, log, verbose)
        out.start()
        sup = Supervisor(proc)
        try:
            while proc.poll() is None:
                try:
                    time.sleep(WATCH_INTERVAL_S)
                    if tail.path is None and out.debug_dir is not None:
                        tail.path = out.debug_dir / "log.csv"
                    for t_s, arm in tail.read():
                        if t_first is None:
                            t_first = time.monotonic()
                            print(f"  policy running (started {t_first - t_launch:.0f} s after launch); "
                                  f"Ctrl+C = stop, arm returns", flush=True)
                        last_t = t_s
                        if detector.update(arm) and early_stop and ended_by is None:
                            ended_by, returned_at = "returned_to_start", round(t_s, 2)
                            print(f"  back at the start pose after {t_s:.1f} s: ending the attempt", flush=True)
                            sup.stop()
                    t = time.monotonic()
                    if ended_by is None and t_first is None and t - t_launch > startup_timeout:
                        ended_by = "watchdog"
                        print(f"  watchdog: no control tick {startup_timeout:.0f} s after launch: stopping", flush=True)
                        sup.stop()
                    elif ended_by is None and t_first is not None and t - t_first > max_seconds + RUN_GRACE_S:
                        ended_by = "watchdog"
                        print(f"  watchdog: still running {RUN_GRACE_S:.0f} s after the time limit: stopping",
                              flush=True)
                        sup.stop()
                    sup.escalate()
                    if t_first is not None and ended_by is None and t - last_status >= STATUS_EVERY_S:
                        last_status = t
                        print(f"  t {last_t:5.1f} / {max_seconds:g} s   arm {detector.distance:4.0f} deg from start"
                              f"{'  (has left the start pose)' if detector.left else ''}", flush=True)
                except KeyboardInterrupt:
                    interrupts += 1
                    if interrupts > 1:
                        sup.force()
                    elif sup.t_int is None:
                        ended_by = ended_by or "operator_stop"
                        print("\n  stopping: the arm returns to its start pose (Ctrl+C again to force)", flush=True)
                        sup.stop()
                    else:
                        print("\n  already stopping: the arm is returning to its start pose (Ctrl+C again to "
                              "force)", flush=True)
        finally:
            if proc.poll() is None:
                sup.shutdown()
            proc.wait()
            out.join(timeout=10)
    rc = proc.returncode
    if ended_by is None:
        ended_by = "time_limit" if out.duration_limit else ("error" if rc != 0 else "exited")
    if rc != 0 and ended_by in ("error", "exited"):
        print("  rollout exited with code", rc, "- last output:\n    " + "\n    ".join(out.tail[-12:]), flush=True)

    # The debug dir is new for every launch (outputs/debug/<RUN>_<time>): move it into the trial.
    if out.debug_dir is not None and out.debug_dir.is_dir():
        shutil.move(str(out.debug_dir), str(trial_dir / "frames"))
        if (trial_dir / "frames" / "log.csv").is_file():
            (trial_dir / "frames" / "log.csv").rename(trial_dir / "actions.csv")
    frames = sorted((trial_dir / "frames").glob("frame_*.jpg"))
    fps = float(os.environ.get("FPS", 30)) / max(1, int(os.environ.get("DEBUG_EVERY", 10)))
    has_video = encode_video(frames, trial_dir / "policy_view.mp4", fps)
    return {
        "ended_by": ended_by, "returncode": rc, "returned_at_s": returned_at, **action_stats(trial_dir / "actions.csv"),
        "camera_stalls": out.camera_stalls, "slow_loop_warnings": out.slow_loop_warnings, "signals": sup.sent,
        "max_seconds": max_seconds, "early_stop": early_stop, "crop": out.crop,
        "started_at": started_at, "finished_at": now(), "wall_s": round(time.monotonic() - t_launch, 1),
        "files": {"log": "rollout.log",
                  "actions": "actions.csv" if (trial_dir / "actions.csv").is_file() else None,
                  "video": "policy_view.mp4" if has_video else None,
                  "frames": "frames" if frames else None},
    }


def encode_video(frames: list[Path], out: Path, fps: float) -> bool:
    """H.264 MP4 (plays in browsers and on the Hub) of the debug frames; False if there are none or no encoder."""
    if not frames:
        return False
    try:
        import av
        from PIL import Image
    except ImportError:
        print("  note: PyAV / Pillow not available: no policy_view.mp4 (the frames are kept)")
        return False
    codec = next((c for c in ("libx264", "h264_videotoolbox", "libopenh264") if c in av.codecs_available), None)
    if codec is None:
        print("  note: no H.264 encoder in PyAV: no policy_view.mp4 (the frames are kept)")
        return False
    try:
        with Image.open(frames[0]) as im:
            w, h = im.size
        w, h = w - w % 2, h - h % 2
        with av.open(str(out), mode="w", options={"movflags": "faststart"}) as container:
            stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            for path in frames:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                rgb = rgb if rgb.size == (w, h) else rgb.crop((0, 0, w, h))
                for packet in stream.encode(av.VideoFrame.from_image(rgb)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return True
    except Exception as e:  # noqa: BLE001  a video failure must never lose the trial
        print(f"  note: policy_view.mp4 not written ({e}); the frames are kept")
        out.unlink(missing_ok=True)
        return False


# ------------------------------------------------------------------------------------------------- results
def results_dir(train_dir: Path, task: EvalTask, ckpt: str, out_name: str = OUT_NAME) -> Path:
    return train_dir / task.run / out_name / ckpt


def new_results(train_dir: Path, task: EvalTask, ckpt: str, trials: int, operator: str | None) -> dict:
    policy = train_dir / task.run / "checkpoints" / ckpt / "pretrained_model"
    try:
        dataset = json.loads((policy / "train_config.json").read_text())["dataset"]["repo_id"]
    except (OSError, ValueError, KeyError, TypeError):
        dataset = None
    return {
        "policy": {"policy": str(policy), "type": "smolvla", "language_conditioned": True, "run": task.run,
                   "checkpoint": ckpt, "training_dataset": dataset},
        "task": {"task_id": task.task_id, "instruction": task.instruction, "setup": task.setup,
                 "success_criterion": task.success},
        "protocol": {"robot": "real SO-101 follower", "launcher": "scripts/real_smolvla.sh (lerobot-rollout)",
                     "trials_planned": trials, "one_attempt_per_trial": True, "judged_by": "operator",
                     "operator": operator, "ends_at": "first of: back at start pose (> 40 deg away, then all arm "
                     "joints within 20 deg), time limit, watchdog, operator stop",
                     "left_start_deg": LEFT_START_DEG, "back_at_start_deg": BACK_AT_START_DEG},
        "render_backend": "real",
        "overall": {}, "per_task": {}, "complete": False,
        "started_at": now(), "updated_at": None,
        "settings": {}, "software": {}, "sessions": [],
        "episodes": [], "discarded": [],
    }


def load_results(out_dir: Path, train_dir: Path, task: EvalTask, ckpt: str, trials: int,
                 operator: str | None) -> dict:
    path = out_dir / "results.json"
    results = json.loads(path.read_text()) if path.is_file() else new_results(train_dir, task, ckpt, trials, operator)
    results["protocol"]["trials_planned"] = trials
    if operator:
        results["protocol"]["operator"] = operator
    return results


def update_summary(results: dict) -> None:
    eps = results["episodes"]
    n, k = len(eps), sum(bool(e["success"]) for e in eps)
    lo, hi = wilson(k, n)
    block = {"episodes": n, "successes": k, "success_rate": round(k / n, 4) if n else 0.0,
             "wilson95": [round(lo, 4), round(hi, 4)]}
    times = [e["rollout_s"] for e in eps if e["success"] and e.get("rollout_s") is not None]
    results["overall"] = block
    results["per_task"] = {results["task"]["task_id"]: {
        "language": results["task"]["instruction"], **block,
        "outcomes": dict(Counter(e["outcome"] for e in eps)),
        "ended_by": dict(Counter(e["ended_by"] for e in eps)),
        "mean_rollout_s_success": round(sum(times) / len(times), 1) if times else None,
    }}
    results["complete"] = n >= results["protocol"]["trials_planned"]
    results["updated_at"] = now()


def save_results(out_dir: Path, results: dict) -> None:
    update_summary(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(out_dir / "results.json", json.dumps(results, indent=2) + "\n")
    rows = [",".join(CSV_FIELDS)]
    for e in results["episodes"]:
        rows.append(",".join(_csv_cell(e.get(f)) for f in CSV_FIELDS))
    write_atomic(out_dir / "episodes.csv", "\n".join(rows) + "\n")


def _csv_cell(v) -> str:
    s = "" if v is None else str(v)
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s


def record(attempt: dict, rel_dir: str) -> dict:
    """The attempt's measurements with its file names made relative to the results folder."""
    files = attempt.pop("files")
    rec = dict(attempt, dir=rel_dir)
    for key, name in files.items():
        rec[key] = f"{rel_dir}/{name}" if name else None
    return rec


def discard(out_dir: Path, trial_dir: Path, index: int, results: dict, attempt: dict | None, reason: str) -> None:
    """Move an attempt that does not count to discarded/ and keep its record."""
    rel = base = f"discarded/ep{index:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    n = 1
    while (out_dir / rel).exists():
        n, rel = n + 1, f"{base}_{n + 1}"
    (out_dir / "discarded").mkdir(parents=True, exist_ok=True)
    shutil.move(str(trial_dir), str(out_dir / rel))
    rec = record(dict(attempt), rel) if attempt else {"dir": rel}
    results["discarded"].append({"index": index, "task_id": results["task"]["task_id"], "reason": reason, **rec,
                                 "discarded_at": now()})


# -------------------------------------------------------------------------------------------- operator prompts
def ask(prompt: str, choices: dict[str, str]) -> str:
    """Repeat the prompt until one of the choices; Ctrl+C / end of input count as 'q'."""
    while True:
        try:
            answer = input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return "q"
        if answer in choices:
            return choices[answer]
        print(f"  please answer one of: {', '.join(repr(c) if c else 'Enter' for c in choices)}")


def ask_text(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


# ----------------------------------------------------------------------------------------------- commands
def parse_max_seconds(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        task_id, _, value = item.partition("=")
        if task_id not in TASKS_BY_ID or not value:
            raise SystemExit(f"--max-seconds expects TASK=SECONDS with TASK in {', '.join(TASKS_BY_ID)}: {item!r}")
        out[task_id] = float(value)
        if out[task_id] <= 0:
            raise SystemExit(f"--max-seconds {item}: must be > 0")
    return out


def preflight(launcher: Path) -> None:
    if not launcher.is_file():
        raise RuntimeError(f"launcher not found: {launcher}")
    if launcher == LAUNCHER:
        for name in ("deploy_policy.py", "rollout_tolerant.py", "rollout_debug.py"):
            if not (PROJECT_ROOT / "scripts" / name).is_file():
                raise RuntimeError(f"scripts/{name} not found")
        if not (os.environ.get("CONDA_EXE") or shutil.which("conda")):
            raise RuntimeError("conda not found (real_smolvla.sh activates the 'lerobot' env): run from an "
                               "initialized shell (`conda activate lerobot`) or set CONDA_EXE")
    if os.environ.get("CROP_RESIZE", "1") != "1":
        raise RuntimeError("CROP_RESIZE=0 disables the debug output this evaluation records (action log, frames)")


def cmd_run(args) -> int:
    max_seconds = parse_max_seconds(args.max_seconds)
    tasks = discover(args.train_dir, args.ckpt, args.tasks)
    preflight(args.launcher)
    settings = {k: os.environ[k] for k in SETTING_KEYS if k in os.environ}
    software = {"git": git_revision(), "lerobot": package_version("lerobot"), "host": platform.node(),
                "python": platform.python_version()}
    print(f"Real-robot evaluation, checkpoint {args.ckpt}, {args.trials} trials per task")
    print(f"robot / camera settings (others: real_smolvla.sh defaults): {settings or 'all defaults'}")
    for task in tasks:
        print(f"  {task.task_id:24s} {task.run}  limit {max_seconds.get(task.task_id, task.max_seconds):g} s  -> "
              f"{results_dir(args.train_dir, task, args.ckpt, args.out_name).relative_to(args.train_dir.parent)}")
    print("Ctrl+C during a rollout stops it (the arm returns to its start pose); twice forces it.\n")

    for task in tasks:
        out_dir = results_dir(args.train_dir, task, args.ckpt, args.out_name)
        results = load_results(out_dir, args.train_dir, task, args.ckpt, args.trials, args.operator)
        limit = max_seconds.get(task.task_id, task.max_seconds)
        if len(results["episodes"]) >= args.trials:
            print(f"== {task.task_id}: {len(results['episodes'])}/{args.trials} trials already recorded, skipped "
                  f"(delete {out_dir} to start over)\n")
            continue
        results["settings"], results["software"] = settings, software
        results["sessions"].append({"started_at": now(), "max_seconds": limit, "early_stop": not args.no_early_stop,
                                    "settings": settings, "software": software})
        save_results(out_dir, results)
        print(f"== {task.task_id}  ({task.run} @ {args.ckpt})")
        print(f"   instruction: {task.instruction}")
        print(f"   setup:       {task.setup}")
        print(f"   success:     {task.success}")
        print(f"   time limit:  {limit:g} s" + ("" if args.no_early_stop else ", or once the arm is back at rest"))
        while len(results["episodes"]) < args.trials:
            index = len(results["episodes"])
            trial_dir = out_dir / "trials" / f"ep{index:03d}"
            if trial_dir.exists():  # left by an interrupted session, never judged
                discard(out_dir, trial_dir, index, results, None, "interrupted session: no verdict recorded")
                save_results(out_dir, results)
            k = sum(bool(e["success"]) for e in results["episodes"])
            choice = ask(f"\n[{task.task_id}] trial {index + 1}/{args.trials} ({k}/{index} so far): reset the "
                         "scene, arm at rest. Enter = start, s = skip task, q = quit: ", {"": "go", "s": "s", "q": "q"})
            if choice == "s":
                break
            if choice == "q":
                print_summary(args.train_dir, args.ckpt, args.out_name, tasks)
                return 0
            attempt = run_attempt(task, args.ckpt, trial_dir, max_seconds=limit, early_stop=not args.no_early_stop,
                                  startup_timeout=args.startup_timeout, launcher=args.launcher, verbose=args.verbose)
            print(f"  ended by {attempt['ended_by']} after {attempt['rollout_s'] or 0:.1f} s of policy control "
                  f"({attempt['ticks']} ticks, max arm excursion {attempt['max_arm_dev_deg'] or 0:.0f} deg, "
                  f"exit {attempt['returncode']})")
            if attempt["ticks"] == 0:
                print("  the policy never ran: 'r' (redo) is probably right")
            verdict = ask("  Outcome? y = success, n = failure, r = redo (discard), q = discard and quit: ",
                          {"y": "success", "n": "failure", "r": "r", "q": "q"})
            if verdict in ("success", "failure"):
                note = ask_text("  Note (optional, Enter to skip): ")
                ep = {"task_id": task.task_id, "index": index, "success": verdict == "success", "outcome": verdict,
                      "note": note, **record(attempt, f"trials/ep{index:03d}")}
                results["episodes"].append(ep)
            else:
                reason = ask_text("  Reason for discarding (optional): ") if verdict == "r" else ""
                discard(out_dir, trial_dir, index, results, attempt, reason or ("redo" if verdict == "r" else "quit"))
            save_results(out_dir, results)
            if verdict == "q":
                print_summary(args.train_dir, args.ckpt, args.out_name, tasks)
                return 0
        o = results["overall"]
        print(f"== {task.task_id}: {o['successes']}/{o['episodes']} successes "
              f"(Wilson 95% [{100 * o['wilson95'][0]:.0f}, {100 * o['wilson95'][1]:.0f}]%) -> {out_dir}\n")
    print_summary(args.train_dir, args.ckpt, args.out_name, tasks)
    return 0


def print_summary(train_dir: Path, ckpt: str, out_name: str, tasks: list[EvalTask]) -> None:
    print("\n| task | run | checkpoint | trials | success | Wilson 95% CI | ended by |")
    print("|---|---|---|---|---|---|---|")
    for task in tasks:
        path = results_dir(train_dir, task, ckpt, out_name) / "results.json"
        if not path.is_file():
            print(f"| {task.task_id} | {task.run} | {ckpt} | 0/? | - | - | - |")
            continue
        r = json.loads(path.read_text())
        o, t = r["overall"], r["per_task"].get(task.task_id, {})
        ended = ", ".join(f"{k} {v}" for k, v in sorted(t.get("ended_by", {}).items(), key=lambda kv: -kv[1]))
        print(f"| {task.task_id} | {task.run} | {ckpt} | {o['episodes']}/{r['protocol']['trials_planned']} | "
              f"{o['successes']}/{o['episodes']} = {100 * o['success_rate']:.0f}% | "
              f"[{100 * o['wilson95'][0]:.0f}, {100 * o['wilson95'][1]:.0f}] | {ended or '-'} |")


def cmd_summary(args) -> int:
    print_summary(args.train_dir, args.ckpt, args.out_name, discover(args.train_dir, args.ckpt, args.tasks))
    return 0


def cmd_push(args) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    owner = None if args.repo_id else (args.repo_owner or api.whoami()["name"])
    ignore = [] if args.include_frames else ["*/frames/*"]
    status = 0
    for task in discover(args.train_dir, args.ckpt, args.tasks):
        out_dir = results_dir(args.train_dir, task, args.ckpt, args.out_name)
        if not (out_dir / "results.json").is_file():
            print(f"{task.task_id}: no results in {out_dir}, skipped")
            continue
        r = json.loads((out_dir / "results.json").read_text())
        o = r["overall"]
        if not r["complete"] and not args.allow_incomplete:
            print(f"{task.task_id}: only {o['episodes']}/{r['protocol']['trials_planned']} trials, skipped "
                  "(--allow-incomplete to push anyway)")
            continue
        repo_id = args.repo_id or f"{owner}/{task.run}"
        path_in_repo = f"{task.run}/{args.out_name}/{args.ckpt}" if args.repo_id else f"{args.out_name}/{args.ckpt}"
        files = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*")
                       if p.is_file() and not p.name.endswith(".tmp")
                       and (args.include_frames or "frames" not in p.relative_to(out_dir).parts))
        size = sum((out_dir / f).stat().st_size for f in files) / 1e6
        print(f"{task.task_id}: {o['successes']}/{o['episodes']} -> {repo_id} ({args.repo_type}) {path_in_repo}/: "
              f"{len(files)} files, {size:.1f} MB")
        if args.dry_run:
            for f in files:
                print(f"    {f}")
            continue
        if not api.repo_exists(repo_id, repo_type=args.repo_type):
            print(f"  {repo_id} does not exist: create it first (hf repo create {repo_id} --repo-type "
                  f"{args.repo_type}) or pass --repo-id")
            status = 1
            continue
        api.upload_folder(repo_id=repo_id, repo_type=args.repo_type, folder_path=out_dir, path_in_repo=path_in_repo,
                          ignore_patterns=ignore + ["*.tmp"],
                          commit_message=f"Real-robot eval of {task.run} @ {args.ckpt} on {task.task_id}: "
                                         f"{o['successes']}/{o['episodes']} successes")
        print(f"  uploaded: https://huggingface.co/{'' if args.repo_type == 'model' else args.repo_type + 's/'}"
              f"{repo_id}/tree/main/{path_in_repo}")
    return status


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", default="080000", help="checkpoint step folder [080000]")
    common.add_argument("--tasks", nargs="+", choices=list(TASKS_BY_ID), help="only these tasks [all with a checkpoint]")
    common.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="training runs [outputs/train]")
    common.add_argument("--out-name", default=OUT_NAME, help="results folder in each run [eval_real]")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="run the trials on the real robot")
    run.add_argument("--trials", type=int, default=10, help="trials per task [10]")
    run.add_argument("--max-seconds", nargs="+", default=[], metavar="TASK=S",
                     help="override a task's time limit, e.g. cube_flip=40")
    run.add_argument("--no-early-stop", action="store_true", help="always run to the time limit")
    run.add_argument("--startup-timeout", type=float, default=180.0,
                     help="seconds from launch to the first control tick before the watchdog stops it [180]")
    run.add_argument("--operator", help="recorded with the results")
    run.add_argument("--verbose", action="store_true", help="echo the launcher's whole output")
    run.add_argument("--launcher", type=Path, default=LAUNCHER, help=argparse.SUPPRESS)  # tests: a fake rollout

    sub.add_parser("summary", parents=[common], help="print the recorded results as a Markdown table")

    push = sub.add_parser("push", parents=[common], help="upload the results to the Hugging Face Hub")
    push.add_argument("--repo-owner", help="user / org of the per-run repos <owner>/<run> [the logged-in user]")
    push.add_argument("--repo-id", help="one repo for every run instead (files under <run>/eval_real/<ckpt>/)")
    push.add_argument("--repo-type", default="model", choices=("model", "dataset"), help="[model]")
    push.add_argument("--include-frames", action="store_true", help="also upload the JPEG frames (the MP4 has them)")
    push.add_argument("--allow-incomplete", action="store_true", help="push tasks with fewer trials than planned")
    push.add_argument("--dry-run", action="store_true", help="only list what would be uploaded")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.train_dir = args.train_dir.resolve()
    if getattr(args, "trials", 1) < 1:
        raise SystemExit("--trials must be >= 1")
    try:
        return {"run": cmd_run, "summary": cmd_summary, "push": cmd_push}[args.command](args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted: every judged trial is saved; an unjudged attempt is discarded on the next run",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
