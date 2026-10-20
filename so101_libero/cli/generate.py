"""Generate verified expert demonstrations into a LeRobot dataset.

    bash scripts/env.sh python -m so101_libero.cli.generate --tasks cylinder_to_tray --num-demos 5 \
        --repo-id local/so101_mini_t1_5 --overwrite

For every task, attempt seeds ``seed_start + task_offset + i`` in order; an attempt is saved only if the expert
finished and the task's goal predicates hold in the final physics state. Failed attempts are logged with their
reason; generation stops after ``--max-attempts`` per task (no infinite retries). Results are consumed in seed
order, so a dataset is reproducible regardless of the number of workers.

Tasks with predefined initial placements (suite so101_tri) take ``--demos-per-placement N`` instead of
``--num-demos``: placement p of a task uses seeds ``task_offset + p * 10_000 + i`` and gets exactly N successful
demonstrations (failed attempts are retried with the next seed, up to ``--max-attempts`` per placement). Each
episode records its task id, placement id and collection seed (side file and per-frame features):

    python -m so101_libero.cli.generate --tasks cube_left --demos-per-placement 20 --workers 6 \
        --repo-id local/so101_tri_cube_left_100
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK_SEED_STRIDE = 100_000  # task k uses seeds seed_start + k * stride + i; eval init states use >= 10_000_000
PLACEMENT_SEED_STRIDE = 10_000  # placement p of a task: seeds task_base + p * stride + i (--demos-per-placement)

_ENV = None
_EXPERT_CFG = None


def _worker_init(task_ids, image_size, render, expert_kw, visual_rand=False, home_noise=0.0):
    import so101_libero  # noqa: F401  (GL setup before mujoco)
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.pick_place import ExpertConfig

    global _ENV, _EXPERT_CFG
    _ENV = {t: SO101TabletopEnv(t, EnvConfig(image_size=image_size, render=render, terminate_on_success=False,
                                             max_steps=episode_limit(t), visual_rand=visual_rand,
                                             home_noise=home_noise)) for t in task_ids}
    _EXPERT_CFG = ExpertConfig(**expert_kw)


def episode_limit(task_id: str) -> int:
    """Steps an expert demonstration may take: 900 (30 s; the limit every existing dataset was generated with), more
    for tasks whose own horizon is longer (cylinder_flip: 40 s horizon -> 50 s)."""
    from so101_libero.tasks.registry import get_task

    return max(900, int(1.25 * get_task(task_id).max_steps))


def _run(job):
    from so101_libero.expert.rollout import run_expert_episode

    task_id, seed, placement = job if len(job) == 3 else (*job, None)
    ep = run_expert_episode(_ENV[task_id], seed, cfg=_EXPERT_CFG, record=True, placement_id=placement)
    out = {"summary": ep.summary()}
    out["summary"]["visual"] = _ENV[task_id].visual_sample  # appearance sample of this episode (None if disabled)
    if ep.success:
        obs = [o for o, _ in ep.frames]
        out["states"] = np.stack([o["observation.state"] for o in obs])
        out["actions"] = np.stack([a for _, a in ep.frames])
        out["images"] = {k: np.stack([o[k] for o in obs]) for k in obs[0] if k.startswith("observation.images.")}
    return out


class _SeedSchedule:
    """Which attempt to run next: one lane of seeds per placement (a single lane for continuous layouts), each lane
    scheduled until its successes plus in-flight attempts reach ``quota``. Results are consumed in submission order and
    every lane walks its seeds in order, so the saved episodes of a lane are its first ``quota`` successes in seed order
    whatever the number of workers. The next job goes to the lane with the fewest saved + in-flight episodes."""

    def __init__(self, lanes: list, quota: int):
        self.lanes, self.quota = lanes, quota
        self.saved = [0] * len(lanes)
        self.pending = [0] * len(lanes)
        self.attempts = [0] * len(lanes)
        self.exhausted = [False] * len(lanes)

    @property
    def total_saved(self) -> int:
        return sum(self.saved)

    def next_job(self):
        open_lanes = [k for k in range(len(self.lanes))
                      if not self.exhausted[k] and self.saved[k] + self.pending[k] < self.quota]
        for k in sorted(open_lanes, key=lambda k: (self.saved[k] + self.pending[k], k)):
            job = next(self.lanes[k], None)
            if job is None:
                self.exhausted[k] = True
                continue
            self.pending[k] += 1
            self.attempts[k] += 1
            return job
        return None

    def done(self, lane: int, success: bool) -> None:
        self.pending[lane] -= 1
        self.saved[lane] += int(success)


def code_hash(files: dict[str, bytes] | None = None) -> str:
    """Hash of the package sources and the robot model: identifies the code that produced a dataset."""
    import hashlib

    h = hashlib.sha256()
    for name, data in (files if files is not None else source_files()).items():
        h.update(name.encode())
        h.update(data)
    return h.hexdigest()[:16]


def source_files() -> dict[str, bytes]:
    """The package sources and the robot model, read once (the code that generates a dataset)."""
    root = Path(__file__).resolve().parents[2]
    files = sorted([*root.glob("so101_libero/**/*.py"), root / "assets/robotstudio_so101/so101.xml"])
    return {str(f.relative_to(root)): f.read_bytes() for f in files}


def snapshot_sources(path: Path, files: dict[str, bytes] | None = None) -> None:
    """Archive the sources next to the dataset (regenerate it bit for bit even after the code changed). Pass the
    files read when generation started, so the archive is exactly the code that ran."""
    import io
    import tarfile

    files = files if files is not None else source_files()
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _visual_rand_config() -> dict:
    import dataclasses

    from so101_libero.sim.randomize import DEFAULT

    return dataclasses.asdict(DEFAULT)


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["cylinder_to_tray"], help="task ids, or a suite name")
    ap.add_argument("--num-demos", type=int, default=5, help="successful demos per task")
    ap.add_argument("--demos-per-placement", type=int, default=None,
                    help="tasks with predefined placements: exactly this many successful demos per placement (the "
                         "task total is this x the number of placements; overrides --num-demos)")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="per task (default: 2 x num-demos + 10); per placement with --demos-per-placement "
                         "(default: 3 x demos-per-placement + 10)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", type=Path, default=None, help="default: $HF_LEROBOT_HOME/<repo-id>")
    ap.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--window", type=int, default=None,
                    help="max episodes in flight (default 2 x workers); finished episodes wait in memory until written, "
                         "so lower it for long episodes on small machines (a flip demo holds ~300 MB of frames)")
    ap.add_argument("--encode-in-process", action="store_true",
                    help="encode videos in the main process instead of LeRobot's per-episode process pool "
                         "(less memory, slower)")
    ap.add_argument("--vcodec", choices=("libsvtav1", "h264", "hevc"), default="libsvtav1",
                    help="video codec (LeRobot default libsvtav1; use h264 on macOS, where SVT-AV1 3.0 crashes "
                         "intermittently when an encoder is destroyed)")
    ap.add_argument("--max-joint-speed", type=float, default=75.0,
                    help="expert: average joint speed per segment, deg/s (min-jerk peaks are 1.875x this)")
    ap.add_argument("--max-grasp-retries", type=int, default=0, help="expert retries a failed grasp (recovery demos)")
    ap.add_argument("--miss-prob", type=float, default=0.0,
                    help="probability of a deliberately offset first grasp attempt (needs --max-grasp-retries)")
    ap.add_argument("--miss-offset", type=float, nargs=2, default=[0.012, 0.02], metavar=("LO", "HI"),
                    help="sideways offset range (m) of a deliberate miss; use >= the object's half width + 3 mm "
                         "(0.018 0.026 for a 3 cm cube) so the miss does not become an unstable edge grasp")
    ap.add_argument("--legacy-expert", action="store_true",
                    help="reproduce datasets generated before 2026-10-17 01:45 (ExpertConfig.legacy_ik)")
    ap.add_argument("--visual-rand", action="store_true",
                    help="appearance-only domain randomization of the images (lights, colours, camera jitter, "
                         "noise; so101_libero/sim/randomize.py), seeded by the episode seed; physics unchanged")
    ap.add_argument("--home-noise-deg", type=float, default=0.0,
                    help="uniform noise added to the 5 arm joints of the start pose (a real arm never starts exactly "
                         "at the home pose)")
    ap.add_argument("--episode-timeout", type=float, default=900.0,
                    help="seconds to wait for one episode result; a worker that died (e.g. an EGL abort) or hangs "
                         "then fails the run with exit code 3 instead of blocking until the job's walltime")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.miss_prob > 0 and args.max_grasp_retries < 1:
        ap.error("--miss-prob needs --max-grasp-retries >= 1 (otherwise every deliberate miss is rejected)")

    import so101_libero  # noqa: F401
    from so101_libero.data.writer import DemoWriter
    from so101_libero.tasks.registry import TASKS, get_suite

    from so101_libero.cli.make_init_states import EVAL_SEED_BASE, rules_hash, scene_hash

    task_ids = [t.task_id for name in args.tasks for t in get_suite(name)]
    all_ids = list(TASKS)
    per_placement = args.demos_per_placement
    if per_placement is not None:
        n_place = {TASKS[t].layout.num_placements for t in task_ids}
        if 0 in n_place:
            ap.error("--demos-per-placement needs tasks with predefined placements (e.g. --tasks so101_tri)")
        if len(n_place) != 1:
            ap.error("all tasks of one --demos-per-placement run must have the same number of placements")
        (n_place,) = n_place
        args.num_demos = per_placement * n_place
        max_attempts = args.max_attempts or 3 * per_placement + 10
        if max_attempts > PLACEMENT_SEED_STRIDE:
            ap.error(f"--max-attempts must stay <= {PLACEMENT_SEED_STRIDE} per placement")
        top = args.seed_start + max(all_ids.index(t) for t in task_ids) * TASK_SEED_STRIDE + n_place * PLACEMENT_SEED_STRIDE
    else:
        max_attempts = args.max_attempts or 2 * args.num_demos + 10
        top = args.seed_start + max(all_ids.index(t) for t in task_ids) * TASK_SEED_STRIDE + max_attempts
    if args.seed_start < 0 or top > EVAL_SEED_BASE:
        ap.error(f"training seeds must stay in [0, {EVAL_SEED_BASE}) (evaluation states use seeds >= {EVAL_SEED_BASE})")
    sources = source_files()  # read before any episode runs: the archive and hash describe the code that ran
    try:  # BLAS library and thread count: part of what bit-exact regeneration depends on (see scripts/env.sh)
        from threadpoolctl import threadpool_info

        blas = [{k: i.get(k) for k in ("internal_api", "version", "num_threads")} for i in threadpool_info()]
    except Exception:  # noqa: BLE001
        blas = None
    started = {"code_hash": code_hash(sources), "blas": blas,
               "scene_hashes": {t: scene_hash(TASKS[t]) for t in task_ids},
               "rules_hashes": {t: rules_hash(TASKS[t]) for t in task_ids}}
    root = args.root or Path(os.environ.get("HF_LEROBOT_HOME", "data")) / args.repo_id
    fps = 30.0
    writer = DemoWriter(args.repo_id, root, fps=fps, image_size=tuple(args.image_size), overwrite=args.overwrite,
                        episode_meta=per_placement is not None, parallel_encoding=not args.encode_in_process,
                        vcodec=args.vcodec)
    import dataclasses
    import math as _math

    from so101_libero.expert.pick_place import ExpertConfig

    expert_kw = {"max_joint_speed": _math.radians(args.max_joint_speed), "legacy_ik": args.legacy_expert,
                 "max_grasp_retries": args.max_grasp_retries, "miss_prob": args.miss_prob,
                 "miss_offset": tuple(args.miss_offset)}
    expert_cfg = dataclasses.asdict(ExpertConfig(**expert_kw))
    ctx = mp.get_context("spawn")
    t0 = time.time()
    stats = {}
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(task_ids, tuple(args.image_size), True, expert_kw, args.visual_rand,
                            _math.radians(args.home_noise_deg))) as pool:
        for task_id in task_ids:
            task = TASKS[task_id]
            base = args.seed_start + all_ids.index(task_id) * TASK_SEED_STRIDE
            if per_placement is None:
                schedule = _SeedSchedule([iter([(task_id, base + i) for i in range(max_attempts)])], args.num_demos)
            else:
                schedule = _SeedSchedule(
                    [iter([(task_id, base + p * PLACEMENT_SEED_STRIDE + i, p) for i in range(max_attempts)])
                     for p in range(task.layout.num_placements)], per_placement)
            fails = collections.Counter()
            # Sliding window of in-flight episodes, consumed in submission order: workers keep simulating while this
            # process writes/encodes; at most `window` episodes of extra work when the target is reached.
            window, pending = args.window or 2 * args.workers, collections.deque()

            def refill():
                while len(pending) < window:
                    job = schedule.next_job()
                    if job is None:
                        return
                    pending.append((job, pool.apply_async(_run, (job,))))

            refill()
            while pending:
                job, fut = pending.popleft()
                try:
                    res = fut.get(timeout=args.episode_timeout)
                except mp.TimeoutError:
                    print(f"[{task_id}] no episode result within {args.episode_timeout:.0f} s: a worker died or hangs "
                          "(an EGL abort shows as 'Fatal Python error' above); aborting", file=sys.stderr, flush=True)
                    pool.terminate()
                    sys.exit(3)
                s = res["summary"]
                lane = job[2] if len(job) == 3 else 0
                if "states" in res:
                    schedule.done(lane, True)
                    record = {k: s[k] for k in ("seed", "task_id", "steps", "success_step", "max_ik_err_mm",
                                                "grasp", "phases", "init_state", "regrasps", "deliberate_miss",
                                                "regrasp_frames", "visual", "placement_id")}
                    record["task"] = task.language
                    if per_placement is not None:
                        record["placement_name"] = task.layout.placement_names[record["placement_id"]]
                    idx = writer.add_episode(task.language, res["states"], res["actions"], res["images"], record)
                    where = f" placement {lane} [{schedule.saved[lane]}/{per_placement}]" if per_placement else ""
                    print(f"[{task_id}] seed {s['seed']}{where}: saved episode {idx} ({s['steps']} frames, "
                          f"slip {s['grasp']['max_slip_mm']} mm)  [{schedule.total_saved}/{args.num_demos}]",
                          flush=True)
                else:
                    schedule.done(lane, False)
                    fails[s["failure"]] += 1
                    writer.add_failure({k: s[k] for k in ("seed", "task_id", "failure", "steps", "phases",
                                                          "init_state", "placement_id")})
                    print(f"[{task_id}] seed {s['seed']}: REJECTED ({s['failure']})", flush=True)
                refill()
            saved = schedule.total_saved
            attempts = saved + sum(fails.values())
            stats[task_id] = {"saved": saved, "attempts_used": attempts, "failures": dict(fails),
                              "expert_success_rate": round(saved / max(attempts, 1), 4), "seed_base": base}
            if per_placement is not None:
                stats[task_id]["per_placement"] = {
                    task.layout.placement_names[p]: {"placement_id": p, "saved": schedule.saved[p],
                                                     "attempts": schedule.attempts[p],
                                                     "seed_base": base + p * PLACEMENT_SEED_STRIDE}
                    for p in range(task.layout.num_placements)}
            print(f"[{task_id}] {saved} demos saved, {sum(fails.values())} rejected: {dict(fails)}", flush=True)
            if saved < args.num_demos:
                print(f"[{task_id}] WARNING: only {saved}/{args.num_demos} demos within {max_attempts} attempts"
                      f"{' per placement' if per_placement else ''}", file=sys.stderr)
    snapshot_sources(root / "meta" / "so101_libero_src.tar.gz", sources)
    writer.finalize({
        "repo_id": args.repo_id, "tasks": task_ids, "num_demos_per_task": args.num_demos,
        "demos_per_placement": per_placement,
        "placement_names": list(TASKS[task_ids[0]].layout.placement_names) if per_placement else None,
        "placement_seed_stride": PLACEMENT_SEED_STRIDE if per_placement else None,
        "instructions": {t: TASKS[t].language for t in task_ids},
        "max_attempts_per_task": None if per_placement else max_attempts,
        "max_attempts_per_placement": max_attempts if per_placement else None,
        "seed_start": args.seed_start, "task_seed_stride": TASK_SEED_STRIDE,
        "image_size": list(args.image_size), "fps": fps, "stats": stats, "git_commit": git_commit(),
        **started, "expert_config": expert_cfg, "visual_rand": args.visual_rand, "home_noise_deg": args.home_noise_deg,
        "visual_rand_config": _visual_rand_config() if args.visual_rand else None,
        "so101_libero_version": so101_libero.__version__, "wall_time_s": round(time.time() - t0, 1),
        "render_backend": os.environ.get("MUJOCO_GL"), "vcodec": args.vcodec,
    })
    print(json.dumps(stats, indent=2))
    print(f"dataset: {root}  ({writer.n_episodes} episodes, {time.time() - t0:.0f} s)")
    if any(v["saved"] < args.num_demos for v in stats.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
