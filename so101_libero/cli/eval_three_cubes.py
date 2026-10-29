"""Evaluate every saved checkpoint of a three-cube SmolVLA run in the simulator, with the full episode logic.

    python -m so101_libero.cli.eval_three_cubes --run-dir outputs/flip_red --task flip              # 20k/40k/60k/80k
    python -m so101_libero.cli.eval_three_cubes --run-dir outputs/multitask --task all              # each on 3 tasks
    python -m so101_libero.cli.eval_three_cubes --run-dir outputs/sort_ryg --task sort --checkpoints 060000 080000
    python -m so101_libero.cli.eval_three_cubes --policy expert --task sort --episodes 3            # test the sweep
    python -m so101_libero.cli.eval_three_cubes --run-dir outputs/flip_red --task flip --summary-only

Every episode is one run of so101_libero/control/episode_controller.py (run_episodes.run_sim_episode): HOME -> the
frozen start state -> SmolVLA -> the simulator's success detector, checked after every executed action -> policy
stopped and the rest of its action chunk discarded -> safe release (if needed) -> safe retreat -> HOME. The episodes are
the task's frozen evaluation states (init_states/<task>.json; 25 per task: 5 layouts x 5 trials, sort: every layout x
starting permutation once); trials of the same start differ by SmolVLA's sampled noise (torch seeded per episode).
The primary metric is the manipulation success (TASK_SUCCESS); returning HOME and episode completion are counted
separately.

Outputs (never silently overwritten):
  <run-dir>/eval/<step>/<task_id>/episodes.csv, episodes.json, summary.json, videos/   one checkpoint on one task. A
      complete evaluation of the same checkpoint (same model file) and settings is reused; --force (or an incomplete or
      different earlier one) moves the old directory aside to <dir>.old_<timestamp> first.
  results/<name>_eval.csv / .json   all episodes and per-checkpoint summaries of the run (rebuilt from the directories
      above, so earlier checkpoints evaluated in another sweep stay in): name = flip | sort | pyramid | multitask
      (``--policy expert``: <name>_expert). Best checkpoint = highest manipulation success rate (multitask: highest mean
      over the three tasks); ties -> the EARLIER checkpoint.

Workers: each evaluation worker process loads the policy once (per checkpoint) and keeps one simulator per task. A
SmolVLA worker needs ~2-2.5 GB of GPU memory; with --device cuda the worker count is lowered to what fits in the free
GPU memory (nvidia-smi; --gpu-mem-per-worker, --no-auto-workers). A worker that dies is replaced and its running
episodes retried once; an episode that raises is recorded as a failed episode with its error.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_SHORT = {"flip_red_cube": "Flip", "sort_ryg": "Sort", "build_pyramid": "Pyramid"}
RESULT_NAME = {"flip_red_cube": "flip", "sort_ryg": "sort", "build_pyramid": "pyramid"}
BEST_RULE = "highest manipulation success rate; ties -> the earlier checkpoint"
BEST_RULE_MULTI = "highest mean manipulation success rate over flip, sort and pyramid; ties -> the earlier checkpoint"


def csv_fields() -> list[str]:
    from so101_libero.cli.run_episodes import CSV_FIELDS

    return [*CSV_FIELDS, "outcome", "error"]


# ------------------------------------------------------------------------------------------------ checkpoints
def step_label(step: str) -> str:
    """'020000' -> '20k' (the user-facing name); anything else unchanged."""
    if step.isdigit():
        n = int(step)
        return f"{n // 1000}k" if n % 1000 == 0 and n > 0 else str(n)
    return step


def discover_checkpoints(run_dir: Path, wanted: list[str] | None) -> list[tuple[str, Path]]:
    """(step, .../pretrained_model) of the run's saved checkpoints in step order (LeRobot naming
    <run>/checkpoints/<6-digit step>/pretrained_model; ``last`` is a link to the newest and skipped)."""
    root = run_dir / "checkpoints"
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist: no checkpoints to evaluate (train first)")
    found = {}
    for d in sorted(root.iterdir()):
        if d.name == "last" or not d.name.isdigit():
            continue
        model = d / "pretrained_model"
        if not (model / "config.json").is_file():
            print(f"WARNING: checkpoint {d} has no pretrained_model/config.json: skipped", file=sys.stderr)
            continue
        found[d.name] = model
    if wanted:
        want = [w if not w.isdigit() else f"{int(w):06d}" for w in wanted]
        missing = [w for w in want if w not in found]
        if missing:
            raise SystemExit(f"checkpoints {missing} not found in {root} (have: {sorted(found)})")
        found = {w: found[w] for w in want}
    if not found:
        raise SystemExit(f"no saved checkpoint with a pretrained_model in {root}")
    return sorted(found.items(), key=lambda kv: int(kv[0]))


def fingerprint(model_dir: str) -> str:
    """Identity of a checkpoint's weights (size and mtime of its model file): an evaluation is reused only for the same
    file, so retraining a run into the same directory never mixes old results in."""
    if model_dir == "expert":
        return "expert"
    f = Path(model_dir) / "model.safetensors"
    try:
        st = f.stat()
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "missing"


def train_config(model_dir: str) -> dict:
    """The parts of LeRobot's train_config.json worth recording next to the evaluation."""
    try:
        tc = json.loads((Path(model_dir) / "train_config.json").read_text())
    except Exception:  # noqa: BLE001  (expert, or an unreadable file)
        return {}
    pol = tc.get("policy") or {}
    return {"dataset": (tc.get("dataset") or {}).get("repo_id"), "dataset_root": (tc.get("dataset") or {}).get("root"),
            "steps": tc.get("steps"), "batch_size": tc.get("batch_size"), "seed": tc.get("seed"),
            "policy_type": pol.get("type"), "pretrained_path": pol.get("pretrained_path"),
            "scheduler_decay_steps": pol.get("scheduler_decay_steps"), "output_dir": tc.get("output_dir")}


# ------------------------------------------------------------------------------------------------ outcomes
def outcome_of(task_id: str, rec: dict) -> str:
    """What happened in a failed episode, from the success detector's last diagnostics (simulator ground truth)."""
    if rec.get("task_success"):
        return "success"
    term = str(rec.get("termination") or "")
    if term.startswith("aborted"):
        return term
    d = rec.get("detector") or {}
    if task_id == "flip_red_cube":
        if float(d.get("lifted_cm") or 0.0) < 2.0:
            return "never_lifted"
        if float(d.get("rotation_deg") or 0.0) < 150.0 or float(d.get("top_turn_deg") or 0.0) < 60.0:
            return "lifted_not_flipped"
        if float(d.get("shift_cm") or 0.0) > 2.0:
            return "flipped_not_back_in_place"
        return "flipped_not_released_or_not_settled"
    if task_id == "sort_ryg":
        errs = d.get("target_errors_cm")
        if errs:
            return f"{sum(1 for e in errs if float(e) <= 2.0)}_of_3_cubes_on_target"
        return "timeout"
    if task_id == "build_pyramid":
        z, off = d.get("green_z_cm"), d.get("green_offset_cm")
        touch = d.get("touch_red_yellow_table") or [False, False, True]
        if z is None:
            return "timeout"
        if float(z) < 3.5:
            return "green_not_on_top"
        if float(off) > 1.0:
            return "green_off_centre"
        if not (touch[0] and touch[1]) or touch[2]:
            return "green_not_supported_by_both"
        if max(d.get("base_shift_cm") or [0.0]) > 1.0:
            return "base_displaced"
        return "stacked_not_released_or_not_settled"
    return "timeout" if term == "timeout" else term or "failure"


# ------------------------------------------------------------------------------------------------ workers
_W: dict = {}


def _init_worker(device: str | None, torch_threads: int) -> None:
    os.environ["SO101_TORCH_THREADS"] = str(torch_threads)
    import so101_libero  # noqa: F401  (offscreen GL before mujoco)

    _W.update(device=device, robots={}, policy=None, policy_key=None, experts={})
    try:
        import torch

        torch.set_num_threads(torch_threads)
    except Exception:  # noqa: BLE001
        pass


def _close_worker() -> None:
    for robot in list(_W.get("robots", {}).values()):
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
    _W["robots"] = {}


def _robot_for(task):
    from so101_libero.control.robots import SimRobot

    robot = _W["robots"].get(task.task_id)
    if robot is None:
        robot = _W["robots"][task.task_id] = SimRobot(task)
        if not _W.get("atexit"):
            import atexit

            # Registered after MuJoCo's own GL atexit handler (made with the first renderer): atexit runs last-in
            # first-out, so the renderers close while the GL context still exists (no EGL errors at worker exit).
            atexit.register(_close_worker)
            _W["atexit"] = True
    return robot


def _policy_for(model: str, task, robot):
    if model == "expert":
        from so101_libero.control.policies import ExpertPolicy

        pol = _W["experts"].get(task.task_id)
        if pol is None:
            pol = _W["experts"][task.task_id] = ExpertPolicy(robot.env)
        return pol
    if _W.get("policy_key") != model:
        _W["policy"], _W["policy_key"] = None, None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        from so101_libero.control.policies import LeRobotPolicy

        _W["policy"] = LeRobotPolicy(model, device=_W.get("device"))
        _W["policy_key"] = model
    return _W["policy"]


def error_result(job: dict, message: str) -> dict:
    """A failed episode that could not run (exception, crashed worker): counted as a failure, never dropped."""
    spec = job["spec"]
    row = {k: "" for k in csv_fields()}
    row.update(checkpoint=job["step"], task_id=job["task_id"], index=spec.get("index"), layout=spec.get("layout_id"),
               layout_name=spec.get("layout_name"), seed=spec.get("seed"), placement_id=spec.get("placement_id"),
               initial_permutation=spec.get("permutation") or "",
               initial_configuration=json.dumps(spec.get("initial_configuration", {}), separators=(",", ":")),
               success=False, returned_home=False, episode_complete=False, policy_steps=0,
               termination=f"error: {message}"[:300], discarded_actions=0, outcome="error", error=message[:300])
    return {"row": row, "record": {"index": spec.get("index"), "error": message}, "error": True}


def _episode(job: dict) -> dict:
    try:
        from so101_libero.cli.run_episodes import record_row, run_sim_episode
        from so101_libero.tasks.registry import get_task

        task = get_task(job["task_id"])
        robot = _robot_for(task)
        policy = _policy_for(job["model"], task, robot)
        rec, _, _ = run_sim_episode(task, job["model"], job["spec"], robot=robot, policy=policy,
                                    detector=job["detector"], max_steps=job["max_steps"], device=_W.get("device"),
                                    video_path=Path(job["video"]) if job["video"] else None,
                                    log=lambda *a, **k: None)
        row = record_row(rec, job["spec"], task, job["step"])
        row["outcome"] = outcome_of(task.task_id, rec)
        row["error"] = ""
        spec = {k: v for k, v in job["spec"].items() if k != "init_state"}
        return {"row": row, "record": {**spec, **rec}, "error": False}
    except Exception as e:  # noqa: BLE001  (one bad episode must not lose the sweep)
        return error_result(job, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}")


def run_jobs(jobs: list[dict], workers: int, device: str | None, torch_threads: int, on_result,
             max_crashes: int = 2, episode_fn=None) -> list[dict]:
    """Run the episodes in ``workers`` spawned processes; ``on_result(i, result)`` as each finishes. A worker that dies
    (native crash in rendering or inference) breaks the pool. The episodes that were running then are retried ONE AT A
    TIME in a one-worker pool, so a crash is charged only to the episode that was running alone when it happened (never
    to healthy episodes that shared the broken pool); an episode that crashed ``max_crashes`` times in all is recorded
    as an error (= failure). The other episodes continue in the full pool."""
    episode_fn = episode_fn or _episode
    done: dict[int, dict] = {}
    crashes = {i: 0 for i in range(len(jobs))}
    suspects: list[int] = []  # running when a shared pool broke: retried alone
    workers = max(1, min(workers, len(jobs)))
    ctx = mp.get_context("spawn")
    while len(done) < len(jobs):
        if suspects:
            i = suspects.pop(0)
            try:
                with cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_init_worker,
                                            initargs=(device, torch_threads)) as pool:
                    r = pool.submit(episode_fn, jobs[i]).result()
            except cf.process.BrokenProcessPool:
                crashes[i] += 1
                print(f"WARNING: episode {jobs[i].get('spec', {}).get('index', i)} crashed its worker while running "
                      f"alone ({crashes[i] + 1} crashes)", file=sys.stderr, flush=True)
                if crashes[i] + 1 >= max_crashes:  # + the crash of the shared pool that made it a suspect
                    done[i] = error_result(jobs[i], f"worker process crashed ({crashes[i] + 1} times)")
                    on_result(i, done[i])
                else:
                    suspects.append(i)
                continue
            done[i] = r
            on_result(i, r)
            continue
        queue = [i for i in range(len(jobs)) if i not in done]
        running: dict = {}
        try:
            with cf.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                                        initargs=(device, torch_threads)) as pool:
                while queue or running:
                    while queue and len(running) < workers:
                        i = queue.pop(0)
                        running[pool.submit(episode_fn, jobs[i])] = i
                    finished, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                    for fut in finished:
                        i = running[fut]
                        r = fut.result()  # raises BrokenProcessPool if the worker died: i stays in `running`
                        del running[fut]
                        done[i] = r
                        on_result(i, r)
        except cf.process.BrokenProcessPool:
            lost = sorted(running.values())
            print(f"WARNING: a worker process died; the {len(lost)} episode(s) that were running are retried one at a "
                  "time", file=sys.stderr, flush=True)
            suspects.extend(i for i in lost if i not in done)
    return [done[i] for i in range(len(jobs))]


def free_gpu_gb() -> float | None:
    """Free memory (GB) of the first visible GPU from nvidia-smi (no CUDA context in this process), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    rows = [line.split(",") for line in out if line.strip()]
    for idx, mem in rows:
        if not visible or idx.strip() == visible:
            return float(mem) / 1024.0
    return float(rows[0][1]) / 1024.0 if rows else None


# ------------------------------------------------------------------------------------------------ summaries
def summarize(rows: list[dict], task, step: str, model: str, requested: int) -> dict:
    from so101_libero.cli.evaluate import wilson

    def truthy(v) -> bool:
        return v is True or str(v) == "True"

    n = len(rows)
    k = sum(truthy(r["success"]) for r in rows)
    lo, hi = wilson(k, n)
    times = [float(r["task_completion_time_s"]) for r in rows if truthy(r["success"]) and r["task_completion_time_s"] not in ("", None)]
    layouts = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (int(r["layout"]) if str(r["layout"]).lstrip("-").isdigit() else 99)):
        d = layouts.setdefault(str(r["layout_name"]), {"episodes": 0, "successes": 0})
        d["episodes"] += 1
        d["successes"] += int(truthy(r["success"]))
    perms = collections.OrderedDict()
    for r in rows:
        if r.get("initial_permutation"):
            d = perms.setdefault(r["initial_permutation"], {"episodes": 0, "successes": 0})
            d["episodes"] += 1
            d["successes"] += int(truthy(r["success"]))
    errors = sum(1 for r in rows if r.get("outcome") == "error")
    return {"checkpoint": step, "label": step_label(step), "checkpoint_path": model, "fingerprint": fingerprint(model),
            "task_id": task.task_id, "task": task.language, "episodes": n, "requested_episodes": requested,
            "successes": k, "success_rate": round(k / n, 4) if n else 0.0, "wilson95": [round(lo, 4), round(hi, 4)],
            "returned_home": sum(truthy(r["returned_home"]) for r in rows),
            "episode_complete": sum(truthy(r["episode_complete"]) for r in rows),
            "mean_completion_time_s_successes": round(float(np.mean(times)), 2) if times else None,
            "mean_policy_steps": round(float(np.mean([float(r["policy_steps"] or 0) for r in rows])), 1) if n else None,
            "per_layout": layouts, "per_permutation": perms or None,
            "outcomes": dict(collections.Counter(r.get("outcome") or "" for r in rows).most_common()),
            "errors": errors, "complete": errors == 0 and n == requested}


def write_group(group_dir: Path, rows: list[dict], records: list[dict], summary: dict) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    fields = csv_fields()
    with open(group_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    (group_dir / "episodes.json").write_text(json.dumps(records, indent=1, default=str))
    (group_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def read_group(group_dir: Path) -> tuple[dict, list[dict]] | None:
    try:
        summary = json.loads((group_dir / "summary.json").read_text())
        with open(group_dir / "episodes.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        return summary, rows
    except Exception:  # noqa: BLE001
        return None


def reusable(group_dir: Path, model: str, indices: list[int], settings: dict) -> bool:
    got = read_group(group_dir)
    if got is None:
        return False
    s, _ = got
    return (s.get("complete") is True and s.get("fingerprint") == fingerprint(model)
            and s.get("episode_indices") == indices and all(s.get(k) == v for k, v in settings.items()))


def move_aside(path: Path) -> Path:
    dest = path.with_name(f"{path.name}.old_{time.strftime('%Y%m%d_%H%M%S')}")
    path.rename(dest)
    return dest


def collect(eval_root: Path, tasks: list, checkpoints: list[tuple[str, str]]) -> dict:
    """Complete per-checkpoint evaluations under ``eval_root`` for ``tasks``: {step: {task_id: (summary, rows)}}. The
    current checkpoints of the run plus any other complete step directories whose checkpoint still has the same model
    file (evaluated in an earlier sweep)."""
    known = dict(checkpoints)
    out: dict = {}
    steps = set(known)
    if eval_root.is_dir():
        steps |= {d.name for d in eval_root.iterdir() if d.is_dir() and (d.name.isdigit() or d.name == "expert")}
    for step in sorted(steps, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s)):
        for task in tasks:
            got = read_group(eval_root / step / task.task_id)
            if got is None or got[0].get("complete") is not True:
                continue
            s = got[0]
            model = known.get(step, s.get("checkpoint_path"))
            if fingerprint(model) != s.get("fingerprint"):
                continue  # the checkpoint was retrained or deleted since: not this run's result
            out.setdefault(step, {})[task.task_id] = got
    return out


def best_of(rates: dict[str, float]) -> str | None:
    """Highest rate; ties -> the earlier checkpoint (steps compare numerically)."""
    if not rates:
        return None
    order = sorted(rates, key=lambda s: (-rates[s], int(s) if s.isdigit() else 0))
    return order[0]


def aggregate(results_dir: Path, name: str, tasks: list, table: dict, run_dir: str, extra: dict) -> dict:
    """Write results/<name>_eval.csv (every episode, by checkpoint then task then index) and <name>_eval.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fields = csv_fields()
    rows = []
    for step in table:
        for task in tasks:
            if task.task_id in table[step]:
                rows += sorted(table[step][task.task_id][1], key=lambda r: int(r["index"]))
    with open(results_dir / f"{name}_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    report = {"run_dir": run_dir, "tasks": [t.task_id for t in tasks], **extra, "checkpoints": {}}
    if len(tasks) == 1:
        t = tasks[0]
        rates = {}
        for step, per in table.items():
            if t.task_id in per:
                report["checkpoints"][step] = per[t.task_id][0]
                rates[step] = per[t.task_id][0]["success_rate"]
        best = best_of(rates)
        report["best"] = {"checkpoint": best, "label": step_label(best) if best else None,
                          "success_rate": rates.get(best), "rule": BEST_RULE}
    else:
        means = {}
        for step, per in table.items():
            entry = {tid: s for tid, (s, _) in per.items()}
            if len(entry) == len(tasks):  # a checkpoint counts for "best" only when evaluated on every task
                means[step] = round(float(np.mean([entry[t.task_id]["success_rate"] for t in tasks])), 4)
            report["checkpoints"][step] = {**entry, "mean_success_rate": means.get(step)}
        best = best_of(means)
        report["best"] = {"checkpoint": best, "label": step_label(best) if best else None,
                          "mean_success_rate": means.get(best), "rule": BEST_RULE_MULTI}
    (results_dir / f"{name}_eval.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def print_summary(report: dict, tasks: list, results_dir: Path, name: str) -> None:
    if not report["checkpoints"]:
        print("no complete evaluation yet")
        return
    if len(tasks) == 1:
        t = tasks[0]
        first = next(iter(report["checkpoints"].values()))
        print(f"\n{TASK_SHORT.get(t.task_id, t.task_id)}   ({t.task_id}: \"{t.language}\"; "
              f"{first['episodes']} episodes per checkpoint)\n")
        print(f"{'checkpoint':10s} {'success':>8s} {'rate':>6s}   95% CI      returned home   complete   "
              "mean time (successes)")
        for step, s in report["checkpoints"].items():
            lo, hi = s["wilson95"]
            mt = f"{s['mean_completion_time_s_successes']:.1f} s" if s["mean_completion_time_s_successes"] else "-"
            print(f"{s['label']:10s} {s['successes']:>4d}/{s['episodes']:<3d} {100 * s['success_rate']:5.0f}%   "
                  f"[{100 * lo:3.0f}, {100 * hi:3.0f}]   {s['returned_home']:>4d}/{s['episodes']:<9d} "
                  f"{s['episode_complete']:>4d}/{s['episodes']:<5d} {mt}")
            lay = "  ".join(f"{k} {v['successes']}/{v['episodes']}" for k, v in s["per_layout"].items())
            print(f"{'':10s}   per layout: {lay}")
            if s.get("per_permutation"):
                pp = "  ".join(f"{k} {v['successes']}/{v['episodes']}" for k, v in s["per_permutation"].items())
                print(f"{'':10s}   per start permutation: {pp}")
            fails = {k: v for k, v in s["outcomes"].items() if k != "success"}
            if fails:
                print(f"{'':10s}   failures: " + ", ".join(f"{k} {v}" for k, v in fails.items()))
        b = report["best"]
        print(f"\nBEST = {b['label']}   ({b['rule']})")
    else:
        print(f"\nMultitask   (one checkpoint, three instructions; per task: successes/episodes)\n")
        print(f"{'Checkpoint':12s}" + "".join(f"{TASK_SHORT[t.task_id]:>10s}" for t in tasks) + f"{'Mean':>9s}")
        for step, per in report["checkpoints"].items():
            cells = []
            for t in tasks:
                s = per.get(t.task_id)
                cells.append(f"{s['successes']}/{s['episodes']:<3d}".rjust(10) if s else f"{'-':>10s}")
            mean = per.get("mean_success_rate")
            print(f"{step_label(step):12s}" + "".join(cells) + (f"{100 * mean:8.0f}%" if mean is not None else f"{'-':>9s}"))
        b = report["best"]
        print(f"\nBEST = {b['label']}   ({b['rule']})")
    print(f"results: {results_dir / f'{name}_eval.csv'}, {results_dir / f'{name}_eval.json'}")


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=None, help="training output dir (with checkpoints/)")
    p.add_argument("--policy", default=None, help="'expert': evaluate the scripted expert instead (tests the sweep)")
    p.add_argument("--task", required=True, help="flip | sort | pyramid | all (= multitask: every checkpoint on all three)")
    p.add_argument("--checkpoints", nargs="*", default=None, help="steps to evaluate (default / 'all': every saved one)")
    p.add_argument("--episodes", type=int, default=25, help="per task and checkpoint (the first N frozen states)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--videos", type=int, default=5, help="record the first N episodes of each checkpoint and task")
    p.add_argument("--device", default="cuda", help="policy device (cuda | cpu)")
    p.add_argument("--detector", choices=("sim", "behavior", "vision"), default="sim",
                   help="success detection (sim = simulator ground truth: the evaluation metric)")
    p.add_argument("--max-steps", type=int, default=None, help="policy timeout per episode (default: the task's)")
    p.add_argument("--force", action="store_true", help="re-run evaluations that exist (the old ones are moved aside)")
    p.add_argument("--summary-only", action="store_true", help="only print and rebuild the summaries from the results")
    p.add_argument("--eval-root", type=Path, default=None, help="default: <run-dir>/eval")
    p.add_argument("--results-dir", type=Path, default=None, help="default: $RESULTS_DIR or <project>/results")
    p.add_argument("--results-name", default=None, help="results/<name>_eval.csv (default: flip|sort|pyramid|multitask)")
    p.add_argument("--torch-threads", type=int, default=2, help="CPU threads per worker")
    p.add_argument("--gpu-mem-per-worker", type=float, default=2.5, help="GB of GPU memory one SmolVLA worker needs")
    p.add_argument("--no-auto-workers", action="store_true", help="never lower --workers to fit the free GPU memory")
    args = p.parse_args(argv)
    if bool(args.run_dir) == bool(args.policy):
        p.error("give exactly one of --run-dir and --policy expert")
    if args.policy not in (None, "expert"):
        p.error("--policy only accepts 'expert' (checkpoints are found in --run-dir)")
    if args.checkpoints in ([], ["all"]):
        args.checkpoints = None
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    import so101_libero  # noqa: F401
    from so101_libero.cli.generate import git_commit
    from so101_libero.cli.run_episodes import episode_specs, task_of

    multi = args.task in ("all", "multitask")
    tasks = [task_of(t) for t in ("flip", "sort", "pyramid")] if multi else [task_of(args.task)]
    name = args.results_name or ("multitask" if multi else RESULT_NAME.get(tasks[0].task_id, tasks[0].task_id))
    if args.policy == "expert":
        checkpoints = [("expert", "expert")]
        if not args.results_name:
            name += "_expert"
        eval_root = args.eval_root or PROJECT_ROOT / "outputs" / "eval_three_cubes_expert"
        run_dir = "expert"
    else:
        checkpoints = [(s, str(m)) for s, m in discover_checkpoints(args.run_dir, args.checkpoints)]
        eval_root = args.eval_root or args.run_dir / "eval"
        run_dir = str(args.run_dir)
    results_dir = args.results_dir or Path(os.environ.get("RESULTS_DIR") or PROJECT_ROOT / "results")
    settings = {"detector": args.detector, "max_steps": args.max_steps}
    extra = {"detector": args.detector, "renderer": os.environ.get("MUJOCO_GL"), "episodes_per_task": args.episodes,
             "git_commit": git_commit()}

    if not args.summary_only:
        specs = {t.task_id: episode_specs(t, args.episodes) for t in tasks}
        for t in tasks:
            if len(specs[t.task_id]) < args.episodes:
                print(f"WARNING: {t.task_id} has only {len(specs[t.task_id])} frozen evaluation states", file=sys.stderr)
        jobs, groups = [], {}
        for step, model in checkpoints:  # checkpoint-major: a worker loads each checkpoint once
            for t in tasks:
                gdir = eval_root / step / t.task_id
                indices = [int(s["index"]) for s in specs[t.task_id]]
                if gdir.exists():
                    if not args.force and reusable(gdir, model, indices, settings):
                        print(f"{step_label(step)} {t.task_id}: complete evaluation exists ({gdir}), reused")
                        continue
                    old = move_aside(gdir)
                    print(f"{step_label(step)} {t.task_id}: earlier evaluation moved aside to {old}")
                groups[(step, t.task_id)] = {"dir": gdir, "model": model, "task": t, "indices": indices,
                                             "pending": len(specs[t.task_id]), "results": {}}
                for k, spec in enumerate(specs[t.task_id]):
                    video = str(gdir / "videos" / f"{t.task_id}_ep{int(spec['index']):03d}.mp4") \
                        if k < args.videos else None
                    jobs.append({"step": step, "model": model, "task_id": t.task_id, "spec": spec, "video": video,
                                 "detector": args.detector, "max_steps": args.max_steps})
        workers = args.workers
        if jobs and args.policy != "expert" and str(args.device).startswith("cuda") and not args.no_auto_workers:
            free = free_gpu_gb()
            if free is not None:
                fit = int(free // max(args.gpu_mem_per_worker, 0.1))
                if fit < workers:
                    print(f"WARNING: {free:.1f} GB free on the GPU fits {fit} SmolVLA worker(s) of "
                          f"~{args.gpu_mem_per_worker:g} GB: using {max(1, fit)} instead of {workers} "
                          "(--no-auto-workers keeps the requested number)", file=sys.stderr, flush=True)
                    workers = max(1, fit)
        if jobs:
            print(f"evaluating {len(groups)} checkpoint x task combination(s), {len(jobs)} episodes, {workers} "
                  f"worker(s), renderer {os.environ.get('MUJOCO_GL')}, device {args.device}", flush=True)
        t0 = time.time()

        def on_result(i: int, res: dict) -> None:
            job = jobs[i]
            g = groups[(job["step"], job["task_id"])]
            g["results"][int(job["spec"]["index"])] = res
            r = res["row"]
            print(f"[{step_label(job['step'])} {job['task_id']} #{r['index']} {r['layout_name']}"
                  f"{' ' + r['initial_permutation'] if r['initial_permutation'] else ''}] success={r['success']} "
                  f"steps={r['policy_steps']} home={r['returned_home']} complete={r['episode_complete']} "
                  f"({r['outcome']})", flush=True)
            if len(g["results"]) == g["pending"]:  # this checkpoint x task is done: write it now
                ordered = [g["results"][k] for k in sorted(g["results"])]
                rows = [x["row"] for x in ordered]
                summary = summarize(rows, g["task"], job["step"], g["model"], g["pending"])
                summary.update(episode_indices=g["indices"], **settings, renderer=os.environ.get("MUJOCO_GL"),
                               device=args.device, train_config=train_config(g["model"]), git_commit=extra["git_commit"],
                               timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
                write_group(g["dir"], rows, [x["record"] for x in ordered], summary)
                print(f"== {step_label(job['step'])} {job['task_id']}: {summary['successes']}/{summary['episodes']} "
                      f"success, returned home {summary['returned_home']}/{summary['episodes']} -> {g['dir']}",
                      flush=True)

        if jobs:
            run_jobs(jobs, workers, args.device if args.policy != "expert" else None, args.torch_threads, on_result)
            print(f"evaluation wall time {time.time() - t0:.0f} s", flush=True)

    table = collect(eval_root, tasks, checkpoints)
    report = aggregate(results_dir, name, tasks, table, run_dir, extra)
    print_summary(report, tasks, results_dir, name)
    incomplete = [f"{step_label(s)}/{t.task_id}" for s, _ in checkpoints for t in tasks
                  if t.task_id not in table.get(s, {})]
    if incomplete:
        print(f"INCOMPLETE (errors or not evaluated): {incomplete}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
