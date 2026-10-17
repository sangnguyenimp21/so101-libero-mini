"""Evaluate a trained policy checkpoint (or the scripted expert) on the fixed evaluation initial states.

    bash scripts/env.sh python -m so101_libero.cli.evaluate --policy outputs/train/act_t1/checkpoints/last/pretrained_model \
        --tasks cylinder_to_tray --episodes 50 --videos 10 --out outputs/eval/act_t1
    bash scripts/env.sh python -m so101_libero.cli.evaluate --policy expert --tasks so101_mini --episodes 50

Success is the task's goal predicate in the simulator (e.g. object resting inside the target container, released
and at rest, for 5 consecutive steps), never training loss or appearance. The policy only gets the observation
(joint state + images) and the instruction; privileged state is read afterwards for failure diagnostics only.
Writes <out>/results.json (summary + per-episode), <out>/episodes.csv and <out>/videos/*.mp4.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def classify(env, task, lifted: bool) -> str:
    """Where the manipulated object ended up (privileged, for diagnostics only)."""
    from so101_libero.tasks.base import PickPlace, PlaceAt, inside_any_container

    ws = env.world
    obj = task.plan.obj if task.plan else task.goal[0].obj
    target = task.plan.target if isinstance(task.plan, PickPlace) else None
    if ws.pos(obj)[2] < -0.05:
        return "fell_off_table"
    if isinstance(task.plan, PlaceAt):
        from so101_libero.tasks.base import OnTable, OnTableInRegion

        pred = next((g for g in task.goal if isinstance(g, OnTableInRegion)), None)
        if not ws.lifted(obj, pred.min_lift if pred else 0.02):  # the predicate's own notion: carried in the jaws
            return "never_lifted" if not lifted else "lift_below_min"
        if ws.touching_robot(obj):
            return "still_held"
        reg, p = task.plan.region, ws.pos(obj)
        if not (reg.x[0] <= p[0] <= reg.x[1] and reg.y[0] <= p[1] <= reg.y[1]):
            return "placed_outside_zone"
        if not OnTable(obj)(ws) or not ws.upright(obj):
            return "in_zone_tipped_or_stacked"
        off = ws.release_offset(obj)
        if off is not None and pred is not None and off > pred.max_release_offset:
            return "in_zone_but_pushed_after_release"
        if pred is not None and float(np.linalg.norm(ws.tcp_pos() - p)) < pred.min_tcp_distance:
            return "in_zone_gripper_still_near"
        return "in_zone_not_settled"  # e.g. still moving when the episode ended
    inside = inside_any_container(ws, obj)
    if inside is not None:
        return "in_target_not_settled_or_held" if inside == target else "in_a_container"
    if not lifted:
        return "never_lifted"
    if ws.touching_robot(obj):
        return "still_held"
    return "dropped_or_placed_wrong"


_STATE = {}


def _init_worker(policy: str, image_size, policy_kw: dict, render: bool = True, visual_rand: bool = False):
    import so101_libero  # noqa: F401

    _STATE["policy"] = policy
    _STATE["image_size"] = tuple(image_size)
    _STATE["render"] = render
    _STATE["visual_rand"] = visual_rand
    _STATE["envs"] = {}
    try:
        if policy != "expert":
            from so101_libero.policy.runner import PolicyRunner

            _STATE["runner"] = PolicyRunner(policy, **policy_kw)
    except Exception as e:  # noqa: BLE001  (report the real cause with every episode instead of a dead pool)
        _STATE["init_error"] = f"{type(e).__name__}: {e}"


def _episode(job: dict) -> dict:
    if "init_error" in _STATE:
        return {"task_id": job["task_id"], "index": job["index"], "seed": job["seed"], "success": False, "steps": 0,
                "outcome": f"error: policy failed to load: {_STATE['init_error']}"[:300], "lifted": False,
                "other_objects_lifted": [], "mean_policy_ms": 0.0}
    try:
        return _run_episode(job)
    except Exception as e:  # noqa: BLE001  (one bad episode must not lose the whole evaluation)
        return {"task_id": job["task_id"], "index": job["index"], "seed": job["seed"], "success": False,
                "steps": 0, "outcome": f"error: {type(e).__name__}: {e}"[:300], "lifted": False, "mean_policy_ms": 0.0}


def _run_episode(job: dict) -> dict:
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.pick_place import PickPlaceExpert
    from so101_libero.expert.rollout import expert_rng
    from so101_libero.tasks.registry import get_task
    from so101_libero.utils.video import label, write_mp4

    task = get_task(job["task_id"])
    key = job["task_id"]
    if key not in _STATE["envs"]:
        _STATE["envs"][key] = SO101TabletopEnv(task, EnvConfig(image_size=_STATE["image_size"], max_steps=job["max_steps"],
                                                                terminate_on_success=True, render=_STATE["render"],
                                                                visual_rand=_STATE.get("visual_rand", False)))
    env = _STATE["envs"][key]
    # The fixed state is used as is; the seed only drives the optional appearance randomization.
    obs, info = env.reset(seed=job["seed"], options={"init_state": job["init_state"]})
    expert = None
    if _STATE["policy"] == "expert":
        expert = PickPlaceExpert(env, task.plan, expert_rng(job["seed"]))
    else:
        import torch

        torch.manual_seed(job["seed"])  # Diffusion / SmolVLA sample noise: same episode -> same result
        _STATE["runner"].reset()
    frames, lifted, t_inf = [], False, 0.0
    obj = task.plan.obj if task.plan else None
    others_lifted: set[str] = set()  # non-target objects the policy picked up (language-grounding diagnostic)
    steps, success, failure = 0, False, None
    while True:
        if job["video"]:
            frames.append(label(env.render(), f"{key} #{job['index']} t={steps} success={success}"))
        t0 = time.perf_counter()
        action = expert.act() if expert else _STATE["runner"].act(obs, task.language)
        t_inf += time.perf_counter() - t0
        if action is None:  # expert finished without success (cannot happen for verified states)
            failure = expert.failure or "expert_stopped"
            break
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        for name, spec in env.h.objects.items():
            if env.world.grasped(name) and env.world.pos(name)[2] - spec.half_height > 0.01:
                if name == obj:
                    lifted = True
                else:
                    others_lifted.add(name)
        if terminated or truncated:
            success = bool(info["is_success"])
            failure = info.get("failure") or (None if success else "timeout")
            break
    if job["video"]:
        frames.append(label(env.render(), f"{key} #{job['index']} t={steps} success={success}"))
    if success:
        outcome = "success"
    elif failure in ("simulation diverged",) or (failure and "fell off" in failure):
        outcome = failure.replace(" ", "_")
    else:
        outcome = classify(env, task, lifted)
        if not lifted and others_lifted:
            outcome = "wrong_object_lifted"
    res = {"task_id": key, "index": job["index"], "seed": job["seed"], "success": success, "steps": steps,
           "outcome": outcome, "lifted": lifted, "other_objects_lifted": sorted(others_lifted),
           "max_lift_cm": round(100 * env.world.max_lift.get(obj, 0.0), 2) if obj else None,
           "mean_policy_ms": round(1e3 * t_inf / max(steps, 1), 2)}
    if job["video"]:
        name = f"{key}_ep{job['index']:03d}_{'success' if success else 'fail'}.mp4"
        write_mp4(frames, Path(job["out"]) / "videos" / name, fps=env.cfg.control_hz)
        res["video"] = f"videos/{name}"
    return res


def _run_parallel(jobs: list[dict], args, image_size, policy_kw: dict, warn, max_crashes: int = 2,
                  episode_fn=None) -> list[dict]:
    """Run episodes in worker processes, at most ``workers`` in flight. A worker that dies (rare native crashes have
    been seen when CPU torch inference and OSMesa share a process) breaks the pool: the pool is rebuilt and the
    episodes that were running are retried; queued ones are not charged. Each episode is deterministic (fixed
    initial state, seeded torch), so a rerun gives the same result. An episode whose worker died ``max_crashes``
    times is recorded as an error (= failure): one retry, two attempts at most. ``episode_fn`` (picklable) replaces
    the episode function in tests."""
    episode_fn = episode_fn or _episode
    if args.workers > 1:
        os.environ.setdefault("SO101_TORCH_THREADS", "1")  # one CPU thread per worker: no nested parallel regions
    done: dict[int, dict] = {}
    crashes = {i: 0 for i in range(len(jobs))}
    while len(done) < len(jobs):
        queue = [i for i in range(len(jobs)) if i not in done]
        running: dict = {}
        try:
            with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                        initargs=(args.policy, image_size, policy_kw, not args.no_render,
                                                  getattr(args, "visual_rand", False))) as pool:
                while queue or running:
                    while queue and len(running) < args.workers:
                        i = queue.pop(0)
                        running[pool.submit(episode_fn, jobs[i])] = i
                    finished, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                    for fut in finished:
                        i = running[fut]
                        r = fut.result()  # raises BrokenProcessPool if the worker died: i stays in `running`
                        del running[fut]
                        done[i] = r
                        print(json.dumps(r), flush=True)
        except cf.process.BrokenProcessPool:
            lost = sorted(running.values())
            warn(f"a worker process died; retrying the {len(lost)} episode(s) that were running in a new pool")
            for i in lost:
                crashes[i] += 1
                if crashes[i] >= max_crashes:
                    j = jobs[i]
                    done[i] = {"task_id": j["task_id"], "index": j["index"], "seed": j["seed"], "success": False,
                               "steps": 0, "outcome": "error: worker process crashed", "lifted": False,
                               "other_objects_lifted": [], "mean_policy_ms": 0.0}
    return [done[i] for i in range(len(jobs))]


def training_dataset_meta(checkpoint: str) -> dict:
    """The so101_libero generation metadata of the dataset a checkpoint was trained on ({} if unavailable)."""
    try:
        tc = json.loads((Path(checkpoint) / "train_config.json").read_text())
        return json.loads((Path(tc["dataset"]["root"]) / "meta" / "so101_libero.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


def training_image_size(checkpoint: str) -> tuple[int, int] | None:
    """Image size of the dataset the checkpoint was trained on (more reliable than the policy config, which for
    fine-tuned pretrained models keeps the base model's shapes)."""
    try:
        tc = json.loads((Path(checkpoint) / "train_config.json").read_text())
        root = Path(tc["dataset"]["root"])
        feats = json.loads((root / "meta" / "info.json").read_text())["features"]
        h, w = feats["observation.images.front"]["shape"][:2]
        return int(h), int(w)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="checkpoint dir (.../pretrained_model) or 'expert'")
    ap.add_argument("--tasks", nargs="+", default=["cylinder_to_tray"], help="task ids or a suite name")
    ap.add_argument("--episodes", type=int, default=50, help="per task (first N fixed init states)")
    ap.add_argument("--videos", type=int, default=5, help="record the first N episodes of each task")
    ap.add_argument("--max-steps", type=int, default=None, help="default: the task horizon")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-action-steps", type=int, default=None, help="ACT/Diffusion: actions executed per inference")
    ap.add_argument("--temporal-ensemble", type=float, default=None, help="ACT temporal ensembling coefficient")
    ap.add_argument("--no-render", action="store_true", help="physics only (expert evaluation without cameras)")
    ap.add_argument("--visual-rand", action="store_true",
                    help="appearance randomization (lights, colours, camera jitter, image noise) seeded per episode; "
                         "the physical initial states stay the frozen ones")
    args = ap.parse_args()
    if args.no_render and args.policy != "expert":
        ap.error("--no-render only works with --policy expert (policies need images)")
    if args.policy != "expert" and not os.environ.get("MUJOCO_GL"):
        ap.error("a policy needs camera images but rendering is off (SO101_RENDER=none or no GL backend): "
                 "use SO101_RENDER=osmesa (CPU) or egl (GPU)")
    if args.no_render:
        args.videos = 0

    import so101_libero  # noqa: F401
    from so101_libero.cli.make_init_states import load_init_states, rules_hash, scene_hash
    from so101_libero.tasks.registry import get_suite

    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)

    tasks = [t for name in args.tasks for t in get_suite(name)]
    image_size = (256, 256)
    policy_kw = {"device": args.device, "n_action_steps": args.n_action_steps,
                 "temporal_ensemble_coeff": args.temporal_ensemble}
    policy_info = {"policy": "scripted expert (privileged state)"}
    if args.policy != "expert":
        import lerobot.policies  # noqa: F401  (registers the policy config classes)
        from lerobot.configs.policies import PreTrainedConfig

        cfg = PreTrainedConfig.from_pretrained(args.policy)
        cams = [f for k, f in cfg.input_features.items() if k.startswith("observation.images.")]
        image_size = training_image_size(args.policy) or (tuple(cams[0].shape[1:]) if cams else image_size)
        language = cfg.type in ("smolvla", "pi0", "pi05", "groot", "xvla")
        policy_info = {"policy": str(args.policy), "type": cfg.type, "language_conditioned": language,
                       **{k: v for k, v in policy_kw.items() if v is not None}}
        if not language and len({t.layout.name for t in tasks}) < len(tasks):
            warn(f"{cfg.type} ignores the instruction, but several evaluated tasks share a scene: it cannot know "
                 "which task to do (useful only as a no-language baseline)")
        dmeta = training_dataset_meta(args.policy)
        policy_info["training_dataset"] = dmeta.get("repo_id")
        policy_info["training_render_backend"] = dmeta.get("render_backend")
        if dmeta.get("render_backend") and dmeta["render_backend"] != os.environ.get("MUJOCO_GL"):
            warn(f"training images were rendered with {dmeta['render_backend']}, this evaluation renders with "
                 f"{os.environ.get('MUJOCO_GL')} (set SO101_RENDER={dmeta['render_backend']} for an identical renderer)")
        for t in tasks:
            old = dmeta.get("scene_hashes", {}).get(t.task_id)
            if old and old != scene_hash(t):
                warn(f"{t.task_id}: the scene changed since the training data was generated ({old} -> {scene_hash(t)})")
    out = args.out or Path("outputs/eval") / (
        "expert" if args.policy == "expert" else Path(args.policy).resolve().parents[2].name + "_" + Path(args.policy).resolve().parents[0].name)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for task in tasks:
        init = load_init_states(task.task_id)
        if init["scene_hash"] != scene_hash(task):
            warn(f"{task.task_id}: the scene changed since the evaluation states were made "
                 f"({init['scene_hash']} -> {scene_hash(task)}); regenerate them with make_init_states --force")
        if init.get("rules_hash") not in (None, rules_hash(task)):
            warn(f"{task.task_id}: the success rules / horizon changed since the evaluation states were made")
        for s in init["states"][: args.episodes]:
            jobs.append({"task_id": task.task_id, "index": s["index"], "seed": s["seed"], "init_state": s["init_state"],
                         "max_steps": args.max_steps or task.max_steps, "video": s["index"] < args.videos,
                         "out": str(out)})
    print(f"evaluating {policy_info} on {len(jobs)} episodes ({len(tasks)} tasks), image size {image_size}, "
          f"renderer {os.environ.get('MUJOCO_GL')}, out={out}", flush=True)
    t0 = time.time()
    results = []
    # Episodes always run in worker processes (also with --workers 1): a native crash in rendering/inference then
    # costs a retry of that episode instead of the whole evaluation.
    results = _run_parallel(jobs, args, image_size, policy_kw, warn)
    errors = [r for r in results if r["outcome"].startswith("error")]
    if errors:
        warn(f"{len(errors)} episode(s) raised an error (counted as failures), e.g. {errors[0]['outcome']}")

    summary = {}
    for task in tasks:
        rs = [r for r in results if r["task_id"] == task.task_id]
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        outcomes = {}
        for r in rs:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        summary[task.task_id] = {"language": task.language, "episodes": len(rs), "successes": k,
                                 "success_rate": round(k / max(len(rs), 1), 4), "wilson95": [round(lo, 4), round(hi, 4)],
                                 "outcomes": outcomes,
                                 "mean_steps_success": round(float(np.mean([r["steps"] for r in rs if r["success"]])), 1)
                                 if k else None}
    n = len(results)
    k = sum(r["success"] for r in results)
    overall = {"episodes": n, "successes": k, "success_rate": round(k / max(n, 1), 4),
               "wilson95": [round(x, 4) for x in wilson(k, n)]}
    report = {"policy": policy_info, "overall": overall, "per_task": summary, "warnings": warnings,
              "complete": not errors,
              "wall_time_s": round(time.time() - t0, 1), "render_backend": os.environ.get("MUJOCO_GL"),
              "visual_rand": bool(args.visual_rand),
              "image_size": list(image_size), "episodes": results}
    if errors and len(errors) == n:  # nothing ran: never leave a results.json that looks like a 0 % policy
        (out / "results_failed.json").write_text(json.dumps(report, indent=2))
        print(f"ALL EPISODES FAILED TO RUN: {errors[0]['outcome']}  ({out / 'results_failed.json'})", file=sys.stderr)
        sys.exit(1)
    (out / "results.json").write_text(json.dumps(report, indent=2))
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "index", "seed", "success", "steps", "outcome", "lifted",
                                          "max_lift_cm", "other_objects_lifted", "mean_policy_ms", "video"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(json.dumps({"overall": overall, "per_task": summary, "warnings": warnings}, indent=2))
    print(f"results: {out / 'results.json'}" + ("" if not errors else f"  (INCOMPLETE: {len(errors)} episode errors)"))
    if errors:  # the files are written (complete: false) but the numbers are not a full evaluation: tell the caller
        sys.exit(2)


if __name__ == "__main__":
    main()
