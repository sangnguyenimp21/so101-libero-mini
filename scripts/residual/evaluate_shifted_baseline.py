"""Evaluate frozen SmolVLA on the expert-validated shifted cube-flip states.

The first pass evaluates every requested state without video.  A second,
deterministic pass reruns up to two failures per set with video enabled.  The
report preserves the per-state training-region tag and aggregates both by set
and by ``in-training-region`` / ``out-of-training-region``.

Run on a GPU node from the repository root:

    MUJOCO_GL=egl python -u scripts/residual/evaluate_shifted_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from so101_libero.cli.evaluate import (
    _run_parallel,
    training_image_size,
    wilson,
)
from so101_libero.cli.make_init_states import rules_hash, scene_hash
from so101_libero.residual.placements import EVALUATION_SETS
from so101_libero.tasks.registry import get_task

DEFAULT_POLICY = Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model")
DEFAULT_STATES = Path("docs/residual/placements")
DEFAULT_OUT = Path("docs/residual/results/baseline_shifted_140")
REGION_TAGS = ("in-training-region", "out-of-training-region")
FAILURE_TYPES = ("never_grasped", "dropped", "lifted_but_not_flipped", "timeout", "other")


def broad_failure_type(result: dict, max_steps: int) -> str | None:
    """Map the repository's detailed cube-flip outcome to the requested broad categories."""
    if result["success"]:
        return None
    outcome = result["outcome"]
    if outcome == "never_lifted":
        return "never_grasped"
    if outcome == "fell_off_table" or "fell_off" in outcome or outcome.startswith("dropped"):
        return "dropped"
    if outcome in {
        "still_held",
        "not_turned",
        "turned_but_not_flat",
        "turned_but_not_in_place_or_not_settled",
    } or result.get("lifted", False):
        return "lifted_but_not_flipped"
    if result["steps"] >= max_steps and not outcome.startswith("error"):
        return "timeout"
    return "other"


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    lo, hi = wilson(successes, n)
    failure_types = Counter(row["failure_type"] for row in rows if not row["success"])
    raw_outcomes = Counter(row["outcome"] for row in rows if not row["success"])
    success_steps = [row["steps"] for row in rows if row["success"]]
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": round(successes / n, 4) if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "failure_types": {name: int(failure_types[name]) for name in FAILURE_TYPES},
        "raw_failure_outcomes": dict(sorted(raw_outcomes.items())),
        "mean_steps_success": round(float(np.mean(success_steps)), 1) if success_steps else None,
    }


def choose_failure_videos(rows: list[dict], per_set: int) -> list[dict]:
    """Prefer two different offset directions when a set has enough failures."""
    if per_set <= 0:
        return []
    selected = []
    for set_id in [spec.name for spec in EVALUATION_SETS]:
        failures = [row for row in rows if row["set_id"] == set_id and not row["success"]]
        if not failures:
            continue
        selected.append(failures[0])
        while len([row for row in selected if row["set_id"] == set_id]) < min(per_set, len(failures)):
            used = {row["offset_label"] for row in selected if row["set_id"] == set_id}
            candidate = next((row for row in failures if row["offset_label"] not in used), None)
            if candidate is None:
                candidate = next(row for row in failures if row["global_index"] not in {x["global_index"] for x in selected})
            selected.append(candidate)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--states-dir", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--failure-videos", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--limit-per-set", type=int, default=None, help="smoke test: evaluate only the first N states per set")
    args = parser.parse_args()

    if os.environ.get("MUJOCO_GL") != "egl":
        parser.error("run this evaluation on the GPU node with MUJOCO_GL=egl")
    args.out.mkdir(parents=True, exist_ok=True)
    task = get_task("cube_flip")

    import lerobot.policies  # noqa: F401  (register policy config classes)
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    image_size = training_image_size(str(args.policy)) or (256, 256)
    policy_kw = {"device": args.device, "n_action_steps": None, "temporal_ensemble_coeff": None}
    runner_args = SimpleNamespace(
        policy=str(args.policy),
        workers=args.workers,
        no_render=False,
        visual_rand=False,
    )

    warnings: list[str] = []

    def warn(message: str) -> None:
        warnings.append(message)
        print(f"WARNING: {message}", file=sys.stderr, flush=True)

    jobs, metadata = [], {}
    expected_scene, expected_rules = scene_hash(task), rules_hash(task)
    for spec in EVALUATION_SETS:
        path = args.states_dir / f"{spec.name}.json"
        data = json.loads(path.read_text())
        if data["scene_hash"] != expected_scene:
            raise RuntimeError(f"{path}: scene hash changed ({data['scene_hash']} -> {expected_scene})")
        if data.get("rules_hash") != expected_rules:
            raise RuntimeError(f"{path}: rules hash changed ({data.get('rules_hash')} -> {expected_rules})")
        states = data["states"] if args.limit_per_set is None else data["states"][: args.limit_per_set]
        for state in states:
            global_index = len(jobs)
            jobs.append(
                {
                    "task_id": task.task_id,
                    "index": global_index,
                    "seed": state["seed"],
                    "init_state": state["init_state"],
                    "max_steps": task.max_steps,
                    "video": False,
                    "out": str(args.out),
                }
            )
            metadata[global_index] = {
                "global_index": global_index,
                "set_id": spec.name,
                "state_index": state["index"],
                "seed": state["seed"],
                "offset_label": state["offset_label"],
                "offset_m": state["offset_m"],
                "training_region_tag": state["training_region_tag"],
            }

    print(
        f"evaluating frozen {cfg.type} baseline on {len(jobs)} shifted states; image_size={image_size}, "
        f"renderer=egl, workers={args.workers}, out={args.out}",
        flush=True,
    )
    started = time.time()
    results = _run_parallel(jobs, runner_args, image_size, policy_kw, warn)
    rows = []
    for job, result in zip(jobs, results):
        row = {**metadata[result["index"]], **result}
        row["failure_type"] = broad_failure_type(row, job["max_steps"])
        rows.append(row)

    errors = [row for row in rows if row["outcome"].startswith("error")]
    if errors:
        warn(f"{len(errors)} episode(s) raised errors and are counted as failures")

    selected = choose_failure_videos(rows, args.failure_videos)
    video_rows = []
    if selected:
        print(f"rerunning {len(selected)} selected failures with video", flush=True)
        by_index = {job["index"]: job for job in jobs}
        video_jobs = [{**by_index[row["global_index"]], "video": True} for row in selected]
        video_results = _run_parallel(video_jobs, runner_args, image_size, policy_kw, warn)
        row_by_index = {row["global_index"]: row for row in rows}
        for original, video_result in zip(selected, video_results):
            target = row_by_index[original["global_index"]]
            if video_result["success"] != target["success"] or video_result["outcome"] != target["outcome"]:
                warn(
                    f"video rerun changed result for global state {target['global_index']}: "
                    f"{target['outcome']} -> {video_result['outcome']}"
                )
                continue
            source = args.out / video_result["video"]
            destination = source.with_name(
                f"{target['set_id']}__{target['offset_label']}__state{target['state_index']:02d}_fail.mp4"
            )
            source.rename(destination)
            target["video"] = str(destination.relative_to(args.out))
            video_rows.append(target)

    per_set = {
        spec.name: summarize([row for row in rows if row["set_id"] == spec.name]) for spec in EVALUATION_SETS
    }
    per_region = {tag: summarize([row for row in rows if row["training_region_tag"] == tag]) for tag in REGION_TAGS}
    overall = summarize(rows)
    report = {
        "policy": {"path": str(args.policy), "type": cfg.type, "device": args.device},
        "states_dir": str(args.states_dir),
        "render_backend": "egl",
        "image_size": list(image_size),
        "complete": not errors and len(rows) == len(jobs),
        "wall_time_s": round(time.time() - started, 1),
        "failure_type_mapping": {
            "never_grasped": ["never_lifted"],
            "dropped": ["fell_off_table", "fell_off_*", "dropped*"],
            "lifted_but_not_flipped": [
                "still_held",
                "not_turned",
                "turned_but_not_flat",
                "turned_but_not_in_place_or_not_settled",
                "any other outcome with lifted=true",
            ],
            "timeout": "reached horizon without a more specific diagnostic",
            "other": "remaining outcomes and runtime errors",
        },
        "overall": overall,
        "per_set": per_set,
        "per_training_region": per_region,
        "failure_videos": [row.get("video") for row in video_rows],
        "warnings": warnings,
        "episodes": rows,
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    fields = (
        "global_index",
        "set_id",
        "state_index",
        "seed",
        "offset_label",
        "offset_m",
        "training_region_tag",
        "success",
        "steps",
        "failure_type",
        "outcome",
        "lifted",
        "max_lift_cm",
        "mean_policy_ms",
        "video",
    )
    with (args.out / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    print(json.dumps({"overall": overall, "per_set": per_set, "per_training_region": per_region}, indent=2))
    print(f"results: {args.out / 'results.json'}", flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
