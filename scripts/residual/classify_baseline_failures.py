"""Classify shifted-baseline failures by actual arm motion.

Every failed episode from ``baseline_shifted_140/results.json`` is rerun with
the same state, seed, frozen policy, renderer, and horizon.  The diagnostic
measures the maximum absolute deviation of the five arm joints from their
reset values over the whole episode (the gripper is excluded because it uses a
0--100 unit rather than degrees).

Categories:
  - stalled: maximum arm-joint deviation < 5 degrees;
  - moved_but_missed: moved at least 5 degrees but never lifted the cube;
  - lifted_but_not_flipped: the cube was lifted but the episode failed.

For two stalled episodes, the current six-joint state and SmolVLA's six
absolute target actions are logged for steps 0--149 (three 50-action chunks).
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

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from so101_libero.cli.evaluate import classify
from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.policy.runner import PolicyRunner
from so101_libero.residual.placements import EVALUATION_SETS
from so101_libero.tasks.registry import get_task

DEFAULT_POLICY = Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model")
DEFAULT_BASELINE = Path("docs/residual/results/baseline_shifted_140/results.json")
DEFAULT_STATES = Path("docs/residual/placements")
DEFAULT_OUT = Path("docs/residual/results/baseline_shifted_140/stall_diagnostics")
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TRACE_STEPS = 150


def load_state_lookup(states_dir: Path) -> dict[tuple[str, int], dict]:
    lookup = {}
    for spec in EVALUATION_SETS:
        data = json.loads((states_dir / f"{spec.name}.json").read_text())
        for state in data["states"]:
            lookup[(spec.name, state["index"])] = state
    return lookup


def run_diagnostic_episode(runner: PolicyRunner, env: SO101TabletopEnv, task, baseline: dict, state: dict) -> dict:
    seed = int(state["seed"])
    obs, _ = env.reset(seed=seed, options={"init_state": state["init_state"]})
    torch.manual_seed(seed)
    runner.reset()

    start = np.asarray(obs["observation.state"], dtype=np.float64).copy()
    per_joint_max = np.zeros(5, dtype=np.float64)
    max_gripper_change = 0.0
    trace_states, trace_actions = [], []
    obj = task.plan.obj
    lifted = False
    steps, success, failure = 0, False, None
    while True:
        current = np.asarray(obs["observation.state"], dtype=np.float64)
        per_joint_max = np.maximum(per_joint_max, np.abs(current[:5] - start[:5]))
        max_gripper_change = max(max_gripper_change, abs(float(current[5] - start[5])))
        action = runner.act(obs, task.language)
        if steps < TRACE_STEPS:
            trace_states.append(current.astype(np.float32).copy())
            trace_actions.append(np.asarray(action, dtype=np.float32).copy())
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        if env.world.grasped(obj) and env.world.pos(obj)[2] - env.h.objects[obj].half_height > 0.01:
            lifted = True
        if terminated or truncated:
            success = bool(info["is_success"])
            failure = info.get("failure") or (None if success else "timeout")
            break

    final_state = np.asarray(obs["observation.state"], dtype=np.float64)
    per_joint_max = np.maximum(per_joint_max, np.abs(final_state[:5] - start[:5]))
    max_gripper_change = max(max_gripper_change, abs(float(final_state[5] - start[5])))
    if success:
        outcome = "success"
    elif failure == "simulation diverged" or (failure and "fell off" in failure):
        outcome = failure.replace(" ", "_")
    else:
        outcome = classify(env, task, lifted)

    return {
        "steps": steps,
        "success": success,
        "outcome": outcome,
        "lifted": lifted,
        "rerun_matches_baseline": success == bool(baseline["success"]) and outcome == baseline["outcome"],
        "start_state": start,
        "final_state": final_state,
        "max_arm_joint_change_deg": float(per_joint_max.max()),
        "max_change_by_arm_joint_deg": per_joint_max,
        "max_gripper_change_units": max_gripper_change,
        "trace_states": np.asarray(trace_states),
        "trace_actions": np.asarray(trace_actions),
    }


def category(row: dict, threshold_deg: float) -> str:
    # Category (c) is a fact from the saved baseline episode. Motion is measured
    # on the replay only to split baseline never-lifted failures into (a)/(b).
    if row["baseline_lifted"]:
        return "lifted_but_not_flipped"
    if row["max_arm_joint_change_deg"] < threshold_deg:
        return "stalled"
    return "moved_but_missed"


def counts_for(rows: list[dict]) -> dict:
    counts = Counter(row["motion_category"] for row in rows)
    return {
        "failed_episodes": len(rows),
        "stalled": counts["stalled"],
        "moved_but_missed": counts["moved_but_missed"],
        "lifted_but_not_flipped": counts["lifted_but_not_flipped"],
    }


def select_trace_rows(rows: list[dict], count: int) -> list[dict]:
    stalled = sorted(
        (
            row
            for row in rows
            if row["motion_category"] == "stalled" and row["rerun_matches_baseline"]
        ),
        key=lambda row: (row["max_arm_joint_change_deg"], row["global_index"]),
    )
    selected = []
    while stalled and len(selected) < count:
        candidate = next((row for row in stalled if row["set_id"] not in {x["set_id"] for x in selected}), stalled[0])
        selected.append(candidate)
        stalled.remove(candidate)
    return selected


def write_trace_csv(path: Path, row: dict) -> None:
    fields = ["step", "chunk_index", "step_in_chunk"]
    fields += [f"state_{name}" for name in JOINT_NAMES]
    fields += [f"action_{name}" for name in JOINT_NAMES]
    fields += ["max_abs_arm_action_minus_state_deg", "abs_gripper_action_minus_state_units"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, (state, action) in enumerate(zip(row["trace_states"], row["trace_actions"])):
            record = {"step": step, "chunk_index": step // 50, "step_in_chunk": step % 50}
            record.update({f"state_{name}": round(float(value), 5) for name, value in zip(JOINT_NAMES, state)})
            record.update({f"action_{name}": round(float(value), 5) for name, value in zip(JOINT_NAMES, action)})
            record["max_abs_arm_action_minus_state_deg"] = round(float(np.max(np.abs(action[:5] - state[:5]))), 5)
            record["abs_gripper_action_minus_state_units"] = round(float(abs(action[5] - state[5])), 5)
            writer.writerow(record)


def trace_chunk_summary(row: dict) -> list[dict]:
    summaries = []
    for chunk in range(3):
        sl = slice(chunk * 50, (chunk + 1) * 50)
        states, actions = row["trace_states"][sl], row["trace_actions"][sl]
        arm_delta = np.abs(actions[:, :5] - states[:, :5])
        summaries.append(
            {
                "chunk_index": chunk,
                "steps": [chunk * 50, (chunk + 1) * 50 - 1],
                "max_observed_arm_change_from_start_deg": round(
                    float(np.max(np.abs(states[:, :5] - row["start_state"][:5]))), 4
                ),
                "mean_abs_predicted_action_minus_state_arm_deg": round(float(arm_delta.mean()), 4),
                "max_abs_predicted_action_minus_state_arm_deg": round(float(arm_delta.max()), 4),
                "mean_abs_predicted_gripper_action_minus_state_units": round(
                    float(np.mean(np.abs(actions[:, 5] - states[:, 5]))), 4
                ),
                "predicted_action_range": {
                    name: [round(float(actions[:, j].min()), 4), round(float(actions[:, j].max()), 4)]
                    for j, name in enumerate(JOINT_NAMES)
                },
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--states-dir", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold-deg", type=float, default=5.0)
    parser.add_argument("--trace-count", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="smoke test: rerun only the first N failed episodes")
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "egl":
        parser.error("run on the GPU node with MUJOCO_GL=egl")

    baseline_report = json.loads(args.baseline.read_text())
    failures = [row for row in baseline_report["episodes"] if not row["success"]]
    if args.limit is not None:
        failures = failures[: args.limit]
    state_lookup = load_state_lookup(args.states_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    task = get_task("cube_flip")
    env = SO101TabletopEnv(task, EnvConfig(image_size=(256, 256), max_steps=task.max_steps, terminate_on_success=True))
    runner = PolicyRunner(args.policy, device=args.device)
    print(f"runner: {runner.describe()}", flush=True)
    print(f"rerunning {len(failures)} failed episodes with motion instrumentation", flush=True)

    started = time.time()
    rows = []
    try:
        for i, baseline in enumerate(failures):
            state = state_lookup[(baseline["set_id"], baseline["state_index"])]
            diag = run_diagnostic_episode(runner, env, task, baseline, state)
            row = {
                "global_index": baseline["global_index"],
                "set_id": baseline["set_id"],
                "state_index": baseline["state_index"],
                "seed": baseline["seed"],
                "offset_label": baseline["offset_label"],
                "training_region_tag": baseline["training_region_tag"],
                "baseline_outcome": baseline["outcome"],
                "baseline_lifted": bool(baseline["lifted"]),
                "rerun_outcome": diag["outcome"],
                "rerun_lifted": diag["lifted"],
                **{key: value for key, value in diag.items() if key not in ("outcome", "lifted")},
            }
            row["motion_category"] = category(row, args.threshold_deg)
            rows.append(row)
            print(
                f"{i + 1:03d}/{len(failures)} {row['set_id']} {row['offset_label']} state={row['state_index']:02d} "
                f"max_arm_change={row['max_arm_joint_change_deg']:.2f} deg -> {row['motion_category']}"
                + (" [REPLAY OUTCOME DIFFERS]" if not row["rerun_matches_baseline"] else ""),
                flush=True,
            )
    finally:
        env.close()

    selected = select_trace_rows(rows, args.trace_count)
    traces = []
    for row in selected:
        name = f"{row['set_id']}__{row['offset_label']}__state{row['state_index']:02d}__first_3_chunks.csv"
        write_trace_csv(args.out / name, row)
        traces.append(
            {
                "set_id": row["set_id"],
                "state_index": row["state_index"],
                "seed": row["seed"],
                "offset_label": row["offset_label"],
                "max_arm_joint_change_deg": round(row["max_arm_joint_change_deg"], 4),
                "csv": name,
                "chunks": trace_chunk_summary(row),
            }
        )

    per_set = {
        spec.name: counts_for([row for row in rows if row["set_id"] == spec.name]) for spec in EVALUATION_SETS
    }
    per_direction = {}
    for spec in EVALUATION_SETS:
        per_direction[spec.name] = {
            label: counts_for(
                [row for row in rows if row["set_id"] == spec.name and row["offset_label"] == label]
            )
            for label, _ in spec.offsets_m
        }
    report = {
        "policy": str(args.policy),
        "baseline": str(args.baseline),
        "threshold_definition": {
            "stalled": f"maximum absolute deviation from reset over arm joints 0-4 is < {args.threshold_deg} degrees",
            "gripper_excluded": True,
        },
        "complete": len(rows) == len(failures),
        "wall_time_s": round(time.time() - started, 1),
        "overall": counts_for(rows),
        "per_set": per_set,
        "per_direction": per_direction,
        "rerun_reproducibility": {
            "matched": sum(row["rerun_matches_baseline"] for row in rows),
            "mismatched": sum(not row["rerun_matches_baseline"] for row in rows),
            "mismatches": [
                {
                    "set_id": row["set_id"],
                    "state_index": row["state_index"],
                    "seed": row["seed"],
                    "baseline_outcome": row["baseline_outcome"],
                    "rerun_outcome": row["rerun_outcome"],
                    "motion_category": row["motion_category"],
                }
                for row in rows
                if not row["rerun_matches_baseline"]
            ],
        },
        "stalled_action_traces": traces,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fields = [
        "global_index",
        "set_id",
        "state_index",
        "seed",
        "offset_label",
        "training_region_tag",
        "baseline_outcome",
        "baseline_lifted",
        "rerun_outcome",
        "rerun_lifted",
        "rerun_matches_baseline",
        "motion_category",
        "max_arm_joint_change_deg",
        "max_gripper_change_units",
    ] + [f"max_change_{name}_deg" for name in JOINT_NAMES[:5]]
    with (args.out / "classification.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {field: row.get(field, "") for field in fields}
            for j, name in enumerate(JOINT_NAMES[:5]):
                record[f"max_change_{name}_deg"] = round(float(row["max_change_by_arm_joint_deg"][j]), 5)
            record["max_arm_joint_change_deg"] = round(float(row["max_arm_joint_change_deg"]), 5)
            record["max_gripper_change_units"] = round(float(row["max_gripper_change_units"]), 5)
            writer.writerow(record)

    print(json.dumps({"overall": report["overall"], "per_set": per_set, "per_direction": per_direction}, indent=2))
    print(f"results: {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
