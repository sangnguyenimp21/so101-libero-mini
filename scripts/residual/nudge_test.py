"""Diagnose stalled SmolVLA episodes with a bounded first-chunk expert nudge.

Only baseline-stalled states from the 2 cm axis and 1.5 cm diagonal sets are
tested.  During the first action chunk, each arm-joint target is moved toward
the scripted expert's collision-checked pre-grasp configuration by at most the
requested number of degrees.  The gripper is unchanged.  From step 50 onward,
the frozen SmolVLA action is used without modification.
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

from so101_libero.cli.evaluate import classify, wilson
from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.expert.rollout import expert_rng, make_expert
from so101_libero.policy.runner import PolicyRunner
from so101_libero.tasks.registry import get_task

DEFAULT_POLICY = Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model")
DEFAULT_BASELINE = Path("docs/residual/results/baseline_shifted_140/results.json")
DEFAULT_CLASSIFICATION = Path(
    "docs/residual/results/baseline_shifted_140/stall_diagnostics/classification.csv"
)
DEFAULT_STATES = Path("docs/residual/placements")
DEFAULT_OUT = Path("docs/residual/results/nudge_test")
TARGET_SETS = ("shift_2cm_axis", "shift_1p5cm_diagonal")


def load_stalled(classification: Path, baseline: Path, states_dir: Path) -> list[dict]:
    baseline_rows = {
        (row["set_id"], int(row["state_index"])): row
        for row in json.loads(baseline.read_text())["episodes"]
    }
    states = {}
    for set_id in TARGET_SETS:
        for state in json.loads((states_dir / f"{set_id}.json").read_text())["states"]:
            states[(set_id, int(state["index"]))] = state
    selected = []
    with classification.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["set_id"], int(row["state_index"]))
            if row["set_id"] in TARGET_SETS and row["motion_category"] == "stalled":
                selected.append({**states[key], "set_id": key[0], "baseline": baseline_rows[key]})
    if len(selected) != 40:
        raise RuntimeError(f"expected 40 target stalled states, found {len(selected)}")
    return selected


def expert_pregrasp_deg(env: SO101TabletopEnv, task, seed: int) -> np.ndarray:
    """Plan the expert grasp without stepping physics and return its pre-grasp arm target."""
    expert = make_expert(env, task.plan, expert_rng(seed))
    obj_pos = env.world.pos(task.plan.obj).copy()
    expert._approach(obj_pos, expert._face_psis(), expert._u(expert.pregrasp_z), expert.grasp_z[0])
    return np.degrees(expert._q[:5]).astype(np.float32)


def run_episode(runner, env, task, state: dict, nudge_deg: float, nudge_steps: int) -> dict:
    seed = int(state["seed"])
    obs, _ = env.reset(seed=seed, options={"init_state": state["init_state"]})
    pregrasp = expert_pregrasp_deg(env, task, seed)
    torch.manual_seed(seed)
    runner.reset()
    lifted = False
    max_arm_change = 0.0
    start = np.asarray(obs["observation.state"], dtype=np.float32).copy()
    steps = 0
    while True:
        base_action = runner.act(obs, task.language)
        action = base_action.copy()
        if steps < nudge_steps:
            action[:5] += np.clip(pregrasp - base_action[:5], -nudge_deg, nudge_deg)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        current = np.asarray(obs["observation.state"], dtype=np.float32)
        max_arm_change = max(max_arm_change, float(np.max(np.abs(current[:5] - start[:5]))))
        obj = task.plan.obj
        if env.world.grasped(obj) and env.world.pos(obj)[2] - env.h.objects[obj].half_height > 0.01:
            lifted = True
        if terminated or truncated:
            success = bool(info["is_success"])
            failure = info.get("failure") or (None if success else "timeout")
            if success:
                outcome = "success"
            elif failure == "simulation diverged" or (failure and "fell off" in failure):
                outcome = failure.replace(" ", "_")
            else:
                outcome = classify(env, task, lifted)
            return {
                "success": success,
                "steps": steps,
                "outcome": outcome,
                "lifted": lifted,
                "max_arm_joint_change_deg": round(max_arm_change, 5),
                "expert_pregrasp_deg": [round(float(x), 5) for x in pregrasp],
            }


def summary(rows: list[dict]) -> dict:
    successes = sum(row["success"] for row in rows)
    if not rows:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": None,
            "wilson95": None,
            "lifted": 0,
            "outcomes": {},
        }
    lo, hi = wilson(successes, len(rows))
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": round(successes / len(rows), 4),
        "wilson95": [round(lo, 4), round(hi, 4)],
        "lifted": sum(row["lifted"] for row in rows),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--states-dir", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nudge-deg", type=float, nargs="+", default=(2.0, 5.0, 10.0))
    parser.add_argument("--nudge-steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "egl":
        parser.error("run on the GPU node with MUJOCO_GL=egl")

    states = load_stalled(args.classification, args.baseline, args.states_dir)
    if args.limit is not None:
        states = states[: args.limit]
    task = get_task("cube_flip")
    env = SO101TabletopEnv(
        task, EnvConfig(image_size=(256, 256), max_steps=task.max_steps, terminate_on_success=True)
    )
    runner = PolicyRunner(args.policy, device=args.device)
    print(f"runner: {runner.describe()}", flush=True)
    rows = []
    started = time.time()
    try:
        for nudge in args.nudge_deg:
            for i, state in enumerate(states):
                result = run_episode(runner, env, task, state, nudge, args.nudge_steps)
                row = {
                    "set_id": state["set_id"],
                    "state_index": int(state["index"]),
                    "seed": int(state["seed"]),
                    "offset_label": state["offset_label"],
                    "nudge_deg": float(nudge),
                    "nudge_steps": args.nudge_steps,
                    **result,
                }
                rows.append(row)
                print(
                    f"nudge={nudge:g} {i + 1:02d}/{len(states)} {row['set_id']} "
                    f"{row['offset_label']} state={row['state_index']:02d}: {row['outcome']}",
                    flush=True,
                )
    finally:
        env.close()

    by_nudge = {
        f"{nudge:g}": summary([row for row in rows if row["nudge_deg"] == nudge])
        for nudge in args.nudge_deg
    }
    by_set = {
        f"{nudge:g}": {
            set_id: summary([row for row in rows if row["nudge_deg"] == nudge and row["set_id"] == set_id])
            for set_id in TARGET_SETS
        }
        for nudge in args.nudge_deg
    }
    report = {
        "protocol": {
            "policy": str(args.policy),
            "target_sets": list(TARGET_SETS),
            "selection": "baseline motion_category == stalled",
            "nudge_steps": args.nudge_steps,
            "arm_nudge_deg": list(args.nudge_deg),
            "gripper_nudge": 0.0,
            "target": "scripted expert collision-checked pre-grasp joint configuration",
            "smolvla_frozen": True,
        },
        "complete": args.limit is None,
        "elapsed_s": round(time.time() - started, 2),
        "by_nudge": by_nudge,
        "by_set": by_set,
        "episodes": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "episodes.csv").open("w", newline="") as handle:
        flat_rows = [{k: v for k, v in row.items() if k != "expert_pregrasp_deg"} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(by_nudge, indent=2), flush=True)


if __name__ == "__main__":
    main()
