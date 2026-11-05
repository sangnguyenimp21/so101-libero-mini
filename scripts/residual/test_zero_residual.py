"""Mandatory equivalence test: zero residual on ten original frozen states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from so101_libero.cli.make_init_states import load_init_states
from so101_libero.residual.residual_env import ResidualConfig, ResidualSO101Env

DEFAULT_POLICY = Path("checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model")
DEFAULT_BASELINE = Path("docs/residual/results/baseline_original_50/results.json")
DEFAULT_OUT = Path("docs/residual/results/zero_residual_test.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    frozen = load_init_states("cube_flip")["states"][: args.episodes]
    baseline = json.loads(args.baseline.read_text())["episodes"][: args.episodes]
    if not all(row["success"] for row in baseline):
        raise RuntimeError("saved SmolVLA baseline did not succeed on all selected states")

    env = ResidualSO101Env(ResidualConfig(checkpoint=args.policy))
    rows = []
    try:
        for i, (state, expected) in enumerate(zip(frozen, baseline)):
            seed = int(state["seed"])
            torch.manual_seed(seed)
            _, _ = env.reset(seed=seed, options={"init_state": state["init_state"]})
            steps, total_reward = 0, 0.0
            while True:
                _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
                steps += 1
                total_reward += reward
                if terminated or truncated:
                    break
            row = {
                "index": i,
                "seed": seed,
                "expected_success": bool(expected["success"]),
                "zero_residual_success": bool(info["is_success"]),
                "steps": steps,
                "total_reward": round(total_reward, 6),
            }
            rows.append(row)
            print(f"{i + 1:02d}/{len(frozen)} seed={seed}: success={row['zero_residual_success']} steps={steps}", flush=True)
    finally:
        env.close()
    matches = all(row["expected_success"] == row["zero_residual_success"] for row in rows)
    report = {
        "complete": len(rows) == args.episodes,
        "matches_saved_baseline": matches,
        "expected_successes": sum(row["expected_success"] for row in rows),
        "zero_residual_successes": sum(row["zero_residual_success"] for row in rows),
        "episodes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
    if not matches:
        raise SystemExit("zero residual does not match the saved SmolVLA baseline")


if __name__ == "__main__":
    main()
