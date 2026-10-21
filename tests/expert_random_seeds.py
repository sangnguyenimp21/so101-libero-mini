"""Expert success on random training-distribution seeds (physics only), saved as JSON.

    SO101_RENDER=none bash scripts/env.sh python tests/expert_random_seeds.py --seeds 100 --out docs/results/expert_random_seeds.json
"""
import argparse
import collections
import json
import multiprocessing as mp
from pathlib import Path


def run(args):
    import so101_libero  # noqa: F401
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.rollout import run_expert_episode

    task, seeds = args
    env = SO101TabletopEnv(task, EnvConfig(render=False, terminate_on_success=False, max_steps=900))
    eps = [run_expert_episode(env, s) for s in seeds]
    return task, {"episodes": len(eps), "successes": sum(e.success for e in eps),
                  "failures": dict(collections.Counter(e.failure for e in eps if not e.success)),
                  "mean_steps": round(sum(e.steps for e in eps) / len(eps), 1),
                  "max_slip_mm": max(e.grasp["max_slip_mm"] for e in eps)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from so101_libero.tasks.registry import TASKS, seed_slot

    jobs = [(t, list(range(k * 100_000 + 5_000, k * 100_000 + 5_000 + a.seeds))) for t in TASKS for k in [seed_slot(t)]]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = dict(pool.map(run, jobs))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"seeds_per_task": a.seeds, "note": "training-distribution seeds k*100000+5000+i "
                                 "(not used by the stored datasets), default expert config", "tasks": res}, indent=2))
    print(json.dumps(res, indent=2))
