"""Create the fixed evaluation initial states of each task (LIBERO-style ``init_states``).

    bash scripts/env.sh python -m so101_libero.cli.make_init_states --tasks so101_mini --num 50

Seeds come from a range disjoint from every training seed (>= 10,000,000). Each candidate state is checked with
the scripted expert (physics only) and kept only if the expert solves it, so evaluation failures are policy
failures, not impossible layouts. The file stores the sampled layouts themselves (not just seeds), plus a hash
of the scene, so the benchmark stays fixed even if the sampler code changes. Episode k of an evaluation uses
state k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EVAL_SEED_BASE = 10_000_000
INIT_DIR = Path(__file__).resolve().parents[2] / "init_states"


def scene_hash(task) -> str:
    """Fingerprint of the compiled scene: the MJCF (made path-independent and stable across compiles) plus the
    bytes of every referenced mesh file."""
    import re

    from so101_libero.sim.scene import ASSETS, build_spec

    spec = build_spec(task.layout.scene_config())
    xml = spec.to_xml()
    xml = re.sub(r'\s+content_type="[^"]*"', "", xml)  # only present before the first compile in a process
    xml = xml.replace(str(ASSETS), "<ASSETS>")
    h = hashlib.sha256(xml.encode())
    for f in sorted(set(re.findall(r'file="([^"]+)"', xml))):
        path = Path(f.replace("<ASSETS>", str(ASSETS)))
        if not path.is_absolute():
            path = ASSETS / "robotstudio_so101" / "assets" / f
        h.update(f.encode())
        h.update(path.read_bytes() if path.exists() else b"missing")
    return h.hexdigest()[:16]


def rules_hash(task) -> str:
    """Fingerprint of the evaluation rules: goal predicates, success hold, horizon, environment timing."""
    from so101_libero.env import EnvConfig

    c = EnvConfig()
    rules = repr((task.goal, task.success_hold_steps, task.max_steps, c.control_hz, c.substeps, c.settle_time))
    return hashlib.sha256(rules.encode()).hexdigest()[:16]


def init_state_path(task_id: str) -> Path:
    return INIT_DIR / f"{task_id}.json"


def load_init_states(task_id: str) -> dict:
    p = init_state_path(task_id)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run python -m so101_libero.cli.make_init_states --tasks {task_id}")
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["so101_mini"])
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--max-candidates", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    import so101_libero  # noqa: F401
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.rollout import run_expert_episode
    from so101_libero.tasks.registry import TASKS, get_suite

    INIT_DIR.mkdir(exist_ok=True)
    for task in [t for name in args.tasks for t in get_suite(name)]:
        path = init_state_path(task.task_id)
        if path.exists() and not args.force:
            print(f"{path} exists, skipping (--force to regenerate)")
            continue
        env = SO101TabletopEnv(task, EnvConfig(render=False, terminate_on_success=False, max_steps=900))
        base = EVAL_SEED_BASE + list(TASKS).index(task.task_id) * 100_000
        states, rejected = [], []
        for i in range(args.max_candidates):
            if len(states) == args.num:
                break
            ep = run_expert_episode(env, base + i)
            entry = {"seed": base + i, "init_state": ep.init_state,
                     "expert": {"success": ep.success, "failure": ep.failure, "steps": ep.steps}}
            (states if ep.success else rejected).append(entry)
        for k, s in enumerate(states):
            s["index"] = k
        if len(states) < args.num:
            raise RuntimeError(f"{task.task_id}: only {len(states)} expert-solvable states in {args.max_candidates}")
        path.write_text(json.dumps({
            "task_id": task.task_id, "language": task.language, "layout": task.layout.name,
            "scene_hash": scene_hash(task), "rules_hash": rules_hash(task), "seed_base": base, "num": len(states),
            "rejected_candidates": [{"seed": r["seed"], "failure": r["expert"]["failure"]} for r in rejected],
            "states": states}, indent=1))
        print(f"{task.task_id}: {len(states)} states ({len(rejected)} candidates rejected) -> {path}")


if __name__ == "__main__":
    main()
