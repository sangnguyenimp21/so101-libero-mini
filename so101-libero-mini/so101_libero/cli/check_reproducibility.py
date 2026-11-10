"""Regenerate a dataset's episodes from their seeds (physics only) and compare the actions bit for bit.

    bash scripts/env.sh python -m so101_libero.cli.check_reproducibility --repo-id local/so101_mini_t1_200 --episodes 20

Uses the expert configuration recorded in meta/so101_libero.json (e.g. legacy_ik for datasets generated before the
IK fixes). Bit-exactness also needs the same BLAS thread setting (scripts/env.sh pins OpenBLAS to 2 threads).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=20, help="number of episodes, spread over the dataset")
    ap.add_argument("--out", type=Path, default=None, help="write the report as JSON here")
    args = ap.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq

    import so101_libero  # noqa: F401
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.expert.pick_place import ExpertConfig
    from so101_libero.cli.generate import episode_limit
    from so101_libero.expert.rollout import run_expert_episode

    root = args.root or Path(os.environ.get("HF_LEROBOT_HOME", "data")) / args.repo_id
    meta = json.loads((root / "meta" / "so101_libero.json").read_text())
    known = set(ExpertConfig.__dataclass_fields__)
    cfg_dict = {k: (tuple(v) if isinstance(v, list) else v) for k, v in meta.get("expert_config", {}).items() if k in known}
    cfg = ExpertConfig(**cfg_dict)
    episodes = [json.loads(line) for line in open(root / "meta" / "so101_libero_episodes.jsonl")]
    data = pd.concat([pq.read_table(f).to_pandas() for f in sorted((root / "data").glob("*/*.parquet"))])
    picks = sorted(set(np.linspace(0, len(episodes) - 1, min(args.episodes, len(episodes))).round().astype(int)))
    envs, rows = {}, []
    for i in picks:
        e = episodes[i]
        if e["task_id"] not in envs:
            envs[e["task_id"]] = SO101TabletopEnv(e["task_id"], EnvConfig(render=False, terminate_on_success=False,
                                                                          max_steps=episode_limit(e["task_id"])))
        ep = run_expert_episode(envs[e["task_id"]], e["seed"], cfg=cfg, record=True,
                                placement_id=e.get("placement_id"))
        new = np.stack([a for _, a in ep.frames]) if ep.frames else np.zeros((0, 6), np.float32)
        old = np.stack(data[data.episode_index == e["episode_index"]]["action"].values)
        same = new.shape == old.shape and np.array_equal(new, old)
        diff = float(np.max(np.abs(new - old))) if new.shape == old.shape else None
        rows.append({"episode_index": e["episode_index"], "seed": e["seed"], "task_id": e["task_id"],
                     "bit_identical": bool(same), "max_abs_diff": diff, "frames": int(len(old))})
        print(json.dumps(rows[-1]), flush=True)
    n_same = sum(r["bit_identical"] for r in rows)
    report = {"repo_id": args.repo_id, "expert_config": {k: v for k, v in cfg_dict.items()}, "checked": len(rows),
              "bit_identical": n_same, "episodes": rows}
    print(f"{n_same}/{len(rows)} episodes regenerated bit-identically")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    sys.exit(0 if n_same == len(rows) else 1)


if __name__ == "__main__":
    main()
