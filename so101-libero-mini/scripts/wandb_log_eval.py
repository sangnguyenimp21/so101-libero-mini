"""Post a finished simulated evaluation (results.json of so101_libero.cli.evaluate) to the W&B run of a training run.

    bash scripts/env.sh python scripts/wandb_log_eval.py --run-dir outputs/train/<run> \
        --results outputs/train/<run>/eval/last_80000/results.json --label last_80000 --videos 4

The run id is read from <run-dir>/wandb/latest-run (where lerobot-train put it); the run is resumed, the per-task
success rates, Wilson intervals and outcome counts are logged under eval/<label>/..., a summary table and a few
evaluation videos are attached, and the run is closed again. Works offline too (synced later with `wandb sync`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path


def run_id_of(run_dir: Path) -> str:
    paths = glob.glob(str(run_dir / "wandb" / "latest-run" / "run-*.wandb"))
    if len(paths) != 1:
        raise SystemExit(f"no (single) W&B run found under {run_dir}/wandb/latest-run (was training run with WANDB=true?)")
    m = re.search(r"run-([^.]+)\.wandb", Path(paths[0]).name)
    if m is None:
        raise SystemExit(f"cannot parse the run id from {paths[0]}")
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="lerobot-train output dir (contains wandb/)")
    ap.add_argument("--results", type=Path, required=True, help="results.json written by so101_libero.cli.evaluate")
    ap.add_argument("--label", default=None, help="eval/<label>/... prefix (default: the results dir name)")
    ap.add_argument("--videos", type=int, default=4, help="evaluation videos to upload per task (0 = none)")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    args = ap.parse_args()

    import wandb

    r = json.loads(args.results.read_text())
    label = args.label or args.results.parent.name
    tc = json.loads((args.run_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json").read_text())
    project = args.project or tc["wandb"].get("project")
    entity = args.entity or tc["wandb"].get("entity")
    run = wandb.init(id=run_id_of(args.run_dir), project=project, entity=entity, resume="allow", dir=str(args.run_dir),
                     mode=os.environ.get("WANDB_MODE", "online"))
    # W&B only accepts monotonically increasing steps on a run: never pass one (evaluations of earlier checkpoints
    # would be dropped); the checkpoint step is logged as a value instead.
    log: dict = {}
    m = re.search(r"(\d+)", label)
    if m:
        log[f"eval/{label}/checkpoint_step"] = int(m.group(1))
    table = wandb.Table(columns=["task", "episodes", "successes", "success_rate", "wilson95_lo", "wilson95_hi", "outcomes"])
    for task, t in r["per_task"].items():
        lo, hi = t["wilson95"]
        log[f"eval/{label}/{task}/success_rate"] = t["success_rate"]
        log[f"eval/{label}/{task}/successes"] = t["successes"]
        log[f"eval/{label}/{task}/wilson95_lo"] = lo
        log[f"eval/{label}/{task}/wilson95_hi"] = hi
        table.add_data(task, t["episodes"], t["successes"], t["success_rate"], lo, hi, json.dumps(t["outcomes"]))
    log[f"eval/{label}/overall/success_rate"] = r["overall"]["success_rate"]
    log[f"eval/{label}/summary"] = table
    if args.videos > 0:
        vids = sorted((args.results.parent / "videos").glob("*.mp4"))
        per_task: dict[str, int] = {}
        for v in vids:
            task = v.name.rsplit("_ep", 1)[0]
            if per_task.get(task, 0) >= args.videos:
                continue
            per_task[task] = per_task.get(task, 0) + 1
            log[f"eval/{label}/video/{v.stem}"] = wandb.Video(str(v), fps=30, format="mp4")
    for k, v in log.items():
        if not isinstance(v, (wandb.Table, wandb.Video)):
            run.summary[k] = v
    run.log(log)
    run.summary[f"eval/{label}/policy"] = json.dumps(r.get("policy", {}))
    run.summary[f"eval/{label}/render_backend"] = r.get("render_backend")
    print(f"logged {label}: overall {r['overall']['success_rate']:.2f} -> {run.get_url() or run.id}")
    run.finish()


if __name__ == "__main__":
    main()
