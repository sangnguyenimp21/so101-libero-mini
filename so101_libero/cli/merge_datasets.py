"""Combine per-task datasets into ONE multitask LeRobot dataset (lerobot 0.4.2 cannot train on several repo ids).

    python -m so101_libero.cli.merge_datasets --out-repo-id local/so101_tri_v2_multitask_300 \
        --repo-ids local/so101_tri_cylinder_aside_100 local/so101_tri_cube_flip_100 local/so101_tri_green_into_tray_100

Uses LeRobot's own ``aggregate_datasets`` (data, videos, per-episode metadata, statistics; tasks are re-indexed by
their instruction string, so every frame keeps its own natural-language task), with file rotation switched off: in
lerobot 0.4.2 a rotation (a destination video file reaching ``video_files_size_in_mb``, 200 MB by default) points every
episode of a multi-file source at the LAST destination file, so its frames decode from the wrong file or past its end
(seen with the AV1 flip dataset on the cluster). One file per camera keeps LeRobot's per-file offsets correct. After
merging, the first and last frame of every episode (both cameras, state, action, task) are decoded from the merged
dataset and compared with its source; any mismatch fails the merge. The sources must share fps, robot type
and features (camera keys and order, joint names, action representation): checked before anything is written. The
project side files are merged too (episode indices shifted), and ``meta/so101_libero.json`` records the sources, the
per-task episode/frame counts and the sampling rule for training: frames are drawn so that every TASK is sampled
equally often (so101_libero/train/lerobot_train_balanced.py), not in proportion to its frame count.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def verify_against_sources(repo_ids: list[str], roots: list[Path], out_repo_id: str, out: Path) -> None:
    """Decode the first and last frame of every episode from the merged dataset and from its source; they must match
    (videos are remuxed, not re-encoded, so the decoded images are identical)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    merged = LeRobotDataset(out_repo_id, root=out, video_backend="pyav")
    keys = [*merged.meta.video_keys, "observation.state", "action"]
    offset, bad = 0, []
    for repo, root in zip(repo_ids, roots):
        src = LeRobotDataset(repo, root=root, video_backend="pyav")
        for e in range(src.meta.total_episodes):
            first = src.meta.episodes[e]["dataset_from_index"]
            last = src.meta.episodes[e]["dataset_to_index"] - 1
            for i in (first, last):
                a, b = src[i], merged[offset + i]
                same = a["task"] == b["task"] and all(torch.equal(a[k], b[k]) for k in keys)
                if not same:
                    bad.append(f"{repo} episode {e} frame {i}")
        offset += src.meta.total_frames
        print(f"verified {repo}: {src.meta.total_episodes} episodes (first and last frames)", flush=True)
    if bad:
        sys.exit(f"merged dataset differs from its sources at {len(bad)} frames, e.g. {bad[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-ids", nargs="+", required=True, help="source datasets, in this order")
    ap.add_argument("--roots", nargs="+", type=Path, default=None, help="default: $HF_LEROBOT_HOME/<repo-id>")
    ap.add_argument("--out-repo-id", required=True)
    ap.add_argument("--out-root", type=Path, default=None, help="default: $HF_LEROBOT_HOME/<out-repo-id>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    home = Path(os.environ.get("HF_LEROBOT_HOME", "data"))
    roots = args.roots or [home / r for r in args.repo_ids]
    if len(roots) != len(args.repo_ids):
        ap.error("--roots needs one path per --repo-ids entry")
    out = args.out_root or home / args.out_repo_id
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists (pass --overwrite to replace it)")
        shutil.rmtree(out)

    metas = [LeRobotDatasetMetadata(r, root=root) for r, root in zip(args.repo_ids, roots)]
    first = metas[0]
    for m, r in zip(metas, args.repo_ids):
        for what, a, b in (("fps", m.fps, first.fps), ("robot_type", m.robot_type, first.robot_type),
                           ("features", m.features, first.features)):
            if a != b:
                sys.exit(f"{r}: {what} differs from {args.repo_ids[0]} ({a} != {b}); cannot combine")
    sides = [json.loads((root / "meta" / "so101_libero.json").read_text()) for root in roots]
    for side, r in zip(sides, args.repo_ids):
        if side.get("image_size") != sides[0].get("image_size"):
            sys.exit(f"{r}: image size {side.get('image_size')} != {sides[0].get('image_size')}")

    print(f"aggregating {len(metas)} datasets -> {out}", flush=True)
    no_rotation = 1_000_000  # MB: never start a second data/video file (see the module docstring)
    aggregate_datasets(args.repo_ids, args.out_repo_id, roots=roots, aggr_root=out,
                       data_files_size_in_mb=no_rotation, video_files_size_in_mb=no_rotation)
    verify_against_sources(args.repo_ids, roots, args.out_repo_id, out)

    # Project side files: episode records with shifted indices, failures, and a summary.
    offset, per_task, episodes, failures = 0, {}, [], []
    for m, root, side in zip(metas, roots, sides):
        for line in open(root / "meta" / "so101_libero_episodes.jsonl"):
            rec = json.loads(line)
            rec["source_episode_index"] = rec["episode_index"]
            rec["episode_index"] += offset
            episodes.append(rec)
            t = per_task.setdefault(rec["task_id"], {"instruction": rec["task"], "episodes": 0, "frames": 0,
                                                     "per_placement": {}})
            t["episodes"] += 1
            t["frames"] += rec["steps"]
            if rec.get("placement_id") is not None:
                k = str(rec["placement_id"])
                t["per_placement"][k] = t["per_placement"].get(k, 0) + 1
        fpath = root / "meta" / "so101_libero_failures.jsonl"
        if fpath.exists():
            failures += [json.loads(line) for line in open(fpath)]
        offset += m.total_episodes
    meta_dir = out / "meta"
    with open(meta_dir / "so101_libero_episodes.jsonl", "w") as f:
        for rec in episodes:
            f.write(json.dumps(rec) + "\n")
    with open(meta_dir / "so101_libero_failures.jsonl", "w") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")
    merged = LeRobotDatasetMetadata(args.out_repo_id, root=out)
    summary = {
        **{k: sides[0][k] for k in ("image_size", "fps", "render_backend", "so101_libero_version") if k in sides[0]},
        "repo_id": args.out_repo_id, "merged_from": [{"repo_id": r, "root": str(root), "episodes": m.total_episodes,
                                                      "frames": m.total_frames, "tasks": side.get("tasks")}
                                                     for r, root, m, side in zip(args.repo_ids, roots, metas, sides)],
        "tasks": [t for side in sides for t in side.get("tasks", [])],
        "per_task": per_task,
        "demos_per_placement": sides[0].get("demos_per_placement"),
        "placement_names": sides[0].get("placement_names"),
        "sampling": "task-balanced: python -m so101_libero.train.lerobot_train_balanced (every task equally often)",
        "total_episodes": merged.total_episodes, "total_frames": merged.total_frames,
        "lerobot_tasks": {int(i): t for t, i in zip(merged.tasks.index, merged.tasks["task_index"])},
    }
    (meta_dir / "so101_libero.json").write_text(json.dumps(summary, indent=2))
    if merged.total_episodes != len(episodes):
        sys.exit(f"episode count mismatch: LeRobot {merged.total_episodes} vs side file {len(episodes)}")
    print(json.dumps({k: summary[k] for k in ("total_episodes", "total_frames", "lerobot_tasks")}, indent=2))
    print(f"merged dataset: {out}")


if __name__ == "__main__":
    main()
