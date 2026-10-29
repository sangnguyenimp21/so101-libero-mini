"""Lightweight sanity check of a LeRobot demonstration dataset (three-cube suite, also any so101_libero dataset).

    python -m so101_libero.cli.check_dataset --repo-id local/smolvla_sort_ryg
    bash scripts/check_dataset.sh sort            # the same, by task name

Reports and checks: number of episodes, task labels (instructions), camera keys, robot state and action dimensions,
episode lengths, NaN / Inf values, missing frames (frame indices contiguous, timestamps = index / fps, episode index
ranges and video segments consistent, the first and last frame of every episode decodable from both cameras), and the
layout distribution (sort: also the starting permutations, and layout x permutation). Exit code 1 if a check fails.
``--decode-all`` decodes every video frame (slow) instead of the first and last of each episode.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", type=Path, default=None, help="default: $HF_LEROBOT_HOME/<repo-id>")
    ap.add_argument("--expect-episodes", type=int, default=None, help="fail unless the dataset has this many episodes")
    ap.add_argument("--decode-all", action="store_true", help="decode every video frame (slow)")
    ap.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = ap.parse_args(argv)

    import so101_libero  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root or Path(os.environ.get("HF_LEROBOT_HOME", "data")) / args.repo_id
    ds = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    meta = ds.meta
    problems: list[str] = []
    report: dict = {"dataset": args.repo_id, "root": str(root)}

    def fail(msg: str) -> None:
        problems.append(msg)
        print(f"  PROBLEM: {msg}")

    print(f"Dataset: {Path(args.repo_id).name}   ({root})")
    n_ep, n_fr = meta.total_episodes, meta.total_frames
    print(f"\nEpisodes: {n_ep}   Frames: {n_fr}   fps: {meta.fps}   robot: {meta.robot_type}")
    report.update(episodes=n_ep, frames=n_fr, fps=meta.fps)
    if args.expect_episodes is not None and n_ep != args.expect_episodes:
        fail(f"{n_ep} episodes, expected {args.expect_episodes}")

    hf = ds.hf_dataset.with_format(None)
    ep_idx = np.asarray(hf["episode_index"], dtype=np.int64)
    task_idx = np.asarray(hf["task_index"], dtype=np.int64)
    tasks = {int(i): str(t) for t, i in zip(meta.tasks.index, meta.tasks["task_index"])}
    ep_task = {int(e): tasks[int(task_idx[ep_idx == e][0])] for e in np.unique(ep_idx)}
    print("\nTask labels:")
    for t, c in sorted(collections.Counter(ep_task.values()).items()):
        print(f"  {c:4d} episodes: {t}")
    report["tasks"] = dict(collections.Counter(ep_task.values()))

    cams = [k for k in meta.features if meta.features[k]["dtype"] in ("video", "image")]
    print("\nCamera keys:")
    for k in cams:
        f = meta.features[k]
        codec = f.get("info", {}).get("video.codec", "")
        print(f"  - {k}  {tuple(f['shape'])} {f['dtype']} {codec}")
    report["camera_keys"] = cams
    if [k for k in cams] != ["observation.images.front", "observation.images.wrist"]:
        fail(f"camera keys {cams} differ from [observation.images.front, observation.images.wrist]")
    for key in ("observation.state", "action"):
        f = meta.features.get(key)
        if f is None:
            fail(f"{key} missing")
            continue
        print(f"\n{'Robot state' if key == 'observation.state' else 'Action'} ({key}): dim {f['shape'][0]}  "
              f"names {f.get('names')}")
        report[f"{key}_dim"] = int(f["shape"][0])
        if tuple(f["shape"]) != (6,):
            fail(f"{key} has shape {f['shape']}, expected (6,)")
    extra = [k for k in meta.features if k.startswith("observation.") and k not in (*cams, "observation.state")]
    if extra:
        fail(f"extra observation features (possibly privileged): {extra}")

    lengths = np.array([int(meta.episodes[e]["length"]) for e in range(n_ep)])
    print(f"\nEpisode lengths (frames): min {lengths.min()}  mean {lengths.mean():.1f}  max {lengths.max()}  "
          f"({lengths.min() / meta.fps:.1f}-{lengths.max() / meta.fps:.1f} s)")
    report["episode_length"] = {"min": int(lengths.min()), "mean": round(float(lengths.mean()), 1),
                                "max": int(lengths.max())}

    states = np.asarray(hf["observation.state"], dtype=np.float32)
    actions = np.asarray(hf["action"], dtype=np.float32)
    nan_s, nan_a = int((~np.isfinite(states)).any(1).sum()), int((~np.isfinite(actions)).any(1).sum())
    print(f"\nNaN/Inf: states {nan_s}  actions {nan_a}")
    report.update(nan_states=nan_s, nan_actions=nan_a)
    if nan_s or nan_a:
        fail(f"non-finite values: {nan_s} state rows, {nan_a} action rows")

    # missing frames: contiguous indices, timestamps, index ranges and video segments per episode
    frame_idx = np.asarray(hf["frame_index"], dtype=np.int64)
    ts = np.asarray(hf["timestamp"], dtype=np.float64)
    missing = 0
    for e in range(n_ep):
        m = ep_idx == e
        ep = meta.episodes[e]
        n = int(m.sum())
        if n != int(ep["length"]) or int(ep["dataset_to_index"]) - int(ep["dataset_from_index"]) != n:
            missing += abs(int(ep["length"]) - n) or 1
            fail(f"episode {e}: {n} data rows, metadata says {ep['length']}")
        if not np.array_equal(frame_idx[m], np.arange(n)) or not np.allclose(ts[m], np.arange(n) / meta.fps, atol=1e-4):
            missing += 1
            fail(f"episode {e}: frame indices / timestamps not contiguous")
        for k in cams:
            span = float(ep[f"videos/{k}/to_timestamp"]) - float(ep[f"videos/{k}/from_timestamp"])
            if abs(span - n / meta.fps) > 1.5 / meta.fps:
                missing += 1
                fail(f"episode {e}: {k} video segment {span:.3f} s for {n} frames")
    decode_errors = 0
    picks = range(n_fr) if args.decode_all else sorted({i for e in range(n_ep) for i in (
        int(meta.episodes[e]["dataset_from_index"]), int(meta.episodes[e]["dataset_to_index"]) - 1)})
    for i in picks:
        try:
            item = ds[int(i)]
            for k in cams:
                v = item[k]
                if tuple(v.shape)[0] != 3 or not bool(np.isfinite(v.numpy()).all()):
                    raise ValueError(f"bad frame {tuple(v.shape)}")
        except Exception as ex:  # noqa: BLE001
            decode_errors += 1
            if decode_errors <= 3:
                fail(f"frame {i}: {type(ex).__name__}: {ex}")
    print(f"\nMissing frames: {missing}   (video frames decoded: {len(picks)} {'all' if args.decode_all else 'first+last per episode'},"
          f" errors {decode_errors})")
    report.update(missing_frames=missing, decode_errors=decode_errors)

    # layouts / permutations: per-frame features of the three-cube datasets, else the side file
    side_path = root / "meta" / "so101_libero_episodes.jsonl"
    side = [json.loads(line) for line in open(side_path)] if side_path.exists() else []
    per_ep = {}
    if "layout_id" in meta.features:
        lay = np.asarray(hf["layout_id"], dtype=np.int64).reshape(-1)
        perm = np.asarray(hf["permutation_id"], dtype=np.int64).reshape(-1)
        for e in range(n_ep):
            m = ep_idx == e
            if len(set(lay[m])) != 1 or len(set(perm[m])) != 1:
                fail(f"episode {e}: layout/permutation not constant within the episode")
            per_ep[e] = (int(lay[m][0]), int(perm[m][0]))
    names = {rec["episode_index"]: rec for rec in side}
    if per_ep:
        layout_names = {v[0]: names.get(e, {}).get("layout_name", f"layout_{v[0]}") for e, v in per_ep.items()}
        print("\nLayouts:")
        lay_counts = collections.Counter(v[0] for v in per_ep.values())
        for k in sorted(lay_counts):
            print(f"  layout_{k} ({layout_names[k]}): {lay_counts[k]}")
        report["layouts"] = {f"layout_{k}": lay_counts[k] for k in sorted(lay_counts)}
        by_task = collections.defaultdict(list)
        for e, v in per_ep.items():
            by_task[ep_task[e]].append(v)
        for t, vs in by_task.items():
            c = collections.Counter(l for l, _ in vs)
            if len(set(c.values())) > 1:
                fail(f"layouts unbalanced for '{t}': {dict(c)}")
        perms = {e: names.get(e, {}).get("permutation") for e, v in per_ep.items() if v[1] >= 0}
        if perms:
            pc = collections.Counter(perms.values())
            print("\nStarting permutations (left, centre, right slot; robot frame):")
            for p in sorted(pc):
                print(f"  {p}: {pc[p]}")
            report["permutations"] = dict(pc)
            print("\nLayout x permutation:")
            table = collections.Counter((per_ep[e][0], perms[e]) for e in perms)
            cols = sorted(pc)
            print("            " + "  ".join(f"{p:>4s}" for p in cols))
            for k in sorted(lay_counts):
                print(f"  layout_{k}   " + "  ".join(f"{table[(k, p)]:4d}" for p in cols))
            if len(pc) > 1 and len({lay for lay, _ in table}) > 1:
                single = [p for p in cols if len({lay for (lay, q) in table if q == p}) == 1]
                if single:
                    fail(f"permutations {single} appear in one layout only")
    elif side:
        pc = collections.Counter(rec.get("placement_name") for rec in side)
        print("\nPlacements (side file):", dict(pc))
    report["problems"] = problems
    print(f"\nResult: {'PASS' if not problems else f'FAIL ({len(problems)} problems)'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
