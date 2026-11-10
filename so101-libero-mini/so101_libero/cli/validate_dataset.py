"""Validate a generated dataset with LeRobot's own loader before training.

    bash scripts/env.sh python -m so101_libero.cli.validate_dataset --repo-id local/so101_mini_t1_5demos

Checks: metadata (fps, features, names, robot type), per-item shapes/dtypes/ranges, frame timing, decodable
videos for every episode, action/state consistency, side metadata, and an open-loop REPLAY: the recorded
actions are re-executed in the simulator from each episode's stored initial state; the replayed joint
trajectory must match the recorded observation.state and the task goal must hold at the end. That proves the
stored actions are exactly what produced the demonstration (no obs/action misalignment or unit mismatch).
Writes a contact sheet of decoded frames to <out>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def check(cond: bool, msg: str, errors: list[str]) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        errors.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--replay-episodes", type=int, default=5, help="episodes to replay in simulation (0 = skip)")
    ap.add_argument("--min-episodes", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="contact sheet dir (default: outputs/dataset_checks/<name>)")
    args = ap.parse_args()

    import so101_libero  # noqa: F401
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    from so101_libero.data.writer import dataset_features
    from so101_libero.env import EnvConfig, SO101TabletopEnv
    from so101_libero.sim.robot import MOTOR_NAMES
    from so101_libero.tasks.registry import TASKS

    root = args.root or Path(os.environ.get("HF_LEROBOT_HOME", "data")) / args.repo_id
    out = args.out or Path("outputs/dataset_checks") / root.name
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    print(f"== loading {root}")
    ds = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    meta = ds.meta
    side = json.loads((root / "meta" / "so101_libero.json").read_text())
    episodes = [json.loads(line) for line in open(root / "meta" / "so101_libero_episodes.jsonl")]
    H, W = side["image_size"]
    print(f"   {meta.total_episodes} episodes, {meta.total_frames} frames, fps {meta.fps}, tasks {list(meta.tasks.index)}")

    print("== metadata")
    check(meta.fps == 30, f"fps == 30 (got {meta.fps})", errors)
    check(meta.robot_type == "so101_follower", f"robot_type so101_follower (got {meta.robot_type})", errors)
    check(meta.total_episodes >= args.min_episodes, f">= {args.min_episodes} episodes", errors)
    want = dataset_features((H, W))
    for key, spec in want.items():
        got = meta.features.get(key)
        check(got is not None and tuple(got["shape"]) == tuple(spec["shape"]) and got["dtype"] == spec["dtype"],
              f"feature {key}: {spec['dtype']} {tuple(spec['shape'])}", errors)
    check(meta.features["action"]["names"] == list(MOTOR_NAMES), "action names = so101_follower motor names", errors)
    extra = [k for k in meta.features if k.startswith("observation.") and k not in want]
    check(not extra, f"no extra (possibly privileged) observation features {extra}", errors)
    check(len(episodes) == meta.total_episodes, "side metadata has one line per episode", errors)
    check(len({e["seed"] for e in episodes}) == len(episodes), "episode seeds are unique", errors)

    print("== items")
    item = ds[0]
    for key in ("observation.images.front", "observation.images.wrist"):
        v = item[key]
        check(tuple(v.shape) == (3, H, W) and v.dtype == torch.float32 and 0 <= float(v.min()) and float(v.max()) <= 1,
              f"{key}: float32 (3,{H},{W}) in [0,1] (got {tuple(v.shape)} {v.dtype})", errors)
    for key in ("observation.state", "action"):
        check(tuple(item[key].shape) == (6,) and item[key].dtype == torch.float32, f"{key}: float32 (6,)", errors)
    check(isinstance(item["task"], str) and item["task"] in [t.language for t in TASKS.values()],
          f"task string is a registered instruction: {item['task']!r}", errors)

    print("== per-episode timing, videos and action/state consistency")
    sheet = []
    hf = ds.hf_dataset.with_format(None)
    ep_idx = np.array(hf["episode_index"])
    frame_idx = np.array(hf["frame_index"])
    ts = np.array(hf["timestamp"], dtype=np.float64)
    states = np.array(hf["observation.state"], dtype=np.float32)
    actions = np.array(hf["action"], dtype=np.float32)
    lags = []
    for e in range(meta.total_episodes):
        m = ep_idx == e
        fi, t = frame_idx[m], ts[m]
        ok = np.array_equal(fi, np.arange(m.sum())) and np.allclose(t, fi / meta.fps, atol=1e-4)
        if not ok:
            check(False, f"episode {e}: frame_index contiguous and timestamp = frame_index / fps", errors)
        s, a = states[m], actions[m]
        lags.append(np.abs(a[:-1, :5] - s[1:, :5]).mean(0))  # arm joints track commands closely
        first = ds.meta.episodes[e]["dataset_from_index"]
        last = ds.meta.episodes[e]["dataset_to_index"] - 1
        if e < 6:
            fr = [ds[i] for i in (first, (first + last) // 2, last)]
            row = [np.concatenate([(f["observation.images.front"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                   (f["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)], 1)
                   for f in fr]
            sheet.append(np.concatenate(row, 1))
        else:
            _ = ds[last]  # decodes the last frame of every episode's videos
    check(True, f"all {meta.total_episodes} episodes: contiguous frames, timestamps = frame_index / {meta.fps}", errors)
    lag = np.mean(lags, 0)
    print(f"   mean |action_t - state_t+1| per arm joint (deg): {np.round(lag, 3).tolist()}")
    check(bool(np.all(lag < 2.0)), "arm joints track the commanded actions (< 2 deg mean one-step lag)", errors)
    Image.fromarray(np.concatenate(sheet, 0)).save(out / "frames_first_mid_last.png")
    print(f"   contact sheet: {out / 'frames_first_mid_last.png'}")

    if args.replay_episodes:
        print(f"== open-loop replay of {min(args.replay_episodes, meta.total_episodes)} episodes (spread over the dataset)")
        envs = {}
        # Spread the replayed episodes over the whole dataset (episodes are written task by task).
        picks = sorted(set(np.linspace(0, meta.total_episodes - 1, min(args.replay_episodes, meta.total_episodes))
                           .round().astype(int).tolist()))
        for e in picks:
            rec = episodes[e]
            task_id = rec["task_id"]
            if task_id not in envs:
                envs[task_id] = SO101TabletopEnv(task_id, EnvConfig(render=False, terminate_on_success=False,
                                                                    max_steps=10_000))
            env = envs[task_id]
            obs, _ = env.reset(options={"init_state": rec["init_state"]})
            m = ep_idx == e
            s_rec, a_rec = states[m], actions[m]
            dev = 0.0
            info = {"is_success": False}
            for t in range(len(a_rec)):
                dev = max(dev, float(np.max(np.abs(obs["observation.state"] - s_rec[t]))))
                obs, _, _, _, info = env.step(a_rec[t])
            ok = dev < 1e-3 and info["is_success"]
            check(ok, f"episode {e} (seed {rec['seed']}): replayed state max dev {dev:.2e} deg/units, "
                      f"goal holds at end: {info['is_success']}", errors)

    print("== result:", "PASS" if not errors else f"FAIL ({len(errors)} problems)")
    (out / "validation.json").write_text(json.dumps({"repo_id": args.repo_id, "root": str(root), "errors": errors,
                                                     "episodes": meta.total_episodes, "frames": meta.total_frames,
                                                     "mean_action_state_lag_deg": lag.tolist()}, indent=2))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
