"""Deploy copy of a checkpoint whose preprocessor crops each real camera frame and resizes it to the training size.

Weights and the other files are symlinked; only policy_preprocessor.json is rewritten, with LeRobot's
image_crop_resize_processor inserted as the first step (before any renaming, so crops use the robot's camera
names). Crop a square to keep the aspect ratio of the 256x256 sim images. Prints the deploy directory.
With --debug-dir, the debug steps of scripts/rollout_debug.py are added too (save the policy's input frames after
the crop, log every action); such a copy only runs through scripts/rollout_debug.py, which registers them.
With --stats-from, the normalization stats come from another checkpoint, or from a LeRobot dataset of this robot
(its meta/stats.json: the stats a checkpoint trained on it would carry, no training needed), instead (e.g. for
lerobot/smolvla_base, whose stats are per pretraining dataset with none for observation.state / action: LeRobot would
then pass both through unnormalized and the arm would head for the all-zero joint pose). A copy without such stats
is refused.

    python scripts/deploy_policy.py CHECKPOINT_DIR DEPLOY_DIR --crop front=0,80,480,480 --crop wrist=0,80,480,480
    python scripts/deploy_policy.py CHECKPOINT_DIR DEPLOY_DIR --crop ... --debug-dir outputs/debug/run1 --debug-every 10
    python scripts/deploy_policy.py SMOLVLA_BASE_DIR DEPLOY_DIR --crop ... --stats-from outputs/train/<run>/checkpoints/last/pretrained_model
    python scripts/deploy_policy.py SMOLVLA_BASE_DIR DEPLOY_DIR --crop ... --stats-from ~/.cache/huggingface/lerobot/<repo id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def stats_keys(path: Path) -> set[str]:
    """Features with stats in a normalizer .safetensors (read from its JSON header, e.g. 'action' of 'action.mean')."""
    with path.open("rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    return {k.rsplit(".", 1)[0] for k in header if k != "__metadata__"}


def state_files(ckpt: Path) -> dict[str, str]:
    """registry_name -> state_file of the stateful (normalizer) steps of a checkpoint's processors."""
    return {s["registry_name"]: s["state_file"] for f in ("policy_preprocessor.json", "policy_postprocessor.json")
            for s in json.loads((ckpt / f).read_text())["steps"] if "state_file" in s}


def dataset_stats(dataset: Path, out: Path) -> Path:
    """Normalizer .safetensors of a LeRobot dataset's meta/stats.json (state and action stats, float32)."""
    import torch
    from safetensors.torch import save_file

    stats = json.loads((dataset / "meta" / "stats.json").read_text())
    save_file({f"{feat}.{name}": torch.tensor(value, dtype=torch.float32)
               for feat in ("observation.state", "action") for name, value in stats[feat].items()}, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="pretrained_model directory")
    parser.add_argument("out", type=Path, help="deploy directory (created or refreshed)")
    parser.add_argument("--crop", action="append", default=[], metavar="CAM=TOP,LEFT,HEIGHT,WIDTH")
    parser.add_argument("--debug-dir", help="save the policy's input frames and an action log here")
    parser.add_argument("--debug-every", type=int, default=10, help="save a frame every N control ticks")
    parser.add_argument("--stats-from", type=Path,
                        help="pretrained_model or LeRobot dataset directory whose normalization stats to use")
    args = parser.parse_args()

    src = args.checkpoint.resolve()
    config = json.loads((src / "config.json").read_text())
    sizes = {tuple(f["shape"][1:]) for f in config["input_features"].values() if f["type"] == "VISUAL"}
    if len(sizes) != 1:
        raise SystemExit(f"cameras of {src} have different training sizes: {sizes}")
    (size,) = sizes
    crops = {}
    for spec in args.crop:
        cam, box = spec.split("=")
        crops[f"observation.images.{cam}"] = [int(v) for v in box.split(",")]

    rewritten = {"policy_preprocessor.json", "policy_postprocessor.json"}
    args.out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        link = args.out / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        if f.name not in rewritten:
            link.symlink_to(f)
    pre = json.loads((src / "policy_preprocessor.json").read_text())
    post = json.loads((src / "policy_postprocessor.json").read_text())
    pre["steps"].insert(0, {"registry_name": "image_crop_resize_processor",
                            "config": {"crop_params_dict": crops, "resize_size": list(size)}})
    if args.debug_dir:
        pre["steps"].insert(1, {"registry_name": "debug_frame_saver",
                                "config": {"out_dir": args.debug_dir, "every": args.debug_every}})
        post["steps"].append({"registry_name": "debug_action_logger", "config": {"out_dir": args.debug_dir}})
    stats_src = args.stats_from.resolve() if args.stats_from else src
    if (stats_src / "meta" / "stats.json").is_file():  # a dataset: its stats serve every normalizer step
        dataset_file = dataset_stats(stats_src, args.out / "dataset_stats.safetensors")
        stats_files = {s["registry_name"]: dataset_file for s in pre["steps"] + post["steps"] if "state_file" in s}
    else:
        stats_files = {name: stats_src / f for name, f in state_files(stats_src).items()}
    for step in pre["steps"] + post["steps"]:
        if "state_file" not in step:
            continue
        link = args.out / step["state_file"]
        link.unlink(missing_ok=True)
        link.symlink_to(stats_files[step["registry_name"]])
        missing = {"observation.state", "action"} - stats_keys(link)
        if missing:
            raise SystemExit(f"{link.resolve()} has no stats for {sorted(missing)} (LeRobot would pass them through "
                             "unnormalized): use --stats-from a checkpoint trained on this robot")
    (args.out / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2) + "\n")
    (args.out / "policy_postprocessor.json").write_text(json.dumps(post, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
