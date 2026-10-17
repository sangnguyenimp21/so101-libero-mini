"""Deploy copy of a checkpoint whose preprocessor crops each real camera frame and resizes it to the training size.

Weights and the other files are symlinked; only policy_preprocessor.json is rewritten, with LeRobot's
image_crop_resize_processor inserted as the first step (before any renaming, so crops use the robot's camera
names). Crop a square to keep the aspect ratio of the 256x256 sim images. Prints the deploy directory.
With --debug-dir, the debug steps of scripts/rollout_debug.py are added too (save the policy's input frames after
the crop, log every action); such a copy only runs through scripts/rollout_debug.py, which registers them.

    python scripts/deploy_policy.py CHECKPOINT_DIR DEPLOY_DIR --crop front=0,80,480,480 --crop wrist=0,80,480,480
    python scripts/deploy_policy.py CHECKPOINT_DIR DEPLOY_DIR --crop ... --debug-dir outputs/debug/run1 --debug-every 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="pretrained_model directory")
    parser.add_argument("out", type=Path, help="deploy directory (created or refreshed)")
    parser.add_argument("--crop", action="append", default=[], metavar="CAM=TOP,LEFT,HEIGHT,WIDTH")
    parser.add_argument("--debug-dir", help="save the policy's input frames and an action log here")
    parser.add_argument("--debug-every", type=int, default=10, help="save a frame every N control ticks")
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
    (args.out / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2) + "\n")
    (args.out / "policy_postprocessor.json").write_text(json.dumps(post, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
