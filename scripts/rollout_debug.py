"""lerobot-rollout with debug output: the exact images the policy gets, and every state/action, written to disk.

Registers two processor steps that scripts/deploy_policy.py --debug-dir puts into the deploy copy of a checkpoint,
then runs lerobot-rollout unchanged (same command-line flags):
  debug_frame_saver   (preprocessor, right after the crop/resize) every `every` control ticks saves
                      frame_<tick>.jpg: front | wrist as the policy sees them (256x256, shown at 2x) + the joint state
  debug_action_logger (postprocessor, last) appends tick, state and the commanded action to log.csv every tick

    python scripts/rollout_debug.py <lerobot-rollout flags>    # used by scripts/real_act.sh when DEBUG=1
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot.processor import ActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_shared = {"tick": -1, "state": None, "t0": None}  # filled by the frame saver, read by the action logger


def _to_bgr(image) -> np.ndarray:
    """(1, 3, H, W) or (3, H, W) float tensor in [0, 1] -> uint8 BGR (H, W, 3)."""
    img = image.detach().float().cpu().numpy()
    img = img[0] if img.ndim == 4 else img
    return cv2.cvtColor((np.clip(img.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


@ProcessorStepRegistry.register("debug_frame_saver")
@dataclass
class DebugFrameSaver(ObservationProcessorStep):
    out_dir: str = "outputs/debug"
    every: int = 10

    def observation(self, observation):
        _shared["tick"] += 1
        _shared["t0"] = _shared["t0"] or time.time()
        state = observation.get("observation.state")
        _shared["state"] = None if state is None else state.detach().float().cpu().numpy().reshape(-1)
        if _shared["tick"] % self.every == 0:
            keys = sorted(k for k in observation if "image" in k)
            panels = []
            for k in keys:
                p = cv2.resize(_to_bgr(observation[k]), (512, 512), interpolation=cv2.INTER_NEAREST)
                cv2.putText(p, k.split(".")[-1], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                panels.append(p)
            img = np.hstack(panels)
            bar = np.full((40, img.shape[1], 3), 30, np.uint8)
            s = _shared["state"]
            text = f"tick {_shared['tick']}  t {time.time() - _shared['t0']:.1f}s  state " + (
                "-" if s is None else " ".join(f"{v:.1f}" for v in s))
            cv2.putText(bar, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(self.out_dir) / f"frame_{_shared['tick']:05d}.jpg"), np.vstack([img, bar]),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        return observation

    def get_config(self):
        return {"out_dir": self.out_dir, "every": self.every}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("debug_action_logger")
@dataclass
class DebugActionLogger(ActionProcessorStep):
    out_dir: str = "outputs/debug"

    def action(self, action):
        path = Path(self.out_dir) / "log.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("tick,time_s," + ",".join(f"state_{j}" for j in JOINTS) + ","
                            + ",".join(f"action_{j}" for j in JOINTS) + "\n")
        a = action.detach().float().cpu().numpy().reshape(-1)
        s = _shared["state"] if _shared["state"] is not None else np.full(len(JOINTS), np.nan)
        t = time.time() - (_shared["t0"] or time.time())
        with path.open("a") as f:
            f.write(f"{_shared['tick']},{t:.3f}," + ",".join(f"{v:.2f}" for v in s) + ","
                    + ",".join(f"{v:.2f}" for v in a) + "\n")
        return action

    def get_config(self):
        return {"out_dir": self.out_dir}

    def transform_features(self, features):
        return features


if __name__ == "__main__":
    from lerobot.scripts.lerobot_rollout import main

    main()
