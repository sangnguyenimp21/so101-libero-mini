"""LeRobot (v3.0 format, lerobot 0.4.2) dataset writing for expert demonstrations.

Policy-visible features are exactly the environment's observation/action interface (``env.py``). Privileged
simulator data (seeds, initial layouts, grasp metrics, failures) goes to side files under ``meta/`` that no
LeRobot policy reads:
  meta/so101_libero.json                  generation config + counts
  meta/so101_libero_episodes.jsonl        one line per saved episode (episode_index, seed, task, init_state, ...)
  meta/so101_libero_failures.jsonl        rejected attempts with their failure reason
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from so101_libero.sim.robot import MOTOR_NAMES

CAMERA_KEYS = ("observation.images.front", "observation.images.wrist")


def dataset_features(image_size: tuple[int, int], cameras: tuple[str, ...] = ("front", "wrist")) -> dict:
    h, w = image_size
    feats = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(MOTOR_NAMES)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(MOTOR_NAMES)},
    }
    for cam in cameras:
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": (h, w, 3),
                                              "names": ["height", "width", "channels"]}
    return feats


class DemoWriter:
    """Thin wrapper over ``LeRobotDataset.create`` / ``add_frame`` / ``save_episode`` / ``finalize``."""

    def __init__(self, repo_id: str, root: Path, fps: float, image_size: tuple[int, int], overwrite: bool = False,
                 image_writer_threads: int = 8):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"{self.root} exists (pass --overwrite to replace it)")
            shutil.rmtree(self.root)
        self.ds = LeRobotDataset.create(repo_id, fps=int(round(fps)), features=dataset_features(image_size),
                                        root=self.root, robot_type="so101_follower", use_videos=True,
                                        image_writer_threads=image_writer_threads, video_backend="pyav")
        self.meta_dir = self.root / "meta"
        self.n_episodes = 0
        self._episodes = open(self.meta_dir / "so101_libero_episodes.jsonl", "w")
        self._failures = open(self.meta_dir / "so101_libero_failures.jsonl", "w")

    def add_episode(self, task: str, states: np.ndarray, actions: np.ndarray, images: dict[str, np.ndarray],
                    record: dict) -> int:
        for t in range(len(actions)):
            frame = {"observation.state": states[t].astype(np.float32), "action": actions[t].astype(np.float32),
                     "task": task}
            for key, arr in images.items():
                frame[key] = arr[t]
            self.ds.add_frame(frame)
        self.ds.save_episode()
        index = self.n_episodes
        self._episodes.write(json.dumps({"episode_index": index, **record}) + "\n")
        self._episodes.flush()
        self.n_episodes += 1
        return index

    def add_failure(self, record: dict) -> None:
        self._failures.write(json.dumps(record) + "\n")
        self._failures.flush()

    def finalize(self, config: dict) -> None:
        self.ds.stop_image_writer()
        self.ds.finalize()
        self._episodes.close()
        self._failures.close()
        (self.meta_dir / "so101_libero.json").write_text(json.dumps(config, indent=2))
