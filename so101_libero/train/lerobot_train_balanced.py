"""``lerobot-train`` (lerobot 0.4.2) with TASK-BALANCED sampling for multitask datasets.

    python -m so101_libero.train.lerobot_train_balanced <the usual lerobot-train arguments>

Everything is LeRobot's own training (policy, processors, optimizer, scheduler, checkpoints, ``--resume``); only the
dataset handed to its DataLoader is wrapped. lerobot-train shuffles frames uniformly, so a task with longer episodes
would be sampled more often than one with shorter ones. The wrapper exposes a virtual index space in which every task
owns the same number of slots (the frame count of the largest task) and maps slot j of task t onto that task's frames,
spread evenly (each frame of a smaller task appears once or twice per epoch). With LeRobot's uniform shuffle over
those slots, every epoch draws each task exactly equally often. Each sample
is still the real frame with its own natural-language ``task`` string, so language conditioning is unchanged. Logged
epochs count virtual samples; ``dataset.num_frames`` / ``num_episodes`` (forwarded) stay the real counts. The mapping is
written to <output_dir>/task_balancing.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch


class TaskBalancedDataset(torch.utils.data.Dataset):
    """Map-style view of a LeRobotDataset in which every task has the same number of indices."""

    def __init__(self, dataset):
        self._ds = dataset
        if hasattr(dataset, "_ensure_hf_dataset_loaded"):
            dataset._ensure_hf_dataset_loaded()
        task_index = np.asarray(dataset.hf_dataset.with_format(None)["task_index"], dtype=np.int64)
        if len(task_index) != len(dataset):
            raise ValueError(f"task_index has {len(task_index)} rows, the dataset {len(dataset)}")
        self._tasks = sorted(int(t) for t in np.unique(task_index))
        self._rows = [np.flatnonzero(task_index == t) for t in self._tasks]
        self._slots = max(len(r) for r in self._rows)
        names = dataset.meta.tasks
        self.summary = {
            "tasks": [{"task_index": t, "task": str(names.index[names["task_index"] == t][0]), "frames": int(len(r))}
                      for t, r in zip(self._tasks, self._rows)],
            "slots_per_task": int(self._slots),
            "virtual_length": int(self._slots * len(self._tasks)),
            "real_frames": int(len(dataset)),
        }

    def __len__(self) -> int:
        return self._slots * len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        t, j = divmod(int(idx), self._slots)
        rows = self._rows[t]
        return self._ds[int(rows[(j * len(rows)) // self._slots])]

    def __getattr__(self, name):  # meta, num_frames, num_episodes, features, ... of the wrapped dataset
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ds, name)


def main() -> None:
    import lerobot.scripts.lerobot_train as lerobot_train

    make_dataset = lerobot_train.make_dataset

    def make_balanced_dataset(cfg):
        ds = TaskBalancedDataset(make_dataset(cfg))
        logging.info("task-balanced sampling: %s", json.dumps(ds.summary))
        if cfg.output_dir is not None:
            out = Path(cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "task_balancing.json").write_text(json.dumps(ds.summary, indent=2))
        return ds

    lerobot_train.make_dataset = make_balanced_dataset
    lerobot_train.main()


if __name__ == "__main__":
    main()
