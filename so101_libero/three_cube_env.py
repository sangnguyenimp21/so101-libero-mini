"""``SO101ThreeCubeEnv``: the shared three-cube scene with one of its tasks (so101_libero/tasks/three_cubes.py).

    from so101_libero.three_cube_env import SO101ThreeCubeEnv
    env = SO101ThreeCubeEnv("sort")                       # flip | sort | pyramid, or a task id
    obs, info = env.reset(options={"placement_id": 7})    # sort: placement = 5 * layout + permutation
    env.task_info()                                       # layout, permutation, cube poses, targets (logs only)

The same robot, cameras, table, action space, observations, control rate, cube models and HOME for every task; only
the reset configuration (placements), the instruction (``env.task.language``) and the success rule differ. It is
``SO101TabletopEnv`` itself: the policy-facing observation is the joint state and the two camera images, nothing else.
``task_info`` reads the hidden simulator state and is meant for logs, the expert and evaluation only.
"""

from __future__ import annotations

import numpy as np

from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.tasks.three_cubes import SUITE, resolve_task_id


class SO101ThreeCubeEnv(SO101TabletopEnv):
    def __init__(self, task: str = "flip", config: EnvConfig | None = None):
        super().__init__(resolve_task_id(task), config)
        if SUITE not in self.task.tags:
            raise ValueError(f"{self.task.task_id} is not a three-cube task (flip | sort | pyramid)")

    def task_info(self) -> dict:
        """Task id, instruction, layout and starting permutation of the current episode, the cubes' settled start poses
        and (sort) the row targets. Privileged: never give this to a policy."""
        pid = (self.init_state or {}).get("placement_id")
        info = {"task_id": self.task.task_id, "instruction": self.task.language,
                **(self.task.layout.placement_info(pid) if pid is not None else {})}
        info["initial_positions"] = {k: np.round(v, 4).tolist() for k, v in self.world.init_pos.items()}
        if self.task.task_id == "sort_ryg":
            info["targets"] = np.round(self.task.goal[0].target_xy(self.world), 4).tolist()
        return info
