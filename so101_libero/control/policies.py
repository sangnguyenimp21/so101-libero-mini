"""Policies behind one interface for the episode controller: ``reset``, ``act(obs, instruction)``, and the two things
the controller needs to stop cleanly: ``pending()`` (actions of the current chunk not executed yet) and
``discard_pending()``.

LeRobot 0.4.2's SmolVLA has no done flag, termination token or stop action: it predicts chunks of ``chunk_size`` (50)
actions and ``select_action`` pops them one per control step from ``policy._queues["action"]`` (a deque), predicting a
new chunk when the queue is empty (lerobot/policies/smolvla/modeling_smolvla.py). The controller therefore detects
success externally after every step, and on success drops whatever is still queued (``discard_pending`` = count the
deque, then ``policy.reset()``, which clears it), so no queued action is ever executed after success.
"""

from __future__ import annotations

import numpy as np


class LeRobotPolicy:
    """A LeRobot checkpoint (SmolVLA, ACT, ...) through so101_libero/policy/runner.py (the checkpoint's own pre/post
    processors; inputs are only the joint state, the two camera images and the instruction)."""

    def __init__(self, checkpoint: str, device: str | None = None, n_action_steps: int | None = None):
        from so101_libero.policy.runner import PolicyRunner

        self.runner = PolicyRunner(checkpoint, device=device, n_action_steps=n_action_steps)
        self.chunks = 0  # chunks predicted in this episode (= inference calls)

    def _queue(self):
        pol = self.runner.policy
        queues = getattr(pol, "_queues", None)
        if isinstance(queues, dict) and "action" in queues:
            return queues["action"]
        return getattr(pol, "_action_queue", None)  # ACT keeps its own deque

    def reset(self) -> None:
        self.runner.reset()
        self.chunks = 0

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        if self.pending() == 0:
            self.chunks += 1
        return np.asarray(self.runner.act(obs, instruction), dtype=np.float32).reshape(6)

    def pending(self) -> int:
        q = self._queue()
        return 0 if q is None else len(q)

    def discard_pending(self) -> int:
        n = self.pending()
        self.runner.reset()
        return n

    def describe(self) -> dict:
        return {"kind": "lerobot", **self.runner.describe()}


class ExpertPolicy:
    """The scripted expert (privileged simulator state) behind the policy interface, for debugging the controller in
    simulation: its actions are counted in virtual chunks of ``chunk_size`` so that stop-on-success reports discarded
    actions exactly like a chunked policy. The expert is (re)built at ``reset`` from the freshly reset scene."""

    def __init__(self, env, seed: int = 0, chunk_size: int = 50):
        self.env, self.seed, self.chunk_size = env, int(seed), int(chunk_size)
        self.expert = None
        self._served = 0
        self._last = None
        self.chunks = 0

    def reset(self) -> None:
        from so101_libero.expert.rollout import expert_rng, make_expert

        self.expert = make_expert(self.env, self.env.task.plan, expert_rng(self.seed))
        self._served, self._last, self.chunks = 0, None, 0

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        if self._served % self.chunk_size == 0:
            self.chunks += 1
        action = self.expert.act()
        if action is None:  # the expert finished (or failed): hold the last command
            action = self._last if self._last is not None else np.asarray(obs["observation.state"], dtype=np.float32)
        self._served += 1
        self._last = np.asarray(action, dtype=np.float32)
        return self._last.copy()

    def pending(self) -> int:
        return (-self._served) % self.chunk_size

    def discard_pending(self) -> int:
        n = self.pending()
        self._served += n  # the rest of the virtual chunk is dropped (the expert is never asked again)
        return n

    def describe(self) -> dict:
        return {"kind": "scripted_expert", "seed": self.seed, "chunk_size": self.chunk_size}
