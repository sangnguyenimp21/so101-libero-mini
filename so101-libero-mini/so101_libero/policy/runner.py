"""Run a trained LeRobot policy checkpoint on environment observations.

The runner receives ONLY what a real robot provides: the observation dict (joint state + camera images) and the
language instruction. Normalization uses the pre/post-processors saved with the checkpoint, so units are the
dataset's (LeRobot so101_follower units) end to end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

POLICY_KEYS = ("observation.state", "observation.images.front", "observation.images.wrist")


class PolicyRunner:
    def __init__(self, checkpoint: str | Path, device: str | None = None, n_action_steps: int | None = None,
                 temporal_ensemble_coeff: float | None = None):
        import lerobot.policies  # noqa: F401  (registers the policy config classes)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.checkpoint = str(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":  # several evaluation workers share the CPU: do not let each grab every core
            torch.set_num_threads(int(os.environ.get("SO101_TORCH_THREADS", "4")))
        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        cfg.device = self.device
        if temporal_ensemble_coeff is not None:  # ACT temporal ensembling: re-plan every step
            if cfg.type != "act":
                raise ValueError("temporal ensembling is an ACT option")
            cfg.temporal_ensemble_coeff = temporal_ensemble_coeff
            cfg.n_action_steps = 1
        elif n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        if temporal_ensemble_coeff is not None or n_action_steps is not None:
            # from_pretrained(config=cfg) does not re-run the config's checks: run them for the overridden values
            # (ACT: n_action_steps <= chunk_size, ensembling needs n_action_steps == 1). Diffusion's limit is not
            # checked by LeRobot (actions beyond it are silently dropped), so check it here.
            cfg.__post_init__()
            if cfg.type == "diffusion" and cfg.n_action_steps > cfg.horizon - cfg.n_obs_steps + 1:
                raise ValueError(f"diffusion: n_action_steps must be <= horizon - n_obs_steps + 1 "
                                 f"= {cfg.horizon - cfg.n_obs_steps + 1}")
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(self.checkpoint, config=cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=self.checkpoint, preprocessor_overrides={"device_processor": {"device": self.device}})
        # Policies trained with --rename_map (e.g. SmolVLA: front -> camera1) carry the renaming in their saved
        # preprocessor; feed environment keys and let it rename. Map each policy input back to its env key.
        inverse = {v: k for k, v in self._rename_map().items()}
        self.input_keys, missing = [], []
        for k in cfg.input_features:
            env_key = inverse.get(k, k)
            (self.input_keys if env_key in POLICY_KEYS else missing).append(env_key)
        if missing and cfg.type != "smolvla":  # SmolVLA skips absent cameras (e.g. smolvla_base's camera3)
            raise ValueError(f"checkpoint expects inputs the environment does not provide: {missing}")

    def _rename_map(self) -> dict[str, str]:
        path = Path(self.checkpoint) / "policy_preprocessor.json"
        if not path.exists():
            return {}
        for step in json.loads(path.read_text()).get("steps", []):
            if "rename" in str(step.get("registry_name", "")):
                return dict(step.get("config", {}).get("rename_map") or {})
        return {}

    def reset(self) -> None:
        self.policy.reset()

    @torch.inference_mode()
    def act(self, obs: dict, task: str) -> np.ndarray:
        batch = {"task": task}
        for k in self.input_keys:
            v = obs[k]
            if k.startswith("observation.images."):
                batch[k] = torch.from_numpy(np.ascontiguousarray(v)).permute(2, 0, 1).float().div(255.0)
            else:
                batch[k] = torch.from_numpy(np.asarray(v, dtype=np.float32))
        action = self.post(self.policy.select_action(self.pre(batch)))
        return action[0].detach().cpu().numpy().astype(np.float32)

    def describe(self) -> dict:
        c = self.cfg
        return {"checkpoint": self.checkpoint, "type": c.type, "device": self.device,
                "n_action_steps": getattr(c, "n_action_steps", None), "chunk_size": getattr(c, "chunk_size", None),
                "temporal_ensemble_coeff": getattr(c, "temporal_ensemble_coeff", None), "inputs": self.input_keys}
