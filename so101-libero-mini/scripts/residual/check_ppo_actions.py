"""Read-only audit of the current best PPO checkpoints on the 20 validation episodes."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from scripts.residual.train_ppo import load_eval_states
from so101_libero.residual.residual_env import ResidualConfig, ResidualSO101Env

states = load_eval_states(Path('docs/residual/placements'), Path('docs/residual/results/baseline_shifted_140/results.json'), 20)
out = {}
for name in ('calm', 'kickstart'):
    run = Path('docs/residual/results') / f'ppo_{name}'
    cfg = json.loads((run / 'config.json').read_text())
    checkpoints = sorted((run / 'checkpoints').glob('ppo_residual_*_steps.zip'),
                         key=lambda p: int(p.stem.split('_')[-2]))
    checkpoint = checkpoints[-1] if checkpoints else run / 'best_model.zip'
    model = PPO.load(checkpoint, device='cpu')
    env = ResidualSO101Env(ResidualConfig(checkpoint='checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model',
        early_arm_scale_deg=cfg['early_arm_scale'], late_arm_scale_deg=cfg['late_arm_scale'],
        early_gripper_scale=cfg['early_gripper_scale'], late_gripper_scale=cfg['late_gripper_scale']))
    variant = {'checkpoint': str(checkpoint), 'selection_seeds': [int(s['seed']) for s in states],
               'deterministic_reference': '9/20 SmolVLA baseline', 'trials': []}
    for trial in (-1, 0, 1, 2):
        magnitudes = []
        successes = 0
        for s in states:
            seed = int(s['seed'])
            torch.manual_seed(1000 + trial)
            np.random.seed(1000 + trial)
            obs, _ = env.reset(seed=seed, options={'init_state': s['init_state'], 'offset_m': s['offset_m']})
            step = 0
            while True:
                action, _ = model.predict(obs, deterministic=(trial == -1))
                obs, _, done, truncated, info = env.step(action)
                if step < 50:
                    magnitudes.append(np.abs(info['residual_action'] * info['residual_scale']))
                step += 1
                if done or truncated:
                    successes += int(info['is_success'])
                    break
        variant['trials'].append({'mode': 'deterministic' if trial == -1 else 'stochastic', 'seed': 1000 + trial, 'successes': successes,
            'mean_abs_first_50': np.mean(magnitudes, axis=0).tolist()})
        print(name, trial, successes, variant['trials'][-1]['mean_abs_first_50'], flush=True)
    env.close()
    out[name] = variant
path = Path('docs/residual/results/ppo_action_audit.json')
path.write_text(json.dumps(out, indent=2))
