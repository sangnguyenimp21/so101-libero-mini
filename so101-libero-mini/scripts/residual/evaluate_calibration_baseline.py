"""Frozen calibration-offset SmolVLA sweep on original cube_flip placements."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from so101_libero.cli.evaluate import wilson
from so101_libero.cli.make_init_states import load_init_states
from so101_libero.policy.runner import PolicyRunner
from so101_libero.residual.calibration import CalibrationOffsetEnv
from so101_libero.tasks.registry import get_task

CKPT = 'checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model'

def episode(env, runner, state, limit):
    seed = int(state['seed'])
    torch.manual_seed(seed)
    options = {'init_state': state['init_state']}
    if 'bias_by_level_deg' in state:
        options['calibration_bias_deg'] = state['bias_by_level_deg'][str(limit)]
    obs, info = env.reset(seed=seed, options=options)
    runner.reset()
    start = np.asarray(obs['observation.state'])[:5].copy()
    max_move = 0.0
    steps = 0
    while True:
        action = runner.act(obs, get_task('cube_flip').language)
        obs, _, done, truncated, info = env.step(action)
        max_move = max(max_move, float(np.max(np.abs(obs['observation.state'][:5] - start))))
        steps += 1
        if done or truncated:
            break
    success = bool(info['is_success'])
    lifted = bool(env.unwrapped.world.lifted('red_cube', 0.02))
    failure = None if success else ('lifted_not_flipped' if lifted else
        'stalled' if max_move < 5 else 'moved_but_missed')
    return {'seed': seed, 'placement_id': state.get('placement_id', state['index'] % 5),
            'bias_deg': env.bias.tolist(), 'success': success, 'lifted': lifted,
            'steps': steps, 'max_arm_motion_deg': max_move, 'failure_type': failure}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('validation','zero_test'), default='validation')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.split == 'zero_test':
        raw = load_init_states('cube_flip')['states'][:10]
        states = [{'index': r['index'], 'seed': r['seed'], 'init_state': r['init_state']} for r in raw]
        levels = [0]
    else:
        states = json.loads(Path('docs/residual/calibration/validation.json').read_text())['episodes']
        levels = [0,2,4,6,8]
    runner = PolicyRunner(CKPT, device='cuda')
    results = {}
    for limit in levels:
        env = CalibrationOffsetEnv(limit)
        rows = []
        for i, state in enumerate(states):
            rows.append(episode(env, runner, state, limit))
            print(f'{a.split} L={limit} {i+1}/{len(states)} success={rows[-1]["success"]}', flush=True)
        env.close()
        n = len(rows); k = sum(r['success'] for r in rows)
        lo, hi = wilson(k,n)
        results[str(limit)] = {'episodes': rows, 'successes': k, 'n': n,
            'wilson95': [lo,hi], 'failures': {t: sum(r['failure_type']==t for r in rows)
                 for t in ('stalled','moved_but_missed','lifted_not_flipped')}}
        a.out.write_text(json.dumps({'split': a.split, 'levels': results}, indent=2))
        print('LEVEL', limit, k, n, results[str(limit)]['failures'], flush=True)
    if a.split == 'zero_test':
        reference = json.loads(Path('docs/residual/results/baseline_original_50/results.json').read_text())['episodes'][:10]
        assert [r['success'] for r in results['0']['episodes']] == [r['success'] for r in reference]
        print('L=0 exactly matches original 10 outcomes', flush=True)

if __name__ == '__main__': main()
