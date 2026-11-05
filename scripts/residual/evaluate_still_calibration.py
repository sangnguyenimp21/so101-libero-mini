"""Evaluate non-learning first-still-moment calibration on the frozen 100-state test."""
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from so101_libero.cli.evaluate import wilson
from so101_libero.policy.runner import PolicyRunner
from so101_libero.residual.calibration import CalibrationOffsetEnv
from so101_libero.residual.still_calibration import FirstStillCalibrator
from so101_libero.tasks.registry import get_task

ROOT = Path('docs/residual/results/calibration')
CKPT = 'checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model'
L = int(json.loads((ROOT/'selection.json').read_text())['chosen_L_deg'])
states = json.loads(Path('docs/residual/calibration/test.json').read_text())['episodes']
reference = json.loads((ROOT/'final_evaluation/results.json').read_text())['sets']['test_Lstar']['baseline']
assert len(states) == len(reference) == 100

runner = PolicyRunner(CKPT, device='cuda')
env = CalibrationOffsetEnv(L)
rows = []
for row, saved in zip(states, reference):
    seed = int(row['seed'])
    assert seed == int(saved['seed']) and row['index'] == saved['index']
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed, options={'init_state': row['init_state'],
        'calibration_bias_deg': row['bias_by_level_deg'][str(L)]})
    runner.reset()
    calibrator = FirstStillCalibrator(L)
    start = obs['observation.state'][:5].copy()
    motion = 0.0
    step = 0
    while True:
        base_action = runner.act(obs, get_task('cube_flip').language)
        command = calibrator.compensate(base_action, env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, info = env.step(command)
        step += 1
        calibrator.observe(command, obs['observation.state'], step)
        motion = max(motion, float(np.max(np.abs(obs['observation.state'][:5] - start))))
        if terminated or truncated:
            break
    success = bool(info['is_success'])
    lifted = bool(env.unwrapped.world.lifted('red_cube', 0.02))
    failure = None if success else ('lifted_not_flipped' if lifted else
        'stalled' if motion < 5 else 'moved_but_missed')
    result = {'index': row['index'], 'seed': seed, 'placement_id': row['placement_id'],
        'bias_true_deg': env.bias.tolist(), 'bias_estimate_deg': calibrator.bias_hat.tolist(),
        'estimated_at_step': calibrator.estimated_at_step, 'success': success,
        'failure_type': failure, 'steps': step}
    rows.append(result)
    print(f"{len(rows)}/100 success={success} estimate_step={calibrator.estimated_at_step}", flush=True)
    # Save after every episode so a PBS wall-time interruption retains the diagnostic record.
    (ROOT/'still_calibration_partial.json').write_text(json.dumps(rows, indent=2))
env.close()

fixed = sum(not b['success'] and r['success'] for b, r in zip(reference, rows))
broken = sum(b['success'] and not r['success'] for b, r in zip(reference, rows))
d = fixed + broken
p = min(1., 2 * sum(math.comb(d, k) for k in range(min(fixed, broken)+1)) / 2**d) if d else 1.
k = sum(r['success'] for r in rows)
lo, hi = wilson(k, len(rows))
summary = {'Lstar_deg': L, 'episodes': len(rows), 'successes': k, 'wilson95': [lo, hi],
    'baseline_successes': sum(b['success'] for b in reference), 'fixed': fixed, 'broken': broken,
    'mcnemar_exact_two_sided_p': p,
    'failure_types': dict(Counter(r['failure_type'] for r in rows if not r['success'])),
    'estimated_episodes': sum(r['estimated_at_step'] is not None for r in rows),
    'mean_abs_bias_estimation_error_deg': np.mean([
        np.abs(np.asarray(r['bias_estimate_deg'][:5]) - np.asarray(r['bias_true_deg'][:5])) for r in rows], axis=0).tolist(),
    'estimator': {'window': 10, 'max_position_span_deg': 0.3,
        'max_command_span_deg': 0.5, 'bias_hat': 'median(measured_state - command)',
        'correction': 'SmolVLA_target - bias_hat'}}
(ROOT/'still_calibration_results.json').write_text(json.dumps({'summary': summary, 'episodes': rows}, indent=2))
print(json.dumps(summary, indent=2), flush=True)
