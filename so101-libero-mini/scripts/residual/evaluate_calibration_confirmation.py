"""Independent frozen confirmation: SmolVLA versus prechosen no-leak final PPO."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from so101_libero.cli.evaluate import wilson
from so101_libero.policy.runner import PolicyRunner
from so101_libero.residual.calibration import CalibrationOffsetEnv
from so101_libero.residual.calibration_residual_env import CalibrationResidualConfig
from so101_libero.residual.calibration_no_leak_env import NoLeakCalibrationResidualEnv
from so101_libero.tasks.registry import get_task
from so101_libero.utils.video import write_mp4, label

ROOT = Path('docs/residual/results/calibration')
CKPT = 'checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model'
L = int(json.loads((ROOT/'selection.json').read_text())['chosen_L_deg'])
OUT = ROOT/'no_leak'/'confirmation_2deg_200'
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT/'no_leak'/'ppo'/'final_model.zip'
EXPECTED_SHA256 = '85469b20688c532492453d73e11fa49607cd6f3b9540f47eb05a01ac1ea6207f'

def confirmation_rows():
    frozen = json.loads(Path('docs/residual/calibration/confirmation_2deg_200.json').read_text())
    rows = frozen['episodes']
    assert L == frozen['offset_limit_deg'] == 2 and len(rows) == 200
    assert [row['seed'] for row in rows] == list(range(40_000_000, 40_000_200))
    assert all(row['placement_id'] == i % 5 for i, row in enumerate(rows))
    return rows

def classify(info, max_motion):
    if info['is_success']: return None
    if info['lifted']: return 'lifted_not_flipped'
    return 'stalled' if max_motion < 5 else 'moved_but_missed'

def baseline(rows, limit):
    env = CalibrationOffsetEnv(limit)
    runner = PolicyRunner(CKPT, device='cuda')
    out = []
    for row in rows:
        seed = int(row['seed']); torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed, options={'init_state': row['init_state'],
            'calibration_bias_deg': row['bias_by_level_deg'][str(limit)]})
        runner.reset(); start = obs['observation.state'][:5].copy(); motion = 0
        while True:
            action = runner.act(obs, get_task('cube_flip').language)
            obs, _, done, truncated, info = env.step(action)
            motion = max(motion, float(np.max(np.abs(obs['observation.state'][:5]-start))))
            if done or truncated: break
        info['lifted'] = env.unwrapped.world.lifted('red_cube', 0.02)
        out.append({'index': row['index'], 'seed': seed, 'bias_deg': env.bias.tolist(),
            'success': bool(info['is_success']), 'failure_type': classify(info,motion)})
    env.close(); return out

def residual(model_path, rows, limit, video_indices=()):
    env = NoLeakCalibrationResidualEnv(CalibrationResidualConfig(limit_deg=limit))
    model = PPO.load(model_path, device='cpu')
    out=[]
    for row in rows:
        seed = int(row['seed']); torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed, options={'init_state': row['init_state'],
            'calibration_bias_deg': row['bias_by_level_deg'][str(limit)]})
        motion=0; mags=[]; frames=[]
        record_video = row['index'] in video_indices
        while True:
            action,_ = model.predict(obs, deterministic=True)
            obs,_,done,truncated,info = env.step(action)
            mags.append(np.abs(info['residual_action']*info['residual_scale']))
            motion = max(motion, info['max_arm_motion_deg'])
            if record_video: frames.append(label(env.env.unwrapped.render(), f"state {row['index']} L={limit}"))
            if done or truncated: break
        result={'index':row['index'],'seed':seed,'success':bool(info['is_success']),
            'failure_type':classify(info,motion),'mean_abs_residual_by_joint':np.mean(mags,axis=0).tolist()}
        if record_video and result['success']:
            path=OUT/'videos'/f'{model_path.stem}_state_{row["index"]:03d}.mp4'
            write_mp4(frames,path)
            result['video']=str(path.relative_to(OUT))
        out.append(result)
    env.close(); return out

def aggregate(base, candidate):
    n=len(base); k=sum(x['success'] for x in candidate); b=sum(x['success'] for x in base)
    fixed=sum(not x['success'] and y['success'] for x,y in zip(base,candidate))
    broken=sum(x['success'] and not y['success'] for x,y in zip(base,candidate))
    d=fixed+broken
    p=min(1., 2*sum(math.comb(d,i) for i in range(min(fixed,broken)+1))/2**d) if d else 1.
    lo,hi=wilson(k,n); blo,bhi=wilson(b,n)
    return {'n':n,'baseline_successes':b,'baseline_wilson95':[blo,bhi],
        'residual_successes':k,'residual_wilson95':[lo,hi],
        'fixed':fixed,'broken':broken,'mcnemar_exact_two_sided_p':p,
        'baseline_failures':dict(Counter(x['failure_type'] for x in base if not x['success'])),
        'residual_failures':dict(Counter(x['failure_type'] for x in candidate if not x['success'])),
        'mean_abs_residual_by_joint':np.mean([x['mean_abs_residual_by_joint'] for x in candidate],axis=0).tolist()}

if __name__ == '__main__':
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, f'checkpoint changed: {digest}'
    rows = confirmation_rows()
    base = baseline(rows, L)
    print('confirmation baseline', sum(x['success'] for x in base), len(base), flush=True)
    cand = residual(MODEL, rows, L)
    assert all(a['index'] == b['index'] and a['seed'] == b['seed'] for a,b in zip(base,cand))
    summary = aggregate(base, cand)
    results = {'offset_limit_deg': L, 'checkpoint': str(MODEL),
               'checkpoint_sha256': digest, 'summary': summary,
               'baseline_episodes': base, 'residual_episodes': cand}
    (OUT/'results.json').write_text(json.dumps(results, indent=2))
    print('confirmation', summary, flush=True)
