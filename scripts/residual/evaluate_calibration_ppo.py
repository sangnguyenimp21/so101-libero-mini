"""Matched frozen calibration test: base versus selected and final residual PPO."""
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from so101_libero.cli.evaluate import wilson
from so101_libero.cli.make_init_states import load_init_states
from so101_libero.policy.runner import PolicyRunner
from so101_libero.residual.calibration import CalibrationOffsetEnv
from so101_libero.residual.calibration_residual_env import CalibrationResidualEnv, CalibrationResidualConfig
from so101_libero.tasks.registry import get_task
from so101_libero.utils.video import write_mp4, label

ROOT = Path('docs/residual/results/calibration')
CKPT = 'checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model'
L = int(json.loads((ROOT/'selection.json').read_text())['chosen_L_deg'])
OUT = ROOT/'final_evaluation'
OUT.mkdir(parents=True, exist_ok=True)

def datasets():
    test = json.loads(Path('docs/residual/calibration/test.json').read_text())['episodes']
    original = load_init_states('cube_flip')['states'][:50]
    normal = [{'seed': s['seed'], 'index': s['index'], 'placement_id': s['index']%5,
        'init_state': s['init_state'], 'bias_by_level_deg': {'0': [0]*6}} for s in original]
    return {'test_Lstar': (test,L), 'original_L0': (normal,0)}

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
    env = CalibrationResidualEnv(CalibrationResidualConfig(limit_deg=limit))
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

results={'Lstar_deg':L,'sets':{}}
for name,(rows,limit) in datasets().items():
    base=baseline(rows,limit)
    entry={'baseline':base,'models':{}}
    for model_name in ('best_model','final_model'):
        path=ROOT/'ppo'/f'{model_name}.zip'
        cand=residual(path,rows,limit)
        fixes=[r['index'] for b,r in zip(base,cand) if not b['success'] and r['success']]
        if name=='test_Lstar' and fixes:
            residual(path,[r for r in rows if r['index'] in fixes[:3]],limit,fixes[:3])
        entry['models'][model_name]={'summary':aggregate(base,cand),'episodes':cand}
        results['sets'][name]=entry
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        print(name,model_name,entry['models'][model_name]['summary'],flush=True)
