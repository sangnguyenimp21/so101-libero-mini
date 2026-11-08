"""Require zero residual to match SmolVLA on ten identical L* validation states."""
import json
from pathlib import Path
import numpy as np
import torch
from so101_libero.residual.calibration_residual_env import CalibrationResidualConfig
from so101_libero.residual.calibration_no_leak_env import NoLeakCalibrationResidualEnv
ROOT=Path('docs/residual/results/calibration')
L=int(json.loads((ROOT/'selection.json').read_text())['chosen_L_deg'])
rows=json.loads(Path('docs/residual/calibration/validation.json').read_text())['episodes'][:10]
base=json.loads((ROOT/'validation_sweep.json').read_text())['levels'][str(L)]['episodes'][:10]
env=NoLeakCalibrationResidualEnv(CalibrationResidualConfig(limit_deg=L))
out=[]
for row in rows:
    seed=int(row['seed']); torch.manual_seed(seed)
    obs,_=env.reset(seed=seed,options={'init_state':row['init_state'],
        'calibration_bias_deg':row['bias_by_level_deg'][str(L)]})
    while True:
        obs,_,done,truncated,info=env.step(np.zeros(6,np.float32))
        if done or truncated: break
    out.append(bool(info['is_success']))
env.close()
expected=[bool(x['success']) for x in base]
(ROOT/'no_leak'/'zero_residual_test.json').write_text(json.dumps({'L_deg':L,'baseline':expected,'zero_residual':out,
    'matches':out==expected},indent=2))
assert out==expected,(out,expected)
print('zero residual matches baseline on 10 L* validation states',flush=True)
