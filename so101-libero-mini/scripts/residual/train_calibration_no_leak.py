"""PPO on original placements with fixed episode-wise arm calibration bias."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from so101_libero.residual.calibration_residual_env import CalibrationResidualConfig
from so101_libero.residual.calibration_no_leak_env import NoLeakCalibrationResidualEnv

ROOT = Path('docs/residual/results/calibration')
STATES = Path('docs/residual/calibration')

def level():
    value = json.loads((ROOT/'selection.json').read_text())['chosen_L_deg']
    if value is None: raise RuntimeError('No preregistered calibration level')
    return int(value)

def frozen_rows(split):
    return json.loads((STATES/f'{split}.json').read_text())['episodes']

def evaluate(model, env, rows, L, stochastic_seed=None):
    outcomes, magnitudes = [], []
    for row in rows:
        seed = int(row['seed'])
        torch.manual_seed(seed if stochastic_seed is None else seed + 100_000 * stochastic_seed)
        obs, _ = env.reset(seed=seed, options={'init_state': row['init_state'],
            'calibration_bias_deg': row['bias_by_level_deg'][str(L)]})
        ep_mags = []
        while True:
            action, _ = model.predict(obs, deterministic=stochastic_seed is None)
            obs, _, done, truncated, info = env.step(action)
            ep_mags.append(np.abs(info['residual_action'] * info['residual_scale']))
            if done or truncated: break
        outcomes.append(bool(info['is_success']))
        magnitudes.extend(ep_mags)
    return {'successes': sum(outcomes), 'n': len(rows), 'outcomes': outcomes,
        'mean_abs_residual_by_joint': np.mean(magnitudes, axis=0).tolist()}

class FrozenEval(BaseCallback):
    def __init__(self, eval_env, rows, L, out, interval):
        super().__init__()
        self.eval_env, self.rows, self.L, self.out, self.interval = eval_env, rows, L, out, interval
        self.next_eval = interval
        self.best = -1
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/'eval_checkpoints').mkdir(exist_ok=True)

    def record(self, steps):
        det = evaluate(self.model, self.eval_env, self.rows, self.L)
        stochastic = [evaluate(self.model, self.eval_env, self.rows, self.L, j) for j in (1,2,3)]
        rec = {'steps': steps, 'deterministic': det, 'stochastic': stochastic}
        model_path = self.out/'eval_checkpoints'/f'eval_{steps:08d}.zip'
        self.model.save(model_path)
        with (self.out/'evaluations.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
        if det['successes'] > self.best:
            self.best = det['successes']
            self.model.save(self.out/'best_model.zip')
            (self.out/'best_evaluation.json').write_text(json.dumps(rec, indent=2))
        print('selection', steps, det['successes'], [r['successes'] for r in stochastic], flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.record(self.num_timesteps)
            self.next_eval += self.interval
        return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--total-timesteps', type=int, default=500_000)
    p.add_argument('--eval-freq', type=int, default=100_000)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--out', type=Path, default=ROOT/'no_leak'/'ppo')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    L = level()
    rows = frozen_rows('selection')
    cfg = lambda i: CalibrationResidualConfig(limit_deg=L, worker_id=i, num_workers=a.num_envs)
    def make(i):
        def factory(): return NoLeakCalibrationResidualEnv(cfg(i))
        return factory
    env = VecMonitor(SubprocVecEnv([make(i) for i in range(a.num_envs)], start_method='forkserver'))
    eval_env = NoLeakCalibrationResidualEnv(CalibrationResidualConfig(limit_deg=L))
    a.out.mkdir(parents=True, exist_ok=True)
    latest = sorted((a.out/'eval_checkpoints').glob('eval_*.zip')) if (a.out/'eval_checkpoints').exists() else []
    if latest and not a.smoke:
        model = PPO.load(latest[-1], env=env, device='cpu')
        print('resuming', latest[-1], model.num_timesteps, flush=True)
    else:
        model = PPO('MlpPolicy', env, policy_kwargs={'net_arch': {'pi':[256,256], 'vf':[256,256]}},
            n_steps=256, batch_size=256, n_epochs=10, learning_rate=1e-4,
            target_kl=0.02, seed=73, device='cpu', verbose=1, tensorboard_log=str(a.out/'tb'))
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.log_std.fill_(-2.0)
    callback = FrozenEval(eval_env, rows, L, a.out, a.eval_freq)
    if latest and not a.smoke:
        best_path = a.out/'best_evaluation.json'
        if best_path.exists(): callback.best = json.loads(best_path.read_text())['deterministic']['successes']
        callback.next_eval = ((model.num_timesteps // a.eval_freq)+1)*a.eval_freq
    else:
        callback.model = model
        callback.record(0)
    remaining = a.total_timesteps - model.num_timesteps
    if remaining > 0:
        model.learn(remaining, callback=callback, reset_num_timesteps=not latest,
                    tb_log_name='calibration_ppo')
    model.save(a.out/'final_model.zip')
    env.close(); eval_env.close()

if __name__ == '__main__': main()
