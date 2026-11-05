"""Build a 300-episode LeRobot cube_flip dataset: 100 original-layout and 200 diverse demos.

The original 100-demo dataset is absent on Vanda, so the first 100 are regenerated
from the same five placements. All collection seeds are in the reserved 20M range.
Only physics-verified successful expert episodes are saved.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from so101_libero.data.writer import DemoWriter
from so101_libero.env import EnvConfig, SO101TabletopEnv
from so101_libero.expert.rollout import run_expert_episode
from so101_libero.residual.placements import training_offset, shifted_init_state
from so101_libero.tasks.registry import get_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo-id', default='local/so101_cube_flip_diverse_300')
    p.add_argument('--seed-start', type=int, default=21_000_000)
    a = p.parse_args()
    if not 20_000_000 <= a.seed_start <= 29_990_000:
        raise ValueError('seed must be in reserved training range')
    env = SO101TabletopEnv('cube_flip', EnvConfig(render=True, terminate_on_success=False, max_steps=900))
    writer = DemoWriter(a.repo_id, a.root, 30, (256, 256), episode_meta=False)
    counts = {'original_regenerated': 0, 'diverse_new': 0}
    attempts = {'original_regenerated': 0, 'diverse_new': 0}
    try:
        for phase, quota in [('original_regenerated', 100), ('diverse_new', 200)]:
            while counts[phase] < quota:
                if attempts[phase] >= quota * 12:
                    raise RuntimeError(f'{phase}: too many expert failures; saved {counts[phase]}/{quota}')
                seed = a.seed_start + attempts['original_regenerated'] + attempts['diverse_new']
                attempt = attempts[phase]
                attempts[phase] += 1
                if phase == 'original_regenerated':
                    placement = counts[phase] % 5
                    ep = run_expert_episode(env, seed, record=True, placement_id=placement)
                    offset = None
                else:
                    offset = training_offset(seed)
                    ep = run_expert_episode(env, seed, record=True,
                                            init_state=shifted_init_state(*offset))
                record = {**ep.summary(), 'phase': phase, 'offset_m': offset}
                if not ep.success:
                    writer.add_failure(record)
                    print(f'{phase} failed seed={seed}: {ep.failure}', flush=True)
                    continue
                observations = [o for o, _ in ep.frames]
                states = np.stack([o['observation.state'] for o in observations])
                actions = np.stack([u for _, u in ep.frames])
                images = {k: np.stack([o[k] for o in observations]) for k in observations[0]
                          if k.startswith('observation.images.')}
                writer.add_episode(get_task('cube_flip').language, states, actions, images, record)
                counts[phase] += 1
                print(f'{phase}: {counts[phase]}/{quota}, seed={seed}, offset={offset}', flush=True)
        writer.finalize({'task': 'cube_flip', 'counts': counts, 'attempts': attempts,
                         'seed_start': a.seed_start, 'original_provenance': 'regenerated; source dataset unavailable',
                         'new_positions': 'uniform continuous training_offset; reserved 20M seeds'})
    finally:
        env.close()
    print(json.dumps({'counts': counts, 'attempts': attempts}), flush=True)


if __name__ == '__main__':
    main()
