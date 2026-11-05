"""Freeze disjoint original-placement states and calibration biases before RL."""
import json
from pathlib import Path
from so101_libero.cli.make_init_states import load_init_states
from so101_libero.residual.calibration import sample_bias

OUT = Path('docs/residual/calibration')
SPECS = {'validation': (32_000_000, 50), 'selection': (33_000_000, 20), 'test': (34_000_000, 100)}
LEVELS = (0, 2, 4, 6, 8)
original = load_init_states('cube_flip')['states']
OUT.mkdir(parents=True, exist_ok=True)
for name, (seed_base, count) in SPECS.items():
    rows = []
    for i in range(count):
        seed = seed_base + i
        state = original[i % 50]['init_state']
        rows.append({'index': i, 'seed': seed, 'placement_id': i % 5, 'init_state': state,
                     'bias_by_level_deg': {str(level): sample_bias(seed, level).tolist() for level in LEVELS}})
    path = OUT / f'{name}.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({'name': name, 'seed_base': seed_base, 'episodes': rows}, indent=2))
    print(path, len(rows))
