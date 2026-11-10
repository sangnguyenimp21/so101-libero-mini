"""Freeze a new, disjoint 200-episode L=2 confirmation set once."""
import json
from pathlib import Path

from so101_libero.cli.make_init_states import load_init_states
from so101_libero.residual.calibration import sample_bias

OUT = Path('docs/residual/calibration/confirmation_2deg_200.json')
SEED_BASE = 40_000_000
COUNT = 200

if OUT.exists():
    raise FileExistsError(OUT)
original = load_init_states('cube_flip')['states']
assert len(original) >= 50
used = set()
for name in ('validation', 'selection', 'test'):
    rows = json.loads((OUT.parent/f'{name}.json').read_text())['episodes']
    used.update(int(row['seed']) for row in rows)
assert all(SEED_BASE + i not in used for i in range(COUNT))
rows = []
for i in range(COUNT):
    seed = SEED_BASE + i
    rows.append({
        'index': i,
        'seed': seed,
        'placement_id': i % 5,
        'init_state': original[i % 50]['init_state'],
        'bias_by_level_deg': {'2': sample_bias(seed, 2).tolist()},
    })
OUT.write_text(json.dumps({
    'name': 'confirmation_2deg_200', 'seed_base': SEED_BASE,
    'offset_limit_deg': 2, 'episodes': rows,
}, indent=2))
print(f'{OUT}: {len(rows)} frozen episodes, seeds {SEED_BASE}-{SEED_BASE + COUNT - 1}')
