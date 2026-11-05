"""Choose smallest validation offset level with 30%-70% baseline success."""
import json
from pathlib import Path
p = Path('docs/residual/results/calibration/validation_sweep.json')
levels = json.loads(p.read_text())['levels']
valid = [int(L) for L in (0,2,4,6,8) if 0.3 <= levels[str(L)]['successes']/levels[str(L)]['n'] <= 0.7]
selection = {'chosen_L_deg': min(valid) if valid else None, 'rule': 'smallest L with success in [30%,70%]',
             'successes': {L: f"{levels[L]['successes']}/{levels[L]['n']}" for L in levels}}
out = p.with_name('selection.json')
out.write_text(json.dumps(selection, indent=2))
print(selection, flush=True)
if not valid:
    raise SystemExit('No level meets preregistered rule; stop before training')
