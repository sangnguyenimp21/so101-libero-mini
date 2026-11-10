"""Per-set paired report for the kickstart 500k matched evaluation."""
import json
from pathlib import Path
from scripts.residual.evaluate_residual import paired_comparison

out = Path('docs/residual/results/kickstart_500k_eval')
report = json.loads((out/'results.json').read_text())
if not report['complete'] or len(report['episodes']) != 190:
    raise RuntimeError('expected complete matched 50 original + 140 shifted evaluation')
sets = {}
lines = ['# Kickstart PPO at 500,000 steps', '',
    'The 500k checkpoint is evaluated with the saved kickstart action scales. '
    'The duplicate `final` column in `results.json` is the same 500k model, '
    'because the existing evaluator requires two model paths.', '',
    '| Set | SmolVLA | Kickstart 500k | Wilson 95% | Stalled | Moved but missed | Lifted not flipped | Fixed | Broken | Exact McNemar p |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for set_id, summaries in report['per_set'].items():
    rows = [r for r in report['episodes'] if r['set_id'] == set_id]
    paired = paired_comparison(rows, 'residual')
    model = summaries['residual']; base = summaries['baseline']
    motion = model['motion_failures']
    sets[set_id] = {'baseline': base, 'kickstart_500k': model, 'paired': paired}
    ci = model['wilson95']
    lines.append(f"| {set_id} | {base['successes']}/{base['episodes']} | "
        f"{model['successes']}/{model['episodes']} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
        f"{motion['stalled']} | {motion['moved_but_missed']} | "
        f"{motion['lifted_but_not_flipped']} | {paired['fixed']} | {paired['broken']} | "
        f"{paired['mcnemar_exact_two_sided_p']:.6f} |")
for name, condition in [('original_50', lambda r: r['set_id']=='original_50'),
                        ('all_shifted_140', lambda r: r['set_id']!='original_50'),
                        ('all_190', lambda r: True)]:
    rows = [r for r in report['episodes'] if condition(r)]
    sets[name+'_paired'] = paired_comparison(rows,'residual')
(out/'paired_per_set.json').write_text(json.dumps(sets,indent=2)+'\n')
(out/'paired_per_set.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
