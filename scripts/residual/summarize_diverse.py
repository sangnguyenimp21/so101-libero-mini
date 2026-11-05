"""Produce matched baseline/fine-tuned summary from the fixed 190 evaluation states."""
import json
from pathlib import Path
from collections import Counter
from scripts.residual.evaluate_shifted_baseline import summarize

root = Path('docs/residual/results/diverse_demo')
base = Path('docs/residual/results')
pairs = []
for label, oldpath, newpath in (
    ('original_50', base/'baseline_original_50/results.json', root/'original_50/results.json'),
    ('shifted_140', base/'baseline_shifted_140/results.json', root/'shifted_140/results.json')):
    old = json.loads(oldpath.read_text())
    new = json.loads(newpath.read_text())
    assert old['complete'] and new['complete']
    def key(row):
        return (row.get('set_id', 'original'), int(row.get('state_index', row['index'])), int(row['seed']))
    prior = {key(row): row for row in old['episodes']}
    current = {key(row): row for row in new['episodes']}
    assert prior.keys() == current.keys() and len(prior) == (50 if label == 'original_50' else 140)
    for k, row in current.items():
        b = prior[k]
        pairs.append({'group': label, 'set_id': row.get('set_id', 'original'), 'key': k,
                      'baseline_success': bool(b['success']), 'finetuned_success': bool(row['success']),
                      'failure_type': row.get('failure_type')})

def aggregate(rows):
    return {'n': len(rows), 'baseline_successes': sum(r['baseline_success'] for r in rows),
            'finetuned_successes': sum(r['finetuned_success'] for r in rows),
            'fixed': sum(not r['baseline_success'] and r['finetuned_success'] for r in rows),
            'broken': sum(r['baseline_success'] and not r['finetuned_success'] for r in rows),
            'fine_tuned_failure_types': dict(Counter(r['failure_type'] for r in rows if not r['finetuned_success']))}
summary = {name: aggregate([r for r in pairs if r['set_id'] == name])
           for name in sorted({r['set_id'] for r in pairs})}
summary['all_190'] = aggregate(pairs)
summary['shifted_140'] = aggregate([r for r in pairs if r['group'] == 'shifted_140'])
(root/'paired_summary.json').write_text(json.dumps(summary, indent=2))
with (root/'paired_summary.md').open('w') as f:
    f.write('# Diverse-demo fine-tuning: matched evaluation\n\n')
    f.write('Original 100 demonstrations were regenerated because the original dataset was unavailable on Vanda. All 200 new demonstrations used reserved training seeds.\n\n')
    f.write('| Set | Baseline | Fine-tuned | Fixed | Broken |\n|---|---:|---:|---:|---:|\n')
    for name, s in summary.items():
        f.write(f"| {name} | {s['baseline_successes']}/{s['n']} | {s['finetuned_successes']}/{s['n']} | {s['fixed']} | {s['broken']} |\n")
print(json.dumps(summary, indent=2))
