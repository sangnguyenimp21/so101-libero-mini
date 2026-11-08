"""Summarize matched frozen-test calibration comparisons after both jobs finish."""
import json
from pathlib import Path

ROOT = Path('docs/residual/results/calibration')
assisted = json.loads((ROOT/'final_evaluation/results.json').read_text())['sets']
no_leak = json.loads((ROOT/'no_leak/final_evaluation/results.json').read_text())['sets']
still = json.loads((ROOT/'still_calibration_results.json').read_text())['summary']

rows = [
    ('SmolVLA alone', assisted['test_Lstar']['models']['best_model']['summary'], 'baseline'),
    ('Simulator-assisted (offset-revealing input), selection-chosen', assisted['test_Lstar']['models']['best_model']['summary'], 'residual'),
    ('Simulator-assisted, final checkpoint', assisted['test_Lstar']['models']['final_model']['summary'], 'residual'),
    ('No-leak, selection-chosen', no_leak['test_Lstar']['models']['best_model']['summary'], 'residual'),
    ('No-leak, final checkpoint', no_leak['test_Lstar']['models']['final_model']['summary'], 'residual'),
]
assert all(r['baseline_successes'] == 61 and r['n'] == 100 for _, r, _ in rows)
assert still['baseline_successes'] == 61 and still['episodes'] == 100
lines = [
    '# Calibration offsets up to 2 degrees: frozen 100-episode test',
    '',
    'All methods use the same original cube-flip placements, initial states, seeds, and per-episode offsets. The selection-chosen checkpoints are the primary PPO results. The final checkpoints are shown separately.',
    '',
    '| Method | Successes | Fixed | Broken | Exact McNemar p vs SmolVLA |',
    '|---|---:|---:|---:|---:|',
]
for label, result, kind in rows:
    if kind == 'baseline':
        lines.append(f"| {label} | {result['baseline_successes']}/100 | — | — | — |")
    else:
        lines.append(f"| {label} | {result['residual_successes']}/100 | {result['fixed']} | {result['broken']} | {result['mcnemar_exact_two_sided_p']:.6g} |")
lines.append(f"| First-still non-learning calibration | {still['successes']}/100 | {still['fixed']} | {still['broken']} | {still['mcnemar_exact_two_sided_p']:.6g} |")
lines += [
    '',
    'At zero offset on the original 50 states: SmolVLA and both simulator-assisted PPO checkpoints scored 50/50; '
    f"the no-leak selection-chosen and final checkpoints scored {no_leak['original_L0']['models']['best_model']['summary']['residual_successes']}/50 and {no_leak['original_L0']['models']['final_model']['summary']['residual_successes']}/50, respectively.",
    '',
    'The no-leak PPO observation contains measured joints, current SmolVLA command, previous offset-free commanded target, command-minus-measured error, chunk index, and previous residual. The first-still comparator uses commands and measured positions only.',
    '',
    'Per-episode results, confidence intervals, failure types, action magnitudes, and videos are in the respective result JSON files.',
]
out = ROOT/'no_leak/comparison.md'
out.write_text('\n'.join(lines) + '\n')
with Path('docs/residual/NOTES.md').open('a') as f:
    f.write('\n### Completed matched no-leak comparison\n\n')
    f.write('\n'.join(lines) + '\n')
print(out)
