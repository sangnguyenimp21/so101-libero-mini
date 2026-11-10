# Kickstart PPO at 500,000 steps

The 500k checkpoint is evaluated with the saved kickstart action scales. The duplicate `final` column in `results.json` is the same 500k model, because the existing evaluator requires two model paths.

| Set | SmolVLA | Kickstart 500k | Wilson 95% | Stalled | Moved but missed | Lifted not flipped | Fixed | Broken | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_50 | 50/50 | 50/50 | [0.929, 1.000] | 0 | 0 | 0 | 0 | 0 | 1.000000 |
| shift_2cm_axis | 19/40 | 20/40 | [0.352, 0.648] | 20 | 0 | 0 | 1 | 0 | 1.000000 |
| shift_1p5cm_diagonal | 17/40 | 20/40 | [0.352, 0.648] | 20 | 0 | 0 | 3 | 0 | 0.250000 |
| shift_3cm_axis_feasible | 0/30 | 0/30 | [0.000, 0.114] | 20 | 0 | 10 | 0 | 0 | 1.000000 |
| shift_4cm_axis_feasible | 0/30 | 0/30 | [0.000, 0.114] | 10 | 20 | 0 | 0 | 0 | 1.000000 |
