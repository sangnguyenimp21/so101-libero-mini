# Calibration offsets up to 2 degrees: frozen 100-episode test

All methods use the same original cube-flip placements, initial states, seeds, and per-episode offsets. Each paired fixed/broken count uses the SmolVLA outcomes measured in that run. The selection-chosen checkpoints are the primary PPO results. The final checkpoints are shown separately.

| Method | Successes | Fixed | Broken | Exact McNemar p vs SmolVLA |
|---|---:|---:|---:|---:|
| SmolVLA alone (simulator-assisted run) | 61/100 | — | — | — |
| Simulator-assisted (offset-revealing input), selection-chosen | 84/100 | 27 | 4 | 3.39532e-05 |
| Simulator-assisted, final checkpoint | 87/100 | 26 | 0 | 2.98023e-08 |
| SmolVLA alone (no-leak run) | 62/100 | — | — | — |
| No-leak, selection-chosen | 63/100 | 3 | 2 | 1 |
| No-leak, final checkpoint | 81/100 | 22 | 3 | 0.000156522 |
| SmolVLA alone (non-learning run) | 63/100 | — | — | — |
| First-still non-learning calibration | 64/100 | 2 | 1 | 1 |

Repeated SmolVLA baseline runs on the same saved states and seeds did not give identical outcomes: the simulator-assisted run scored 61/100 and the non-learning run scored 63/100. Four episode outcomes changed between those two baseline runs. Treat small cross-run differences with caution; paired fixed/broken counts are computed within each run.

At zero offset on the original 50 states: SmolVLA and both simulator-assisted PPO checkpoints scored 50/50; the no-leak selection-chosen and final checkpoints scored 50/50 and 50/50, respectively.

The no-leak PPO observation contains measured joints, current SmolVLA command, previous offset-free commanded target, command-minus-measured error, chunk index, and previous residual. The first-still comparator uses commands and measured positions only.

Per-episode results, confidence intervals, failure types, action magnitudes, and videos are in the respective result JSON files.
