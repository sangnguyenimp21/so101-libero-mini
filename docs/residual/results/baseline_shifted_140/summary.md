# SmolVLA baseline on shifted cube-flip states

Policy: `checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model`  
Evaluation: EGL, CUDA, one worker, deterministic per-state seeds, 450-step horizon.  
Result: **36/140 successes (25.7%), Wilson 95% CI [19.2%, 33.5%]**.

The `in-training-region` label refers to the proposed residual-RL training rectangle (`dx` -1.5 to +3.0 cm, `dy` -2.0 to +2.0 cm). It does not mean these positions were in SmolVLA's original demonstration data, which covered only the five default/axis-1-cm placements.

## Per evaluation set

| Set | Successes | Wilson 95% CI | Never grasped | Dropped | Lifted, not flipped | Timeout | Other | Mean success steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Axis 2 cm | 19/40 (47.5%) | 32.9–62.5% | 20 | 0 | 1 | 0 | 0 | 263.9 |
| Diagonal 1.5 cm | 17/40 (42.5%) | 28.5–57.8% | 20 | 0 | 3 | 0 | 0 | 267.3 |
| Feasible axes 3 cm | 0/30 (0%) | 0–11.4% | 20 | 0 | 10 | 0 | 0 | — |
| Feasible axes 4 cm | 0/30 (0%) | 0–11.4% | 29 | 0 | 1 | 0 | 0 | — |

## By proposed residual training region

| Group | Successes | Wilson 95% CI | Never grasped | Dropped | Lifted, not flipped | Timeout | Other | Mean success steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| In-training-region | 27/80 (33.8%) | 24.4–44.6% | 50 | 0 | 3 | 0 | 0 | 267.2 |
| Out-of-training-region | 9/60 (15.0%) | 8.1–26.1% | 39 | 0 | 12 | 0 | 0 | 260.3 |

Failure types are mutually exclusive. The detailed simulator outcomes are mapped as follows:

- **Never grasped:** the cube was never carried at least 1 cm in the gripper (`never_lifted`).
- **Dropped:** the cube fell off the table or a detailed outcome explicitly reported a drop.
- **Lifted, not flipped:** the cube was lifted, but remained held, was not turned, or was turned without satisfying the final in-place/settled predicate.
- **Timeout:** the horizon was reached without a more specific simulator diagnosis.
- **Other:** runtime errors or remaining outcomes. There were no runtime errors.

All 104 failures reached the 450-step horizon, but each received a more specific diagnostic, so the timeout-only count is zero. Raw outcomes are retained in `results.json` and `episodes.csv`.

## Failure videos

- Axis 2 cm: [forward](videos/shift_2cm_axis__forward__state00_fail.mp4), [left](videos/shift_2cm_axis__left__state02_fail.mp4)
- Diagonal 1.5 cm: [forward-left](videos/shift_1p5cm_diagonal__forward_left__state00_fail.mp4), [backward-left](videos/shift_1p5cm_diagonal__backward_left__state02_fail.mp4)
- Feasible axes 3 cm: [forward](videos/shift_3cm_axis_feasible__forward__state00_fail.mp4), [left](videos/shift_3cm_axis_feasible__left__state01_fail.mp4)
- Feasible axes 4 cm: [forward](videos/shift_4cm_axis_feasible__forward__state00_fail.mp4), [left](videos/shift_4cm_axis_feasible__left__state01_fail.mp4)

Machine-readable outputs: `results.json` and `episodes.csv`.
