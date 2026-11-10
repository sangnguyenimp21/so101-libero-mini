# Motion classification of SmolVLA baseline failures

All 104 failed episodes from `baseline_shifted_140` were replayed with joint-motion instrumentation. The five arm joints are measured in degrees; the gripper is excluded from the 5° threshold because it uses a 0–100 command scale.

- **Stalled:** maximum absolute change of any arm joint from its reset value was under 5°.
- **Moved but missed:** at least one arm joint moved by 5° or more, but the saved baseline episode never lifted the cube.
- **Lifted but not flipped:** the saved baseline episode lifted the cube but failed the success predicate.

## Overall

| Failed episodes | Stalled | Moved but missed | Lifted but not flipped |
|---:|---:|---:|---:|
| 104 | 70 | 19 | 15 |

The split has a large numerical margin: stalled replay trajectories moved by roughly 0.4–1.0°, while normal-motion trajectories moved by roughly 87–95°.

## Per set

| Set | Failed | Stalled | Moved but missed | Lifted but not flipped |
|---|---:|---:|---:|---:|
| Axis 2 cm | 21 | 20 | 0 | 1 |
| Diagonal 1.5 cm | 23 | 20 | 0 | 3 |
| Feasible axes 3 cm | 30 | 20 | 0 | 10 |
| Feasible axes 4 cm | 30 | 10 | 19 | 1 |

## Per direction

| Set | Direction | Failed | Stalled | Moved but missed | Lifted but not flipped |
|---|---|---:|---:|---:|---:|
| Axis 2 cm | forward | 10 | 10 | 0 | 0 |
| Axis 2 cm | backward | 1 | 0 | 0 | 1 |
| Axis 2 cm | left | 10 | 10 | 0 | 0 |
| Axis 2 cm | right | 0 | 0 | 0 | 0 |
| Diagonal 1.5 cm | forward-left | 10 | 10 | 0 | 0 |
| Diagonal 1.5 cm | forward-right | 0 | 0 | 0 | 0 |
| Diagonal 1.5 cm | backward-left | 10 | 10 | 0 | 0 |
| Diagonal 1.5 cm | backward-right | 3 | 0 | 0 | 3 |
| Feasible axes 3 cm | forward | 10 | 10 | 0 | 0 |
| Feasible axes 3 cm | left | 10 | 10 | 0 | 0 |
| Feasible axes 3 cm | right | 10 | 0 | 0 | 10 |
| Feasible axes 4 cm | forward | 10 | 0 | 10 | 0 |
| Feasible axes 4 cm | left | 10 | 10 | 0 | 0 |
| Feasible axes 4 cm | right | 10 | 0 | 9 | 1 |

## Predicted-action traces for two stalled episodes

Each CSV logs the current six-value joint state and SmolVLA's six absolute target actions at every step for the first three 50-action chunks (150 rows).

### Diagonal 1.5 cm, forward-left, state 28

Episode maximum arm-joint change: **0.4206°**.  
Full trace: [CSV](shift_1p5cm_diagonal__forward_left__state28__first_3_chunks.csv)

| Chunk | Mean absolute arm action−state | Maximum absolute arm action−state | Mean absolute gripper action−state |
|---:|---:|---:|---:|
| 0 | 0.0492° | 0.3100° | 0.0240 units |
| 1 | 0.0482° | 0.3203° | 0.0198 units |
| 2 | 0.0440° | 0.3316° | 0.0169 units |

### Axis 2 cm, left, state 22

Episode maximum arm-joint change: **0.5381°**.  
Full trace: [CSV](shift_2cm_axis__left__state22__first_3_chunks.csv)

| Chunk | Mean absolute arm action−state | Maximum absolute arm action−state | Mean absolute gripper action−state |
|---:|---:|---:|---:|
| 0 | 0.0561° | 0.3764° | 0.0176 units |
| 1 | 0.0508° | 0.2323° | 0.0184 units |
| 2 | 0.0590° | 0.4791° | 0.0242 units |

At step 0 the reset arm state is approximately `[0, -94, 82, 68, 0]°`. SmolVLA's targets remain within fractions of a degree of those values throughout all three chunks. This confirms that the stalled episodes are caused by the model predicting near-hold actions, not by a large command being physically blocked.

## Replay reproducibility note

98/104 replay outcomes matched the saved baseline outcome exactly. Six normal-motion boundary cases produced a different detailed outcome on replay; five were already baseline `lifted but not flipped` episodes, and one 4 cm-right baseline `never_lifted` episode moved significantly on both the directional pattern and diagnostic replay. These differences do not affect the three-way classification. Both action-trace episodes exactly reproduced their baseline `never_lifted` outcome.

Machine-readable outputs: `summary.json` and `classification.csv`.
