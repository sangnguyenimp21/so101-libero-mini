# Language-conditioned SmolVLA policies for an SO-101 arm, trained in simulation

Report on `smolvla_tri_cube_flip_100`, `smolvla_tri_green_push_100` and `smolvla_tri_cylinder_right_100` (2026-11-01).

## Summary

- **Training.** A pretrained vision-language-action model (SmolVLA, `lerobot/smolvla_base`, 450M parameters) was
  fine-tuned separately on three tabletop tasks for the SO-101 arm (5 joints + gripper):
  - flipping a cube onto its side;
  - pushing a block 5 cm forward;
  - placing a cylinder to the right of a cube.
  Each task used 100 scripted demonstrations generated in a MuJoCo contact simulation.
- **Simulator results.** Evaluated in the simulator on 50 start states per task, the final checkpoints succeed:
  - cube flip: 50/50 (Wilson 95% CI 0.93–1.00);
  - block push: 49/50 (0.90–1.00);
  - cylinder placement: 48/50 (0.87–0.99).
- **Interpretation caveat.**
  - For the flip and the push, the test start states are the same five layouts used in training. That result
    measures closed-loop imitation, not generalization.
  - The cylinder task is the only one evaluated on unseen start positions, and those lie inside the training range.
- **Real robot.** Results are qualitative only: the cube-flip policy works; the other two are being tested.

## 1. Problem setting

- **Robot.** The SO-101 is a low-cost 5-DoF arm with a parallel gripper and 6 servo motors.
  - Control is by absolute joint-position targets at 30 Hz.
  - Joint units match the LeRobot `so101_follower` driver: degrees for 5 joints, gripper 0–100.
  - The simulator therefore exposes exactly the real robot's interface.
- **Observations.**
  - Two RGB cameras at 256×256: a fixed front camera facing the robot, placed at a measured real camera pose, and a
    wrist camera on the gripper.
  - The 6 joint positions.
  - A natural-language instruction.
- **Simulation.** MuJoCo 3.3.7 rigid-body contact.
  - Physics timestep 2.08 ms, i.e. 16 substeps per control step.
  - Elliptic friction cones and multi-contact collision detection.
  - Gripper torque capped at 1.2 N·m to mirror the real follower.
  - Objects are held by friction only (no welding or attachment).
  - The robot model is MuJoCo Menagerie's `robotstudio_so101`.
- **Scene (one fixed scene for all tasks).** Coordinates: x points forward from the robot base, y to the robot's left.
  - a light gray cylinder, Ø3 × 6 cm, 30 g;
  - a green half-cylinder block, Ø3.5 × 3 cm, 15 g;
  - an all-red 3 cm cube, 20 g, rotated −20.4° so that one face points at the shoulder axis;
  - a white tray fixed on the robot's right. It is unused, but kept so that the scene stays identical across tasks.
  - Friction coefficient 1.0. Masses and friction are **assumed, not identified** from the real objects.

## 2. Tasks and success criteria

Success means all goal predicates hold for 10 consecutive control steps (0.33 s). They are evaluated on privileged
simulator state, not on images.

| task | instruction | goal predicate (summary) | horizon |
|---|---|---|---|
| cube_flip | "Pick up the red cube and flip it onto its side." | the original top face ends within 15° of vertical (upside down does not count); cube flat on the table, ≤ 2 cm from its start, lifted ≥ 2 cm on the way; released, still, gripper ≥ 4 cm away; other objects undisturbed (≤ 2 cm, ≤ 20°, untouched) | 450 steps (15 s) |
| green_push_forward | "Push the green semicircular block forward 5 cm." | block displaced 5 ± 1 cm forward, ≤ 1.5 cm sideways, still flat (≤ 20°); never lifted > 1 cm (pushed, not carried); others undisturbed | 450 steps |
| cylinder_right_of_cube | "Put the light gray cylinder to the right of the red cube." | cylinder upright in a zone 11.5–18 cm right of the centre line, ≥ 4 cm to the robot's right of the cube and ≤ 3 cm in front of or behind it; carried ≥ 2 cm high (not shoved); cube moved ≤ 1 cm; others undisturbed | 600 steps (20 s) |

**Start-state distributions.**

| task | how start states are drawn | distinct start states |
|---|---|---|
| cube_flip, green_push_forward | **5 fixed placements**: the default, or all three objects shifted together by 1 cm left, right, forward or back. Nothing else is random. Arm always in the same folded rest pose. | 5 in training, the same 5 at test |
| cylinder_right_of_cube | cube and block at the default spot; cylinder **uniform in a 4 × 6 cm rectangle** (x 0.19–0.23 m, y 0.11–0.17 m), stratified over six 2 × 2 cm cells | 100 in training, 50 new ones at test (inside the same rectangle) |

## 3. Demonstration data

- **Expert.** A scripted controller that does not learn:
  - damped-least-squares inverse kinematics with checked convergence;
  - minimum-jerk trajectories capped at a 75°/s average joint speed;
  - collision-checked poses;
  - physics checks at every phase boundary (grasped, lifted, still held, placed).
  Failed attempts are discarded.
- **Variation between demonstrations.** Only timing (×0.9–1.15), heights, release points, push gap, and for the
  cylinder its start position.
- **No recovery behaviour.** No demonstration contains a missed grasp or a correction.
- **Ending.** Every demonstration unfolds from rest, performs the task, returns to rest and holds for 2 s.
- **Strategies.**
  - Flip: a single grasp with a 45° outward-tilted approach, a lift to 9–10 cm, a 180° wrist roll (the cube turns
    90°, always towards the robot), and a release with a ~2 cm drop.
  - Push: gripper closed and pointing down behind the block, a straight 5 cm push at 4 cm/s, then back off.
  - Cylinder: a 45° side grasp with the wrist camera on top, a carry over the other two objects, and an upright
    placement.

| task | expert success (physics) | demo length (steps @30 Hz) | expert reaches goal at step | demos | frames |
|---|---|---|---|---|---|
| cube_flip | 50/50 | 352–386 | 238–261 | 100 (20 per placement) | ≈37k |
| green_push_forward | 50/50 | 271–293 | 141–162 | 100 (20 per placement) | ≈28k |
| cylinder_right_of_cube | 60/60 | 380–425 | 270–310 | 100 | ≈40k (estimated from lengths) |

- **Format.** LeRobot v3.0 datasets: 30 fps; front and wrist video plus joint state; actions are absolute joint
  targets; each episode is labelled with its instruction.
- **Determinism.** Seeds are deterministic per task, placement and episode. A replay of 10 recorded episodes in the
  simulator reproduces the joint trajectories and the goal for every dataset.

## 4. Model and training

- **Architecture.** SmolVLA:
  - the vision-language backbone SmolVLM2-500M-Video-Instruct (first 16 layers used);
  - a flow-matching "action expert" transformer that cross-attends to the backbone's features (width 0.75× the
    backbone's, self-attention every 2nd layer);
  - it predicts a **chunk of 50 future actions** (1.67 s) per inference, with 10 integration steps;
  - all 50 actions are executed before re-planning;
  - observation history: 1 frame.
- **Parameters.** 450.0M in total.
  - Trained: ≈99.9M (22%): the action expert (98.2M), the action/time MLPs (1.6M) and the state/action projections
    (0.08M).
  - Frozen: the vision encoder (86.4M), the connector (11.8M) and the language model (≈252M).
- **Inputs.** Images padded and resized to 512×512, with no pixel normalization. Joint state and actions are z-scored
  with dataset statistics. Instructions are tokenized to 48 tokens. The front and wrist cameras map to the pretrained
  model's camera slots 1 and 2.
- **Optimization.**
  - AdamW: lr 1e-4, β = (0.9, 0.95), weight decay 1e-10, gradient-norm clip 10.
  - 1,000 warm-up steps, then cosine decay to 2.5e-6 over the full run.
  - Batch 32, BF16 mixed precision, no image augmentation, seed 1000.

| run | steps | samples seen | ≈ epochs |
|---|---|---|---|
| cube_flip | 100k | 3.2M | ≈86 |
| green_push_forward | 80k | 2.56M | ≈91 |
| cylinder_right_of_cube | 80k | 2.56M | ≈64 |

- **Compute.** One NVIDIA H100 96 GB per run, ~3.5–5 h of training each.
- **Checkpoints.** Saved every 20k steps; all of them were evaluated.
- **Training curves: not available locally.** W&B was off; the loss logs exist only in the cluster job logs.
- **Software.** lerobot 0.4.2, torch 2.7.1, transformers 4.57.6, mujoco 3.3.7, Python 3.10.

## 5. Evaluation protocol (simulation)

- **Episodes.** 50 episodes per checkpoint and task. Each episode starts from a frozen start state that the expert
  had solved (0 rejected).
  - Flip and push: 10 per placement.
  - Cylinder: 8–9 per 2 × 2 cm cell, from seeds disjoint from training.
- **Rollouts.** One rollout per start state. The physics is deterministic; the only randomness is the flow-matching
  noise of the policy. An episode stops at success or at the horizon.
- **Rendering.** Evaluation used CPU (OSMesa) rendering. The training data was rendered with OSMesa (flip, cylinder)
  or EGL (push). The two renderers differ by a mean 0.6–1.0/255 per pixel.
- **Reported statistics.** Success rate with a Wilson 95% interval, a failure-mode label per episode, and the mean
  number of steps to success.

## 6. Results (simulation)

Success by checkpoint, as successes/50 with the Wilson 95% interval:

| task | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| cube_flip | 50 [0.93, 1.00] | 50 [0.93, 1.00] | 50 [0.93, 1.00] | 50 [0.93, 1.00] | 50 [0.93, 1.00] |
| green_push_forward | 44 [0.76, 0.94] | 47 [0.84, 0.98] | 49 [0.90, 1.00] | 49 [0.90, 1.00] | not trained |
| cylinder_right_of_cube | 36 [0.58, 0.83] | 47 [0.84, 0.98] | 48 [0.87, 0.99] | 48 [0.87, 0.99] | not trained |

Failure modes:

| task | 20k | 40k | 60k | 80k |
|---|---|---|---|---|
| green_push_forward | pushed too little 5, too far 1 | too little 2, too far 1 | too little 1 | too little 1 (timed out at 450 steps) |
| cylinder_right_of_cube | fell off table 6, placed outside zone 3, never lifted 3, in zone but not settled 2 | fell off 1, never lifted 2 | never lifted 1, fell off 1 | fell off table 2 |

- **Speed.** Mean steps to success at the final checkpoint, against the expert's goal step:
  - cube flip: 252 vs 238–261;
  - push: 149 vs 141–162;
  - cylinder: 298 vs 270–310.
  The policies run at expert speed.
- **Training length.** Success saturates by 40–60k steps for push and cylinder. The flip is at its ceiling from 20k.

## 7. Real-robot observations (qualitative; no success rates measured)

- **Deployment setup.**
  - MacBook (Apple MPS), lerobot 0.6.1.
  - Camera frames centre-cropped to a square and resized to 256×256.
  - Same instruction strings as in training.
  - Objects placed at the simulated positions.
- **Calibration offset.** The real arm's joint zero differs from the model's: elbow +8.9°, shoulder lift +2.1°. Left
  uncorrected, this puts grasps ~2.2–2.5 cm too high. A corrected calibration, fitted to automatic table-touch
  probes, is used for every deployment.
- **cube_flip.** Works on the real arm. **Not measured:** a trial count or success rate.
- **Earlier cylinder model** (not one of the three; it motivated `cylinder_right_of_cube`). It was trained on the 5
  fixed placements and reached ≈49–50/50 in simulation, but on the real arm it started only when the cylinder lay
  within ~3–4 cm of its training image position. In 7 real runs:
  - 4 never committed to a grasp;
  - 3 committed, and all 3 closed on air;
  - none recovered (the demonstrations contain no recoveries).
  The likely cause is a mismatch in front-camera framing: in the real image the robot appears ~25% smaller than in
  the simulator. The new cylinder task therefore trains on a continuous 4 × 6 cm start area.
- **green_push_forward and cylinder_right_of_cube.** Real-arm testing is in progress. **Not measured** yet.

## 8. Limitations and threats to validity

1. **The test set equals the training set for flip and push.** The 50 test episodes cover the same 5 start layouts
   as the 100 demonstrations. 50/50 and 49/50 show faithful closed-loop reproduction, including recovery from the
   policy's own small deviations. They do not show robustness to new positions. Only the cylinder task tests new
   start positions, and only by interpolation inside the training rectangle.
2. **Checkpoints were selected on the test set.** There is no held-out validation split, so picking the best
   checkpoint by its test score is optimistic. The differences between checkpoints are small, though.
3. **One training seed per task, n = 50 per estimate.** The intervals are ±5–12 percentage points, and run-to-run
   variance across seeds is **not measured**.
4. **No baselines or ablations on this suite.** There is no ACT or Diffusion Policy comparison, no multitask model
   result, no data-size ablation, and no test without the language input. A task-balanced multitask job exists but
   has no result reported here.
5. **Success is judged from privileged state.** The all-red cube looks the same flipped and unflipped. The policy has
   no memory and its last frames resemble its first, so on the real arm it may start a second flip. Runs are cut
   after the first flip.
6. **Narrow demonstration diversity.** Scripted, near-identical strategies; only one flip direction demonstrated; no
   recovery demonstrations; no visual domain randomization in these datasets.
7. **Sim-to-real gaps.** Assumed masses and friction; a ~2 cm release drop in the flip; the pushed block ends at the
   edge of the front image; front-camera framing mismatch; joint calibration offset (corrected).

## 9. Reproducibility

- **Code.** Repository `so101-libero-mini`, commit `b791e48`, which contains:
  - the task definitions: `so101_libero/tasks/tri_tasks.py`;
  - the expert: `so101_libero/expert/tri_tasks.py`;
  - the SLURM jobs `slurm/train_smolvla_{cube_flip,green_push,cylinder_right}.slurm`, which generate the data,
    validate it, train and evaluate end to end;
  - the frozen test states: `init_states/<task>.json`, each with a hash of the compiled scene.
- **Outputs.** The checkpoints and per-episode results are in `outputs/train/<run>/`:
  - `checkpoints/<step>/pretrained_model`;
  - `eval/<step>/results.json`, `episodes.csv` and videos.
- **Datasets.** Regenerable bit-exactly on the same platform from the seeds. Datasets made on macOS and on Linux are
  equivalent but not bit-identical.
