# so101_three_cubes: flip, sort and pyramid with SmolVLA (simulation → demonstrations → training → evaluation → robot)

One SO-101 tabletop scene with three plain 3 cm cubes (red, yellow, green), three tasks, five global workspace layouts,
scripted experts, LeRobot datasets of exactly 100 demonstrations per task, direct (no SLURM) SmolVLA training scripts
(20k/40k/60k/80k checkpoints), and one episode controller used by simulated evaluation, simulated inference and the
real robot: **HOME → SmolVLA → task success detected → policy stopped at once, queued actions discarded → safe release
(if needed) → safe retreat → return HOME → finished.**

| task (`--task`) | task id | instruction (exact, every episode) |
|---|---|---|
| `flip` | `flip_red_cube` | Flip the red cube and place it back at its original position. |
| `sort` | `sort_ryg` | Arrange the cubes from left to right as red, yellow, green. |
| `pyramid` | `build_pyramid` | Place the green cube centered on top of the red and yellow cubes. |

## Quick start (from the project root, on this server)

```bash
bash scripts/setup_conda_env.sh            # once (done here): conda env so101lm = lerobot[smolvla] 0.4.2, mujoco 3.3.7, OSMesa, smolvla_base
bash scripts/debug_task.sh flip            # look at everything before training (images/videos under outputs/debug/flip)
bash scripts/collect_flip.sh               # 100 demos (+ validation); likewise collect_sort.sh, collect_pyramid.sh
bash scripts/check_dataset.sh flip
bash scripts/train_flip_smolvla.sh         # 80k steps, checkpoints every 20k (BATCH_SIZE=16|32|64 ... to override)
bash scripts/eval_flip.sh                  # every checkpoint, 25 episodes each -> results/flip_eval.csv, BEST = ...
bash scripts/run_sim.sh --task flip --checkpoint outputs/flip_red/checkpoints/060000/pretrained_model
bash scripts/run_real.sh --task flip --checkpoint outputs/flip_red/checkpoints/060000/pretrained_model
```

All scripts are plain bash and run directly; `scripts/local_env.sh` activates the runtime (conda env `so101lm`, caches
in `~/so101lm`, offline Hugging Face hub, rendering backend) and every script sources it.

## Runtime and versions

The project is written for **LeRobot 0.4.2** (dataset format v3.0). This server had no runtime for it, so
`scripts/setup_conda_env.sh` (the project's own recipe for machines without the cluster image) created the conda env
`so101lm`: Python 3.10, lerobot[smolvla] 0.4.2, torch 2.7.1+cu126, transformers 4.57.6, mujoco 3.3.7 (the version the
scene hashes in `init_states/` need), OSMesa, and the `lerobot/smolvla_base` + `SmolVLM2-500M-Video-Instruct` weights
in `~/so101lm/cache` (the unused ONNX exports of SmolVLM2 were removed: 4.8 GB). The other conda env with LeRobot
(`afp`, LeRobot 0.6.1, no MuJoCo) belongs to another project and was not touched.

Rendering: `SO101_RENDER=auto` (default) picks EGL on this machine (RTX A5000): ~5 ms per control step with both cameras,
vs ~110-150 ms with OSMesa (`SO101_RENDER=osmesa`). EGL worked in 8 worker processes and next to CUDA in one process
here (the README's EGL warning is about some cluster nodes). Datasets record their renderer; evaluate with the same one.

## Scene (robot frame: +x = forward, +y = the robot's LEFT; metres; base origin 3.9 cm behind the shoulder's pan axis)

- Robot, table, cameras, physics: the project's (measured real front camera, wrist camera, black arm, 30 Hz
  joint-position control, 16 physics substeps). The same compiled scene for all three tasks.
- Three 3 cm cubes, solid colours, nothing else on them: red (0.85, 0.12, 0.10), yellow (0.95, 0.78, 0.10), green
  (0.13, 0.62, 0.22) RGBA; 20 g each (assumed), friction 1.0. No markers, textures, arrows, tags or simulation-only cues
  (tested: one plain box geom per cube, no material). All three cubes are in every task.
- **HOME** (`ThreeCubeScene.home_deg`, LeRobot so101_follower units, `use_degrees=True`):
  `[0, -60, 35, 85, 0, 10]` = the project's HOME_DEG arm pose with the gripper at 10/100. The TCP is 13 cm above the
  table at x 18.5 cm, tilted 30° forward, ≥ 5 cm from every cube in every layout. Every episode starts there; the finisher
  returns there. Override: `--home-pose 0,-60,35,85,0,10` or `SO101_HOME_POSE` (then the arm parks elsewhere; the
  policies were trained on episodes that start at the default HOME).
- **Five layouts** (global offsets of everything a task places, and of its targets): `center` (0, 0), `left` (+2 cm
  left), `right` (2 cm right), `forward` (+0.8 cm), `back` (−0.8 cm). Forward/back are 8 mm because the flip's red cube has
  a 1.6 cm band: the 45° flip grasp reaches it only ≥ 18.5 cm from the pan axis and the front image ends ~26 cm in front of
  the base (≥ 6 px margin at 24 cm). `bash scripts/debug_task.sh <task> layouts` renders and checks all of them.

Start configuration, layout `center` (real table: forward = distance in front of the shoulder's pan axis, i.e. the
vertical axis of the base motor; left/right = from the robot's centre line, the robot's own left):

| task | red | yellow | green |
|---|---|---|---|
| flip | 19.3 cm forward, centre line, one face toward the robot | 13.6 forward, 10.5 left | 13.6 forward, 10.5 right |
| sort | start row 13.3 cm forward, slots 9 cm left / centre / 9 cm right, permuted (below) | same row | same row |
| pyramid | 17.6 forward, 2.5 right (red on the robot's left of yellow) | 17.6 forward, 5.5 right (1 mm gap: touching) | 15.1 forward, 9.0 left |

Other layouts: add the offset (e.g. `left`: every y +2 cm). Flip: the red cube faces the pan axis in every layout (layouts
`left`/`right` turn it by ±5.9°); sort and pyramid cubes are square to the table. Sort targets (`center`): a row 19.3 cm
forward, red 5 cm left, yellow on the centre line, green 5 cm right (5 cm centre to centre; the row moves with the layout).
Sort starting permutations (letters = left, centre, right slot): GRY, YGR, RGY, GYR, YRG (never already sorted); every
layout is combined with every permutation (25 placements), so no permutation belongs to one layout.

## Tasks, experts and the success rules (simulation)

Success (simulation ground truth, `so101_libero/tasks/base.py`, tolerances in `ThreeCubeTolerances`) must hold
**10 consecutive control steps** (0.33 s). Ground truth is used for resets, experts, evaluation and diagnostics only;
policies get the joint state, the two camera images and the instruction, nothing else.

- **Flip** (`FlippedInPlace` + `Undisturbed`). A plain cube cannot be turned upside down by a 5-DoF SO-101 with one
  grasp at table height: its approach can only lean in the vertical plane through the pan axis, and grasp and release
  both need ≥ ~40° downward tilt (with tilts β₁, β₂ the start-top face ends at best cos⁻¹(−cos(β₁+β₂)) ≈ 100° from up
  and settles on a side face). The flip is therefore the one-grasp motion that is possible and the same every time:
  grasp the red cube across two side faces with the approach tilted 45° outward, lift, **roll the wrist 180° about the
  approach axis** (always the same grasp side and roll direction: `FlipInPlace(psi_deg=-90, roll_dir=-1)`; the other
  side hangs the wrist-camera mount into the table), set the cube down where it started (from ~2 cm: after the roll the
  camera mount hangs below the TCP). The cube turns by a 180° rotation; its top face ends up facing the robot.
  Success: hidden body orientation rotated ≥ 150° from the start **and** the start-top face turned ≥ 60° away from up
  (a 180° spin about the vertical axis is not a flip), lying flat, ≤ 2 cm from its start, carried ≥ 2 cm high,
  released (no robot contact), at rest, TCP ≥ 4 cm away; yellow and green moved ≤ 2 cm, tilted ≤ 20°, untouched.
- **Sort** (`SortedRow`): red, yellow, green each within 2 cm of its row target (5 cm apart: the order is implied), flat
  on the table, released, at rest, TCP ≥ 4 cm from every cube. Expert: red, then yellow, then green; each picked with the
  jaws closing along the row (the moving jaw on the outer side, away from the other cubes), turned on the way, put on its
  target with the jaws closing front-back (never reaching between row neighbours), released, lifted to 10 cm.
- **Pyramid** (`OnTopOfPair`): green's centre within 1 cm of the red/yellow midpoint, its bottom within 4 mm of their
  tops, touching red and yellow and not the table (half supported by each), flat; red and yellow moved ≤ 1 cm, still on
  the table; everything released and at rest, TCP ≥ 4 cm from green. Expert: top-down pick of green, carried over the
  midpoint (turned square to the base on the way), lowered to 3-6 mm above it, released, lifted to 10 cm.

Demonstrations start at HOME and **end right after the manipulation**: release, a short retreat straight up (~10 cm), a 1 s
hold. Returning HOME is not in the data: the deterministic finisher does it at run time.

## Episode execution (the same for evaluation, simulated inference and the real robot)

`so101_libero/control/episode_controller.py`:
`INITIALIZE → MOVE_TO_HOME → PREPARE_SCENE → RUN_POLICY → SUCCESS | TIMEOUT | ABORTED → STOP_POLICY → SAFE_RELEASE →
SAFE_RETREAT → RETURN_HOME → FINISHED`.

- **Stop on success.** After every executed action the completion detector is updated; when it confirms (the condition
  held `confirm_steps` consecutive steps) the loop ends before any further policy call.
- **Remaining SmolVLA actions are cancelled.** LeRobot 0.4.2's SmolVLA has no done flag, termination token or stop
  action (checked in `lerobot/policies/smolvla/modeling_smolvla.py`): it predicts 50-action chunks into
  `policy._queues["action"]` and pops one per step. On any ending, `discard_pending()` counts that deque and calls
  `policy.reset()`, which empties it; the count is logged (`discarded_actions`). Nothing queued is executed afterwards.
- **Safe release** (`EpisodeFinisher.safe_release_if_needed`): nothing happens if the jaws are open and free
  (commanded and measured open, no object felt). Every success rule requires a released cube, so after a success that
  is always the case and a finished arrangement is never disturbed. Otherwise (timeout or abort, possibly mid-grasp)
  the arm first holds still 0.3 s so the jaws settle, the state is judged again, and unless the jaws are now open and
  free they open over 0.6 s while the arm does not move, then 0.4 s settle: an object still held is put down where it
  is (no recovery attempts). "Holding" = both jaws touching a cube (simulation; one jaw = uncertain = open) or a closed
  command the jaws cannot follow (real robot; anything but "commanded open and following" counts as uncertain).
- **Safe retreat** (`safe_retreat`): if the TCP is lower than 10 cm, it goes straight up (numpy IK on the extracted SO-101
  chain, orientation kept, `so101_libero/control/kinematics.py`, identical to MuJoCo's FK within 1e-6 m) by ≥ 5 cm at
  5 cm/s, so the open jaws slide off the cubes vertically.
- **Return HOME** (`return_home`, also `move_to_home` before an episode): one min-jerk joint move (≤ 40°/s average per
  joint), 0.5 s settle, success = every arm joint within 3° of HOME.
- **Timeouts and failures.** Policy timeout = the task's horizon (flip 600 steps = 20 s, sort 1050 = 35 s, pyramid
  450 = 15 s; `--max-steps`, real robot also `--max-seconds`). ABORTED on: camera failure / blank frame, invalid
  observation, non-finite action, an action > 15° beyond a joint limit, a policy exception, a communication error, a
  simulation failure, Ctrl-C. Every ending goes through stop → discard → release-if-needed → retreat → HOME. On the
  real robot nothing moves on its own after something went wrong: after Ctrl-C you are asked first (ENTER: park; `n`:
  leave it holding); after a bus error the finisher is skipped; a Ctrl-C or error DURING release / retreat / HOME stops
  there (`finisher_interrupted`, the arm holds); and any episode that did not end cleanly at HOME ends the session
  (no next episode). Every automatic move to HOME from a pose that is not HOME is confirmed first.
- **Logged separately:** `task_success` (manipulation; the primary metric), `returned_home`, `episode_complete` (back at
  HOME, holding nothing, the finisher not interrupted), printed as `TASK_SUCCESS`, `RETURN_HOME_SUCCESS`,
  `EPISODE_COMPLETE`. `success_basis` says what `task_success` rests on: simulator ground truth (`sim`); the learned
  behaviour finished, result not verified (`behavior`: printed as `TASK_SUCCESS [unverified]`); behaviour finished AND
  the camera check at HOME passed (`vision`: `BEHAVIOR_COMPLETE` when the policy stops, then `TASK_SUCCESS` or
  `TASK_FAILED` after the check).

### Completion detectors (`so101_libero/control/completion/`)

| mode | where | flip | sort | pyramid |
|---|---|---|---|---|
| `sim` | simulation (default) | ground truth above | ground truth | ground truth |
| `behavior` | real robot (default) | 1 grasp→release cycle, jaws open, TCP ≥ 7 cm, arm still 0.33 s, +5 steps | 3 cycles, same | 1 cycle, same |
| `vision` | real robot (success = behaviour AND this check) | behavior + red cube back within 2.5 cm of its start position (front camera, at HOME) | behavior + row R-Y-G, 3.5-6.5 cm spacing, aligned (front camera, at HOME, 5 frames) | behavior + red/yellow side by side (blobs ≤ 5.5 cm apart) and green blob where a stacked cube projects (at HOME, 5 frames) |

A grasp→release cycle = the gripper is commanded closed but blocked well above the command for ≥ 5 steps (an object in
the jaws; in simulation a 3 cm cube holds it at ~18/100 while the command is 0, an empty close follows to ~0), then
commanded open and following freely. A missed grasp or a dropped cube is not a cycle. On all 35 expert demonstrations
(every layout and permutation) the behaviour detector fired 9-16 steps (0.3-0.5 s) after the ground truth (flip 16, sort
13, pyramid 9) and never before it, inside the demonstrations' final 1 s hold.
Vision does not gate stopping: while the policy holds still after a task its open jaws hang over the result and hide it
from the front camera, so the check runs after RETURN_HOME, on 5 consecutive frames. Colour locator accuracy in
simulation: table cubes 0.3 mm mean / 1.3 mm max error (`python -m so101_libero.control.completion.vision --sim-check`).

**Limits of real-world success detection.**
- **Flip: not verifiable by camera.** The red cube is a plain cube, so before and after a flip it looks the same and
  no RGB check can tell whether it was turned; none is attempted. The real detector confirms that the learned flip
  behaviour ran to its end (one grasp, one release, lifted, still), and `vision` adds only a position check (back
  near its start). A flip that is dropped and lands on the wrong face, or a grasp released without turning, is
  reported as complete.
- **Sort and pyramid** with `behavior` confirm the end of the behaviour, not the result. `vision` checks the result but
  depends on the HSV colour bands (tuned to the simulated colours: check them on your camera with
  `python -m so101_libero.control.completion.vision --image <front frame>`), on the camera pose matching the calibrated
  front camera, and for the pyramid on an area heuristic (one camera cannot measure height).
- The behaviour detector depends on the gripper thresholds (`GripperCycleTracker`, policy units) and on the policy
  holding still after the task, as the demonstrations do. A policy that never stops moving runs to the timeout, and is
  then parked HOME the same way.

## Commands

| step | command | output |
|---|---|---|
| inspect a task (layouts, cameras, one expert demo + success trace, dataset replay, stop-on-success and return-home demo) | `bash scripts/debug_task.sh flip` (`sort`, `pyramid`; modes `layouts`, `expert`, `replay`, `controller`, `all`; `--layout K`, `--perm GRY`) | `outputs/debug/<task>/` |
| 100 demos per task | `bash scripts/collect_flip.sh`, `collect_sort.sh`, `collect_pyramid.sh` (`WORKERS=12`) | `~/so101lm/datasets/lerobot/local/smolvla_{flip_red,sort_ryg,pyramid}` |
| dataset sanity check | `bash scripts/check_dataset.sh flip` (`sort`, `pyramid`, `multitask`) | printed + `results/dataset_check_<dataset>.json` |
| train (80k, ckpt every 20k) | `bash scripts/train_flip_smolvla.sh`, `train_sort_smolvla.sh`, `train_pyramid_smolvla.sh` | `outputs/{flip_red,sort_ryg,pyramid}/checkpoints/0{2,4,6,8}0000/pretrained_model` |
| multitask (300 demos, task-balanced) | `bash scripts/train_multitask_smolvla.sh` (merges the three datasets first) | `outputs/multitask/checkpoints/...` |
| evaluate 20k/40k/60k/80k | `bash scripts/evaluate_all.sh` (or `eval_flip.sh`, `eval_sort.sh`, `eval_pyramid.sh`, `eval_multitask.sh`) | `results/{flip,sort,pyramid,multitask}_eval.{csv,json}`, `<run>/eval/<step>/<task_id>/` |
| simulated inference | `bash scripts/run_sim.sh --task flip --checkpoint <ckpt>` (`--episodes`, `--layout`, `--video`, `--detector behavior`) | `outputs/run_sim/<task>_<time>/` |
| real robot | `bash scripts/run_real.sh --task flip --checkpoint <ckpt> --port ... --front-camera ... --wrist-camera ...` | `outputs/run_real/<task>_<time>/` |

Training settings (top of `scripts/train_smolvla_task.sh`, all overridable from the environment): `BASE_MODEL`
[lerobot/smolvla_base], `DATASET`, `OUTPUT_DIR`, `STEPS` [80000], `SAVE_FREQ` [20000], `BATCH_SIZE` [32], `LR_DECAY_STEPS`
[= STEPS], `PRECISION` [bf16], `NUM_WORKERS` [8], `SEED` [1000], `DEVICE` [cuda], `WANDB` [false], `MIN_FREE_GB` [12],
`GPU_MEM_GB` [unset: a hard cap on this run's GPU memory for a GPU shared with someone else's job; if the run would need
more it fails with out-of-memory itself instead of growing into the other job's memory
(`so101_libero/train/capped_train.py`, which also prints the peak memory)].
Every run starts from `BASE_MODEL` (never from another task). Re-running continues from `checkpoints/last`; a finished
run is not retrained; an output dir without checkpoints is renamed aside, never deleted. On a resume, LeRobot restores the
run's own settings; an explicitly given `BATCH_SIZE`, `SAVE_FREQ`, `NUM_WORKERS` or `LOG_FREQ` is passed on (e.g.
`BATCH_SIZE=16` after running out of memory), while a different `DATASET`, `BASE_MODEL` or `SEED` is refused (use a new
`OUTPUT_DIR`). `DEVICE=cpu` hides the GPU from Accelerate (`CUDA_VISIBLE_DEVICES=""`, `ACCELERATE_USE_CPU=true`): plumbing
tests only. Launch settings (the effective values) and logs: `logs/train/<run>_<time>.{json,log}`; LeRobot's
`train_config.json` is in every checkpoint.

Evaluation: 25 episodes per checkpoint and task on the frozen states `init_states/<task>.json` (flip and pyramid: 5
layouts × 5 trials, the trials differ in SmolVLA's sampling noise, seeded per episode; sort: every layout × permutation
once), with the full controller. CSV columns: checkpoint, task, layout, seed, initial configuration, initial permutation,
success, task completion time, policy steps, returned_home, episode_complete, termination, discarded actions,
goal_holds_after_home, outcome, ... BEST = the highest success rate; ties go to the earlier checkpoint. Multitask: every
checkpoint on all three tasks, a table `Checkpoint / Flip / Sort / Pyramid / Mean`. Existing complete evaluations are
reused; `FORCE=1` re-runs and moves the old ones aside.

## SLURM: one H100 96 GB per task

Four submit scripts, one job per model, each end to end (runtime, dataset, 80k-step training, evaluation of every
checkpoint), all running at the same time on four GPUs:

```bash
sbatch slurm/train_flip_h100.sbatch
sbatch slurm/train_sort_h100.sbatch
sbatch slurm/train_pyramid_h100.sbatch
sbatch slurm/train_multitask_h100.sbatch      # waits for (or makes) the three task datasets, merges them
# account / partition / longer time if needed:  sbatch --account=<acct> --time=36:00:00 slurm/train_flip_h100.sbatch
# another batch size:                           BATCH_SIZE=64 sbatch slurm/train_flip_h100.sbatch
```

- Each job asks for `--gres=gpu:h100-96:1`, 32 CPUs, 128 GB RAM, 24 h (`#SBATCH` lines of the wrappers; same partition
  and GPU naming as the repo's other SLURM jobs). The shared driver is `slurm/three_cubes_job.sbatch`; the settings are in
  `slurm/three_cubes.env` (every one overridable from the submitting shell).
- Batch 128 by default (~25 GB by the memory measured on this project: ~1.3 GB + 0.18 GB per sample), 16 data workers,
  BF16. The training log prints the real peak GPU memory at steps 1, 10 and every 1000 (`peak GPU memory:`) and
  `updt_s` every 200 steps: 80,000 x `updt_s` is the training time. H100 speed was not measured; the project measured
  ~0.17 s/step at batch 32 on an H200 (80k ≈ 3.8 h), and a bigger batch makes every step slower while the step count
  stays 80k, so batch 128 is expected to take several times longer than batch 32. `BATCH_SIZE=32` is the fastest.
- Each checkpoint is evaluated (25 episodes, 8 workers, the full controller) as soon as it is saved, on the same GPU
  while training continues; at the end the missing ones, then the table with BEST. Rendering is EGL when
  `tests/egl_cuda_workers_selftest.py` passes on the node (EGL next to CUDA aborted on some cluster nodes), else OSMesa.
- Resubmitting a job continues it: training resumes from `checkpoints/last`, finished evaluations are reused, complete
  datasets are kept. The first job on a cluster creates the conda env (`scripts/setup_conda_env.sh`, ~15 min, needs
  internet on the node; the other jobs wait for it). Logs: `liberomini-so101-3c-<task>-<jobid>.out` in the project
  root and `logs/slurm/<task>_<jobid>/`.
- Get the project onto the cluster (the changes are not committed): e.g.
  `rsync -a --exclude outputs --exclude logs --exclude __pycache__ ~/projects/so101-libero-mini/ <cluster>:~/so101-libero-mini/`.
  The jobs generate missing datasets themselves (same seeds, same code: the same demonstrations); copying the ones made
  here saves that step: `rsync -a ~/so101lm/datasets/lerobot/local/smolvla_* <cluster>:~/so101lm/datasets/lerobot/local/`.
- The dry run of the driver (through `slurm/train_pyramid_h100.sbatch`, on this server's CPU with tiny settings, no
  SLURM) passed every stage, including a checkpoint evaluated while training continued. Nothing ran on an H100.

## Real robot (nothing here has run on hardware yet)

`scripts/run_real.sh` runs the same controller on the SO-101 follower through the project's real-robot I/O
(`scripts/real/run_policy.py`: LeRobot `so101_follower` with `use_degrees=True`, two OpenCV cameras, the centre square of
each frame resized to 256x256, per-step clamps, joint offsets). Before the first run, follow `docs/real_setup/DEPLOY.md`
(calibration zero and `--joint-offset`, front-camera alignment with `scripts/camera_align.py` /
`scripts/real/check_cameras.py`, then a `--no-send` dry run whose first predicted action must be close to the measured
pose). Specific to these tasks:
1. Environment on the robot machine: the so101lm recipe plus the motor driver, `pip install "lerobot[feetech]==0.4.2"`
   (missing on this server: no robot is attached here). Copy the run directory (e.g. `outputs/flip_red`) over.
2. Set the cubes up exactly as one simulated layout (the script prints the layout's cube positions from the pan axis
   and waits for ENTER; `--layout-for-setup 0-4`, sort `--permutation GRY` etc.). The flip's red cube must have one face
   pointing at the base motor's axis; the pyramid's red and yellow cubes touch, red on the robot's left.
3. Run it from a terminal: before every episode the arm is brought to HOME (you confirm the move if it is not there
   already), then you set up the scene and press ENTER; after the episode it parks there again. Without a terminal it
   refuses to start unless you pass `--unattended`. `--max-seconds` bounds the policy time; Ctrl-C stops the policy and
   you are asked before the arm moves; an episode that does not end cleanly at HOME ends the session. Commands go out at
   most every 1/30 s (also right after a slow chunk inference), clipped to the joint limits and 6°/step.
   `--no-send` is a true dry run: it observes and infers on the real cameras and joints and never moves the arm (no
   HOME move, no finisher). `--seed` seeds SmolVLA's sampling noise per episode (seed + episode).
4. The behaviour detector's gripper thresholds assume the simulated gripper mapping (a 3 cm cube holds the closed
   gripper at ~18/100, an empty close follows the command to ~0). Check your gripper's value with a cube in the jaws
   (`python scripts/real/run_policy.py --robot real --read-pose`) and set `--holding-margin` (default 8), `--closed-cmd`
   (10), `--open-cmd` (20), `--min-tcp-z` (0.07 m), `--still-steps` (10) on `run_real.sh` if your arm differs.

## GPU, time and disk on this server

- GPU: one RTX A5000 (24 GB), shared with another person's training job (~18.6 GB, 100% utilization, 2026-10-29).
  Measured SmolVLA training memory (peak reserved by PyTorch; the CUDA context adds ~0.5 GB) and speed while sharing the
  GPU with that job: batch 8 = 2.8 GB, 0.33 s/step; batch 16 = 4.25 GB, 0.60 s/step (memory flat after step 10). So the
  flip model was started with `GPU_MEM_GB=4.5 BATCH_SIZE=16 NUM_WORKERS=6 bash scripts/train_flip_smolvla.sh`
  (80k steps ≈ 13 h while sharing). Batch 32, the project's recipe, would need ~7 GB by the same trend (extrapolated, not
  measured): use it when the GPU is free (no gradient accumulation in lerobot-train 0.4.2). To check another batch size, run a short probe:
  `GPU_MEM_GB=<cap> BATCH_SIZE=<b> STEPS=40 OUTPUT_DIR=outputs/bs_probe bash scripts/train_flip_smolvla.sh
  --save_checkpoint=false` (prints the peak memory; delete `outputs/bs_probe` afterwards).
- Time: the project measured ~0.17 s/step at batch 32 on an H200 (80k ≈ 3.8 h). On this A5000, sharing it with the
  other job: 0.60 s/step at batch 16 (80k ≈ 13 h). Each `LOG_FREQ` (200) line of the training log reports `updt_s`.
- Disk: a checkpoint is 1.26 GB (865 MB weights + 394 MB optimizer state); 4 runs × 4 checkpoints ≈ 20 GB. `/home/users`
  had ~26 GB free after the datasets. The training script refuses to start below `MIN_FREE_GB` (12). The optimizer
  state (`training_state/`) is only needed to resume; deleting it from finished intermediate checkpoints frees 394 MB each.

## Files

New: `so101_libero/tasks/three_cubes.py` (scene, layouts, tasks), `so101_libero/three_cube_env.py`
(`SO101ThreeCubeEnv`), `so101_libero/expert/three_cubes.py`, `so101_libero/control/` (`episode_controller.py`,
`episode_finisher.py`, `home.py`, `kinematics.py`, `policies.py`, `robots.py`, `completion/{base,flip_completion,
sort_completion,pyramid_completion,vision}.py`), `so101_libero/sim/cameras.py`, `so101_libero/cli/{run_episodes,
eval_three_cubes,check_dataset,debug_three_cubes}.py`, `init_states/{flip_red_cube,sort_ryg,build_pyramid}.json`,
`scripts/{local_env,three_cubes_common,collect_task,collect_flip,collect_sort,collect_pyramid,check_dataset,
train_smolvla_task,train_flip_smolvla,train_sort_smolvla,train_pyramid_smolvla,train_multitask_smolvla,eval_task,
eval_flip,eval_sort,eval_pyramid,eval_multitask,evaluate_all,run_sim,run_real,debug_task}.sh`, `tests/test_three_cubes.py`.

Changed (backwards compatible; the existing tasks' scene hashes and datasets are unchanged): `tasks/base.py` (new
predicates and plans, `WorldState.touching/flat/rotation_from_start/start_top_turn`), `tasks/registry.py` (the suite,
seed slots 17, 18 (+19, 20 reserved for sort's 25 placements), 21), `expert/rollout.py` (dispatch, grasp monitor follows
the cube being handled), `env.py` (`render_observations` switch), `sim/scene.py` (camera spec moved to `sim/cameras.py`),
`data/writer.py` + `cli/generate.py` (per-frame `layout_id`, `permutation_id`), `tests/test_eval_utils.py` (25 states
for the three-cube tasks).
