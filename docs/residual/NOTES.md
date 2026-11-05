# Residual RL project notes

<!-- RESIDUAL_CURRENT_SUMMARY_START -->
## Current summary

- **Calibration-offset experiment in progress.** Added a wrapper that applies an episode-fixed arm-joint target bias and returns true measured joint positions. Original five cube-flip placements only; gripper bias is zero.
- Frozen three disjoint new seed ranges: validation 32M (50 states), checkpoint selection 33M (20 states), test 34M (100 states). Each state stores full initial pose and precomputed offsets for L = 0, 2, 4, 6, 8 degrees. Training will use 22M seeds.
- L* is not selected yet. The validation sweep must finish, and the selected level and sweep must be committed before PPO training. The named kickstart, audit, and diverse-demo jobs are untouched.
<!-- RESIDUAL_CURRENT_SUMMARY_END -->

## Step 1 — Understand the code (2026-11-05)

Branch: `residual-rl`.

### Environment and compute check

- Ran `nvidia-smi`: NVIDIA A40, 46,068 MiB total memory, 0 MiB in use. The GPU is visible and idle. This confirms the device only; CUDA availability inside the intended Python environment remains Step 2.
- Read `README.md`, `docs/TRI_TASKS.md`, `so101_libero/env.py`, `so101_libero/policy/runner.py`, `so101_libero/cli/evaluate.py`, and `so101_libero/tasks/tri_tasks.py`.

### Verified interface and episode behavior

- The environment observation is a dict containing `observation.state` (6 float32 motor values: five arm joint positions in degrees plus gripper 0–100), and, with rendering enabled, `observation.images.front` and `observation.images.wrist` (uint8 RGB images, 256×256 by default).
- Actions are six absolute joint-position targets in the same LeRobot units. `env.step()` converts/clips them to simulator controls. `cube_flip` has a 450 control-step horizon; the task uses the shared tri-scene and five placements.
- `PolicyRunner.act()` calls LeRobot's `policy.select_action()` on each invocation. The runner itself has no explicit Python action queue. SmolVLA's chunk selection/caching is inside LeRobot's policy implementation; the checkpoint's configured `n_action_steps` controls how many selected actions it returns before replanning. Therefore “runner is called only every 50 steps” is imprecise: callers may invoke `act()` each step while the policy internally returns queued chunk actions. Verify the checkpoint's `n_action_steps` in Step 2 when loading it.
- `_obs()` renders every configured camera whenever called. `reset()` renders once and `step()` renders after every control step. `EnvConfig(render=False)` disables rendering but also removes camera keys, so it cannot be used directly with SmolVLA. There is no built-in option to render only on chunk boundaries; a wrapper would need to cache the last images and refresh them when the policy needs a new chunk.
- On a successful step, `env.step()` returns reward `1.0` (otherwise `0.0`), and `info["is_success"]` reflects the held success predicate. `TurnedOnSide` checks tracked body orientation, that the cube rests flat on its side within 2 cm of its initial position, is released and settled, and was carried at least 2 cm above the table. `WorldState.max_lift["red_cube"]` records the highest cube-bottom height above the table while pinched; `world.lifted("red_cube", 0.02)` reads the lift condition. Because the cube starts on the table, this is effectively 2 cm above its start height. There is no built-in one-time lift reward.
- The evaluator is `so101_libero/cli/evaluate.py`, not `cli/evaluate.py` at repository root. It loads the first N committed frozen states, reports Wilson intervals and outcome breakdowns, and can record videos. Its task-specific diagnostic names differ from the proposed report shorthand (for example `never_lifted`, `not_turned`, and `turned_but_not_flat`).

### Safe placement customization and scene hash

- Existing cube_flip placements are `FixedPlacementLayout` entries in `so101_libero/tasks/tri_tasks.py`. They translate the cylinder, green block, and cube together by 1 cm, preserving each object's yaw. Current center coordinates are about (0.20, 0.13), (0.225, 0.03), and (0.232, -0.072) metres for cylinder, green block, and cube.
- `env.reset(options={"init_state": state})` accepts an explicit complete initial state (`robot_qpos` plus item `(x, y, yaw)` poses) and applies it without changing the task layout or compiled scene. A wrapper can copy a known placement's state and shift the desired group coordinates, retaining yaw and avoiding edits to `init_states/` or the task/scene definitions. The environment does not collision-check arbitrary supplied states; validate settling, object spacing/table boundaries, and scripted-expert success before using any such state. The success/failure logic uses the settled positions captured at reset.
- `init_states/cube_flip.json` includes a scene hash. The source describes the five poses as layout placements; an explicit `init_state` changes reset poses, not scene geometry. Avoid changing scene/task definitions; keep custom placements as runtime reset data and leave the committed frozen evaluation states untouched. Confirm exact hash behavior in the planned placement validation before relying on it.

### Corrections / cautions to the supplied project facts

- `cli/evaluate.py` is at `so101_libero/cli/evaluate.py`.
- The runner's `act()` method is called per observation; chunk caching is delegated to LeRobot `select_action()`. The number of actions per inference must be confirmed from the actual checkpoint/config rather than assumed solely from the runner.
- Camera rendering is performed on every observation, including every environment step. Rendering cannot currently be skipped while preserving image observations via a config switch.
- The environment's success reward is sparse (`1.0` on success), with no lift bonus; the residual wrapper must implement the proposed one-time lift reward itself.
- The initial cube placement is at z determined by its half-height and settles onto the table. `max_lift` tracks cube-bottom height above the table while pinched, which matches the proposed 2 cm lift threshold for this task.

### Commands/results

- `git branch --show-current` → `residual-rl`.
- `nvidia-smi` → A40 detected, idle.
- Read-only source review completed; created this log. No existing project code or evaluation state was changed.

### Next

Wait for Valentin's review before starting Step 2, per the step-by-step project instructions.

## Step 2 — Setup check (2026-11-05)

### Environment

- Conda environment: `/scratch/e1638612/miniforge3/envs/so101lm`, Python 3.10.22.
- Packages: LeRobot 0.4.2, MuJoCo 3.3.7, Stable-Baselines3 2.8.0, Transformers 4.57.6, PyTorch 2.7.1+cu126.
- CUDA works: `torch.cuda.is_available()` is `True`; one NVIDIA A40 is visible (compute capability 8.6).

### Rendering

- Tested both `MUJOCO_GL=osmesa` and `MUJOCO_GL=egl` by constructing `SO101TabletopEnv('cube_flip')` and resetting placement 0.
- Both backends successfully returned the 6-value state and front/wrist RGB images with shape `(256, 256, 3)`.

### Checkpoint

- Listed the Hugging Face repository `tsangb34/smolvla_tri_cube_flip_100` through `HfApi.list_repo_files()` without downloading file contents. The final checkpoint folder is `checkpoints/100000/pretrained_model/`.
- An initial unrestricted download was stopped because it was fetching every 20k snapshot and training state. Resumed only the final inference checkpoint with:
  `hf download tsangb34/smolvla_tri_cube_flip_100 --include 'checkpoints/100000/pretrained_model/*' --local-dir checkpoints/smolvla_cube_flip`
- Local inference checkpoint: `checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model` (about 907 MB model; total retained download directory about 937 MB).
- Verified the safetensors header opens and contains 500 tensors. LeRobot parses the config as `smolvla`, with `chunk_size=50` and `n_action_steps=50`. This confirms the policy internally serves 50 queued actions per inference chunk when `PolicyRunner.act()` is called each control step.
- Removed the downloaded/partial local `020000`, `040000`, `060000`, and `080000` snapshot directories and their matching download-cache entries, freeing about 5.4 GB. Kept the complete 100k files.
- Added `checkpoints/` to `.gitignore`; `git check-ignore` confirms the model is ignored.

### Problems / cautions

- Parsing `PreTrainedConfig` requires importing `lerobot.policies` first so the SmolVLA config class is registered. `PolicyRunner` and the evaluator already do this.
- The interrupted broad download left a small partial `checkpoints/last` tree and a few evaluation/config files. They are not used; the selected path is explicitly the 100k `pretrained_model` folder.

### Next

Wait for Valentin's review before Step 3 (five-episode baseline smoke evaluation, followed by 50 episodes if the smoke test is correct).

## Step 3 — Baseline reproduction (in progress, 2026-11-05)

### Model cache stall and diagnosis

- The first five-episode OSMesa attempt appeared stalled for about 14 minutes. The worker was in Linux `D` state (`folio_wait_bit_common`), used about 1% CPU and 0% GPU, and had not finished policy initialization.
- The traceback after stopping it showed the block was inside Transformers/PyTorch while reading and casting the upstream SmolVLM2 backbone. No environment step had run, so OSMesa was not the cause.
- `HF_HOME` was unset and resolved to `/home/svu/e1638612/.cache/huggingface`. The complete 2,029,990,624-byte `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` backbone was stored there.
- Moved the Hugging Face cache to `/scratch/e1638612/hf_cache` and persisted `HF_HOME=/scratch/e1638612/hf_cache` with `conda env config vars set -n so101lm`. Reactivating the environment now resolves both `HF_HOME` and `HF_HUB_CACHE` under scratch; the old home cache path is gone.
- The stalled worker was launched explicitly with `MUJOCO_GL=osmesa`, confirmed by its evaluation startup log (`renderer osmesa`).

### One-episode EGL diagnostic

- Added `scripts/residual/debug_episode.py`, an in-process diagnostic that prints policy-loading milestones, reset completion, step 1, every 50 steps, and termination. This avoids hiding initialization inside the evaluator's subprocess.
- Command: `MUJOCO_GL=egl python -u scripts/residual/debug_episode.py` (state index 0, frozen seed 11,400,000).
- Policy loading from scratch completed in 217.2 s. During loading it briefly entered `D` state while reading the 907 MB task checkpoint, but it progressed and completed.
- EGL environment creation and reset completed normally.
- First SmolVLA inference took 67.6 s (one-time warm-up). `env.step()` took about 2–5 ms in the printed samples; queued policy actions took about 4–6 ms.
- Episode result: success at step 247, maximum tracked lift 8.1 cm, rollout elapsed 70.7 s after reset. This reproduces success on the first original frozen state.
- Conclusion: the apparent hang was slow network-filesystem model loading plus first-inference warm-up. The simulation and EGL renderer are healthy.

### Next

Rerun the five-state smoke evaluation using the scratch Hugging Face cache. If all five pass, run all 50 original frozen states with exactly the same evaluation settings.

### Five-episode original-state smoke result

- Command used EGL, CUDA, one worker, the final 100k checkpoint, the first five frozen states, and one video:
  `MUJOCO_GL=egl python -m so101_libero.cli.evaluate --policy checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model --tasks cube_flip --episodes 5 --videos 1 --workers 1 --device cuda --out docs/residual/results/baseline_original_smoke`
- Result: **5/5 successes**, no warnings or episode errors. Success steps were 247, 265, 252, 248, and 252 (mean 252.8). Maximum tracked lifts were 7.71–8.06 cm.
- Output: `docs/residual/results/baseline_original_smoke/results.json` and `episodes.csv`.
- Video: `docs/residual/results/baseline_original_smoke/videos/cube_flip_ep000_success.mp4` (8.27 s, 454 KiB).
- The warm cache made the complete five-episode evaluator run finish in about 84 seconds, much faster than the first diagnostic load.

### Next

Run the full 50 original frozen states with the same EGL/CUDA/one-worker settings and no videos. Do not proceed to Step 4 until this baseline matches the expected result or any discrepancy is understood.

### Full 50-episode original baseline result

- Command:
  `MUJOCO_GL=egl python -m so101_libero.cli.evaluate --policy checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model --tasks cube_flip --episodes 50 --videos 0 --workers 1 --device cuda --out docs/residual/results/baseline_original_50`
- Result: **50/50 successes (100%)**, Wilson 95% CI **[92.86%, 100%]**, no warnings or episode errors. This exactly reproduces the reported baseline.
- Every placement succeeded 10/10. Mean success step was 250.2. The complete evaluator wall time was 147.3 s with the warm scratch cache.
- Saved outputs: `docs/residual/results/baseline_original_50/results.json` and `episodes.csv`. Report metadata records `render_backend: egl`, image size 256×256, and `complete: true`.

### Step 3 status

Complete. The setup and final 100k checkpoint are correct. Use this exact frozen-state set and EGL evaluation configuration when comparing the later residual policy on original placements.

### Next

Wait for Valentin's review before Step 4 (timing script and OSMesa/EGL timing comparison).

## Step 4 — Timing (2026-11-05)

### Implementation

- Added `scripts/residual/timing.py`. It benchmarks physics-only and two-camera `env.step()`, SmolVLA's first chunk and later chunks, one complete frozen episode, and process/GPU memory. It writes JSON results and estimates 2,000 episodes for one and eight environments.
- MuJoCo backends are measured in separate processes because `MUJOCO_GL` is selected at import time.

### Measurements

| measurement | EGL | OSMesa |
|---|---:|---:|
| physics-only `env.step()` mean (100 steps) | 1.004 ms | 1.064 ms |
| two-camera `env.step()` mean (100 steps) | 2.524 ms | 200.889 ms |
| first policy chunk after load | 0.646 s | 0.465 s |
| warmed 50-action chunk mean (3 chunks) | 139.7 ms | 205.9 ms |
| complete frozen episode | 2.774 s / 247 steps | 57.458 s / 250 steps |
| policy actions within that episode | 2.004 s | 2.933 s |
| policy load with warm filesystem cache | 25.5 s | 21.5 s |

- EGL rendered steps are about **80x faster** than OSMesa here. Physics-only speed is essentially identical.
- Both full episodes succeeded. The success step differed slightly (247 EGL, 250 OSMesa), consistent with the documented tiny image differences between renderers. Baseline and future residual evaluation must continue using EGL for a fair comparison.
- The earlier cold-cache diagnostic took 217 s to load and 67.6 s for the first inference; after the files and CUDA kernels were warm, loads took 21–26 s and the first chunk took <0.7 s. Startup cost is one-time per process.
- One policy process increased GPU use by about 1.588 GiB and resident CPU memory by about 1.0 GiB; maximum total RSS was about 3.1–3.2 GiB. Eight independent copies are estimated at 12.7 GiB GPU memory and roughly 25 GiB total process RSS, within the A40's 44.4 GiB usable memory and the node's 250 GiB RAM.

### Estimate for 2,000 training episodes

The range uses the measured successful episode length as the lower end and scales to the 450-step horizon as the conservative upper end.

| backend | 1 environment | 8 environments, ideal scaling |
|---|---:|---:|
| EGL | 1.54–2.81 h | 0.19–0.35 h |
| OSMesa | 31.92–57.46 h | 3.99–7.18 h |

- The eight-environment values are ideal lower bounds. Independent policies share one GPU, so inference contention will reduce scaling; PPO optimization also adds time. Measure actual throughput in the Step 8 smoke run before choosing the final worker count.
- The planned residual wrapper will reuse camera images between 50-step chunk boundaries. That should reduce rendering work further, especially for OSMesa, but EGL remains the clear choice on this node.
- EGL is below 12 hours even for the conservative single-environment rollout estimate, so Step 11's backup plan is not needed based on speed.

### Result files

- `docs/residual/results/timing_env_egl.json`
- `docs/residual/results/timing_env_osmesa.json`
- `docs/residual/results/timing_policy_egl.json`
- `docs/residual/results/timing_policy_osmesa.json`

### Step 4 status

Complete.

### Next

Wait for Valentin's review. Step 5 must first propose the continuous training offsets and frozen ±2/±3 cm evaluation states; do not implement or run the expert validation until approved.

## Configuration before Step 5 (2026-11-05)

- Persisted `MUJOCO_GL=egl` in the `so101lm` conda environment with `conda env config vars set -n so101lm MUJOCO_GL=egl`. Reactivating the environment now sets both `MUJOCO_GL=egl` and `HF_HOME=/scratch/e1638612/hf_cache`.
- Added `os.environ.setdefault("MUJOCO_GL", "egl")` before MuJoCo-related imports in every current residual Python script (`debug_episode.py`, `timing.py`). Explicit commands can still override it with `MUJOCO_GL=osmesa` for backend comparisons.
- No residual shell or PBS job scripts exist yet. Future residual job scripts must explicitly export `MUJOCO_GL=egl` as well as relying on the conda environment default.

## Step 5 — New placements (stopped on expert failure, 2026-11-05)

### Approved design and implementation

- Added `so101_libero/residual/placements.py`. It copies the original default cube-flip initial state and translates all three movable objects together, preserving the robot pose, object spacing, and every yaw. The existing task and `init_states/cube_flip.json` are unchanged.
- Continuous training offsets use independent uniform `dx, dy` samples in `[-0.03, 0.03]` m, deterministically generated from training-only seeds `20,000,000` through `29,999,999`.
- Defined four proposed frozen evaluation sets, each with four interleaved offsets and ten seeds per offset (40 states): axis-aligned 2 cm, axis-aligned 3 cm, axis-aligned 4 cm, and diagonals `(±2, ±2)` cm. Their disjoint seed bases are 31,000,000, 31,100,000, 31,200,000, and 31,300,000.
- Added `scripts/residual/make_placements.py`. It requires a visible NVIDIA GPU and `MUJOCO_GL=egl`, checks representative training placements and all unique evaluation layouts, then runs the scripted expert sequentially. It deliberately stops at the first failure and has no seed rejection or replacement path. Frozen JSON files and render sheets are written only if every expert check passes.
- Geometry checks cover settling, yaw, table-only contacts, table bounds, object/tray separation, and actual front-camera visibility. A conservative three-pixel full-bounding-box margin is recorded but is not a safety failure when an object still has sufficient rendered pixels. This distinction was needed at the `(forward 3 cm, right 3 cm)` training corner: all physical checks passed and the objects had 354–413 visible pixels, but their bounding boxes crossed that margin.

### GPU-node validation and stopping condition

- Node check: `GN-A40-035`, NVIDIA A40. All placement validation and rendering commands used `MUJOCO_GL=egl` on this node.
- A preliminary timing/smoke check of the first 4 cm forward state (seed 31,200,000) succeeded in 375 steps and took 2.36 s. This was used only to estimate the full sequential command at 7–8 minutes.
- Approved full command:
  `source /scratch/e1638612/miniforge3/bin/activate && conda activate so101lm && MUJOCO_GL=egl python -u scripts/residual/make_placements.py`
- The initial launch exposed a script import-path bug before validation began; the script was fixed to add the repository root to `sys.path`.
- Training geometry probes passed after treating the image margin as informational rather than a collision/safety condition.
- **Axis-aligned 2 cm: 40/40 scripted-expert checks passed.**
- Axis-aligned 3 cm: the forward state at seed 31,100,000 passed, then the first backward state at seed **31,100,001 failed** after 18 steps. Exact failure:
  `plan:no feasible grasp (unreachable pos=[0.204, -0.067, 0.016])`.
- Validation stopped immediately as requested. The remaining 38 axis-aligned 3 cm states, all 4 cm states in the full run, and all diagonal states were not run. Renders were not generated, and none of the four frozen evaluation JSON files were finalized.
- Partial machine-readable results are in `docs/residual/placements/validation_report.json`: `all_ok=false`, 40/40 passes for `shift_2cm_axis`, and 1/2 passes for the attempted `shift_3cm_axis` states.

### Status / decision needed

Step 5 is incomplete and paused for Valentin's decision. Do not replace the failing seed, change the expert, shrink the requested shift, or continue validation without explicit approval.

## Step 5 follow-up — Reach diagnosis and feasibility map (2026-11-05)

### Why the 3 cm-backward expert failed

- Diagnosed the unchanged state at offset `dx=-0.03 m, dy=0`, seed 31,100,001. The settled cube centre is `(0.202, -0.072)` m, only **17.83 cm from the shoulder pan axis**.
- Both permitted cube-grasp orientations have a reachable pre-grasp at 8 cm TCP height, but the low grasp at 1.65 cm TCP height is outside the expert's conservative endpoint limits.
- With the expert's 4-degree endpoint margin, the elbow saturates at **86.0 degrees** and the best IK solution remains about **14.5 mm** from the target. With the path solver's smaller 1.15-degree margin, IK succeeds at approximately **88.85 degrees**; the physical joint limit is 90 degrees.
- Therefore this is a near-base reach/elbow-limit problem caused by the expert's deliberate safety margin, not a bad reset, collision, or random seed. The physical arm can mathematically reach the pose only very close to its elbow limit, where the expert refuses to place a segment endpoint because it would leave no path slack.

### Expert-only feasibility grid

- Added `scripts/residual/check_feasibility_grid.py`. It does not create frozen evaluation states. It translates the full movable-object group over a 17 x 17 grid: `dx,dy = -4.0 ... +4.0 cm` in 0.5 cm increments.
- Each of the 289 cells uses one unique recorded seed, 32,000,000 through 32,000,288. It runs a complete physics-only scripted-expert episode and saves the CSV after every cell so the run is resumable.
- Approved command, run on `GN-A40-035` with the NVIDIA A40 and EGL:
  `source /scratch/e1638612/miniforge3/bin/activate && conda activate so101lm && MUJOCO_GL=egl python -u scripts/residual/check_feasibility_grid.py`
- Result: **211/289 expert successes; 78 failures**.
  - 56 failures: low grasp unreachable near the base.
  - 21 failures: gripper or moving jaw would collide with the fixed white tray, mostly in the backward-right part of the grid.
  - 1 isolated failure at `(dx=0, dy=-2.5 cm)`: `blocked`. Its surrounding cells pass, so this is likely rollout/seed sensitivity rather than a geometric boundary. The one-seed grid is a screening map, not proof that every green point succeeds for every seed.
- Minimum successful grid `dx` changes with `dy`: `-2.0 cm` for `dy <= +1.0 cm`, `-1.5 cm` for `dy=+1.5...+2.5 cm`, and `-1.0 cm` for `dy>=+3.0 cm`. Every tested point from those boundaries through `dx=+4 cm` passed except the isolated blocked cell.
- Outputs:
  - `docs/residual/placements/expert_feasibility_grid.csv`
  - `docs/residual/placements/expert_feasibility_grid.png`
  - `docs/residual/placements/expert_feasibility_grid_summary.json`

### Proposed replacement design — waiting for approval

- **Training region:** sample continuously and uniformly from the conservative rectangle `dx in [-1.5, +3.0] cm`, `dy in [-2.0, +2.0] cm`. All 90 tested grid cells in this rectangle passed. It retains shifts beyond the original +/-1 cm data while staying 0.5 cm inside most of the observed backward reach boundary. Use the existing training-only 20,000,000-series seeds.
- **Frozen evaluation A — axis 2 cm:** four directions x 10 seeds = 40 states. These already passed 40/40 in the earlier strict validation.
- **Frozen evaluation B — diagonal 1.5 cm:** `(±1.5, ±1.5) cm`, four corners x 10 seeds = 40 states. All four grid-screening cells passed. This replaces `(±2, ±2) cm`, whose backward-left corner failed the feasibility screen.
- **Frozen evaluation C — feasible axis 3 cm:** forward, left, and right only x 10 seeds = 30 states. Do not include backward 3 cm; it is outside the expert-feasible area.
- **Frozen evaluation D — feasible axis 4 cm:** forward, left, and right only x 10 seeds = 30 states. This is the beyond-training test. Do not include backward 4 cm.
- Report the direction omission explicitly rather than pretending the feasible workspace is symmetric. After approval, every proposed frozen state must again pass the scripted expert; stop and report the first failure, with no silent seed replacement. Generate render sheets only after all checks pass, on the GPU node.

No replacement frozen JSON files have been created. The earlier proposed set definitions in code are not finalized and must be revised only after Valentin approves this replacement design.

## Step 5 completion — Approved frozen sets (2026-11-05)

### Final design

- Valentin approved the replacement design with one addition: every state is tagged `in-training-region` or `out-of-training-region`, and later policy results must be reported separately for those groups.
- The continuous training rectangle is inclusive: `dx in [-1.5, +3.0] cm`, `dy in [-2.0, +2.0] cm`. `so101_libero/residual/placements.py` is the single source of truth for these bounds, the seeded training sampler, and state classification.
- Final frozen sets:
  - `shift_2cm_axis`: 40 states, four directions. 30 in-region and 10 out-of-region; the ten 2 cm-backward states are out-of-region.
  - `shift_1p5cm_diagonal`: 40 states, four diagonal corners. All 40 are in-region.
  - `shift_3cm_axis_feasible`: 30 states, forward/left/right only. The ten forward states are in-region; the 20 left/right states are out-of-region.
  - `shift_4cm_axis_feasible`: 30 states, forward/left/right only. All 30 are out-of-region and beyond training.
- Total: **140 states: 80 in-training-region, 60 out-of-training-region**. Each state stores `training_region_tag`; each file and the validation report store normalized per-tag counts, including explicit zero counts.
- Backward 3 cm and 4 cm remain explicitly omitted because the feasibility map showed that the scripted expert cannot safely reach them. There is no silent seed rejection or replacement.

### Final GPU-node validation

- Approved and ran on `GN-A40-035` / NVIDIA A40 with EGL:
  `source /scratch/e1638612/miniforge3/bin/activate && conda activate so101lm && MUJOCO_GL=egl python -u scripts/residual/make_placements.py`
- All geometry probes and **all 140/140 scripted-expert episodes passed**. The separate group result is **80/80 in-training-region** and **60/60 out-of-training-region**.
- All JSON files were re-read and checked for state count, tag count, and expert success. The 4 cm montage was visually inspected; front, wrist, and overview panels correctly show the three approved directions.
- Frozen files and render sheets:
  - `docs/residual/placements/shift_2cm_axis.{json,png}`
  - `docs/residual/placements/shift_1p5cm_diagonal.{json,png}`
  - `docs/residual/placements/shift_3cm_axis_feasible.{json,png}`
  - `docs/residual/placements/shift_4cm_axis_feasible.{json,png}`
  - `docs/residual/placements/validation_report.json`
- `validation_report.json` has `all_ok=true` and an `evaluation_group_summary` with the two required groups.

### Step 5 status

Complete. Do not start Step 6 until Valentin reviews these results. Step 6 must evaluate the frozen SmolVLA baseline on exactly these saved states and report both per-set results and the aggregate in-training-region versus out-of-training-region split.

## Step 6 — SmolVLA baseline on shifted states (2026-11-05)

### Implementation and protocol

- Added `scripts/residual/evaluate_shifted_baseline.py`. It reads the four frozen Step 5 JSON files directly, verifies scene/rules hashes, preserves exact initial states/seeds/tags, and uses the repository's existing policy runner, cube-flip diagnostics, and Wilson interval calculation.
- Settings match the reproduced original baseline: final 100k cube-flip checkpoint, CUDA, EGL, 256 x 256 images, one worker, deterministic Torch seed per state, 450-step horizon, and no appearance randomization.
- The first pass evaluates all states without video. A second deterministic pass reruns up to two actual failures per set with video. Video reruns must match the original outcome before being attached to the report.
- Broad failure mapping requested for this experiment:
  - `never_grasped`: detailed outcome `never_lifted` (cube never carried at least 1 cm in the gripper).
  - `dropped`: fell off table or an explicit dropped outcome.
  - `lifted_but_not_flipped`: lifted but still held, not turned, turned but not flat/in place/settled, or another lifted failure.
  - `timeout`: reached the 450-step horizon without a more specific diagnostic.
  - `other`: remaining outcomes and runtime errors.
- A four-state smoke run (one per set) completed before the full command; all four happened to be `never_lifted`, and the output format/tag aggregation passed.
- Approved full command:
  `source /scratch/e1638612/miniforge3/bin/activate && conda activate so101lm && MUJOCO_GL=egl python -u scripts/residual/evaluate_shifted_baseline.py --out docs/residual/results/baseline_shifted_140 --failure-videos 2 --workers 1`

### Results

- **Overall: 36/140 = 25.7%, Wilson 95% CI [19.2%, 33.5%].** Mean success step: 265.5.
- Per set:
  - Axis 2 cm: **19/40 = 47.5%**, CI [32.9%, 62.5%]; failures = 20 never grasped, 0 dropped, 1 lifted/not flipped, 0 timeout, 0 other; mean success step 263.9.
  - Diagonal 1.5 cm: **17/40 = 42.5%**, CI [28.5%, 57.8%]; failures = 20 never grasped, 0 dropped, 3 lifted/not flipped, 0 timeout, 0 other; mean success step 267.3.
  - Feasible axes 3 cm: **0/30 = 0%**, CI [0%, 11.4%]; failures = 20 never grasped, 0 dropped, 10 lifted/not flipped, 0 timeout, 0 other; no success-step mean.
  - Feasible axes 4 cm: **0/30 = 0%**, CI [0%, 11.4%]; failures = 29 never grasped, 0 dropped, 1 lifted/not flipped, 0 timeout, 0 other; no success-step mean.
- By proposed residual training region:
  - In-training-region: **27/80 = 33.8%**, CI [24.4%, 44.6%]; failures = 50 never grasped, 0 dropped, 3 lifted/not flipped, 0 timeout, 0 other; mean success step 267.2.
  - Out-of-training-region: **9/60 = 15.0%**, CI [8.1%, 26.1%]; failures = 39 never grasped, 0 dropped, 12 lifted/not flipped, 0 timeout, 0 other; mean success step 260.3.
- Raw failure outcomes across all sets: 89 `never_lifted`, 12 `still_held`, 2 `turned_but_not_in_place_or_not_settled`, and 1 `not_turned`. All failures reached the horizon, but each had a more specific cause, hence zero timeout-only failures.
- Directional behavior is strong rather than random seed noise: for the 2 cm and diagonal sets, some directions consistently succeeded while others consistently missed the cube. At 3 cm, SmolVLA had no successes; it sometimes lifted in one direction but did not finish/release the flip. At 4 cm it almost always failed to lift.
- Important terminology: `in-training-region` means inside the **proposed future residual-RL training rectangle**, not inside SmolVLA's demonstration distribution. SmolVLA's original data still covered only the default and axis-1-cm placements.

### Outputs and verification

- `docs/residual/results/baseline_shifted_140/results.json`: complete 140-episode report, per-set and per-region summaries, mapping, raw outcomes, and episode records.
- `docs/residual/results/baseline_shifted_140/episodes.csv`: one row per state.
- `docs/residual/results/baseline_shifted_140/summary.md`: compact human-readable tables and video links.
- Eight failure videos (two per set) are under `docs/residual/results/baseline_shifted_140/videos/`. Every video has 451 frames / 15.03 s and reproduced its original failure outcome exactly.
- Verification: `complete=true`, 140 episode rows, exactly 80 in-region and 60 out-of-region, eight videos, no warnings or runtime errors.

### Step 6 status

Complete. The shifted baseline leaves substantial headroom for residual RL, especially at 3 cm and 4 cm. Do not start Step 7 until Valentin reviews the results.

## Step 6 follow-up — Stalled-failure diagnosis (2026-11-05)

- Added `scripts/residual/classify_baseline_failures.py` and reran all 104 failed shifted-baseline states with the same frozen checkpoint, states, seeds, EGL renderer, and horizon while recording joint motion.
- Definition: maximum absolute deviation from the reset value over the five arm joints and all 450 steps. The gripper is excluded from the 5-degree threshold because it uses a 0–100 scale. Categories are: stalled `<5 deg`; moved-but-missed `>=5 deg` and baseline never lifted; lifted-but-not-flipped from the saved baseline's `lifted=true` fact.
- Result: **70 stalled, 19 moved but missed, 15 lifted but not flipped**.
- Per set:
  - Axis 2 cm: 20 stalled, 0 moved/missed, 1 lifted/not flipped.
  - Diagonal 1.5 cm: 20 stalled, 0 moved/missed, 3 lifted/not flipped.
  - Feasible axes 3 cm: 20 stalled, 0 moved/missed, 10 lifted/not flipped.
  - Feasible axes 4 cm: 10 stalled, 19 moved/missed, 1 lifted/not flipped.
- Directional structure:
  - 2 cm forward and left: all 10 failures in each direction stalled; the one backward failure lifted but did not finish; right had no failures.
  - Diagonal forward-left and backward-left: all 10 failures in each direction stalled; the three backward-right failures lifted but did not finish; forward-right had no failures.
  - 3 cm forward and left: all 10 in each direction stalled; all 10 right failures lifted but did not finish.
  - 4 cm forward: 10 moved/missed; left: 10 stalled; right: 9 moved/missed and 1 lifted/not flipped.
- The threshold has a very large margin: stalled replays moved only about 0.4–1.0 degrees, while normal-motion replays moved about 87–95 degrees.

### First-three-chunk action evidence

- Logged 150 state/action pairs (three 50-action chunks) for two stalled episodes:
  - Diagonal 1.5 cm forward-left state 28: episode maximum movement 0.4206 deg. Per-chunk mean absolute arm target-minus-state differences were 0.0492, 0.0482, and 0.0440 deg; maxima were 0.3100, 0.3203, and 0.3316 deg.
  - Axis 2 cm left state 22: episode maximum movement 0.5381 deg. Per-chunk means were 0.0561, 0.0508, and 0.0590 deg; maxima were 0.3764, 0.2323, and 0.4791 deg.
- Mean gripper target-minus-state difference stayed around 0.017–0.024 units. SmolVLA therefore truly outputs near-hold targets at the folded reset configuration; the robot is not stuck while receiving a large motion command.

### Reproducibility caveat and outputs

- 98/104 replay outcomes exactly matched the saved baseline outcome. Six normal-motion boundary cases changed their detailed outcome on replay (including one success); CUDA/render/physics numerical sensitivity can change whether a near-boundary flip settles. Five were already baseline lifted/not-flipped states. One 4 cm-right baseline never-lifted state replayed a significant-motion partial flip; it remains moved-but-missed based on the baseline lift fact and the strong direction-wide motion pattern.
- Both stalled trace episodes reproduced their baseline `never_lifted` outcome exactly, so the stay-still conclusion is unaffected.
- Outputs: `docs/residual/results/baseline_shifted_140/stall_diagnostics/{summary.md,summary.json,classification.csv}` plus the two first-three-chunk CSV traces.

## Phase A — Early expert-direction nudge diagnostic (2026-11-05)

### Protocol

- Added `scripts/residual/nudge_test.py`. It selects exactly the 40 baseline-stalled states in the two target sets: 20 from `shift_2cm_axis` and 20 from `shift_1p5cm_diagonal`.
- For each reset, the unchanged scripted expert plans its collision-checked pre-grasp joint configuration. During steps 0--49, each of SmolVLA's five arm-joint targets is shifted toward that configuration, capped independently at 2, 5, or 10 degrees. The gripper remains SmolVLA-controlled. From step 50 onward, actions are exactly the frozen SmolVLA outputs.
- This is a diagnostic use of privileged simulator state through the scripted expert; it is not the learned residual policy. States, seeds, EGL rendering, checkpoint, 450-step horizon, and success predicate match the saved shifted baseline.
- Full command: `MUJOCO_GL=egl python -u scripts/residual/nudge_test.py`.

### Results

| first-chunk nudge cap | all stalled states | axis 2 cm | diagonal 1.5 cm | lifted failures |
|---:|---:|---:|---:|---:|
| 2 degrees | 0/40 | 0/20 | 0/20 | 0 |
| 5 degrees | **5/40** | 0/20 | **5/20** | 0 |
| 10 degrees | 0/40 | 0/20 | 0/20 | 4 still held |

- Wilson 95% intervals over all 40 states: 2 degrees [0%, 8.76%], 5 degrees [5.46%, 26.11%], 10 degrees [0%, 8.76%].
- Five degrees is the best tested early scale and is also the requested minimum. Ten degrees was not better: it caused four diagonal episodes to lift without completing/releasing the flip. Axis-2-cm stalled states did not recover at any tested cap.
- Interpretation: a small early displacement can break part of the folded-state failure mode, but the correct magnitude is narrow and a one-chunk fixed-direction displacement is insufficient for the axis set. The residual policy needs sustained closed-loop feedback and distance shaping rather than simply maximizing the early correction.
- Outputs: `docs/residual/results/nudge_test/results.json` and `episodes.csv`. The report is complete with 120 rows.

### Decision for Phase B

- Use an early arm scale of 5 degrees for steps 0--49 and a later arm scale in the requested 2--3 degree range. Keep the gripper scale separately configurable in its native 0--100 units.

## Phase B / Step 7 — Residual environment (2026-11-05)

### Implementation

- Added `so101_libero/residual/residual_env.py` without changing the existing simulator, policy runner, task, expert, or training code.
- `ResidualSO101Env` loads SmolVLA once, sets evaluation mode, disables gradients on every parameter, and calls it only through the existing inference-only runner. SmolVLA remains frozen.
- Residual action space is `[-1, 1]^6`. The final absolute target is clipped to the simulator's joint limits after `base_action + scale * residual_action`.
- Time-dependent scale uses the Phase A winner: 5 degrees per arm joint for steps 0--49, then 2.5 degrees. Gripper scale is separately configurable and defaults to 5 native units both early and late.
- The 20-value residual observation is: normalized measured joints (6), normalized current SmolVLA action (6), step within the 50-action chunk (1), cube world position (3), and cube world quaternion (4).
- **The cube pose and expert-derived grasp point are privileged simulator signals. This environment is sim-only.** `use_cube_pose=False` is supported for a future ablation, reducing the observation to 13 values.
- Dense reward is potential progress toward the expert's collision-checked grasp point: `previous_distance - current_distance` in metres (scale 1.0), plus 0.2 once on the first 2 cm lift and 1.0 on task success. Progress shaping avoids rewarding the agent for merely waiting near the target.
- Training resets use only reserved training seeds and remain inside the approved rectangle. Added deterministic `weighted_training_offset()` to `placements.py`: 75% focused sampling at radial shifts 1.5--2.5 cm, 25% uniform over the full rectangle. Rejection sampling enforces the asymmetric approved bounds.
- A 1,000-seed probe gave 1,000/1,000 inside-region offsets and 858/1,000 at 1.5--2.5 cm radial distance (the total exceeds 75% because part of the uniform component also falls in that band).

### Mandatory zero-residual test

- Added `scripts/residual/test_zero_residual.py` and ran it on the first ten frozen original states against the saved reproduced baseline.
- Command: `MUJOCO_GL=egl python -u scripts/residual/test_zero_residual.py`.
- Result: **10/10 zero-residual successes, exactly matching 10/10 saved SmolVLA-alone successes.** Success steps were 247, 264, 252, 248, 249, 251, 256, 244, 250, and 247.
- Output: `docs/residual/results/zero_residual_test.json`, with `matches_saved_baseline=true`.
- Step 7 mandatory condition passes; it is safe to proceed to PPO smoke testing.

## Phase C / Step 8 — PPO implementation and smoke run (2026-11-05)

### Implementation

- Added `scripts/residual/train_ppo.py` using Stable-Baselines3 PPO and eight `SubprocVecEnv` workers.
- Actor and critic MLPs are `[256, 256]`. After SB3 constructs the policy, the actor mean output layer's weights and bias are explicitly zeroed, so deterministic initial residual actions are exactly zero. Initial `log_std` is set to -1.0.
- Default rollout settings: 8 environments, 256 steps per environment (2,048 transitions/update), batch size 256, 10 epochs, learning rate 3e-4, gamma 0.99, GAE lambda 0.95. SmolVLA copies run on CUDA; the small PPO network runs on CPU.
- Frozen periodic evaluation uses only the saved axis-2-cm and diagonal-1.5-cm states, whose 31M-series seeds are disjoint from the 20M-series training seeds. Checkpoints, best model, final model, JSONL evaluations, monitor CSV, and TensorBoard logs are supported.
- Resume is supported with `--resume CHECKPOINT.zip`; actor initialization is not repeated on resume.

### Smoke-run problems and fixes

1. The first launch passed initial evaluation but SB3 supplied model seeds 73--80 on its first vector reset. The wrapper mistakenly treated them as placement seeds and correctly rejected them as outside the reserved training range. Fixed the wrapper so all non-frozen training resets always use each worker's disjoint 20M-series sequence; SB3's optimizer seed cannot leak into placement selection.
2. The second launch stopped before rollout because TensorBoard was absent from the conda environment. Installed `tensorboard==2.21.0` (plus its pip dependencies) into `so101lm`, as TensorBoard logging is a Phase D requirement.
3. The third launch succeeded. These were two different blocking conditions; neither repeated three times.

### Successful smoke run

- Command: `MUJOCO_GL=egl python -u scripts/residual/train_ppo.py --run-dir docs/residual/results/ppo_smoke --total-timesteps 2048 --eval-episodes 8 --eval-freq 0 --checkpoint-freq 10000`.
- Initial deterministic evaluation: **4/8 residual successes versus 4/8 saved baseline successes**, with the same per-state outcomes. This confirms the zero actor mean starts at baseline behavior on the target shifted states.
- PPO collected and optimized 2,048 transitions with eight environments at 21 aggregate steps/s; rollout time was 93 seconds after initialization.
- Saved `docs/residual/results/ppo_smoke/final_model.zip` and a TensorBoard event under `ppo_smoke/tb/PPO_1/`.
- Post-run check: actor parameters are finite; mean absolute output-layer weight changed from exactly zero to 0.00165, and six log standard deviations remained near -1.0. The smoke rollout was only 256 steps per worker, so no worker completed a full 450-step failure episode; an empty monitor body is expected.
- At 21 steps/s, 1,000,000 total transitions project to about 13.2 hours of rollout, plus PPO updates and periodic evaluation. The full job should fit within the 24-hour limit but is too long for an interactive session.

## Phase D / Step 9 — Full PBS training submission (2026-11-05)

### Batch job

- Added `scripts/residual/train_ppo.pbs` with project `personal-e1638612`, one GPU, 36 CPUs, 24-hour wall time, `MUJOCO_GL=egl`, eight environments, 1,000,000 target transitions, 20 frozen evaluation episodes every 100,000 steps, checkpoints every 100,000 steps, and TensorBoard logs.
- The script is resumable: it finds the latest `ppo_residual_*_steps.zip`, extracts completed steps, and trains only the difference to the 1,000,000-step target. It exits without duplicating training if the target checkpoint already exists.
- Estimated runtime from the successful smoke throughput: 14--18 hours including PPO updates and ten periodic evaluations.
- Submission command: `mkdir -p docs/residual/results/ppo_main && qsub scripts/residual/train_ppo.pbs`.
- PBS job: **`1523564.stdct-mgmt-02`**, name `so101_residual_ppo`, queue `batch_gpu`.
- State immediately after submission: `Q` (queued), scheduler comment `Insufficient amount of resource: node_pool`. This means no matching GPU node was free at that moment; submission itself succeeded.
- Batch log path: `docs/residual/results/ppo_main/pbs.log`.

## Phase E / Step 10 — Final evaluation pipeline prepared (2026-11-05)

- Added `scripts/residual/evaluate_residual.py`. It evaluates a trained PPO model deterministically on exactly the same 50 original and 140 shifted states/seeds used by the saved SmolVLA baselines.
- Outputs include matched per-episode JSON/CSV; Wilson intervals; the Step 6 broad failure types; stalled / moved-but-missed / lifted-but-not-flipped failure classification; per-set and in/out-training-region summaries; an original-50 regression check; all-shifted and two-target-set aggregates; success-versus-shift and training-curve plots; and up to three videos selected only where the residual succeeds and the saved baseline failed.
- The report explicitly records `privileged_cube_pose=true` and `sim_only=true`.
- Smoke-tested with the Phase C model on two original frozen states: both residual results matched the two saved baseline successes. Empty-set aggregation in limited smoke mode is handled explicitly.
- The full command will be run only after PBS training produces `docs/residual/results/ppo_main/best_model.zip`:
  `MUJOCO_GL=egl python -u scripts/residual/evaluate_residual.py`.
- Training PBS job `1523564.stdct-mgmt-02` started successfully on `GN-A40-018` at 16:23 Singapore time.
- Submitted dependent evaluation job **`1523590.stdct-mgmt-02`** with `afterok:1523564.stdct-mgmt-02`. It is held until training succeeds, then automatically runs all 190 states, writes plots/videos, calls `finalize_results.py` to replace the short summary at the top of this file and append the measured Phase E results, and commits only the final result paths.

## Phase D restart — Calm and kickstart variants

### Why the first full run was stopped

- Valentin observed that frozen evaluation was **5/20 at 100,000** and **5/20 at 200,000** transitions, versus **9/20 for SmolVLA alone** on those exact states. Both evaluations lifted 10/20 cubes, so the learned residual often preserved pickup but spoiled completion/release of the flip.
- Verified the two records in `docs/residual/results/ppo_main/evaluations.jsonl`. At 100k the paired target subset had 1 baseline failure fixed and 5 baseline successes broken; at 200k it had 0 fixed and 4 broken.
- Cancelled training job `1523564.stdct-mgmt-02` and dependent evaluation job `1523590.stdct-mgmt-02` as explicitly requested. PBS history records training exit status 143 (user cancellation) after 34:28 wall time and 231,424 transitions. Existing checkpoints/logs were preserved.

### Safer training/model-selection changes

- Added PPO `--target-kl`; both replacement runs use 0.02, learning rate 1e-4, and initial log-standard-deviation -2.0.
- Every frozen evaluation now saves its exact model under `eval_checkpoints/eval_*_steps.zip`. The timestep-zero model is saved as an evaluated checkpoint and starts as `best_model` with its 9/20 baseline-equivalent result. A periodic checkpoint replaces it only on a **strictly higher** success count. `best_evaluation.json` records why the selected checkpoint won.
- **calm:** residual arm scale 5 degrees for steps 0--49, then 1.5 degrees.
- **kickstart:** residual scale is active for steps 0--49 (5 degrees for arm joints, 5 native gripper units), then exactly zero for all six action dimensions. The residual therefore cannot alter SmolVLA after the first chunk.
- Both retain eight workers, 1,000,000 target transitions, evaluation/checkpointing every 100,000 transitions, and the same 20 frozen selection states.

### Expanded matched evaluation

- Final evaluation now runs both `best_model.zip` (the primary reported model) and `final_model.zip` on all 50 original plus 140 shifted states.
- On the 80 requested target states (axis 2 cm + diagonal 1.5 cm), it reports paired `fixed` (baseline failed, residual succeeded), `broken` (baseline succeeded, residual failed), both-success, and both-failed counts.
- It computes the exact two-sided McNemar binomial p-value and the continuity-corrected McNemar chi-square statistic for both best-eval and final checkpoints.
- The dual-model integration smoke test passed on two original frozen states: baseline, best, and final all succeeded.

### Replacement PBS submissions

- Calm training: **`1523669.stdct-mgmt-02`**; dependent calm evaluation: **`1523673.stdct-mgmt-02`**.
- Kickstart training: **`1523671.stdct-mgmt-02`**; dependent kickstart evaluation: **`1523674.stdct-mgmt-02`**.
- Both training jobs entered `batch_gpu` in queued state because no node was immediately free. Both evaluation jobs are correctly held with their respective `afterok` dependency.
- Each evaluation automatically runs both the selected best-eval checkpoint and the final checkpoint, updates this file, and commits its scoped result files. A Git lock serializes finalization if both evaluations finish close together.

## Diverse-demo follow-up (2026-11-05)

- Confirmed `residual-rl` branch and preserved the running calm `1523669` and kickstart `1523671` jobs and their held evaluations. No files in those run directories were edited.
- Read `train_ppo.py`: the 20 selection episodes are a balanced prefix of two files from the 140 frozen shifted test states. Their 31M-series seeds are disjoint from 20M-series training seeds, but they are **not a separate validation set**. Corrected the short summary above.
- The 100 original LeRobot demonstrations were not found at the README's `/scratch/e1583535/datasets/lerobot/local/so101_tri_cube_flip_100` nor under the current user's scratch. The pipeline regenerates 100 from the five original placements and records that provenance. The 200 additional episodes use `training_offset` and new 21M-series training seeds only; expert failures are rejected and logged.
- Added `generate_diverse_demos.py`, separate generation/fine-tune/evaluation PBS scripts, paired-summary generator, and the read-only PPO action audit script. Generation uses the repository `DemoWriter`, and fine-tuning uses `scripts/train.sh` with the existing 100k checkpoint as `--policy.path`; frozen original 50 and shifted 140 evaluations reuse the existing scripts and start-state files.
- Note correction: kickstart `config.json` records late arm scale zero but late gripper scale 5.0, so the gripper can still receive later residual corrections. The audit loads the actual saved scales.
- Static verification: all new Python scripts compiled with `py_compile`. Runtime results are pending PBS jobs.

### PBS submissions

- Action audit: `qsub scripts/residual/check_ppo_actions.pbs` -> `1523908.stdct-mgmt-02` (queued). It picks the latest numbered checkpoint at execution time, measures deterministic first-50-step mean absolute action by joint, then runs three stochastic 20-state trials. It reads only the PPO runs. The selection states are part of the 140 frozen test states, so the code's name `validation` was inaccurate.
- Demo generation: `qsub scripts/residual/diverse_generate.pbs` -> `1523909.stdct-mgmt-02` (queued), 24-hour cap.
- Fine-tuning: `qsub -W depend=afterok:1523909.stdct-mgmt-02 scripts/residual/diverse_train.pbs` -> `1523914.stdct-mgmt-02` (held), 20k optimization steps from the 100k checkpoint, 24-hour cap.
- Evaluation: `qsub -W depend=afterok:1523914.stdct-mgmt-02 scripts/residual/diverse_eval.pbs` -> `1523915.stdct-mgmt-02` (held), original 50 plus all shifted 140, paired fixed/broken summary, 12-hour cap.
- Outputs: `docs/residual/results/ppo_action_audit.json`, `/scratch/e1638612/datasets/lerobot/local/so101_cube_flip_diverse_300`, `outputs/train/smolvla_cube_flip_diverse_300`, and `docs/residual/results/diverse_demo/`. Monitor with `qstat -awn1` and the corresponding PBS logs.

## Calibration-offset experiment (2026-11-05)

### Phase A: wrapper and frozen sets

- Added `so101_libero/residual/calibration.py`: each reset samples five independent arm-joint biases from uniform [-L,L] using an independent deterministic seed stream; gripper bias always zero. `step` sends `clip(command+b)` to the unchanged simulator. Cameras and measured joint positions are returned unchanged. Bias and commanded/executed targets are recorded in info.
- Added `make_calibration_states.py`: generated validation (50, seeds 32,000,000-32,000,049), selection (20, 33,000,000-33,000,019), and test (100, 34,000,000-34,000,099). Each uses only the original five placements and saves full initial states plus biases for all sweep levels. These ranges are disjoint from training (22M) and old frozen states (11M/31M).
- Added baseline evaluator and PBS sweep script. Failure categories: stalled if max absolute arm movement <5 degrees and never lifted; moved-but-missed if >=5 degrees and never lifted; lifted-not-flipped if lifted but unsuccessful. Wilson 95% intervals use the existing repository function.
- Added separate PPO environment with 31-value observation (true joints, current base action, previous executed target, target-minus-state error, chunk index, previous residual). It exposes no cube pose or expert target; reward is only first-lift 0.2 plus success 1.0. Residual scale is L+2 degrees for arms, 2 native gripper units.
- The L=0 ten-episode comparison passed: 10/10 successes, exactly matching the saved SmolVLA-alone outcomes on those same states and seeds. Command: `PYTHONPATH=. MUJOCO_GL=egl python -u scripts/residual/evaluate_calibration_baseline.py --split zero_test --out docs/residual/results/calibration/zero_test.json`. Static `py_compile` passed for new files.

### Phase C-E pipeline prepared

- Added `train_calibration_ppo.py`: eight SB3 subprocess environments, [256,256] actor/critic, zero actor mean, log-std -2, learning rate 1e-4, target KL 0.02. Initial and every 100k-step selection evaluation saves the exact model; best selection uses deterministic success only. Each selection point also logs three stochastic-seed success counts and per-joint mean absolute applied residual (degrees for arms, native units for gripper). No test-state access during training.
- Added `test_calibration_zero.py`: ten L* validation states must match saved SmolVLA-alone outcomes when residual actions are exactly zero. A failure stops the training PBS job before smoke or PPO.
- Added `evaluate_calibration_ppo.py`: paired baseline versus best and final checkpoints on frozen 100-state test and original 50 L=0, with Wilson CIs, the three failure types, fixed/broken, exact two-sided McNemar, per-joint mean residual, and up to three fixed-case videos per checkpoint.
- Added PBS scripts for selection/commit, smoke plus training, and dependent final evaluation. Selection job commits the complete validation sweep and L* before the training job can start. If no level is 30%-70%, selection exits nonzero and the afterok training dependency cannot release. Estimated sweep up to 8 h; PPO smoke plus 500k transitions and periodic evaluations 12-20 h; final 450 matched episodes up to 16 h. All use EGL.
- Static `py_compile` passed. The training and final-evaluation scripts await measured L*; their runtime results are pending.
