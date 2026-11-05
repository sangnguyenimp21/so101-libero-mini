# AGENTS.md — Residual RL for SmolVLA (SO-101, cube_flip)

Read this whole file before doing anything. It is the shared context for every session.

## 0. Who you are working with

- User: Valentin, PhD student at NUS, **beginner in robotics and RL**. Joined the project late; has **~5 days**.
- Explain what you are doing in plain language, one step at a time. Avoid jargon or define it briefly.
- The project belongs to a team (Sang, Andrew) for the course "Robot Learning in the Era of Foundation Models" (Prof. Duan Jiafei). Their work is done; Valentin adds a new contribution on top.

## 1. Rules (always follow)

1. Work only on git branch `residual-rl`. New code goes in `so101_libero/residual/`, new scripts in `scripts/residual/`, notes and results in `docs/residual/`.
2. **Do not modify existing files** (tasks, init_states, env, runner, expert, training code) without asking first. Prefer wrappers and new files.
3. **One step at a time** (Section 7). After each step: summarize what you did, show how to verify it, and **stop and wait** for Valentin's OK.
4. Before any command that runs longer than ~5 minutes, or any PBS job submission, show the command and estimated time, and wait for OK.
5. Keep a running log in `docs/residual/NOTES.md`: decisions, commands, results, problems. Update it at the end of every step. A new session must be able to continue from NOTES.md alone.
6. Never silently change evaluation settings between the baseline and the residual model. Same start states, same seeds, same episode count.
7. If something in this file turns out to be wrong for the actual code, say so, explain, and propose a fix. The facts below come from the repo docs and should be verified in Step 1.
8. Commit working code often with clear messages. Push to the user's fork/branch when asked.

## 2. Compute environment (NUS HPC, Vanda cluster)

- **You are already running on the Vanda cluster** (inside an interactive GPU session, started by Valentin from his MacBook over SSH). Do not ssh anywhere. If the session ends, Valentin restarts it.
- **Login node: no heavy compute.** Interactive/background processes there are killed after 30 min CPU time. Use it only for editing, git, and submitting jobs.
- GPU: NVIDIA A40 48 GB, one GPU node = 36 CPU cores, 250 GB RAM.
- PBS project: `personal-e1638612` (1000 GPU hours).
  - Interactive (max 2 h): `qsub -I -P personal-e1638612 -l select=1:ngpus=1 -l walltime=02:00:00`
  - Batch (max 24 h): script with `#PBS -P personal-e1638612`, `#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1`, `#PBS -l walltime=24:00:00`. Submit with `qsub job.sh`, monitor with `qstat -awn1`.
- Working dir: `/scratch/e1638612/` (files idle >60 days are purged, no backup — keep code in git).
- Python env: conda env `so101lm` (Python 3.10, `lerobot[smolvla]==0.4.2`, `transformers==4.57.6`, `mujoco==3.3.7`, `stable-baselines3`). Activate with `source /scratch/e1638612/miniforge3/bin/activate && conda activate so101lm`.
- Rendering: repo defaults to CPU rendering (OSMesa). The A40 may support EGL (`MUJOCO_GL=egl`); test it in Step 4, it could be much faster.
- Check whether compute nodes have internet; if not, download packages/checkpoints from the login node.
- Note: the repo's `env.sh` may set `HF_HUB_OFFLINE=1`; download checkpoints first.

## 3. The existing project (facts from repo docs — verify in Step 1)

Repo: https://github.com/sangnguyenimp21/so101-libero-mini

- Simulator: MuJoCo, SO-101 arm. Gymnasium env `SO101TabletopEnv` in `so101_libero/env.py`. Create with `SO101TabletopEnv('cube_flip')`; `env.reset(options={"placement_id": k})` selects a placement.
- Policy runner: `so101_libero/policy/runner.py` loads a LeRobot checkpoint with its processors and exposes `act(observation, instruction)`.
- Observation: two cameras (front, wrist) + `observation.state` = 6 motor values.
- Action: **6 absolute joint targets**, degrees for 5 arm joints, 0..100 for gripper. Control at **30 Hz**.
- SmolVLA: chunk size 50, **executes all 50 actions per inference** (open loop for ~1.7 s). So SmolVLA is called only every 50 steps.
- Episode horizon for cube_flip: **450 steps** (15 s). Expert finishes around step 240–260.
- Placements: 5 only — default, and 1 cm left/right/forward/back. Objects move together; orientations never change. **The training data covers only ±1 cm.**
- The cube is yawed −20.4° to face the pan axis so the 45°-tilted grasp closes across two faces. Large shifts may break the grasp geometry.
- Evaluation: 50 frozen start states; episode k uses placement k mod 5 (10 per placement). Reported result: cube_flip 50/50 (on the training placements).
- Success for cube_flip: `TurnedOnSide`-style predicate from the simulator's tracked cube orientation (not the image). A flipped red cube looks identical to an unflipped one.
- No recovery demos and no randomization were used for the tri tasks. Appearance randomization (`--visual-rand`) exists in the code but was not used here.
- Useful tools in repo: `render_placements --expert-check` (verify expert solves a placement), `cli/evaluate.py` (evaluation).

Checkpoints (Hugging Face):
- cube_flip: `tsangb34/smolvla_tri_cube_flip_100` → download to `checkpoints/smolvla_cube_flip`
- green_push: `tsangb34/smolvla_tri_green_push_100` (not needed now)

## 4. Research question and hypothesis

**Question:** Can a small residual network trained with RL improve a frozen SmolVLA policy at cube positions outside its ±1 cm training range?

**Hypothesis:** SmolVLA alone degrades as the cube shift grows (±2, ±3 cm). Because SmolVLA runs open loop for 50 steps per chunk, a residual policy that reacts at every step can correct small errors and raise success, without hurting performance on the original placements.

## 5. Method: residual RL

- **Base policy:** SmolVLA cube_flip checkpoint, **frozen** (no gradient, eval mode).
- **Final action:** `a = a_smolvla + scale * a_residual`, with `a_residual ∈ [-1, 1]^6`.
  - Default scale: 2.0° per arm joint, 5.0 units for the gripper. Configurable. Clip final action to joint limits.
- **Residual observation** (vector):
  - 6 joint positions (normalized),
  - 6 SmolVLA actions for this step (normalized),
  - step index within current chunk / 50,
  - optional `use_cube_pose` (default True): cube position + orientation from the simulator. **Privileged, sim-only**; must be reported as such. Also run/plan an ablation with it False if time allows.
- **Reward (configurable):** +0.2 once when cube first lifted ≥ 2 cm above its start height; +1.0 at success (existing predicate); episode ends at success or horizon. Optionally a small per-step penalty on residual magnitude (default 0).
- **Reset distribution for training:** cube group offset sampled uniformly within ±3 cm in x and y (continuous), keeping orientation as in the task. Must not collide with other objects or leave the table. Training seeds must be disjoint from evaluation seeds.
- **Algorithm:** Stable-Baselines3 PPO, `MlpPolicy`, net_arch [256, 256] for actor and critic.
  - **Initialize the actor's mean output layer to zero** so the residual starts at 0 (robot initially behaves exactly like SmolVLA).
  - Initial log-std ≈ −1.0 (configurable).
  - Parallel envs with `SubprocVecEnv` (as many as CPU cores/memory allow; each holds or shares a SmolVLA copy — measure memory).
  - Periodic evaluation callback logging success rate; save model checkpoints; TensorBoard logging.
- **Speed:** render cameras only when SmolVLA needs a new chunk (every 50 steps) if the env allows; the residual does not use images.

Why residual (for the report): avoids computing flow-matching log-likelihoods (hard for PPO); SmolVLA cannot forget; cheap; corrections stay small and safe.

## 6. Evaluation protocol

- Evaluation sets (frozen start states, fixed seeds, not used in training):
  - **Original:** the existing 50 states (±1 cm).
  - **Shift 2 cm:** 4 directions × 10 = 40 states (more if cheap).
  - **Shift 3 cm:** 4 directions × 10 = 40 states.
  - Every new placement must be verified solvable by the scripted expert.
- Compare **SmolVLA alone** vs **SmolVLA + residual** on exactly the same states.
- Report: successes / N with 95% Wilson confidence intervals; failure-type breakdown (never grasped / dropped / lifted but not flipped / other); training curve (success vs episodes); 2–3 videos where residual succeeds and baseline fails.
- **Must check:** residual does not reduce success on the original placements.

## 7. Step-by-step plan (do one step, then stop for review)

**Step 1 — Understand the code (no changes).** Read README.md, docs/TRI_TASKS.md, env.py, policy/runner.py, cli/evaluate.py, tasks/tri_tasks.py. Write in NOTES.md: obs/action format; how runner.act() handles chunks (one action per call from a queue?); whether env.step() renders every step and whether rendering can be skipped; how success and lift height can be read; how to set arbitrary cube offsets safely without changing existing init_states or scene hash. Flag anything in this file that is wrong.

**Step 2 — Setup check.** Verify conda env, GPU (`torch.cuda.is_available()`), MuJoCo rendering backend, and download the checkpoint.

**Step 3 — Reproduce baseline.** Run the existing evaluation on cube_flip: 5 episodes smoke test, then all 50. Expect ~50/50. If not, stop and debug setup.

**Step 4 — Timing.** `scripts/residual/timing.py`: time per SmolVLA chunk (excluding warm-up), per env.step with and without rendering, OSMesa vs EGL, full episode. Estimate wall time for 2,000 training episodes with 1 and N parallel envs. If >12 h even in parallel, propose the backup plan (Step 11).

**Step 5 — New placements.** Propose a design (wait for OK) for: continuous cube offsets for training, and frozen evaluation states at ±2 and ±3 cm. Verify with the expert; save renders to `docs/residual/placements/`.

**Step 6 — Baseline on shifted positions.** Evaluate SmolVLA alone on the shifted sets. Table: shift | successes | failure types. If success is still ~100% at 3 cm, propose adding 4–5 cm shifts.

**Step 7 — Residual env.** `so101_libero/residual/residual_env.py` (Section 5). **Mandatory test:** with residual ≡ 0, success on 10 original-placement episodes matches SmolVLA alone. Do not continue if it fails.

**Step 8 — PPO training script.** `scripts/residual/train_ppo.py` (Section 5). Smoke run ~2,000 steps with 2 envs; confirm initial success ≈ baseline.

**Step 9 — Training job.** PBS batch script `scripts/residual/train_ppo.pbs` (project `personal-e1638612`, 1 GPU, 36 CPUs, ≤24 h), resumable from checkpoints. Show estimated runtime before submission. While running, help interpret the training curve; if flat, suggest fixes (smaller scale, larger lift reward, more envs, longer training).

**Step 10 — Final evaluation.** `scripts/residual/evaluate_residual.py` (Section 6). Outputs to `docs/residual/results/`: table with Wilson CIs, success-vs-shift plot (two lines), training curve, failure breakdown, videos.

**Step 11 — Backup plan (only if residual RL is infeasible in time).** Self-improvement loop: run SmolVLA on shifted training placements, keep successful episodes in LeRobot format, merge with the original 100 demos, fine-tune SmolVLA from the existing checkpoint (VLM frozen, action expert trained, as in the original setup), evaluate on the same shifted sets.

## 8. Timeline (5 days)

- Day 1: Steps 1–4. Day 2: Steps 5–7. Day 3: Steps 8–9 (training overnight). Day 4: Step 10 + slides. Day 5: buffer.
- Scope: **cube_flip only.** Other tasks only if everything is done early.

## 9. Final deliverables

- Code on branch `residual-rl`, reproducible with documented commands.
- `docs/residual/NOTES.md` (full log) and `docs/residual/results/` (tables, plots, videos).
- A short summary for slides: method diagram description, main table, success-vs-shift plot, limitations (sim only, one task, small corrections, privileged cube pose if used).
