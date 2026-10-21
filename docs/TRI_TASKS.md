# so101_tri: three language-conditioned tasks in one scene

One tabletop scene (a white tray fixed on the robot's right, a red cube with one white face, a green half-cylinder, a
light gray cylinder), exactly five initial placements, three tasks, 100 physically executed demonstrations per task,
and four SmolVLA fine-tuning jobs that evaluate every saved checkpoint in the simulator. The environment is built like
`stack_red_on_blue`: a layout and tasks in the task registry, the scripted IK expert, the `generate` CLI and the shared
SLURM driver. Definitions: `so101_libero/tasks/tri_tasks.py` (geometry, directions, placements, tolerances, tasks)
and `so101_libero/expert/tri_tasks.py` (demonstration controllers).

This is the second version of the suite (2026-10-21). The first one (`cube_left`, `green_into_box`, `cylinder_flip`,
with a white box on the robot's left) was retired. Its objects stood too close to each other and to the arm, and the
cylinder flip needed three grasps and a regrasp station. Its task ids, init states, jobs, datasets and models are
removed; its seed slots stay reserved. The older tray-on-the-left tasks (`red_cube_to_white_tray`,
`red_cube_far_to_white_tray`) were removed at the same time.

![placements](media/tri_tasks/placements.png)

Rows: the five placements. Columns: front camera (the policy's 256x256 view), wrist camera (policy input), and an
overview with the aside zone drawn in (not a policy input).
Example demonstrations (front | wrist, as recorded in the datasets):
- [cylinder aside](media/tri_tasks/expert_cylinder_aside_seed1300000_p0.mp4)
  ([key frames](media/tri_tasks/expert_cylinder_aside_seed1300000_p0_keyframes.png))
- [cube flip](media/tri_tasks/expert_cube_flip_seed1420000_p2.mp4)
  ([key frames](media/tri_tasks/expert_cube_flip_seed1420000_p2_keyframes.png))
- [green block into the tray](media/tri_tasks/expert_green_into_tray_seed1540000_p4.mp4)
  ([key frames](media/tri_tasks/expert_green_into_tray_seed1540000_p4_keyframes.png))

## Directions

All placements and goals are built from one `Directions` object (`TRI_SCENE.directions`). Directions are in the
robot's own frame, never the table edges:

| name | world vector | meaning | in the front image |
|---|---|---|---|
| FRONT | +x | away from the robot base, towards the front camera | down |
| LEFT | +y | the robot's own left | right (the camera faces the robot) |
| RIGHT | -y | the robot's own right | left |

Mirroring the whole scene and its goals is one setting: `Directions(left=(0.0, -1.0))`.

## Scene (metres; world origin = robot base on the table, 3.9 cm behind the shoulder's pan axis)

| item | size | pose (default placement) | mass | collision geometry |
|---|---|---|---|---|
| white tray | 15 cm long (FRONT-BACK) x 9 cm wide x 5 cm tall outside; 1 mm walls and bottom, inside 14.8 x 8.8 x 4.9 cm; open top | centre (0.075, -0.125), fixed to the table beside the robot base on its RIGHT | welded (n/a) | 4 wall boxes + bottom box; the bottom's collision slab also reaches 1 cm into the table (invisible) |
| gray cylinder | 3 cm diameter x 6 cm tall, upright | (0.20, 0.13): left-front | 30 g | cylinder |
| green block | half-cylinder, 3.5 cm diameter x 3 cm long, 1.75 cm tall, flat face down, extrusion axis along FRONT | (0.225, 0.03): centre-front, in front of the folded arm | 15 g | exact convex mesh (32 arc segments), stiffer contacts (solref 0.005 s, priority 1) |
| red cube | 3 cm; its top face white | (0.232, -0.072): right-front, yaw -20.4° (turned to face the pan axis) | 20 g | box (the white face is a visual-only patch: no collision, no mass) |
| aside zone | 7 x 7 cm, not drawn in the policy's views | x 0.05-0.12, y 0.12-0.19: the robot's left, beside the base, opposite the tray | n/a | none (part of the goal) |

Spacing, measured by `render_placements` over all five placements:
- objects ≥ 5.8 cm apart between footprints (~10 cm centre to centre);
- 17.9-21.6 cm from the shoulder's pan axis;
- every object ≥ 3.4 cm from the folded arm at rest and while it unfolds (the closest: the green block and the
  jaw in placement 4);
- every object ≥ 5.1 cm from the tray.

Friction is 1.0 (torsional 0.005, rolling 0.0001), like every other item in this project. Masses and friction are
assumptions, not measurements. The tray is welded to the world (fix the real one to the table). The cameras are the
project's (measured front camera, wrist camera on the gripper's camera mount) and are unchanged.

**Five placements, nothing else random.** The cylinder, green block and cube move together. The tray, aside zone,
table, robot base and cameras never move. Orientations never change. The arm starts in the fixed folded rest pose, as
in `stack_red_on_blue`.

| id | name | group translation | cylinder | green block | cube |
|---|---|---|---|---|---|
| 0 | default | 0 | (0.20, 0.13) | (0.225, 0.03) | (0.232, -0.072) |
| 1 | left | 1 cm LEFT | (0.20, 0.14) | (0.225, 0.04) | (0.232, -0.062) |
| 2 | right | 1 cm RIGHT | (0.20, 0.12) | (0.225, 0.02) | (0.232, -0.082) |
| 3 | forward | 1 cm FRONT | (0.21, 0.13) | (0.235, 0.03) | (0.242, -0.072) |
| 4 | backward | 1 cm BACK | (0.19, 0.13) | (0.215, 0.03) | (0.222, -0.072) |

`env.reset(options={"placement_id": k})` selects a placement; without it the placement is drawn from the reset seed.
`info["placement_id"]` and `env.init_state["placement_id"]` report it. `python -m so101_libero.cli.render_placements`
checks all five ([results](results/tri_tasks/placements_check.json), all pass):
- settled positions equal the nominal ones (< 1 mm) and orientations are unchanged;
- every object rests on the table, touching only the table;
- the spacing above (gaps between objects, to the tray, to the pan axis, to the arm at rest and while unfolding);
- every object, the tray and the aside zone lie inside the 256x256 front image;
- each task's expert succeeds from each placement.

## Tasks

All three use the same compiled scene and placements (suite `so101_tri`, registry indices 13-15). Success means every
goal predicate holds for 10 consecutive control steps (0.33 s, settled). Tolerances live in `TriTolerances`.

| task id | instruction | goal (privileged state, `tasks/base.py`) | horizon |
|---|---|---|---|
| `cylinder_aside` | Pick up the light gray cylinder and put it aside. | `OnTableInRegion(aside_zone)`: centre inside the zone, standing upright (≤ 15°) directly on the table, ≥ 6 cm from its start; carried ≥ 2 cm high and released within 3 cm of where it rests (carried there, not shoved); no robot contact, < 2 cm/s, TCP ≥ 4 cm away. `Undisturbed(cube, green)`: each ≤ 2 cm moved, ≤ 20° tilted, untouched, not in the tray | 450 steps (15 s) |
| `cube_flip` | Pick up the red cube and flip it onto its side. | `TurnedOnSide`: the white (start-top) face within 15° of vertical, i.e. facing sideways (upside down does not count), and the cube lying flat on a face (≤ 15°) directly on the table, ≤ 2 cm (horizontal) from its start; carried ≥ 2 cm high; released, < 2 cm/s, TCP ≥ 4 cm away. `Undisturbed(cylinder, green)` | 450 steps |
| `green_into_tray` | Pick up the green semicircular block and put it into the white tray. | `RestingInContainer`: centre ≥ 3 mm inside the inner walls, resting on the tray floor in any orientation (centre ≤ max half-extent + 5 mm above it); carried ≥ 2 cm high; released, < 2 cm/s, TCP ≥ 4 cm away. `Undisturbed(cylinder, cube)` | 450 steps |

The environment tracks each object's orientation from the start of the episode, so the cube's turn is measured on
the body, not guessed from the image.

## Demonstration controllers (`so101_libero/expert/tri_tasks.py`)

The expert stack is the project's: damped-least-squares IK at every control step with checked convergence, min-jerk
timing under the 75°/s joint-speed cap, and collision-checked poses. Physics checks run at every phase boundary
(grasped, lifted, still held, placed). Actions are absolute joint targets through the position actuators; grasps are
purely frictional (nothing is welded, attached or teleported). Every demonstration unfolds from the rest pose (the
unfold is collision-checked against the objects), does the task, parks at rest again and holds 2 s, so the policy
learns to stop.
- **cylinder_aside:** top-down grasp at the cylinder's middle, jaws turned for the most clearance. Lift until its
  bottom is 5-6 cm up, then carry it to a random free spot in the aside zone (≥ 2 cm inside the zone's edges and ≥ 2 cm
  from other items) that is reachable at every height of the plan. The gripper keeps one orientation the whole time it
  holds the cylinder, so the cylinder stays upright (the least outward tilt, 0/15/30°, for which the whole plan is
  reachable). Lower until it is 2-4 mm above the table, release, retreat, park.
- **cube_flip (one grasp):** the cube faces the pan axis, so a gripper whose approach is tilted 45° outward (in the
  vertical plane through the pan axis, the only plane the 5-DoF arm can aim in) closes across its two side faces. Lift
  to a TCP height of 9-10 cm, then roll the wrist 180° about the tilted approach axis: the cube turns 90° (its white top
  face ends up facing the robot, its outer face ends up down). Lower over its start spot and release. After the roll,
  the wrist-camera mount hangs below the TCP on the robot side, so the release happens at the lowest collision-free
  height, a ~2 cm drop onto the new face. Retreat, park. The roll direction that stays collision-free (and the release
  height) is checked before the grasp.
- **green_into_tray:** top-down grasp across the two flat semicircular end faces. Carry it over the 5 cm walls, turn
  the jaws along the tray's long side, and lower it to 5-9 mm above the floor at a reachable spot inside the tray
  (searched on a grid, jaws clear of the walls). Release, retreat, park.

Expert success from the start states (physics only, 10 seeds x 5 placements): cylinder_aside 50/50, cube_flip 50/50,
green_into_tray 50/50. In-hand slip stays ≤ 1.8 mm. A demonstration takes 295-382 steps including the park and the
2 s hold, and the goal first holds at step 187-261. Failed attempts are discarded and retried with the next seed.

## Demonstration collection

```bash
# local runtime matching the cluster (python 3.10, lerobot[smolvla] 0.4.2, transformers 4.57.6, mujoco 3.3.7):
conda create -n so101lm -c conda-forge python=3.10 ffmpeg=7.1.1 && conda activate so101lm
pip install "lerobot[smolvla]==0.4.2" "transformers==4.57.6" "mujoco==3.3.7" threadpoolctl
# (on the cluster: bash scripts/setup_conda_env.sh, or simply submit a training job, which runs it)

# look at the scene / launch the environment
MUJOCO_GL=cgl python -m so101_libero.cli.render_placements --task cylinder_aside --out outputs/placements --expert-check
MUJOCO_GL=cgl python -m so101_libero.cli.render_video --task cube_flip --seeds 1420000 --placement 2 --out outputs/videos
python -c "import so101_libero; from so101_libero.env import SO101TabletopEnv; env = SO101TabletopEnv('cube_flip'); \
obs, info = env.reset(options={'placement_id': 2}); print(info, {k: v.shape for k, v in obs.items()})"

# 100 successful demonstrations per task = exactly 20 per placement (MUJOCO_GL=cgl on macOS, egl/osmesa on Linux).
# Cluster / Linux: the defaults (AV1 video, LeRobot's parallel encoders) are fine:
python -m so101_libero.cli.generate --tasks cube_flip --demos-per-placement 20 --workers 8 --repo-id local/so101_tri_cube_flip_100
# macOS laptop (how the datasets below were made): H.264 video, because LeRobot's default SVT-AV1 3.0 encoder crashes
# intermittently on macOS when an encoder is destroyed (libdispatch semaphore trap, kills the process); and a small
# window, because finished episodes wait in memory until written:
for t in cylinder_aside cube_flip green_into_tray; do
  python -m so101_libero.cli.generate --tasks $t --demos-per-placement 20 --workers 4 --window 4 --vcodec h264 \
      --encode-in-process --repo-id local/so101_tri_${t}_100
done
# combined multitask dataset (300 demonstrations)
python -m so101_libero.cli.merge_datasets --out-repo-id local/so101_tri_v2_multitask_300 \
    --repo-ids local/so101_tri_cylinder_aside_100 local/so101_tri_cube_flip_100 local/so101_tri_green_into_tray_100
# inspection: LeRobot loader, features, timing, videos, open-loop replay of 10 episodes in the simulator
python -m so101_libero.cli.validate_dataset --repo-id local/so101_tri_v2_multitask_300 --replay-episodes 10
```

Datasets go to `$HF_LEROBOT_HOME/<repo-id>` (LeRobot format v3.0, lerobot 0.4.2). On a SLURM node, a training job
generates any missing per-task dataset itself, with the same settings.

Every dataset has the same policy interface as all other datasets in this project:
- `observation.images.front` and `observation.images.wrist`: 256x256 RGB video, in that order;
- `observation.state`: the 6 `so101_follower` joint positions, in degrees with the gripper at 0-100
  (`use_degrees=True` units);
- `action`: absolute joint targets in the same units;
- 30 fps, and `task` = the natural-language instruction of the episode.

Per-episode metadata:
- per-frame `placement_id` and `collection_seed` (int64; LeRobot never feeds non-`observation.*` keys to a policy);
- `meta/so101_libero_episodes.jsonl`: task id, instruction, placement id and name, seed, initial state, phases and
  grasp-quality metrics;
- `meta/so101_libero.json`: generation config, per-placement attempts/successes, seed ranges, code hash and a source
  snapshot.

Seeds: task slot k uses `k * 100,000 + placement * 10,000 + i` (cylinder_aside 1,300,000+, cube_flip 1,400,000+,
green_into_tray 1,500,000+). Each placement keeps its first 20 successes in seed order, independent of the worker
count. Evaluation states: `init_states/<task>.json`, 50 per task (10 per placement, seeds 11,300,000+ / 11,400,000+ /
11,500,000+), all solved by the expert (0 candidates rejected).

TBD_DATASETS

## Multitask data and sampling

lerobot 0.4.2 cannot train on several repo ids (`make_dataset` raises `NotImplementedError`), so `merge_datasets`
builds one combined dataset with LeRobot's own `aggregate_datasets`:
- tasks are re-indexed by instruction string, so every frame keeps its own instruction;
- it first checks that fps, robot type and features (camera keys and order, joint names) are identical;
- normalization statistics are aggregated by LeRobot;
- file rotation is switched off (one video file per camera). In lerobot 0.4.2, a rotation points every episode of a
  multi-file source at the last destination file, so frames decode from the wrong file or past its end (this broke the
  first version's multitask job on the cluster). After merging, the first and last frame of every episode are decoded
  from the merged dataset and compared with its source, and any mismatch fails the merge.

The three tasks' episodes differ in length, so plain uniform frame sampling would favour the longest. The multitask
job therefore trains with `python -m so101_libero.train.lerobot_train_balanced` instead. This is `lerobot-train`
itself (same arguments, checkpoints and resume), with the dataset wrapped so that every task has the same number of
sample slots per epoch. Each sample is still a real frame with its own instruction. The mapping is written to
`<output_dir>/task_balancing.json`.

## SmolVLA fine-tuning (SLURM, one H100 96 GB, one job per run)

| script | data | output dir / log |
|---|---|---|
| `slurm/train_smolvla_cylinder_aside.slurm` | 100 cylinder demos | `outputs/train/smolvla_tri_cylinder_aside_100` / `liberomini-so101lm-smolvla-cyl-aside-<jobid>.out` |
| `slurm/train_smolvla_cube_flip.slurm` | 100 cube-flip demos | `outputs/train/smolvla_tri_cube_flip_100` / `liberomini-so101lm-smolvla-cube-flip-<jobid>.out` |
| `slurm/train_smolvla_green_into_tray.slurm` | 100 green-block demos | `outputs/train/smolvla_tri_green_into_tray_100` / `liberomini-so101lm-smolvla-green-tray-<jobid>.out` |
| `slurm/train_smolvla_multitask.slurm` | all 300, tasks sampled equally | `outputs/train/smolvla_tri_v2_multitask_300` / `liberomini-so101lm-smolvla-tri-multi-<jobid>.out` |

Each script is a thin wrapper, like `slurm/smolvla_stack_red_on_blue.sbatch`, around the shared driver
`slurm/smolvla_job.sbatch` (the former `smolvla_red_cube_to_white_tray.sbatch`, now task-neutral). The driver's
stages:
1. environment (conda env via `scripts/setup_conda_env.sh`, or `ENV_ACTIVATE`);
2. evaluation states (`init_states/<task>.json`, committed) and a scene-hash check;
3. dataset: used if present, else generated (per task) or merged (multitask);
4. validation (10 replayed episodes);
5. training;
6. **evaluation of every saved checkpoint** (`EVAL_CKPTS=all`, the default in `slurm/tri_tasks.env`).

Training settings:
- fine-tunes `PRETRAINED` (default `lerobot/smolvla_base`), never from scratch;
- `--steps=100000` optimizer updates, cosine LR decay over the same 100k steps;
- `--save_freq=20000`: checkpoints 020000, 040000, 060000, 080000 and 100000, all kept;
- BF16, batch 32 (configurable with `BATCH_SIZE`);
- front/wrist renamed to `camera1`/`camera2`; each sample's instruction is the language input.

Evaluation: after training, each of the five checkpoints runs on the task's 50 frozen start states in the simulator
(episode k = state k, which uses placement k mod 5, so 10 per placement). Results go to
`$OUT/eval/<step>/results.json` (success rate per task with a 95 % interval, counts of where the object ended up),
`episodes.csv` (one row per episode: index, seed, success, steps, outcome) and 10 videos. The job log ends with a table of all checkpoints
(`python -m so101_libero.cli.summarize $OUT/eval/*/results.json`). The multitask job evaluates every checkpoint on all
three tasks. A checkpoint takes ~15 min per task: the per-task jobs need ~4-5 h of training plus ~1.5 h of evaluation
(12 h limit), the multitask job ~4-5 h plus ~4 h. If a job hits its time limit, resubmit the same command: training
resumes from `checkpoints/last` (`--resume=true`, restoring the optimizer, scheduler and RNG state) and finished
evaluations are skipped. `EVAL_CKPTS=last` or `none` shortens a job.

Site settings (scratch dir, environment activation, pretrained checkpoint, dataset ids, batch size, evaluation,
account/partition) are in `slurm/tri_tasks.env`. Nothing is submitted automatically.

```bash
# from the project root on the cluster; account/partition either on the command line, or in slurm/tri_tasks.env
# (SBATCH_ACCOUNT / SBATCH_PARTITION) and then `( source slurm/tri_tasks.env && sbatch <script> )` in a subshell:
sbatch --account=<account> --partition=<partition> slurm/train_smolvla_cylinder_aside.slurm
sbatch --account=<account> --partition=<partition> slurm/train_smolvla_cube_flip.slurm
sbatch --account=<account> --partition=<partition> slurm/train_smolvla_green_into_tray.slurm
sbatch --account=<account> --partition=<partition> --time=16:00:00 slurm/train_smolvla_multitask.slurm
# larger batch (fits in 96 GB; give it more time):  BATCH_SIZE=64 sbatch --time=20:00:00 slurm/train_smolvla_cube_flip.slurm
```

Recommended: push the code and submit; each job generates its dataset on the cluster (20 demos per placement, AV1,
~0.5-1 h on the node before training). All four jobs can be submitted together: a per-dataset lock makes the
multitask job wait for a per-task dataset that a single-task job is generating, then reuse it. Copying the Mac
datasets (`rsync -a ~/.cache/huggingface/lerobot/local/so101_tri_* <cluster>:~/so101lm/datasets/lerobot/local/`)
skips the generation, but the job re-validates every dataset by replaying episodes in the simulator, and episodes
made on the Mac (arm64) may not replay exactly on the cluster's Linux nodes. A failed validation stops the job before
training.

The new dataset ids and output dirs differ from the first version's, so leftovers of it on the cluster are never
picked up. They can be deleted (paths assume the defaults `SCRATCH_ROOT=~/so101lm` and the project root as the
working directory):
```bash
rm -rf ~/so101lm/datasets/lerobot/local/so101_tri_{cube_left,green_into_box,cylinder_flip}_100* \
       ~/so101lm/datasets/lerobot/local/so101_tri_multitask_300* \
       ~/so101lm/datasets/lerobot/local/so101_real_cube{,_far}_white_tray_100* \
       outputs/train/smolvla_tri_{cube_left,green_into_box,cylinder_flip}_100 outputs/train/smolvla_tri_multitask_300 \
       outputs/train/smolvla_real_cube{,_far}_white_tray_100
```

TBD_TRAINCHECK

## Running a trained policy on the real SO-101

Copy the run from the cluster (`outputs/train/smolvla_tri_*`) to this Mac's `outputs/train/`, then:
```bash
RUN=smolvla_tri_cylinder_aside_100 CKPT=100000 TASK=cylinder_aside bash scripts/real_smolvla.sh
RUN=smolvla_tri_cube_flip_100 CKPT=100000 TASK=cube_flip bash scripts/real_smolvla.sh
RUN=smolvla_tri_green_into_tray_100 CKPT=100000 TASK=green_into_tray bash scripts/real_smolvla.sh
RUN=smolvla_tri_v2_multitask_300 CKPT=100000 TASK=cube_flip bash scripts/real_smolvla.sh   # one model, any of the three
```

The script sends the exact training instruction for each task id (any other `TASK` text is sent as is). Set up the
table as in simulation, measuring forward from the shoulder's pan axis (the vertical axis of the base motor) and
sideways from the robot's centre line:
- start with the arm folded at rest;
- white tray fixed with its centre 3.6 cm in front of the pan axis and 12.5 cm to the robot's right, long side
  pointing forward;
- gray cylinder standing 16.1 cm in front, 13 cm left;
- green block 18.6 cm in front, 3 cm left, flat face down, its long axis pointing forward;
- red cube 19.3 cm in front, 7.2 cm right, white face up, turned 20° so one face points at the pan axis;
- or that group of three shifted by 1 cm;
- keep the aside zone clear: 1-8 cm in front of the pan axis, 12-19 cm left.

Before the real arm, `python scripts/real/run_policy.py --robot sim --task-id cube_flip --policy <checkpoint>` runs the
same policy loop in the simulator.

## Files

Suite files:
- `so101_libero/tasks/tri_tasks.py`: scene configuration, directions, placements, tolerances and the three tasks
  (registered in `tasks/registry.py`, slots 13-15).
- `so101_libero/expert/tri_tasks.py`: the three demonstration controllers.
- `so101_libero/cli/render_placements.py`: renders and checks of the placements (spacing, clearances, image, experts).
- `so101_libero/cli/merge_datasets.py`: the combined multitask dataset.
- `so101_libero/train/lerobot_train_balanced.py`: task-balanced `lerobot-train`.
- `slurm/train_smolvla_{cylinder_aside,cube_flip,green_into_tray,multitask}.slurm` and `slurm/tri_tasks.env`.
- `init_states/{cylinder_aside,cube_flip,green_into_tray}.json`.
- `docs/TRI_TASKS.md`, `docs/media/tri_tasks/`, `docs/results/tri_tasks/`.

Shared code extended for it (the remaining tasks are unchanged: their scene and rules hashes still match
`init_states/`):
- `sim/objects.py`: half-cylinder mesh shape, containers fixed to the table, per-object contact softness, an optional
  visual-only top-face colour.
- `sim/scene.py`: mesh assets and fixed containers.
- `tasks/base.py`: `FixedPlacementLayout`; predicates `RestingInContainer`, `TurnedOnSide` and `Undisturbed`; plans
  `PlaceInBox`, `PlaceUprightAt` and `TipOver`; start-orientation tracking.
- `tasks/registry.py`: seed slots that survive removals (`retire`, `seed_slot`).
- `env.py`: `placement_id`.
- `expert/rollout.py`: expert dispatch by plan, placement ids, grasp monitor through the roll.
- `cli/generate.py`: `--demos-per-placement`, per-task episode limit, `--window` / `--encode-in-process` (memory),
  `--vcodec` (H.264 on macOS).
- `data/writer.py`: per-frame episode metadata.
- `cli/{render_video,make_init_states,check_reproducibility,evaluate}.py`: placements, seed slots, expert dispatch.
- `scripts/real_smolvla.sh`: the three task ids and their instructions.
- `slurm/smolvla_job.sbatch` (renamed from `smolvla_red_cube_to_white_tray.sbatch`): required `TASKS` / `REPO_ID` /
  `OUT`, plus `PRETRAINED`, `TRAIN_LAUNCHER`, `DATASET_ROOT`, `MERGE_FROM`, `ENV_ACTIVATE` and a per-dataset generation
  lock. `smolvla_red_cube_to_black_box.sbatch` and `smolvla_stack_red_on_blue.sbatch` run it with their old defaults.

Removed in this version: the tasks `cube_left`, `green_into_box`, `cylinder_flip`, `red_cube_to_white_tray` and
`red_cube_far_to_white_tray` with their predicates and plans (`MovedBy`, `FlippedInPlace`, `MoveBy`,
`FlipViaStation`), the white tray of `sim/objects.py`, their init states, SLURM jobs, media and results.

## Limitations

- **The cube flip ends with a ~2 cm drop.** After the 180° roll, the wrist-camera mount (4-8 cm beside the roll axis)
  hangs below the TCP on the robot side, so the gripper cannot lower the cube to the table from its middle. It
  releases at the lowest collision-free height and the cube falls ~2 cm onto its new face. All 50 test attempts land
  flat within 2 cm, but a real cube may bounce differently.
- **Only one flip direction is demonstrated.** The single tilted grasp always tips the cube towards the robot (white
  face ends facing the robot). The goal accepts any side, but the data shows one.
- **The cube's white face is visual only.** It has no thickness or mass. Mark the real cube's top face the same way;
  without the mark, "on its side" and "upright" look alike to the cameras.
- **The cube must face the pan axis.** It is turned -20.4° so that the 45° tilted grasp closes across two faces. A
  real cube placed square to the table edges is not the trained start state.
- Object masses and friction, the contact softness and the green block's stiffer contacts are not identified from
  the real objects. The 1 mm tray walls are thin collision geometry (fine for gentle contacts).
- The demonstrations are scripted and differ only in timing, heights, release points and (for the cylinder) the spot
  in the aside zone; the start states are the five placements. There are no recovery demonstrations: the base
  expert's `--max-grasp-retries` / `--miss-prob` are not wired into these controllers.
- Datasets collected on macOS (CGL rendering, H.264 video) are valid but not bit-identical to ones generated on the
  Linux cluster (EGL/OSMesa rendering, AV1 video); training decodes either with `--dataset.video_backend=pyav`. Do not
  merge datasets with different codecs (the merge refuses differing features).
