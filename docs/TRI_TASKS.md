# so101_tri: three language-conditioned tasks in one scene

One tabletop scene (fixed white box, red cube, green half-cylinder, light gray cylinder), exactly five initial
placements, three tasks, 100 physically executed demonstrations per task, and four SmolVLA fine-tuning jobs. The
environment is built like `stack_red_on_blue`: a layout and tasks in the task registry, the scripted IK expert, the
`generate` CLI and the same SLURM driver. Definitions: `so101_libero/tasks/tri_tasks.py` (geometry, directions,
placements, tolerances, tasks) and `so101_libero/expert/tri_tasks.py` (demonstration controllers).

![placements](media/tri_tasks/placements.png)

Rows: the five placements; columns: front camera (the policy's 256x256 view), wrist camera (policy input), overview.
Example demonstrations (front | wrist, as recorded): [move the cube](media/tri_tasks/expert_cube_left_seed1000000_p0.mp4)
([key frames](media/tri_tasks/expert_cube_left_seed1000000_p0_keyframes.png)) ·
[green block into the box](media/tri_tasks/expert_green_into_box_seed1110000_p1.mp4)
([key frames](media/tri_tasks/expert_green_into_box_seed1110000_p1_keyframes.png)) ·
[flip the cylinder](media/tri_tasks/expert_cylinder_flip_seed1230000_p3.mp4)
([key frames](media/tri_tasks/expert_cylinder_flip_seed1230000_p3_keyframes.png)).

## Directions

All placements and goals are built from one `Directions` object (`TRI_SCENE.directions`), the robot's own frame, never
the table edges:

| name | world vector | meaning | in the front image |
|---|---|---|---|
| FRONT | +x | away from the robot base, towards the front camera | down |
| LEFT | +y | the robot's own left | right (the camera faces the robot) |
| RIGHT | -y | the robot's own right | left |

Mirroring the whole scene and its goals is one setting: `Directions(left=(0.0, -1.0))`.

## Scene (metres; world origin = robot base on the table, 3.9 cm behind the shoulder's pan axis)

| item | size | pose (default placement) | mass | collision geometry |
|---|---|---|---|---|
| white box | 15 cm wide (LEFT-RIGHT) x 9 cm deep (FRONT-BACK) x 5 cm tall outside; 1 mm walls and bottom, inside 14.8 x 8.8 x 4.9 cm; open top | centre (0.08, 0.14), fixed to the table, beside the robot base on its left (9.5 mm clear of the arm at rest) | welded (n/a) | 4 wall boxes + bottom box; the bottom's collision slab also reaches 1 cm into the table (invisible) |
| red cube | 3 cm | (0.16, 0.14): FRONT of the box, centred on it, 2 cm gap | 20 g | box |
| green block | half-cylinder, 3.5 cm diameter x 3 cm long, 1.75 cm tall, flat face down, extrusion axis along FRONT | (0.22, 0.14): 6 cm FRONT of the cube's centre | 15 g | exact convex mesh (32 arc segments), stiffer contacts (solref 0.005 s, priority 1) |
| gray cylinder | 3 cm diameter x 6 cm tall, upright, uniformly light gray | (0.22, 0.08): 6 cm RIGHT of the green block's centre | 30 g | cylinder |

Friction 1.0 (torsional 0.005, rolling 0.0001) like every other item in this project. Masses and friction are
assumptions, not measurements. The box is welded to the world (it is fixed in the real setup). The cameras are the
project's (measured front camera, wrist camera on the gripper's camera mount), unchanged.

**Five placements, nothing else random.** The cube, green block and cylinder move together; the box, table, robot base
and cameras never move; orientations never change; the arm starts in the fixed folded rest pose (as
`stack_red_on_blue`).

| id | name | group translation | cube centre |
|---|---|---|---|
| 0 | default | 0 | (0.16, 0.14) |
| 1 | left | 1 cm LEFT | (0.16, 0.15) |
| 2 | right | 1 cm RIGHT | (0.16, 0.13) |
| 3 | forward | 1 cm FRONT | (0.17, 0.14) |
| 4 | backward | 1 cm BACK | (0.15, 0.14) |

`env.reset(options={"placement_id": k})` selects a placement; without it the placement is drawn from the reset seed.
`info["placement_id"]` and `env.init_state["placement_id"]` report it. `python -m so101_libero.cli.render_placements`
checks all five ([results](results/tri_tasks/placements_check.json), all pass):
- settled positions equal the nominal ones (< 1 mm) and orientations are unchanged;
- the spacing is exactly 60.0 mm FRONT (green block from the cube) and 60.0 mm RIGHT (cylinder from the green block);
- every object rests on the table, touching only the table;
- every object, the box and the cube's 5 cm-left destination lie inside the 256x256 front image (the tightest is the
  green block's near edge in placement 3, 8 px from the bottom);
- 380-455 visible pixels per object;
- each task's expert succeeds from each placement.

## Tasks

All three use the same compiled scene and placements (`suite so101_tri`, registry indices 10-12). Success = every goal
predicate holds for 10 consecutive control steps (0.33 s, settled). Tolerances live in `TriTolerances`.

| task id | instruction | goal (privileged state, `tasks/base.py`) | horizon |
|---|---|---|---|
| `cube_left` | Pick up the red cube and place it 5 cm to the left. | `MovedBy`: displaced 5 cm LEFT of where it was at the start of the episode, ±1 cm along and ±1 cm FRONT-BACK; on the table; original top face up (≤ 15°); yaw within 15° (mod 90°); carried ≥ 2 cm high and released within 3 cm of where it rests (not shoved); no robot contact; < 2 cm/s; TCP ≥ 4 cm away. `Undisturbed(green, cylinder)`: each ≤ 2 cm moved, ≤ 20° tilted, untouched | 450 steps (15 s) |
| `green_into_box` | Pick up the green semicircular block and put it into the white box. | `RestingInContainer`: centre ≥ 3 mm inside the inner walls, resting on the box floor in any orientation (centre ≤ max half-extent + 5 mm above it); carried ≥ 2 cm high; released, at rest, TCP ≥ 4 cm away. `Undisturbed(cube, cylinder)` | 450 steps |
| `cylinder_flip` | Pick up the light gray cylinder at its middle, flip it upside down, and place it back in its original position. | `FlippedInPlace`: body axis within 15° of straight DOWN (the env tracks orientation, so this only holds after a real 180° turn; the cylinder looks identical either way up); centre ≤ 1.5 cm from its start, standing on an end (height ± 4 mm); lifted ≥ 3 cm clear of the table; released, at rest, TCP ≥ 4 cm away. `Undisturbed(cube, green)` | 1200 steps (40 s) |

The environment also records each object's lowest body-axis z (`world.min_up_z`) every step, so lifting and replacing
the cylinder is distinguishable from flipping it.

## Demonstration controllers (`so101_libero/expert/tri_tasks.py`)

The expert stack is the project's: damped-least-squares IK at every control step with checked convergence, min-jerk
timing under the 75°/s joint-speed cap, and collision-checked grasp poses. Physics checks run at every phase boundary
(grasped, lifted, still held, placed). Actions are absolute joint targets through the position actuators; grasps are
purely frictional (nothing is welded, attached or teleported).
- **cube_left:** unfold from rest; top-down grasp across the left/right faces (clear of the box wall behind and the
  green block in front); carry with the jaw yaw fixed in the world (the cube keeps its yaw); put it down 5 cm LEFT of
  its start (±2 mm jitter; where it sits in the jaws is measured and compensated); release; park at rest; 2 s hold.
- **green_into_box:** top-down grasp across the two flat semicircular end faces; carry over the 5 cm walls; turn the
  jaws along the box's long side; lower to 5-9 mm above the floor at a reachable spot inside the box (searched on a
  grid, jaws clear of the walls); release; park at rest.
- **cylinder_flip (three grasps).** A one-grasp flip is impossible for this arm in view, so the demo uses a regrasp
  station (the design you chose):
  1. Top-down grasp at the cylinder's middle, carry it to a station 30 cm in front of the pan axis (just beyond the
     front camera's table view; the wrist camera sees it) and stand it there. While the cylinder is held, the gripper
     keeps one fixed 30° outward tilt with tangential jaws, so it stays vertical.
  2. Horizontal side grasp at its middle; lift; roll the wrist 180° about the horizontal approach axis; release it on
     its other end.
  3. Top-down grasp again; carry it back to its start; release; end hovering above it (see Limitations for why it
     doesn't park).

  The stages are told apart by arm pose and jaw opening (32° approach, 20° table release, 40° side release).

Expert success from the start states (physics only, 10 seeds x 5 placements): cube_left 50/50, green_into_box 50/50,
cylinder_flip 47/50 (the 3 failures topple after the station drop). Failed attempts are discarded and retried with
the next seed.

## Demonstration collection

```bash
# local runtime matching the cluster (python 3.10, lerobot[smolvla] 0.4.2, transformers 4.57.6, mujoco 3.3.7):
conda create -n so101lm -c conda-forge python=3.10 ffmpeg=7.1.1 && conda activate so101lm
pip install "lerobot[smolvla]==0.4.2" "transformers==4.57.6" "mujoco==3.3.7" threadpoolctl
# (on the cluster: bash scripts/setup_conda_env.sh, or simply submit a training job, which runs it)

# look at the scene / launch the environment
MUJOCO_GL=cgl python -m so101_libero.cli.render_placements --task cube_left --out outputs/placements --expert-check
MUJOCO_GL=cgl python -m so101_libero.cli.render_video --task cylinder_flip --seeds 1200000 --placement 0 \
    --max-steps 1500 --out outputs/videos
python -c "import so101_libero; from so101_libero.env import SO101TabletopEnv; env = SO101TabletopEnv('cube_left'); \
obs, info = env.reset(options={'placement_id': 2}); print(info, {k: v.shape for k, v in obs.items()})"

# 100 successful demonstrations per task = exactly 20 per placement (MUJOCO_GL=cgl on macOS, egl/osmesa on Linux).
# Cluster / Linux: the defaults (AV1 video, LeRobot's parallel encoders) are fine:
python -m so101_libero.cli.generate --tasks cube_left --demos-per-placement 20 --workers 8 --repo-id local/so101_tri_cube_left_100
# macOS laptop (how the datasets below were made): H.264 video, because LeRobot's default SVT-AV1 3.0 encoder crashes
# intermittently on macOS when an encoder is destroyed (libdispatch semaphore trap, kills the process); and a small
# window, because finished episodes wait in memory until written (a flip demo holds ~300 MB of frames):
for t in cube_left green_into_box cylinder_flip; do
  python -m so101_libero.cli.generate --tasks $t --demos-per-placement 20 --workers 4 --window 4 --vcodec h264 \
      --encode-in-process --repo-id local/so101_tri_${t}_100
done
# combined multitask dataset (300 demonstrations)
python -m so101_libero.cli.merge_datasets --out-repo-id local/so101_tri_multitask_300 \
    --repo-ids local/so101_tri_cube_left_100 local/so101_tri_green_into_box_100 local/so101_tri_cylinder_flip_100
# inspection: LeRobot loader, features, timing, videos, open-loop replay of 10 episodes in the simulator
python -m so101_libero.cli.validate_dataset --repo-id local/so101_tri_multitask_300 --replay-episodes 10
```

Datasets go to `$HF_LEROBOT_HOME/<repo-id>` (LeRobot format v3.0, lerobot 0.4.2). On a SLURM node, a training job
generates any missing per-task dataset itself, with the same settings.

Every dataset holds the same policy interface as all other datasets in this project:
- `observation.images.front` and `observation.images.wrist`: 256x256 RGB video (AV1), in that order;
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

Seeds: task k uses `k * 100,000 + placement * 10,000 + i` (cube_left 1,000,000+, green 1,100,000+,
flip 1,200,000+). Each placement keeps its first 20 successes in seed order, independent of the worker count.
Evaluation states: `init_states/<task>.json`, 50 per task (10 per placement), expert-verified.

**Collected datasets** (this Mac, 2026-10-20, `MUJOCO_GL=cgl`, H.264; in `~/.cache/huggingface/lerobot/local/`;
details in [results/tri_tasks/datasets.json](results/tri_tasks/datasets.json)):

| repo id | episodes | per placement | frames | frames / episode | expert attempts | size |
|---|---|---|---|---|---|---|
| `local/so101_tri_cube_left_100` | 100 | 20 x 5 | 27,179 | 262-282 (9.1 s mean) | 100 (0 rejected) | 63 MB |
| `local/so101_tri_green_into_box_100` | 100 | 20 x 5 | 30,017 | 283-318 (10.0 s) | 100 (0 rejected) | 73 MB |
| `local/so101_tri_cylinder_flip_100` | 100 | 20 x 5 | 80,244 | 777-830 (26.7 s) | 106 (6 toppled after the station drop) | 182 MB |
| `local/so101_tri_multitask_300` (merged) | 300 | 20 x 5 x 3 tasks | 137,440 | | | 314 MB |

Inspection (`validate_dataset`, [results](results/tri_tasks/)) passes for all four datasets:
- LeRobot loader, features and dtypes, contiguous frames at 30 fps, decodable videos, one side-file line per episode
  and unique seeds;
- per-frame `placement_id` / `collection_seed` match the side file;
- arm tracking under 2° mean lag;
- open-loop replay of 10 episodes per task (12 for the merged set, spread over all placements): the replayed joint
  states equal the recorded ones exactly, and the goal holds at the end.

Frozen evaluation states: the expert solves 30/30 ([results](results/tri_tasks/eval_expert_frozen_states.json)).
A balanced training run on the merged set gives each task 80,244 sample slots per epoch (flip frames are 58 % of the
data).

## Multitask data and sampling

lerobot 0.4.2 cannot train on several repo ids (`make_dataset` raises `NotImplementedError`), so `merge_datasets`
builds one combined dataset with LeRobot's own `aggregate_datasets`:
- tasks are re-indexed by instruction string, so every frame keeps its own instruction;
- it first checks that fps, robot type and features (camera keys and order, joint names) are identical;
- normalization statistics are aggregated by LeRobot.

A flip demonstration has ~3x the frames of the others, so plain uniform frame sampling would see the flip task ~60 %
of the time. The multitask job therefore trains with `python -m so101_libero.train.lerobot_train_balanced` instead.
This is `lerobot-train` itself (same arguments, checkpoints and resume), with the dataset wrapped so that every task
has the same number of sample slots per epoch. Each sample is still a real frame with its own instruction. The mapping
is written to `<output_dir>/task_balancing.json`.

## SmolVLA fine-tuning (SLURM, one H100 96 GB, one job per run)

| script | data | output dir / log |
|---|---|---|
| `slurm/train_smolvla_cube_left.slurm` | 100 cube demos | `outputs/train/smolvla_tri_cube_left_100` / `liberomini-so101lm-smolvla-cube-left-<jobid>.out` |
| `slurm/train_smolvla_green_into_box.slurm` | 100 green-block demos | `outputs/train/smolvla_tri_green_into_box_100` / `liberomini-so101lm-smolvla-green-box-<jobid>.out` |
| `slurm/train_smolvla_cylinder_flip.slurm` | 100 flip demos | `outputs/train/smolvla_tri_cylinder_flip_100` / `liberomini-so101lm-smolvla-cyl-flip-<jobid>.out` |
| `slurm/train_smolvla_multitask.slurm` | all 300, tasks sampled equally | `outputs/train/smolvla_tri_multitask_300` / `liberomini-so101lm-smolvla-tri-multi-<jobid>.out` |

Each script is a thin wrapper, like `slurm/smolvla_stack_red_on_blue.sbatch`, around the existing driver
`slurm/smolvla_red_cube_to_white_tray.sbatch`. The driver's stages:
1. environment (conda env via `scripts/setup_conda_env.sh`, or `ENV_ACTIVATE`);
2. evaluation states;
3. dataset: used if present, else generated (per task) or merged (multitask);
4. validation (10 replayed episodes);
5. training;
6. optional evaluation (`EVAL_CKPTS`, default `none`).

Training settings:
- fine-tunes `PRETRAINED` (default `lerobot/smolvla_base`), never from scratch;
- `--steps=100000` optimizer updates, cosine LR decay over the same 100k steps;
- `--save_freq=20000`: checkpoints 020000, 040000, 060000, 080000 and 100000, all kept;
- BF16, batch 32 (configurable);
- front/wrist renamed to `camera1`/`camera2`.

Resubmitting the same command resumes from `checkpoints/last` (`--resume=true`, restoring the optimizer, scheduler and
RNG state). Site settings (scratch dir, environment activation, pretrained checkpoint, dataset ids, batch size,
account/partition) are in `slurm/tri_tasks.env`. Nothing is submitted automatically.

```bash
# from the project root on the cluster; account/partition either in slurm/tri_tasks.env (SBATCH_ACCOUNT /
# SBATCH_PARTITION, then `source slurm/tri_tasks.env` first) or on the command line:
sbatch --account=<account> --partition=<partition> slurm/train_smolvla_cube_left.slurm
sbatch --account=<account> --partition=<partition> slurm/train_smolvla_green_into_box.slurm
sbatch --account=<account> --partition=<partition> slurm/train_smolvla_cylinder_flip.slurm
sbatch --account=<account> --partition=<partition> slurm/train_smolvla_multitask.slurm
# larger batch (fits in 96 GB; give it more time):  BATCH_SIZE=64 sbatch --time=20:00:00 slurm/train_smolvla_multitask.slurm
```

Recommended: push the code and submit; each job generates its dataset on the cluster (20 demos per placement, AV1,
~0.5-1.5 h on the node before training). All four jobs can be submitted together: a per-dataset lock makes the
multitask job wait for a per-task dataset that a single-task job is generating, then reuse it. Copying the Mac datasets
(`rsync -a ~/.cache/huggingface/lerobot/local/so101_tri_* <cluster>:~/so101lm/datasets/lerobot/local/`) skips the
generation, but the job re-validates every dataset by replaying episodes in the simulator. Episodes made on the Mac
(arm64) may not replay exactly on the cluster's Linux nodes, especially the flip's station drop, and a failed
validation stops the job before training.

Checked against the installed lerobot 0.4.2 (local `so101lm` env): the exact argument list the driver builds ran 2
training steps on CPU through the balanced launcher on a merged 3-task dataset (equal slots per task, finite loss,
checkpoints written), resumed from `checkpoints/last` to step 3 (`training_state/` restored), and the checkpoint
loaded in `PolicyRunner` and produced 6-d actions in the new environment.

## Changed files

New:
- `so101_libero/tasks/tri_tasks.py`: scene configuration, directions, placements, tolerances and the three tasks
  (registered in `tasks/registry.py`).
- `so101_libero/expert/tri_tasks.py`: the three demonstration controllers.
- `so101_libero/cli/render_placements.py`: renders and checks of the placements.
- `so101_libero/cli/merge_datasets.py`: the combined multitask dataset.
- `so101_libero/train/lerobot_train_balanced.py`: task-balanced `lerobot-train`.
- `slurm/train_smolvla_{cube_left,green_into_box,cylinder_flip,multitask}.slurm` and `slurm/tri_tasks.env`.
- `init_states/{cube_left,green_into_box,cylinder_flip}.json`.
- `docs/TRI_TASKS.md`, `docs/media/tri_tasks/`, `docs/results/tri_tasks/`.

Extended (additive; existing tasks unchanged: their scene and rules hashes still match `init_states/`):
- `sim/objects.py`: half-cylinder mesh shape, containers fixed to the table, optional per-object contact softness.
- `sim/scene.py`: mesh assets and fixed containers.
- `tasks/base.py`: `FixedPlacementLayout`; `MovedBy`, `RestingInContainer`, `FlippedInPlace` and `Undisturbed`;
  plans `MoveBy`, `PlaceInBox` and `FlipViaStation`; start-orientation and flip tracking.
- `env.py`: `placement_id`.
- `expert/rollout.py`: expert dispatch by plan, placement ids, multi-grasp grasp monitor.
- `cli/generate.py`: `--demos-per-placement`, per-task episode limit, `--window` / `--encode-in-process` (memory),
  `--vcodec` (H.264 on macOS).
- `data/writer.py`: per-frame episode metadata.
- `cli/{render_video,make_init_states,check_reproducibility,evaluate}.py`: placements and expert dispatch.
- `slurm/smolvla_red_cube_to_white_tray.sbatch`: `PRETRAINED`, `TRAIN_LAUNCHER`, `DATASET_ROOT`, `MERGE_FROM` and
  `ENV_ACTIVATE`, with defaults that keep the existing jobs unchanged.

## Limitations

- **The flip cannot be a single grasp with this robot in view.** Picking and placing at table height after a 180° turn
  requires a horizontal side grasp at the cylinder's middle. With `wrist_flex` at ±95°, that grasp is reachable only
  ≥ ~27-29 cm from the pan axis, beyond the front camera's view of the table. A wider sim wrist range doesn't help:
  the arm then collides with its own base.
- **The station release is a drop.** After the 180° roll, the wrist-camera mount (4-8 cm beside the roll axis) hangs
  below the cylinder. The flipped cylinder is therefore released from the lowest collision-free height, a ~5.5 cm
  drop onto its end, with a snap release so the pad does not spin it. This is physical, but it is the least
  realistic step (~6 % of attempts topple and are discarded). The final placement at the original position is gentle.
- **The flip is the hardest task for a learned policy:**
  - ~27 s episodes with three grasps;
  - the station is outside the front view, so only the wrist camera sees it;
  - the cylinder looks the same either way up, so the stage must be inferred from the arm pose. For the same reason
    the flip demonstrations end hovering rather than parked at rest: a parked final frame would look exactly like the
    first one.
- Object masses and friction, the contact softness and the green block's stiffer contacts are not identified from
  the real objects. The 1 mm box walls are thin collision geometry (fine for gentle contacts).
- The demonstrations are scripted and differ only in timing, heights and release points (the start states are the
  five placements). There are no recovery demonstrations by default: the base expert's `--max-grasp-retries` /
  `--miss-prob` work for the cube and box tasks, not for the flip.
- Datasets collected on macOS (CGL rendering, H.264 video) are valid but not bit-identical to ones generated on the
  Linux cluster (EGL/OSMesa rendering, AV1 video); training decodes either with `--dataset.video_backend=pyav`. Do not
  merge datasets with different codecs (the merge refuses differing features).
