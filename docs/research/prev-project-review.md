# Review: `/scratch/e1583535/projects/lerobot-mini-project` (MuJoCo SO-101 lift)

**What I looked at.** I read all of `sim_mujoco/` and the training, eval and dataset-check scripts, plus `sim/leisaac_ext/act_policy.py`. I read every `logs/mujoco_pipeline/*.log`, every `outputs/**/eval/*.json`, and the dataset metadata under `/scratch/e1583535/datasets/lerobot/local/so101_*`. I also re-ran the expert with physics only, using a copy of the package in the scratchpad. Nothing in the project was modified.

## 1. Architecture

**Scene** (`env.py:_build`)
- Built with `MjSpec` on Menagerie `robotstudio_so101/so101.xml` at commit 0059d433. The robot is unchanged apart from recolouring.
- Added on top: an infinite table plane at `table_z`; floor, walls and props that are visual only (`contype=0`); a "key" directional light with shadows and a "fill" light; a free-joint `object`.
- The `front` camera is built from position, look-at point and roll. The wrist camera is Menagerie's `wrist_cam` (fovy 61), with an optional extra rotation, offset and fovy per profile.
- World frame: base at the origin, +x forward, +y to the robot's left, z up.

**Control**
- One env step is one 30 Hz frame, made of 6 `mj_step` calls at dt = 1/180 s (Menagerie's implicitfast integrator, elliptic cone, impratio 10).
- The action is 6 **absolute joint targets** in "sim units": each arm joint's MJCF range maps to −100..100 and the gripper's range maps to 0..100 (`env.py:258-268`). Targets are clipped to the actuator ctrlrange and sent to Menagerie's position actuators (kp 998.22, kv 2.731, forcerange ±2.94 N·m).
- Degrees per unit: shoulder_pan 1.1, shoulder_lift 1.0, elbow 0.97, wrist_flex 0.95, wrist_roll 1.6 (unit 0 = +2.8°), gripper 1.1 over −10°..100°.

**Objects**
- Default cube: 3 cm, 30 g, μ 1, condim 3, solref 0.01.
- Random-size objects: density 400 kg/m³, or 250–700 kg/m³ per episode (3 cm cube ≈ 6.8–18.9 g); μ 0.6–1.2; condim 4.
- `office3`: the real 3 cm cube or Ø3×3 cm cylinder; 20 % of training cells use 2.7–4.0 cm.

**Expert** (`expert.py`) — plans once at reset from the true object pose:
1. Joint-space smoothstep approach to a "high" point: fingertips at hover (0.07 m) + `approach_clearance` (0.06 m) above the object centre, fingers pointing down, jaw axis on a cube face or at the least wrist-roll change.
2. IK line down to hover, then IK line to the grasp point (fingertips 7–9 mm below the object centre).
3. Close over 0.6 s.
4. Lift with position-only IK (`w_rot=0`) until the fingertips are at 0.13 m.
5. Hold 0.6 s. Phase durations vary ±15 %.

IK is damped least squares (λ=1e-4, step clip 0.2 rad, 200 or 500 iterations) on site `gripperframe`. Plans that bring the wrist-camera mount within `min_camera_clearance` of the arm (35 mm in office2/office3, 15 mm in home2) are dropped. **There are no weld, mocap or teleport hacks**: the object qpos is written only at reset, so grasps are purely frictional.

**Success** (`env.py:461,502`): object centre more than `table_z + 0.10` m for 5 consecutive steps. Eval times out at 12 s (360 steps).

**Dataset** (`generate.py`)
- LeRobot v3.0, fps 30, `robot_type=so101_follower`.
- Features: `observation.state` (6), `action` (6), names `<joint>.pos`; `observation.images.front` and `.wrist`, 480×640 video.
- Video codec is AV1 (`libsvtav1`, yuv420p, crf 30, g 2), the LeRobot 0.4.2 defaults.
- Each frame stores the observation *before* the action. Only successful episodes are kept. `sim_mujoco_generation.json` and `sim_mujoco_failures.json` record per-episode seed, placement and randomization draws.

**Training** (`sim/train_act_hpc.sh`)
- `lerobot-train --policy.type=act --batch_size=8 --dataset.video_backend=pyav --num_workers=10`.
- Everything else is the ACT default (checked in `outputs/act_office3_cube/checkpoints/100000/pretrained_model/train_config.json`): `chunk_size=100`, **`n_action_steps=100`**, no temporal ensembling, ResNet18 with ImageNet weights, lr 1e-5, `kl_weight` 10, MEAN_STD normalization, `use_imagenet_stats=true`, image transforms off.
- Speed is about 0.043 s/step on an H200 (100k steps ≈ 72 min); final loss 0.020–0.027.
- `pbs/mujoco_pipeline.pbs` alternates training (`RESUME=1`) with an eval every `EVAL_FREQ` steps.

**Eval** (`eval_act.py`, which uses `LocalACTPolicy`)
- Loads with `ACTPolicy.from_pretrained`, then `make_pre_post_processors(policy_cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor":{"device":…}})`.
- Observations go through `prepare_observation_for_inference` (uint8 HWC → float/255 CHW), then `select_action` and the postprocessor.
- Images are raw renders. Seeds start at 100000, or held-out placements are used with `--split val|test`.

## 2. Evidence of results

**Expert success (generation logs):**

| Profile / object | Saved / attempts | Log |
|---|---|---|
| home cube | 300/305 (98.4 %) | 665907 |
| office2 cube (v1 and v2) | 300/300 | 667412, 667989 |
| office2 cylinder | 300/329 (91.2 %) and 300/326 (92.0 %) | 667413, 667990 |
| home2 cube | 100/100 | 668489 |
| office3 cube | train 80/81, val 10/11, test 10/10 (drops are all "unsafe") | 669447 |
| office3 cylinder | train 80/80, val 10/14, test 10/10 (drops are all "unsafe") | 669489 |

My physics-only re-run got 8/8 on every profile and 6/6 on office3 training placements. The `real` profile does not load any more: `assets/real/` was deleted in commit f7d5ee6.

**ACT success in sim** (`success_rate` in `outputs/**/eval/step_*.json`):

| Run | Demos | Eval episodes | Success over checkpoints |
|---|---|---|---|
| `legacy/act_home_cube` | 300 | 50 | 42, 50, 54, 56, 56, 52 % (50k–100k) |
| `legacy/act_office_cube` | 100 | 50 | 36–42 % |
| `legacy/act_office_cylinder` | 100 | 50 | 12–18 % |
| `legacy/act_office2_cube` (old raised start pose; README says not to use) | 300 | 50 | 54–82 % (best 82 % at 90k) |
| `legacy/act_office2_cylinder` (same old start pose) | 300 | 50 | 70–82 % |
| `act_office3_cube` | 80 | 20, held-out val cells | 5, 5, 10, 0, 5, 0, 5, 0, 5 % (10k–90k) |

- **office3 failures at 90k:** never_reached 12, missed_grasp 3, knocked_over 3, pushed_away 1. The policy mostly never gets to the object.
- **No sim result exists for four runs:** `act_office2_{cube,cylinder}_v2`, `act_home2_cube` and `act_office3_cylinder`. Every eval ended in `Aborted (core dumped)` with no Python traceback (logs 667989, 667990, 668489, 669489; the first home eval, 665907, also aborted).
  - The pipeline swallows these failures (`|| echo WARNING`), so these runs trained for hours with no evaluation.
  - It depends on the node: evals aborted on hopper-17/20/21/30 and ran on hopper-15/16/19/24/26/28. Generation, which renders with EGL but has no CUDA, worked on the same nodes.
- **`act_office3_cube` never reached its test split.** At 100k it died with `OSError: [Errno 28] No space left on device` inside `wandb_logger.log_policy`, while staging the W&B model artifact.
- **No real-arm success numbers** for any MuJoCo-trained policy are recorded in the repo.

## 3. Correctness concerns (most serious first)

1. **The expert targets approach poses it cannot reach.** With `approach_clearance=0.06` (home, office, office2, office3), the "high" point is out of reach near the base. IK returns a compromise 9–25 cm away from it.
   - Median miss: home 111 mm, office 93 mm, office2 125–134 mm, office3 116–120 mm. For comparison, default is 0.1 mm and home2 19 mm.
   - Example (office3, training cell 0): target fingertip height 0.145 m, reached 0.223 m with the fingers 49° off vertical; the elbow swings from 80° to −73° during the approach.
   - Planned joint speeds have a median of 200–250 °/s and a maximum of 450 °/s (default profile: 80–100 °/s).
   - 31 of 80 office3 demos contain target jumps above 8 units/frame. The worst are shoulder_lift 15.9 units/frame (episode 78, frame 52) and wrist_roll about 10 units/frame during the lift (episode 8, frames 137–141), because the position-only lift leaves wrist roll free.
   - wrist_flex is driven to its +100 limit.
   - The real-arm runner caps speed at 90 °/s (`local/run_act_policy.py`, `MAX_SPEED`).
   - The `expert_ik_error` metadata (up to 0.15 m / 1.0) reflects this, but it also includes meaningless rotation error from the `w_rot=0` lift.

2. **Grasps are far easier in sim than on the real arm.**
   - Menagerie's gripper geoms have `priority=1`, condim 6 and μ 1. So every jaw–object contact uses μ 1 whatever the object's `friction_range` (measured contact friction `[1,1,0.005,5e-4,5e-4]`). The friction randomization only affects object–table contact.
   - The close target (8 units ≈ −1.2°) is well past contact, so the gripper actuator saturates at 2.94 N·m. Summed normal force on a 14.5 g cube is 79 N (about 550× its weight), with 2–6 mm interpenetration.
   - The real LeRobot follower limits the gripper to 50 % torque and 50 % current (`so101_follower.py:158-165`).

3. **Success check is lenient.** There is no tilt or contact check and only a 5-step hold, and eval stops at the first success. Held objects end tilted 17–25° (median per profile), up to 84°. The `knocked_over` category (tilt > 45°) also counts a cube resting on another face as knocked over.

4. **ACT runs 100-step chunks open-loop** (3.3 s, about 2 inferences per episode). `--n_action_steps` and temporal ensembling were never tried.

5. **Units are non-standard.** They are a linear map of the MJCF ranges, not LeRobot's `RANGE_M100_100`. The real-arm code converts through `use_degrees` assuming zero joint offsets, and passes the gripper's 0..100 straight through. The office3 start pose sets shoulder_lift to −100° (the model's limit) while the real arm reads −102.9°, so the sim start sits on the joint limit (state minimum −101.7).

6. **Not enough data for the office3 task.** office3 has 80 demos (one per cell) over a U-shaped region of about 17×38 cm, heavy appearance randomization (24 procedural table textures, random background RGB 0.15–0.95, light ±40 %, shadows p=0.5) and spatially held-out validation. office2 had 300 demos in a 5×5 cm square. Which factor drives the ~5 % vs ~80 % gap is UNVERIFIED.

7. **Image gaps between training and use.** Training decodes AV1 at crf 30, while eval and the real arm use raw frames. The wide-angle wrist lens is approximated by a pinhole camera with fovy 94–108, so barrel distortion is not modelled.

8. **Infrastructure.**
   - The cause of the eval aborts is UNVERIFIED. `mujoco/egl` picks the first EGL device that initializes, and `MUJOCO_EGL_DEVICE_ID` was never set.
   - With `TMPDIR` on NFS `/scratch`, multiprocessing cleanup prints hundreds of harmless `Errno 16 … '.__dpc…'` tracebacks per log.
   - W&B model-artifact staging ran out of disk (see Section 2).

9. **Minor.** Eval prints "seeds 100000.." even in split mode. The val/test *datasets* are generated but never used. One val placement is one that the expert itself rejects as unsafe.

**Checked and found OK**
- `action[t] ≈ state[t+1]`: office3 mean |Δ| is [0.21, 0.32, 0.40, 0.18, 0.04, 2.98] units, against [0.31, 0.68, 1.21, 0.43, 0.10, 3.30] for `state[t]`.
- Timestamps step by exactly 1/30 s.
- The policy sees only the two images and the joints; no privileged state leaks in.
- Training and eval use the same pre/post-processors.

## 4. Reusable ideas

**Front cameras** (640×480; roll is about the viewing axis; table at z=0 unless noted):

| Profile | Real camera | pos (m) | lookat (m) | fovy° | roll° | pitch down° |
|---|---|---|---|---|---|---|
| office3 | D435 RGB (OpenCV 1) on a stand in front, ~51 cm up | (0.3972, −0.0534, 0.5081) | (0.1840, −0.0685, 0.1700) | 42.5 | −1.71 | 57.7 |
| office2 | D435 31 cm above the work area | (0.2221, −0.0397, 0.3138) | (0.1328, 0.0018, 0.0304) | 42.5 | 18.2 | 70.8 |
| home2 | home webcam (OpenCV 0); table_z −0.042 | (0.3572, 0.0069, 0.3775) | (0.0257, 0.0287, −0.0422) | 39.7 | 1.4 | 51.6 |
| home | home webcam | (0.370, 0.099, 0.486) | (0.100, −0.021, −0.174) | 42.3 | 9.0 | 65.9 |
| office | office webcam, front-left | (0.290, 0.123, 0.414) | (−0.028, −0.247, −0.186) | 54.6 | −5.1 | 50.9 |
| default | — | (0.60, 0, 0.42) | (0.17, 0, 0.02) | 55 | 0 | 42.9 |

- **Per-episode camera jitter:** office3 and home2 ±1.5 cm / ±2°; office2 ±2 cm / ±3°.
- **Wrist camera:** extra rotation in the camera frame of (9.1, 1.0, 1.2)° for office2/office3 (jaw silhouette IoU 0.92 vs 0.64 according to the code; not checked by me). home2 uses (−2.3, 11.2, −5.5)° plus a position offset of (17, −3, 1) mm. Fitted fovy is 99–101.
- **Start poses read from the real arms** (arm degrees, then gripper 0–100):
  - office3: [3.4, −102.9, 96.0, 72.1, −92.0], 8
  - office2: [7.5, −97.5, 93.2, 70.0, −87.6], 0
  - home2: [1.6, −101.7 (sim uses −98.5), 96.9, 72.9, −7.3], 15.5
- **Real frame handling:** frames are centre-cropped to 4:3 and resized to 640×480 (`run_act_policy.py:read_rgb`), so vertical fovy is the number to match. That the D435's 640×480 mode keeps a 42.5° vertical FOV is UNVERIFIED.
- **How to verify the cameras:** render the folded pose and blend it over `sim_mujoco/assets/office3/photo_{front,wrist}_folded.png`. The reference fit is `assets/office3/front_fit_overlay.png` (13 poses, IoU 0.78). The 13 capture photos in `local/setup_captures/` are not on Hopper.

**Physics:** nothing was tuned beyond Menagerie defaults. The only values taken from the real rigs are the gripper close targets (office arm 8 units, home arm 15) and the object densities.

**EGL rendering on Hopper** (`sim/hpc_env.sh`, `sim_mujoco/setup_hpc.sh`)
- Run with `singularity exec -e --nv`.
- `/scratch/e1583535/opt/egl_shim` holds a copy of the host's `/usr/lib64/libGLdispatch.so.0` and a `10_nvidia.json` pointing at `libEGL_nvidia.so.0`; it still exists.
- Set `LD_LIBRARY_PATH=$EGL_SHIM/lib`, `__EGL_VENDOR_LIBRARY_FILENAMES`, `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.
- Measured 52–102 env steps/s (physics plus two 640×480 cameras) on an H200.
- To verify: run `tools/render_check.py` in a ~5 min GPU job. Also set `MUJOCO_EGL_DEVICE_ID`, add `PYTHONFAULTHANDLER=1`, and run eval as its own process whose failure stops the pipeline.

**PBS pattern**
- Headers: `-P CFP05-CF-002 -q auto -l select=1:ngpus=1 -j oe -k oed`. Each GPU comes with 12 CPUs and 225 GB.
- Job body: `cd $PBS_O_WORKDIR; module load singularity`, logs to `logs/<x>/$PBS_JOBID.log`.
- `qsub -v` splits on commas, so values cannot contain commas. Chain jobs with `-W depend=afterany:<id>`.
- Add `--wandb.disable_artifact=true`, and keep `TMPDIR` off NFS.

**Design ideas worth keeping**
- Held-out spatial cells (`placements.py`): k-means on a 5 mm grid into 80/10/10 cells, val/test chosen by farthest-point, 7.5 mm margin, seed offsets 0 / 1e7 / 2e7 per split.
- `workspace_scan.py`: derives the spawn region from where the object is on the table, at least 20 px inside the image, and liftable by the expert with the camera clear.
- Per-episode generation and failure metadata with the git commit, which makes datasets reproducible.
- Eval failure taxonomy.
- Per-camera colour gain and saturation applied after rendering.
- Procedural table textures that avoid the object's hue.

**Environment notes for whoever runs things next**
- The login node's `/tmp` (4.9 GB, shared) hit 100 % full during this review. Sibling downloads in `scratchpad/research` are about 0.9 GB.
- Inside the container on the login node, the new project's `env.sh` sets `MUJOCO_GL=osmesa`. A bare `import mujoco` then fails with `AttributeError: 'NoneType' object has no attribute 'glGetError'`. Setting `MUJOCO_GL=disable` or `SO101_RENDER=none` works for physics-only runs.

Analysis scripts are in `/tmp/claude-24441/-scratch-e1583535/cc20221c-31b8-4cc2-b268-3cd4267ef119/scratchpad/research/scripts/`: `analyze_expert.py`, `ik_detail.py`, `contact_check.py`, `ds_check.py`, `summarize_evals.py`, `multi.sh`.