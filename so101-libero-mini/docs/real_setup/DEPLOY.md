# Deploying the red-cube policy on the real SO-101

Task: **"Pick up the red cube and place it aside."** The policy picks the 3 cm red cube up from the table in front of
the robot and puts it down on a fixed spot on the robot's **left** (its right-hand side as the front camera sees it).
Trained purely in simulation (`so101_libero`, task `red_cube_aside`); nothing here has been run on hardware yet, so
the checklist below is what must be verified on the real rig before the first autonomous run.

## 1. What the policy expects

| item | value |
|---|---|
| scene | robot base at the table edge facing the workspace; **only the red cube** on the table, 14–24 cm in front of the base, within ±8 cm sideways; nothing else in the camera view (no trays, no other objects) |
| the "aside" zone | 8–18 cm in front of the base, 13–20 cm to the robot's **left** |
| front camera | as measured in `PROMPT.md`: 37 cm in front of the base, 56 cm above the table (67 cm from the base), looking at the base, rolled 2.3°, the policy image is the **centre square** of the frame resized to 256×256 (vertical field of view 45°); the robot stands in the upper middle of that square (`outputs/sim_front_home.png`, `docs/media/sim_front_home.png`) |
| wrist camera | the gripper camera, centre square of its frame resized to 256×256 |
| joints | LeRobot `so101_follower` with `use_degrees=true`: 5 arm joints in degrees (0 = middle of each calibrated range), gripper 0 (closed) … 100 (open) |
| start pose | home `[0, -60, 35, 85, 0]` with the gripper at 0; the demonstrations start within ±3° of it |
| gripper commands the policy produces | ~38 while approaching (jaw gap ≈ 5.3 cm in sim), 0 to grasp, ~27 to release: check that 38 opens the real jaws clearly wider than the cube and that 0 closes them on it |
| control | 30 Hz absolute joint targets; ACT chunks of 50 actions (1.7 s) executed between inferences; the demonstrations never change a joint by more than 5.3° per step, so a 5°/tick `max_relative_target` is consistent |
| episode | pick + place takes the expert 5–7 s; 20–30 s is plenty |

## 2. Checkpoint

`outputs/train/act_real_cube_300_dr/checkpoints/last/pretrained_model` (ACT, trained on 300 demonstrations with
randomized lighting, colours and camera jitter; see README → Results, real-setup rows, for the simulated success rates
of this and the nominal-appearance run `act_real_cube_300`). Both runs are logged in the W&B project `cs6283`.

## 3. Before the first run (once)

1. **Calibration zero.** Torque off, put the arm by hand in the simulator's home pose (base centred, upper arm leaning
   back, gripper pointing down over the table in front of the robot; compare with `docs/media/sim_front_home.png`),
   then `python scripts/real/run_policy.py --robot real --read-pose`: the printed differences to the home pose are
   per-joint offsets; a correct calibration gives a few degrees at most. Larger values: pass them as
   `--joint-offset NAME=DEG` (path B) or recalibrate.
2. **Front camera alignment.** `python scripts/camera_align.py live --cam 3 --width 1280 --height 720` (your tool:
   the sim robot outline in cyan must coincide with the real one in orange), or
   `python scripts/real/check_cameras.py --sim-reference outputs/sim_front_home.png` for a 50/50 blend with the sim
   render at the home pose.
3. **Cube and table.** Red cube of 3 cm, table clear, no strong shadows from people standing next to the camera.
4. **Dry run.** `python scripts/real/run_policy.py --policy <ckpt> --no-send --log-dir outputs/real/dry` (hardware
   connected, nothing sent): the first predicted action must be close to the measured pose (the script warns above
   15°); look at `outputs/real/dry/episode_000/policy_view.mp4` — it shows exactly what the policy sees.

## 4. Running

Path A, your existing wrapper (`lerobot-rollout` with the deploy copy that crops and resizes inside the preprocessor;
the arm starts from wherever it is, so place it near the home pose first, torque off):

```bash
RUN=act_real_cube_300_dr CKPT=last DURATION=30 bash scripts/real_act.sh
```

Path B, the standalone loop (same crop/resize code path as the simulator test, moves to the home pose first, per-step
safety clamp, logs what the policy saw):

```bash
# the same loop in simulation (no hardware)
SO101_RENDER=osmesa bash scripts/env.sh python scripts/real/run_policy.py --robot sim --task-id red_cube_aside \
    --policy outputs/train/act_real_cube_300_dr/checkpoints/last/pretrained_model --seconds 15 --log-dir outputs/real/sim_check
# the real arm (defaults: port /dev/tty.usbmodem5B610366661, front camera 3 at 1280x720, wrist camera 0 at 640x480)
python scripts/real/run_policy.py --policy outputs/train/act_real_cube_300_dr/checkpoints/last/pretrained_model \
    --log-dir outputs/real/run1 --seconds 25
```

`python scripts/real/run_policy.py --help` lists every option (device, camera sizes/indices, `--max-step-deg`,
`--max-relative-target`, `--joint-offset`, `--n-action-steps`, `--temporal-ensemble`, `--torque-off-at-exit`).

### SmolVLA alternative (`outputs/train/smolvla_real_cube_300`)

A language-conditioned SmolVLA fine-tuned from `lerobot/smolvla_base` on the same 300 demonstrations (30k steps,
batch 32; W&B `cs6283`). It must get the training instruction **"Pick up the red cube and place it aside."** (both
scripts default to it). It needs the SmolVLA extras (`transformers` etc.) in your LeRobot env, and it pauses briefly at
each 50-action chunk while it plans (a GPU or `mps` is strongly recommended; the first call warms up for several seconds).

```bash
bash scripts/real_smolvla.sh                       # lerobot-rollout path; RUN=smolvla_real_cube_300, TASK=red_cube_aside by default
python scripts/real/run_policy.py --policy outputs/train/smolvla_real_cube_300/checkpoints/last/pretrained_model \
    --log-dir outputs/real/smolvla_run1 --seconds 30   # standalone loop (go-home, clamps, logging)
```

## 5. If it does not work

| symptom | likely cause / what to try |
|---|---|
| arm moves to the wrong place from the first chunk | camera not aligned (step 3.2) or calibration offsets (3.1); the dry-run video shows the robot at the wrong place in the image |
| gripper does not close on the cube / opens too little | gripper 0..100 calibration differs from the sim mapping: check what 38 and 0 do on the real gripper |
| reaches the cube but misses the grasp, retries | expected to some degree; the policy was trained with recovery demonstrations (it re-opens, rises, tries again) |
| hesitates / stops between chunks | inference time: use `--device mps` (or cuda); ACT inference is < 20 ms on a GPU, ~200 ms on a CPU |
| jerky motion | lower `--max-step-deg` / `MAX_REL`, or `--n-action-steps 25` (re-plans every 0.8 s), or `--temporal-ensemble 0.01` (re-plans every step; in simulation this was worse than full chunks for the earlier task) |
| cube placed outside the zone | the zone is a fixed spot relative to the base: check the camera alignment and that the base stands where the camera measurement was made |

## 6. Simulation evidence

See README → Results (real-setup rows) and `docs/results/eval/act_real_cube_300*.json`: success rates on 50 frozen
initial states, nominal appearance and under randomized appearance. Expert videos: `docs/media/expert_red_cube_aside_*`.
