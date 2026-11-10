# Design notes

Decisions behind the simulator, the expert and the data pipeline, including the problems found while building
them (each was reproduced, root-caused and fixed, not patched over).

## Simulation

- **Model.** MuJoCo Menagerie `robotstudio_so101` (derived from TheRobotStudio `so101_new_calib.xml`). Its jaws have
  10 (fixed) / 6 (moving) primitive collision geoms plus convex-hull meshes, with horizontal contact normals; TheRobotStudio's MJCF uses convex hulls of
  the jaw meshes, which fill the space between the jaws and cannot pinch a 3 cm object (0/6 grasps in a test).
- **Scene composition** with `mujoco.MjSpec` on the unmodified vendored XML. The table is a box with its top at
  z = 0; containers are free bodies (base plate + walls; the round one is a ring of 24 overlapping boxes) whose mass
  is split over their geoms by volume; objects are single primitives with condim 4.
- **Timing.** 30 Hz control (LeRobot SO-101 recording rate), 16 physics substeps of 2.083 ms, implicitfast,
  elliptic friction cones, impratio 10, 50 solver iterations.
- **multiccd.** Cylinder-on-box and box-on-cylinder contacts go through MuJoCo's general convex collider, which
  returns one contact point. Without `mjENBL_MULTICCD` the resting cylinder sank 2.5 mm into the table and kept
  rocking (0.3-1 rad/s measured), the round container sank 5 mm and wobbled, and an object released in the tray kept
  a ~0.06 m/s velocity, so the "at rest" success test failed randomly (2/10 and later 3/10 expert episodes). With
  it everything rests.
- **Gripper force.** The model's 2.94 N·m actuator limit pinches with ~80 N (old project measurement) and mm-deep
  interpenetration; the real LeRobot follower runs the gripper at 50 % torque. 1.2 N·m gives ~33 N total normal
  force, ~1.1-1.4 mm penetration and 0.5-2.2 mm slip of the object in the jaws during carrying.
- **Elbow limit.** Above ~90° the forearm hits the shoulder, so the elbow range/ctrlrange is capped at 90°.
- **Rendering.** EGL on GPU nodes (2-5 ms per frame), OSMesa/llvmpipe on CPU (decimated visual meshes:
  ~90 + 45 ms per front + wrist frame with 4 threads on an idle node, 200-400 ms per pair under load). Renders of the same state differ by < 1/255
  on average (edges). EGL contexts next to CUDA aborted on some GPU nodes (hopper-24 in one process; hopper-30 with
  several worker processes although a one-process self-test passed), so evaluations on GPU nodes render with OSMesa
  and run the policy on the GPU. A rare native crash was also seen with CPU torch inference next to OSMesa; episodes
  therefore run in worker processes and a crashed episode is retried once (two attempts at most).

## Expert (`so101_libero/expert/`)

- **IK.** Damped least squares on the 5 arm joints with a 6-D error (position + orientation). Top-down grasps
  are exactly determined for this kinematics (pan = azimuth, lift/elbow = reach/height, flex = pitch, roll = jaw
  yaw). Tilted approaches must lie in the vertical plane through the **pan axis** (at x = 0.039 m, not the world
  origin); using the world origin made every off-axis tilted target unreachable.
- **Reachability.** A vertical gripper is reachable up to ~8 cm TCP height for radii 0.13-0.22 m from the pan
  axis; higher, wrist_flex (±95°) runs out and the gripper must tilt outward (5-45°). The expert takes the least
  tilt that converges, chooses segment *end* poses 4° inside the joint limits (end poses on a limit made nearby
  poses infeasible) and follows each segment with a smooth tilt profile (endpoint interpolation plus a smooth bump
  when needed). An earlier per-step tilt search produced 7°/step joint jumps when a path crossed the limit.
- **Grasp.** The jaw closing direction is aligned with a cube face (any yaw for the cylinder), close to radial
  (small wrist roll); candidates whose open jaws come within 12 mm of another item are tried last, and candidates
  whose grasp / pre-grasp configuration collides (robot with itself, the table or another item) are skipped. The wrist-camera mount hits
  the shoulder when the object is close to the base and the roll is large (found from a stalled descent).
  The TCP approaches 6 mm towards the fixed-jaw side (~3.4 mm clearance to the fixed jaw's tip spheres), the jaws
  open to 32° (~5.3-5.6 cm gap), then the moving jaw pushes the object against the fixed jaw.
- **Motion.** Joint-space min-jerk move to the pre-grasp, then Cartesian straight lines / arcs around the pan axis
  with IK at every control step, durations from a 75°/s average joint speed (min-jerk peaks are ~1.9x higher and
  the ×0.9-1.15 timing randomization can shorten a segment by 10 %: measured peaks ≤ 157.5°/s) and ≤ 12-15 cm/s TCP
  speed. `--max-joint-speed` in `cli.generate` slows it down.
- **Checks.** grasp formed (both jaws touching), object lifted and still pinched after lift / transport / lowering,
  goal predicate at the end, arm tracking error (> 8° for 3 steps = blocked), joint-target jumps rejected at plan
  time. Any failure rejects the episode with a recorded reason (`meta/so101_libero_failures.jsonl`); seeds are
  tried in order with an attempt cap (default 2 x demos + 10).
- **Recovery demonstrations (optional, `--max-grasp-retries N --miss-prob p`).** A failed grasp (or a drop right
  after lifting) is retried: open, rise, re-plan from the object's current pose. With probability p the first
  attempt is deliberately offset 12-20 mm along the jaw width so the data contains recoveries. The offset attempt
  is recorded like any other expert action (in `so101_mini_t1_200_recovery`: 60/200 episodes, 9.4 % of frames);
  episodes record `deliberate_miss` and `regrasp_frames`, so these frames can be excluded later. Measured effect:
  ACT reaches 94 % at 40k steps instead of 74 %, and retries after a missed grasp (README Results).
- **Release.** Inside containers the jaws open only to 20° (4.2 cm), with the TCP shifted 1 cm towards the
  fixed jaw, so neither jaw touches the walls; on the table (relative placement) the object is lowered to 3-6 mm.

## Data

- Frame t = (observation before acting, action t). Actions are the absolute joint targets the expert sent;
  `validate_dataset` replays them open-loop from the stored initial state and reproduces every recorded state
  exactly (deterministic physics), which rules out timing or unit mismatches between observations and actions.
- Seeds: task k (registry order) trains on seeds `k * 100000 + i`; evaluation states use seeds ≥ 10,000,000.
  Regenerating from seeds reproduces bit-identical actions.
- Initial states where the goal already holds are resampled (needed for relative-placement goals).

## Adversarial review (2026-10-17) and resulting fixes

Six independent reviewers (privileged-information leaks, units/timing, success/physics, expert/IK, LeRobot API
usage, evaluation protocol/operations) each listed defects; every finding was then checked by a separate skeptic
that tried to refute it, usually with a reproduction script. Confirmed and fixed:

| Finding | Fix |
|---|---|
| `RelativeTo` could be satisfied by moving or holding the reference cube (39/50 eval states credited a success while the red cylinder never moved) | reference must rest on the table, untouched, within 1 cm of its start position |
| `OnTable` accepted objects on a container floor (+4.7 mm vs a 5 mm tolerance) | outside every container footprint, 2 mm tolerance |
| `InContainer` accepted flipped / tipped / held containers | container upright, resting on the table, untouched and at rest; object centre above the floor |
| MuJoCo's auto-reset after a blow-up teleported everything to the origin, which counted as success | divergence (warning counters or non-finite state) ends the episode as a failure |
| `classify()` crashed the whole evaluation for the relative-placement task | handles every plan type; per-episode errors are recorded as failures |
| scene hash differed between the first and later compiles (`content_type` attributes) and depended on the checkout path | stable, path-independent hash (+ mesh bytes) and a separate rules hash |
| robot base had no collision geometry | low box collider (+ base/shoulder exclusion); no demonstration touches it, datasets regenerate bit for bit |
| nothing kept training seeds out of the evaluation range | `generate` refuses seeds >= 10,000,000 |
| datasets did not record the scene / code; evaluation could not detect a renderer or scene change | datasets store scene/rules/code hashes, the expert config and a source snapshot; `evaluate` warns and records mismatches |
| replay validation only covered the first task of multi-task datasets | replayed episodes are spread over the dataset |
| inference overrides (`n_action_steps`, temporal ensembling) skipped LeRobot's config checks | the config's checks are re-run after overriding |
| resuming training silently reset `STEPS` to its default; relative paths resolved in the wrong directory | resume keeps the run's budget unless `STEPS` is set; paths made absolute |
| `train.pbs` evaluated with EGL even for OSMesa-rendered data, swallowed evaluation failures, could train on incomplete datasets | renderer taken from the dataset metadata, failures propagate to the exit code, completeness + validation gate |
| evaluation did not seed torch (Diffusion/SmolVLA sampling) and a crashed worker hung the pool | per-episode `torch.manual_seed`; episodes run in a `ProcessPoolExecutor` that is rebuilt after a crash, retrying only the running episodes |
| IK fallback seeds used pan = +azimuth (the shoulder frame is flipped, so they pointed at the mirror side); tilted approaches were aimed at the TCP, which is 5 mm off the wrist-roll axis, so tilted poses with a non-radial jaw yaw were slightly inconsistent (rarely unreachable close to the base) | fixed; `ExpertConfig.legacy_ik=True` / `generate --legacy-expert` reproduces the datasets generated before the fix (verified 200/200 bit-identical) |

Accepted as is (documented): images/contacts lag `observation.state` by one 2 ms physics substep (recomputing the
kinematics would change what the expert reads and break bit-exact regeneration; ≤ 0.35° of motion); expert speed
limits are per-segment averages (min-jerk peaks are ~1.9x); AV1 compression of training videos vs raw renders at
evaluation (LeRobot's encoder settings are fixed); WANDB project now passed explicitly.
Rejected after verification: "ACT silently ignores the instruction" (true, but intended: it is the no-language
baseline, and evaluation now flags it), "side files are dropped by lerobot-edit-dataset" (only with tools this
project does not use).

## Second verification round (2026-10-17, after the policies were trained)

A workflow ran every documented command in a sandbox, audited the docs against the code and the requested
deliverables, and reviewed the code added after the first review (each finding re-checked by a skeptic). Fixed:
evaluation now always runs episodes in worker processes (a single-worker run could lose everything to a native
crash), reports policy-loading errors per episode, never writes a `results.json` for a run in which nothing ran,
and charges a crash only to the episodes that were running; the grasp monitor is reset on a regrasp (recovery
episodes had reported ~60 mm "slip" from the failed attempt; the dataset side files were corrected by replay);
the generator archives the sources read at start; the `train.pbs` validation cache must match the dataset; numpy's
OpenBLAS is pinned to 2 threads (a 1-thread pool rounds differently and 64-thread pools wasted CPU); Diffusion's
`n_action_steps` limit is checked; `smoke.pbs` evaluates with OSMesa; `generate.pbs` checks every task's demo count;
an EGL-shim setup script was added; numbers and commands in the README were corrected (peak speed, expert-eval
command, slip statistics, stale render reference).

A final audit (numbers vs stored results, fresh-eyes README review, command spot-checks) found that the crash-retry
cap in `evaluate` never triggered (the crashed episode was removed from the running set before its result raised),
fixed and covered by `tests/test_eval_crash_retry.py`; it also corrected ~20 small numbers in the docs and made the
README examples safe to copy (new output names). Historical measurements quoted above (e.g. 0/6 grasps with the
TheRobotStudio MJCF, the 7°/step jumps, 39/50 exploit successes) come from development runs that were not stored.

## Real-setup round (2026-10-17 afternoon): matching the physical robot, task `red_cube_aside`

Request: make the simulation match the user's physical setup (`docs/real_setup/PROMPT.md`: near-black arm, the front
camera as measured on the real rig) and train a deployable policy for "pick up the red cube and place it aside".

- **Scene.** `CameraSpec` gained a `roll` (degrees about the viewing axis, applied after the look-at orientation);
  `FRONT_CAMERA` is now the measured camera: position (0.374, 0.009, 0.557) m, looking at (0.005, 0.003, 0), roll
  2.3°, fovy 45° (vertical FOV of the square image the policy sees: the real pipeline takes the centre square of the
  camera frame and resizes it to 256x256, which the simulated camera renders directly). All robot materials are
  overridden to matte near black (`SceneConfig.robot_rgba`, specular 0.15, shininess 0.2); table, floor, items,
  lights and the wrist camera are unchanged. Physics is untouched, so the frozen evaluation states of the old tasks
  were rebuilt with identical physical states (0 rejected candidates, same seeds) under the new scene hash.
  `outputs/sim_front_home.png` is the home-pose check render (robot upper middle, steep view, workspace below).
- **Visible workspace.** With this camera the table is visible up to x ≈ 0.26 m in front of the base (the bottom
  edge of the image) and about ±0.25 m sideways, so the old start regions (x up to 0.26) put items on the image
  edge; the new task keeps items at x ≤ 0.24.
- **Task** `red_cube_aside` (suite `so101_real`, registry index 4 → training seeds 400000+, evaluation seeds
  10,400,000+): a single 3 cm red cube (`RED_CUBE`, same mass/friction as the blue cube) on an otherwise empty
  table, start region x 0.14–0.24, y ±0.08, yaw ±45°. "Aside" is a fixed zone on the robot's left,
  `ASIDE_ZONE` = x 0.08–0.18, y 0.13–0.20 (the lower right of the front image). Nothing else is on the table: the
  user's demo table will hold only the cube (a policy trained with the trays as distractors would need their exact
  look; distractors are a possible later extension of the layout).
- **Goal predicate** `OnTableInRegion(obj, region)`: the cube rests on the table (not in a container, ≤ 2 mm
  height tolerance), upright (< 30° tilt), its centre inside the zone, released (no robot contact), at rest
  (< 5 cm/s), moved ≥ 4 cm from its start, and **was lifted ≥ 2 cm at some point in the episode**. The lift
  requirement is per-episode history, not a function of the current state alone: `WorldState.track()` (called by the
  environment after every control step and reset at `_apply_init`) keeps the highest point each object's bottom
  reached. Without it, pushing the cube into the zone would count as "pick up ... and place". The rules hash covers
  the predicate (regions included), so changing the zone invalidates the evaluation states as intended.
- **Expert plan** `PlaceAt(obj, region, margin=0.025)`: same grasp as the other tasks, then an arc at carry height to
  a random spot inside the zone shrunk by the margin (cube half-size + jitter), chosen among up to 20 candidates that
  are clear of other items and reachable (IK at carry and release height), then lowering to 3–6 mm above the table,
  a 20° release, retreat and hold. Physics-only check on 80 training seeds: 80/80 successes, 152–210 steps
  (5–7 s), max slip in the jaws 2.45 mm. With recovery demonstrations and a 3° start-pose noise (the settings used
  for the datasets below) 97/100: the 3 rejections were two unreachable pre-grasp/carry poses for a cube pushed to
  ~0.13 m from the base and one drop (all recorded, none saved).
- **Deliberate-miss offset.** The recovery mechanism's 12–20 mm sideways miss, tuned for the Ø3 cm cylinder,
  produced marginal edge grasps on the cube (3 drops in 60 episodes: both jaws touch the cube's edge, "grasped"
  holds, the cube slips during transport). `generate --miss-offset LO HI` (default unchanged) sets it; the cube
  datasets use 18–26 mm (≥ half width + 3 mm), which gives clean misses followed by re-grasps (25 re-grasps in 100
  episodes, 1 drop).
- **Start-pose noise.** `generate --home-noise-deg D` adds uniform noise to the five arm joints of the start pose
  (`EnvConfig.home_noise`); a real arm never starts exactly at the home pose. The evaluation states keep the exact
  home pose.
- **Evaluation states** for `red_cube_aside`: 50 states from 51 candidates (one rejected: "no collision-free grasp
  yaw", a cube close to the base at an unlucky yaw), expert 158–199 steps.
- **W&B.** Training logs go to the user's project (`WANDB=true WANDB_PROJECT=cs6283 WANDB_ENTITY=...` on
  `pbs/train.pbs`; `scripts/train.sh` passes `--wandb.entity/--wandb.mode`). GPU nodes reach api.wandb.ai (probe job
  `pbs/net_probe.pbs`, credentials from `~/.netrc` inside the container); `train.pbs` still checks reachability and
  falls back to offline logging with a `wandb sync` hint. After the evaluation, `scripts/wandb_log_eval.py` resumes
  the run and posts the simulated success rates, Wilson intervals, outcome counts and a few evaluation videos.
- **Appearance randomization** (`so101_libero/sim/randomize.py`, `generate/evaluate --visual-rand`,
  `EnvConfig.visual_rand`). Per episode, seeded by the episode seed (`default_rng([seed, 4242])`, image noise from
  `[seed, 4243]`), the model's visual fields are resampled in place: the two lights (position ±0.35 m, re-aimed at the
  workspace, diffuse/specular gain 0.6–1.2 with a ±10 % warm/cool tint), the headlight (0.7–1.15), the table material
  (0.7–1.06 brightness, ±8 % per channel), floor (0.6–1.4), robot materials (0.6–1.8 of the 0.12 near black), item
  colours (0.75–1.15, ±10 % per channel), the front camera (position and look-at ±1.5 cm, roll ±1.5°, fovy ±2°), the
  wrist camera (±3 mm, ±2°, fovy ±5°), plus per-episode image post-processing (contrast/brightness ±10 %, gaussian
  noise σ ≤ 3/255, a 30 % chance of a 0.3–0.8 px blur). The light/table ranges are narrower than first planned: the
  nominal scene already sits at the saturation knee (15 % of front-image pixels ≥ 250), and ×1.5 gains washed out
  the table in a tenth of the samples (measured on 96 samples: p90 saturation now 6 %, robot/table contrast ≥ 76/255,
  red/blue dominance of the objects never below 4× / 2×). Physics is untouched (qpos bit-identical with and without
  randomization, unit-tested), so datasets stay bit-reproducible in states and actions and the open-loop replay
  validation still applies; the appearance sample is stored per episode (`visual` in the episodes side file) and the
  config in `meta/so101_libero.json`. Camera and light placement live in `MjData` (`mj_camlight`), so the randomizer
  is given the data at reset: otherwise frame 0 of an episode would show the previous episode's camera (found by the
  module's author, 17/255 mean difference). The nominal evaluation states are evaluated both as they are and with
  `--visual-rand` (same physical states, randomized looks) to measure the appearance robustness of each policy.

### Review of this round (adversarial workflow, partly cut short by an API spend limit) and fixes

Confirmed and fixed:
- **"Placed" was under-specified.** `OnTableInRegion` accepted (a) a cube resting on *another* face as "not upright"
  (false negative, and `classify` called it "not settled"), (b) a lift of any kind (a flicked cube hopping 2 cm, no
  grasp) and (c) a lift somewhere else followed by shoving the cube into the zone, and (d) success 5 steps after the
  release with the jaws still around the cube (knocking it over while retreating would not count). Now
  `WorldState.track()` records the carry height and the held position **only while the object is pinched between
  both jaws**; the predicate requires the carry (≥ 2 cm), a release point within 5 cm of where the cube rests, the
  TCP ≥ 4 cm away from the cube, and boxes may rest on any face. `classify` uses the predicate's own definitions
  (`lift_below_min`, `in_zone_tipped_or_stacked`, `in_zone_but_pushed_after_release`, `in_zone_gripper_still_near`)
  and records `max_lift_cm`. The rules hash changed, so the evaluation states were rebuilt (same 50 states, expert
  158–199 steps); the nominal dataset's demonstrations satisfy the stricter goal (open-loop replay check).
- **Evaluations with episode errors looked complete.** `evaluate` now exits with code 2 when any episode errored
  (results are still written with `complete: false`), so `train.pbs`/`eval_real.pbs` flag the run.
- **W&B offline fallback** would have posted the evaluation into a second offline run; now the evaluation is posted
  only when online, and the job prints the exact `wandb sync` + `wandb_log_eval.py` commands at the end otherwise.
  A resumed training also receives `--wandb.mode`.
- **`scripts/camera_align.py` ignored `CameraSpec.roll`** in its sim camera matrix (numerically verified against
  `scene.camera_quat` now). Its printed marker sheet is placed 17.5 cm in front of the pan axis and would reach
  x ≈ 0.31 m, beyond the measured camera's view (x ≤ 0.26): use the outline mode (default) with this camera.
- Documented gaps: the simulated table extends 22 cm behind the base (the real one ends there); a cube at the
  far-left corner of the start region touches the bottom image border (its top face is always visible).

Infrastructure: the randomized-data generation job landed on hopper-30, where EGL aborts in worker processes (all
10 workers died, the pool then waited for the walltime). `pbs/generate.pbs` now runs `tests/egl_workers_selftest.py`
(spawned workers rendering with randomization) and falls back to OSMesa; `generate --episode-timeout` (900 s) fails
the run with exit code 3 instead of hanging when a worker dies.

## Task `red_cube_to_box` (2026-10-17 evening): variable cube size, black box on the left

- **Task.** "Pick up the red cube and put it in the black box." Goal `In(red_cube, black_tray)` (the existing tray,
  inner 12 × 9 cm, 2.5 cm walls; the user's choice for the real box). Layout `real_box_tabletop_v1`: cube start region as
  in `red_cube_aside`, tray centred at (0.15, 0.15) ± 2 cm, yaw ± 10°, i.e. the robot's left / lower right of the front
  image. At y > 0.17 the tray's outer wall left the front image, hence the region (renders of the four corner cases
  checked). Suite `so101_real`, registry index 5.
- **Per-episode size.** `ObjectSpec.scale_range` (`RED_CUBE_VAR`: 0.9-4/3 of the 3 cm cube = 2.7-4.0 cm edge).
  `Layout.sample_sizes` draws one factor per such object *after* the home-pose noise and *before* the poses (layouts
  without a range draw nothing, so every existing task keeps its random stream); footprints for the collision-free
  placement use the episode size. The size is stored in the initial state (`"sizes": {"red_cube": [hx, hy, hz]}`), so
  datasets' side files and the evaluation states carry it, and `SceneHandles.apply_sizes` writes it into the compiled
  model at `_apply_init`: geom size, bounding sphere, geom AABB, the body's BVH box and inertia (mass kept at 20 g,
  inertia always scaled from the compiled value so the model depends only on the requested size, not on earlier
  episodes), then `mj_setConst`. `SceneHandles.objects` returns the specs at the episode size, so predicates, the
  expert and the placement height all see it. Checked: an expert rollout with a runtime-resized cube equals one with a
  model compiled at that size to 4e-15 m; re-applying a stored state after another episode is bit-identical
  (`test_per_episode_cube_size_roundtrip`). Only uniform scaling of boxes is implemented.
- **Expert.** The approach and deliberate-miss offsets shift by the difference in half width from 3 cm (so the fixed
  jaw keeps its 6 mm clearance and misses stay clean), and the release opening inside containers grows by 1.1° per mm
  of extra width (4 cm cube: ~31° instead of 20°, the jaws would otherwise barely let go). All three are exactly
  unchanged for 3 cm objects, so existing datasets remain reproducible bit for bit. Physics-only check on 120 training
  seeds with the dataset settings (2 retries, 30 % deliberate misses, 3° home noise): 112/120 successes (156-278
  steps); failures are 5 × "no collision-free grasp yaw" and 2 × unreachable (cube near the base), 1 drop of a 3.96 cm
  cube. By size: 2.7-3.0 cm 18/18, 3.0-3.35 37/40, 3.35-3.7 31/32, 3.7-4.0 26/30.
- **Evaluation states**: 50 of 51 candidates (one "no collision-free grasp yaw"), edges 2.8-4.0 cm (mean 3.4), expert
  156-193 steps.
- **Training.** Dataset `local/so101_real_cube_box_500` (500 demos, same expert settings as `so101_real_cube_300`,
  nominal looks); SmolVLA from `smolvla_base`, 80k steps, batch 32, cosine decay over 80k (`--policy.scheduler_decay_steps`:
  the 30k-step run's learning rate reached its floor exactly at 30k, so its flat end was the schedule, not
  convergence), checkpoints every 20k all evaluated to measure what steps beyond 30k buy.

