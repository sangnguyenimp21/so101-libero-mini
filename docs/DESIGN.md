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
