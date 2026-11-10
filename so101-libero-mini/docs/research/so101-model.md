# SO-101 MuJoCo model evaluation (MuJoCo 3.3.7, research only)

**Verdict:** vendor the Menagerie `robotstudio_so101` model. In a scripted top-down pick test it grasped a 3 cm cube and a Ø3 cm cylinder 10/10 times. TheRobotStudio's `so101_new_calib.xml` grasped 0/6, because its jaw collision shapes are convex hulls that fill the gap between the jaws.

**Paths.** Everything is under `R=/tmp/claude-24441/-scratch-e1583535/cc20221c-31b8-4cc2-b268-3cd4267ef119/scratchpad/research`:
- Clones: `$R/menagerie/mujoco_menagerie/robotstudio_so101` and `$R/soarm100/SO-ARM100/Simulation/SO101`.
- My scripts: `$R/scripts/{summary,jaw,jaw2,ik,workspace*,grasp,camera,specapi}.py`.
- Raw outputs: `$R/out/*.txt`.

**Note on `env.sh`:** the project's `scripts/env.sh` changed while I was working. It now picks osmesa automatically on the login node, and `import mujoco` then fails with `'NoneType' object has no attribute 'glGetError'`. Prefix calls with `SO101_RENDER=none`. Because of `singularity exec -e`, only the variables `env.sh` lists are passed into the container.

## 1. Sources

| | Menagerie `robotstudio_so101` | SO-ARM100 `Simulation/SO101` |
|---|---|---|
| Repo HEAD | `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35` (2026-10-14) | `a758567c3978dfeefe282ede0500085a48fe8f78` (2026-10-06) |
| Model history | `a0f6f50` initial (2025-12-19); `c96a32d` camera PCB, moved `wrist_cam` (2026-09-23); `3abca2e` wrist_roll upper-limit fix (2026-09-28) | new_calib.xml last changed `aec17bb` "Update actuator model params #141" (2025-11-27); old_calib `ac69024`; `*_camera.{xml,urdf}` `f555c6c` #176 (2026-09-06) |
| License | Apache-2.0 | Apache-2.0 |
| Files | `so101.xml`, `scene.xml`, `scene_box.xml`, 19 STL (18 MB) | `so101_{new,old}_calib.{xml,urdf}`, `so101_new_calib_camera.{xml,urdf}`, `so101_leader_new_calib.urdf`, `scene.xml`, `joints_properties.xml`, assets (19 MB) |

The Menagerie README says it copied `so101_new_calib.xml` at `aec17bb`, then:
- switched to implicitfast,
- used the default forcerange instead of 3.35,
- added primitive collisions plus gripper solver parameters,
- added a camera mount.

## 2. Model facts

**Kinematic tree.** Both new_calib models have the same kinematics; Menagerie only rounded the numbers.
- `base` has no collision geoms. Chain: `base` → `shoulder`[shoulder_pan] → `upper_arm`[shoulder_lift] → `lower_arm`[elbow_flex] → `wrist`[wrist_flex] → `gripper`[wrist_roll] → `moving_jaw_so101_v1`[gripper].
- Menagerie adds a welded child body `camera_mount` under `gripper`.
- Every hinge has local axis `0 0 1`.
- World axes at q=0, checked by FK:
  - pan is −z, so +pan swings the arm toward −y.
  - lift, elbow and wrist_flex are +y, so + lowers the TCP.
  - wrist_roll is −x, i.e. the approach direction.
  - gripper is about −y, and + opens.

**Joint ranges (new_calib):**

| Joint | Range (rad) | Range (deg) |
|---|---|---|
| shoulder_pan | ±1.91986 | ±110 |
| shoulder_lift | ±1.74533 | ±100 |
| elbow_flex | ±1.69 | ±96.83 (XML comment: "5-degree calibration offset") |
| wrist_flex | ±1.65806 | ±95 |
| wrist_roll | −2.74385…+2.84121 | −157.2…+162.8 |
| gripper | −0.17453…+1.74533 | −10…+100 |

**old_calib differences:**
- shoulder_lift −190…+10°, elbow_flex −10…+180°, wrist_roll ±160°.
- Zero is the arm fully extended. An FK check gives new = old + (lift +90°, elbow −90°).
- Sites are renamed `base` / `gripper`.
- Actuators: kp 17.8, kv 0, forcerange ±3.35.

**Model comparison:**

| | Menagerie `so101.xml` | SO-ARM100 `so101_new_calib.xml` |
|---|---|---|
| Solver options | implicitfast, dt 0.005, elliptic, impratio 10, iterations 10 / ls 20 | none set, so defaults: Euler, 0.002, pyramidal, impratio 1 |
| Joint dynamics (class `sts3215`) | damping 0.60, armature 0.028, frictionloss 0.052 | same |
| Actuators | `position`, kp 998.22, kv 2.731, gear 1, ctrlrange = joint range, forcerange ±2.94 N·m (class default, = 30 kg·cm) | same kp/kv, but each actuator sets forcerange ±3.35 explicitly (stale `joints_properties.xml` still says kp 17.8) |
| Arm collision | 9 boxes, class `collision`, group 3, condim 3 | 13 full-resolution meshes (MuJoCo uses their convex hulls), group 3, contype/conaffinity 1, condim 3, solref 0.02 |
| Gripper collision | 18 primitives in class `collision_gripper` (group 3): thin plates `fixed_jaw_box4-7` and `moving_jaw_box2/3`, tip spheres r 0.75 mm, palm boxes, `camera_box1/2`. Plus 3 split hull meshes in class `collision_gripper_mesh` (group 4, maxhullvert 64). All condim 6, friction 1/5e-3/5e-4, solref 0.01 1, priority 1, mass 0 | fixed jaw = one hull of the whole `wrist_roll_follower_so101_v1` part |
| Visual geoms | 19 meshes in group 2, contype 0; lens cylinder in group 1 | 17 meshes in group 2 |
| Meshes | 18 (173,564 verts). `moving_jaw_so101_gripper_v1.stl` is declared but unused; `wrist_roll_follower_so101_gripper_v1.stl` is unreferenced | 13 (160,796 verts); camera variant 15 |
| Sites | `baseframe`; `gripperframe` on body `gripper` at (0.012, −0.0002, −0.0981), quat (0.7071, 0, 0.7071, 0): x-axis = approach, z-axis = closing direction (fixed → moving jaw) | `gripperframe` at (−0.0079, −0.0002, −0.0981), which sits on the fixed-jaw tip |
| Cameras | `wrist_cam` on `camera_mount`: pos (−0.00296, 0.05436, −0.05310), quat (0.958280, −0.285792, −0.001556, 0.004547), fovy 61 | none; the camera variant has camera meshes but no `<camera>` element |
| Keyframes | only `scene_box.xml` has one (`pickup`) | none |
| nq / nu | 6 / 6 (`scene_box.xml`: 13 / 6) | 6 / 6 |
| Compile time / step time | 0.85 s / 25 µs | 0.96 s / 26 µs |

## 3. Loaded-model results

**Forward kinematics:**
- qpos=0 is a "Γ" pose: upper arm at 76° elevation, forearm horizontal, gripper pointing +x.
  - Menagerie `gripperframe`: (0.391, −0.001, 0.246). Tip spheres: fixed (0.394, 0.000, 0.226), moving (0.395, −0.001, 0.243).
  - SO-ARM100 `gripperframe`: (0.391, 0.000, 0.227).
- The `pickup` keyframe has arm q = [0, 0.02, 27.1, 67.4, 90.8, 41.7]°. `gripperframe` is then at (0.208, 0.012, 0.014) with approach (−0.08, 0, −1.00).
- Holding a pose against gravity gives about 0.03° error and at most 0.57 N·m (shoulder_lift).
- LeIsaac's rest pose [0, −100, 90, 50, 0, −10]° is collision-free.
- At the elbow limit of +96.8° the forearm self-collides with the shoulder, so the real usable elbow maximum is about 90°.

**Workspace (Menagerie model, TCP = gripperframe):**
- 200k uniform random configurations give x −0.34…0.48, y ±0.44, z −0.22…0.53. Maximum horizontal reach is 0.44 m from the pan axis at (0.0388, 0).
- Top-down reach (approach −z) was checked with IK plus collision checks against a table at z=0, along y=0:

| TCP height z (m) | Reachable x (m) |
|---|---|
| 0.015 | 0.11–0.31 (0.17–0.31 if the wrist camera faces the base) |
| 0.05 | 0.17–0.29 |
| 0.08 | 0.155–0.25 |
| 0.09 | 0.19–0.215 |
| ≥ 0.10 | none (wrist_flex hits its 95° limit) |

- The full pan range of ±110° is reachable at radius 0.15–0.25 m.
- Example: at (0.22, 0, 0.015) with the jaws perpendicular to the radial direction, q = [3.7, −0.2, 26.5, 63.7, 96.5]°.
- A practical table patch is radius 0.13–0.27 m from the pan axis, e.g. x 0.15–0.29 and |y| ≤ 0.15.

**Jaw opening** (collision gap near the tips; Menagerie primitives match the visual meshes within about 1 mm):

| Gripper q (deg) | −10 | 0 | 5 | 10 | 15 | 20 | 30 | 45 |
|---|---|---|---|---|---|---|---|---|
| Gap (mm) | 2–2.5 | 16 | 22.5 | 29.5 | 36 | 43 | 54–58 | 72–79 |

- Near the tips the gap is roughly 16 mm + 1.37 mm per degree.
- The visual jaws touch at about −11.5…−12°, just beyond the −10° limit.
- At fully open (100°) the moving tip is about 111 mm from the fixed jaw's inner face.
- MuJoCo never simulates jaw-on-jaw contact (parent–child filter), so the −10° limit is the stop. Objects thinner than about 2.5 mm cannot be pinched.
- A 3 cm cube or cylinder fits from q ≈ 10° and is gripped at 9–10.7°. Opening to 25–30° gives margin.
- In the SO-ARM100 model, the fixed-jaw hull sticks 1.5 / 7.5 / 13 / 17 mm into the grasp space at 5 / 15 / 25 / 35 mm above the tip. At 25 mm above the tip, its effective gap at q = 10 / 20 / 30° is 17 / 26.5 / 37.5 mm, versus 33 / 43 / 54.5 mm in reality.

**Scripted pick test** (`grasp.py`):
- Procedure: damped-least-squares IK to a top-down pose, open 30°, command close to −10°, lift to z 0.065, hold 2 s, then move 10 cm sideways in 0.8 s.
- Objects: 30 g; random position x 0.16–0.26, |y| ≤ 0.10; random yaw.
- "Slip" is object movement relative to the gripper; "pen" is the deepest object–jaw penetration.

| Model / variant | 3 cm cube | Ø3×5 cm cylinder |
|---|---|---|
| Menagerie as-is (dt 5 ms) | 10/10; slip ≤ 0.9 mm (≤ 1.2 after transport); pen ≤ 2.9 mm | 10/10 |
| Menagerie, dt 2 ms, gripper force ±1.2 N·m | 10/10; slip ≤ 0.3 mm (≤ 0.7); pen ≤ 1.7 mm | 10/10 |
| Menagerie, gripper force ±1.0 N·m | 10/10 | 10/10 |
| Menagerie, 4 cm cube, open 40° | 8/10 (both failures: cube yawed ~43°, so a fingertip landed on it) | – |
| SO-ARM100 as-is | 0/6 | – |
| SO-ARM100 + implicitfast / elliptic / impratio 10, and with solref 0.01 on all geoms | 0/6 (also 0/6 with the extra solref change) | – |

- Why SO-ARM100 fails: each hull pair produces a single contact, and the fixed-hull contact normal tilts 22° ([0, 0.93, −0.37]). The jaws sink 15–20 mm into the cube and close fully to −10°.
- Why Menagerie works: it gets about 10 contacts per jaw, all with horizontal normals.
- With the TCP moved to x = 0.006 (too close to the fixed jaw), the cube score dropped to 7/10. Keep at least 4 mm clearance on the fixed-jaw side.

## 4. Mapping to the real LeRobot SO-101 (lerobot 0.4.2)

**Evidence:**
- SO-ARM100 `Simulation/SO101/README.md`:
  - "New Calibration (Default): Each joint's virtual zero is set to the **middle** of its joint range"
  - Old calibration zero is "fully extended horizontally"
  - Gripper note: "In LeRobot … `0` = fully closed, `100` = fully open. This mapping is **not yet reflected** in the current URDF and MuJoCo files."
- `robots/so101_follower/so101_follower.py`:
  - `norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100`
  - gripper is `Motor(6,"sts3215",MotorNormMode.RANGE_0_100)` in all modes
  - `calibrate()`: "Move … to the middle of its range of motion" → `set_half_turn_homings()` (sets that pose to 2047) → `record_ranges_of_motion()`
  - `configure()`: P_Coefficient 16, I 0, D 32; gripper Max_Torque_Limit 500 (50%), Protection_Current 250
  - config default is `use_degrees: bool = False`
- LeRobot docs `so101.mdx` (v0.4.2): "move the robot to the position where all joints are in the middle of their ranges".
- `motors/motors_bus.py::_normalize`:
  - DEGREES: `(val-(min+max)/2)*360/4095`
  - RANGE_M100_100: `(val-min)/(max-min)*200-100`
  - RANGE_0_100: `…*100`
  - So in every mode, LeRobot's zero is the midpoint of the **recorded** range.
- `model/kinematics.py::RobotKinematics(urdf_path, target_frame_name="gripper_frame_link", joint_names=None)`: `forward_kinematics(joint_pos_deg)` feeds `np.deg2rad(...)` straight into the URDF joints.
- The v0.4.2 examples `examples/phone_to_so100/teleoperate.py` and `so100_to_so100_EE` use `use_degrees=True` with `./SO101/so101_new_calib.urdf`.
- `lerobot_find_joint_limits.py` says "we are using the new calibration method for so101 and so100".

**Conclusion:**
- **Yes.** With `use_degrees=True`, `q_sim = np.deg2rad(q_lerobot)` for the five body joints, which is LeRobot's own convention.
- It is exact only if the operator's recorded min/max straddle the CAD range symmetrically. Expect a few degrees of offset per joint; check by commanding the q=0 "Γ" pose on the real arm.
- That joint signs match is implied by LeRobot using the URDF directly, but is UNVERIFIED on hardware.
- With the 0.4.2 default (`use_degrees=False`, values in ±100), use LeIsaac's affine map. Source: `LightwheelAI/leisaac@24d3bcd`, `utils/robot_utils.py::convert_lerobot_action_to_leisaac`, limits in `assets/robots/lerobot.py`:

```python
USD_DEG = dict(shoulder_pan=(-110,110), shoulder_lift=(-100,100), elbow_flex=(-100,90),
               wrist_flex=(-95,95), wrist_roll=(-160,160), gripper=(-10,100))
MOTOR = {**{k: (-100,100) for k in USD_DEG}, "gripper": (0,100)}
q_rad = np.deg2rad((v-m0)/(m1-m0)*(j1-j0)+j0)
```

**Gripper (0–100 in every mode):**
- `q_grip = deg2rad(-10 + 1.10*v)`.
- Checked against the geometry: v=0 → −10° (about 2 mm at the tips); v≈9 → 0° (16 mm); v≈20 → 12° (about 32 mm).
- Whether real 100 equals sim 100° is UNVERIFIED. Measure the tip gap with calipers at v = 0 / 25 / 50 and fit.

**Frame conventions:**
- LeRobot's URDF `gripper_frame_link` (rpy 0 π 0) has z = approach and sits on the fixed-jaw tip.
- Menagerie's site has x = approach and sits 2 cm toward the moving jaw.

## 5. Recommendation

**Vendor** Menagerie `robotstudio_so101` at `62ab00f`, unmodified, keeping its LICENSE and README. The two unused STLs (about 1.4 MB) can be dropped. Apply changes at load time instead of editing the file (this compiles in 3.3.7):

```python
spec = mujoco.MjSpec.from_file(".../robotstudio_so101/so101.xml")
spec.body("gripper").add_site(name="tcp", pos=[0.012,-0.0002,-0.0981],
                              quat=[0.7071068,0,0.7071068,0], group=4)  # x=approach, z=fixed→moving
spec.actuator("gripper").forcerange = [-1.2, 1.2]   # real gripper capped at 50% torque
spec.joint("elbow_flex").range = [-1.69, 1.5708]; spec.actuator("elbow_flex").ctrlrange = [-1.69, 1.5708]
spec.option.timestep = 0.002                        # 0.005 also passed all tests
```

**Physics settings:**
- Keep implicitfast, elliptic cones and impratio 10.
- Keep the gripper primitives as they are (priority 1, condim 6, friction 1/5e-3/5e-4, solref 0.01, which stays ≥ 2× the timestep).
- No noslip was needed in the tests.
- Objects: condim 3–4, friction about 1, 20–50 g, width at least 5 mm. Mount the base on a table plane at z=0.
- Do not use the SO-ARM100 MJCF for grasping; making it work would mean rebuilding Menagerie's jaw primitives.

**Grasp planning:**
- TCP x ≥ −0.008 + w/2 + 0.004 (w = object width), so the fixed jaw has at least 4 mm clearance.
- Open until the gap is at least w + 10 mm, i.e. q ≈ (w_mm − 6) / 1.37 degrees.
- Only go straight down below TCP z ≈ 0.08 m; above that, tilt the gripper.
- The sim actuators have no velocity limit, so cap commanded joint speed in the demo generator. The real servo speed is UNVERIFIED.

**Wrist camera:**
- Menagerie's `wrist_cam` sits on the +y face of the wrist-roll part, 54 mm off the roll axis and 53 mm toward the tips.
  - Its optical axis is tilted 33° from the approach axis toward the jaws.
  - In the grasp pose it is 6 cm above the table, and both tips appear in the lower half of a 640×480 image: fixed tip at (287, 352) px, moving tip at (459, 349) px.
- The community alternative is SO-ARM100's Hex-Nut 32×32 UVC mount (`so101_new_calib_camera.xml`).
  - The camera module sits at (0.0025, 0.073, 0.007) in the gripper frame.
  - Its lens points along body +z, which is (0, −0.424, −0.906) in the gripper frame (25° tilt).
  - That puts it about 3.6 cm further from the tips than Menagerie's camera.
  - To render from it, add `<camera name="wrist_cam" pos="0 0 0.027" quat="0 1 0 0" fovy="…"/>` inside body `wrist_camera`.
- SO-ARM100's README recommends the innomaker 32×32 UVC module run at 640×480 @ 30 fps.
- UNVERIFIED: which physical mount Menagerie's `wrist_roll_follower_so101_camera_mount.stl` represents, the real FOV, and the image-up direction. Measure these on the real arm and copy them into the camera definition.