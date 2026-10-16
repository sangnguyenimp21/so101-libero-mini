# LIBERO design study for an SO-101 MuJoCo mini-benchmark

**Sources.** I read all of these directly:
- LIBERO at commit `8f1084e` (2025-03-15). The clone is at `/scratch/e1583535/tmp/cc/research/libero_clone/LIBERO` and is symlinked from `/tmp/claude-24441/-scratch-e1583535/cc20221c-31b8-4cc2-b268-3cd4267ef119/scratchpad/research/libero_clone`. LIBERO paths below are relative to `libero/libero/`.
- The versions LIBERO pins: robosuite 1.4.0 and bddl 1.0.1 (PyPI source tarballs).
- The installed LeRobot 0.4.2, plus LeRobot `main` (`envs/libero.py` and `docs/source/libero.mdx`).
- OpenVLA `experiments/robot/libero/run_libero_eval.py`.
- The HF files `meta/info.json`, `meta/stats.json` and `meta/tasks.parquet`.

## 1. Task spec (BDDL), initial-placement distributions, fixed init states

Task files live at `bddl_files/<suite>/<task>.bddl`. They are parsed by `envs/bddl_utils.py::robosuite_parse_problem`, which sits on `bddl.parsing.scan_tokens`. That function **lowercases the whole file**.

```lisp
(define (problem LIBERO_Tabletop_Manipulation)     ; -> scene class via TASK_MAPPING
 (:domain robosuite)
 (:language Pick the akita black bowl between the plate and the ramekin and place it on the plate)
 (:regions
   (plate_region (:target main_table) (:ranges ((0.05 0.19 0.07 0.21))))  ; xmin ymin xmax ymax; several rects = union
   (cabinet_region (:target main_table) (:ranges ((0.02 -0.28 0.04 -0.26))) (:yaw_rotation ((2.661 2.724))))
   (top_region (:target wooden_cabinet_1)))        ; no ranges -> site authored in the object MJCF
 (:fixtures main_table - table wooden_cabinet_1 - wooden_cabinet)
 (:objects akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl plate_1 - plate)
 (:obj_of_interest akita_black_bowl_1 plate_1)     ; used only for segmentation
 (:init (On akita_black_bowl_1 main_table_between_plate_ramekin_region) (On wooden_cabinet_1 main_table_cabinet_region))
 (:goal (And (On akita_black_bowl_1 plate_1))))
```

**Regions**
- A region's key is `<target>_<name>`.
- Table regions become invisible `TargetZone` box sites on the table body. Half-extents are rect/2, half-height is 0.007, and z is +0.02 (`envs/problems/libero_tabletop_manipulation.py::_load_sites_in_arena`).
- Object regions are box sites already written into the asset XML: `basket_1_contain_region`, `wooden_cabinet_1_top_region`, and `flat_stove_1_cook_region` (size `0.075 0.075 0.0025` in `assets/articulated_objects/flat_stove.xml`).
- The same site is used both to place objects and to check goals.

**How init atoms become samplers** (`envs/bddl_base_domain.py::_add_placement_initializer`)

| Init atom | Sampler |
|---|---|
| `(On obj table_region)` | `TableRegionSampler` |
| fixtures | `MultiRegionRandomSampler(rotation=yaw_rotation)` |
| `(On obj object_site)` | `SiteRegionRandomSampler` |
| `(On obj other_obj)` | `ObjectBasedSampler` (xy offset 0) |
| `(In obj site)` | `InSiteRegionRandomSampler` |
| `(Open\|Close\|Turnon\|Turnoff X)` | joint samplers over `default_*_ranges` |

- `yaw_rotation` is only applied to fixtures. Movable objects use the `rotation` set on their class, which is mostly a fixed upright offset.

**Sampling procedure** (`envs/regions/base_region_sampler.py::MultiRegionRandomSampler.sample`)
1. Pick one rect uniformly.
2. Sample x and y uniformly. With `ensure_object_boundary_in_range`, the range is first shrunk by the object's `horizontal_radius`.
3. Reject the sample if footprint circles overlap. Retry up to 5000 times, then raise `RandomizationError`. `ControlEnv.reset` retries without limit.
- `_reset_internal` runs: joint samplers, then `mj_step1`, then table/fixture placement, then site-conditional placement, then on-object placement.
- The RNG is the global `np.random`; `env.seed(s)` just calls `np.random.seed(s)`.

**The spread is tiny.** I loaded the 50 stored states of spatial task 0 with a restricted unpickler:
- Objects vary by about 3 cm in xy, even though the rects are 2 cm. That is consistent with the radius shrink inverting the bounds (low > high), which `np.random.uniform` accepts silently.
- Orientations are constant.
- Robot joints vary by about ±0.05 rad (robosuite `initialization_noise="default"` is Gaussian with σ = 0.02).

**Fixed eval states**
- `init_files/<suite>/<task>.pruned_init` is a `torch.save` of a float64 array of shape `(50, 1+nq+nv)`: 92 columns for spatial, 110 for object, 79 for goal; 10 and 90 vary.
- Each row is the flattened MuJoCo state `[time=0, qpos, qvel=0]`, captured before settling (objects at spawn height z = 0.97).
- `.init` files hold 100 states (spatial/10/90) or 50 (object). Goal has only `.pruned_init`.
- How states were pruned is UNVERIFIED; the repo has no generator script.
- Loading: `Benchmark.get_task_init_states(i)` calls `torch.load`.
- Applying: `ControlEnv.set_init_state(s)` does `sim.set_state_from_flattened(s); sim.forward(); check_success(); _post_process(); _update_observables(force=True)`.
- It must come **after** `env.reset()`. With `hard_reset=True`, a reset rebuilds the model and resamples placements.

**Why fixed states matter:** results no longer depend on sampler code, RNG streams or library versions. LIBERO's lifelong eval runs states 0–19 (env k gets state k). OpenVLA and LeRobot `main` run episode k from state k, so their numbers are comparable.

## 2. Goal predicates and success

From `envs/predicates/base_predicates.py`:
```python
class In(BinaryAtomic):
    def __call__(self, arg1, arg2): return arg2.check_contact(arg1) and arg2.check_contain(arg1)
class On(BinaryAtomic):
    def __call__(self, arg1, arg2): return arg2.check_ontop(arg1)
class Up(BinaryAtomic):            # absolute world z; no suite uses it
    def __call__(self, arg1): return arg1.get_geom_state()["pos"][2] >= 1.0
```

From `envs/object_states/base_object_states.py` and `envs/objects/site_object.py`:
```python
# ObjectState.check_ontop (object on object), using sim.data.body_xpos origins
return (this_pos[2] <= other_pos[2]) and self.check_contact(other) \
       and np.linalg.norm(this_pos[:2] - other_pos[:2]) < 0.03
# SiteObjectState.check_contact -> always True
# SiteObject.in_box:
total_size = np.abs(this_mat @ self.size); ub = pos + total_size; lb = pos - total_size; lb[2] -= 0.01
return np.all(other_position > lb) and np.all(other_position < ub)
# SiteObject.under (On with a site): d = this_mat @ (p_obj - p_site)
#   size_z-0.005 < d_z < size_z+0.10 and |d_xy| < size_xy, plus contact(parent, obj) if the site has a parent
```

**What these mean**
- `In` is true when the object's **body origin** is inside the region's approximate box.
- `In` only works with site regions: `ObjectState.check_contain` calls `in_box`, which only the site classes define.
- `On(obj, object)` requires contact, the object being above, and xy centering within 3 cm.
- `On(obj, region)` is a point-in-slab test above the site.
- Contact uses robosuite's `MujocoEnv.check_contact`, which scans `sim.data.contact[:ncon]` for geom names in each model's `contact_geoms`.
- `Open`, `Close`, `TurnOn` and `TurnOff` compare a joint's qpos to thresholds such as `default_open_ranges` (`envs/objects/articulated_objects.py`).

**How success is checked**
- A goal is a flat AND of atoms (bddl `package_predicates`).
- `BDDLBaseDomain.step` re-checks it on every 20 Hz step: `obs, reward, done, info = super().step(action); done = self._check_success()`. Reward is 1.0 on success.
- Evaluators latch success (`dones[k] = dones[k] or done[k]`) and stop that episode.
- There is no hold time and no requirement that the gripper has released the object.

**Goal atoms per suite**

| Suite | Goal atoms |
|---|---|
| spatial | 10 On |
| object | 10 In (basket) |
| goal | 7 On, 1 In, 1 Open, 1 Turnon |
| libero_10 | 9 In, 7 On, 2 Close, 2 Turnon |

## 3. Benchmark organization

- `benchmark/__init__.py` defines `Task(NamedTuple)` with fields `name, language, problem, problem_folder, bddl_file, init_states_file`. `init_states_file` is `<name>.pruned_init`. Task lists are in `benchmark/libero_suite_task_map.py`.
- **Suites:**
  - `libero_spatial`: same objects, bowl location varies.
  - `libero_object`: different target item, always into the basket.
  - `libero_goal`: one scene, 10 different goals.
  - `libero_10`: long-horizon, mostly two sub-goals.
  - `libero_90`.
- **API:** `benchmark.get_benchmark_dict()[name](task_order_index=0)` or `get_benchmark(name)`. Instances expose `.n_tasks`, `.get_task(i)`, `.get_task_bddl_file_path(i)`, `.get_task_init_states(i)` and `.get_task_demonstration(i)`.
- **Ordering:** `task_orders` holds 21 permutations for lifelong-learning order; `libero_90` only supports order 0.
- `LIBERO_100` is registered, but `task_maps` has no entry for it, so creating it raises KeyError.
- **There are two instruction strings per task:**
  - `task.language` comes from the filename via `grab_language_from_filename`, which strips `SCENEk_`. Example: "pick up the black bowl between the plate and the ramekin and place it on the plate".
  - `env.language_instruction` is the lowercased BDDL `:language`: "pick the akita black bowl between …".
  - LeRobot, OpenVLA and the HF `tasks.parquet` all use the filename version.

## 4. Observations, actions, protocol

**Observations**
- Cameras are `agentview` (front view, set per scene class in `_setup_camera`) and `robot0_eye_in_hand`.
- Keys: `agentview_image`, `robot0_eye_in_hand_image`, `robot0_joint_pos` (7), `robot0_gripper_qpos` (2), `robot0_eef_pos`, `robot0_eef_quat` (xyzw), and `<obj>_pos`/`<obj>_quat`.
- Resolution is 128×128 by default in LIBERO (`ControlEnv`); OpenVLA, the HF datasets and LeRobot use 256×256.

**Image flip**
- robosuite sets `IMAGE_CONVENTION="opengl"`, so raw frames are stored bottom-up.
- The HF data is rotated 180° relative to raw frames. LeRobot's `LiberoProcessorStep` applies `torch.flip(img, dims=[2,3])` at eval; OpenVLA does `img[::-1, ::-1]  # rotate 180 degrees to match train preprocessing`.
- mujoco 3.3.7's `Renderer.render()` already applies `np.flipud` (renderer.py line 248), so a native-MuJoCo env gets upright images.

**Actions**
- `OSC_POSE`, 7-D, each value in [-1, 1].
- `controllers/config/osc_pose.json` sets `output_max` to `[0.05]*3 + [0.5]*3` (metres and radians per step), `kp` 150, `control_delta: true`.
- Gripper: −1 opens, +1 closes. `PandaGripper.format_action` moves the target by `0.01*sign(a)` per 2 ms substep, so a full stroke takes about 8 env steps.
- 20 Hz control = 25 substeps of 0.002 s. Horizon is 1000; past it robosuite raises `ValueError`.

**Eval protocols**

| Protocol | Episodes per task | Settle steps | max_steps | Stop rule |
|---|---|---|---|---|
| LIBERO (`lifelong/metric.py::evaluate_one_task_success`, `configs/eval/default.yaml`) | `n_eval: 20` (states 0–19) | 5 all-zero actions | 600 | latch success |
| OpenVLA | 50 (all 50 states) | 10 × `[0,0,0,0,0,0,-1]` | 220/280/300/520/400 (spatial/object/goal/10/90) | break on `done` |
| LeRobot docs | 10, across 4 suites (400 total), averaged over 3 seeds | — | — | report a Wilson 95% interval |

## 5. Demonstrations and LeRobot datasets

**Original HDF5 format** (`scripts/create_dataset.py`; HF repo `yifengzhu-hf/LIBERO-datasets`; 50 demos per task is UNVERIFIED here)
- Per episode, `data/demo_i/` holds `actions (T,7)`, `states (T,1+nq+nv)`, `robot_states`, `rewards`, `dones` (1 on the last step), and `obs/` with `agentview_rgb`, `eye_in_hand_rgb` (128²), `gripper_states` (2), `joint_states` (7), `ee_states` (6), `ee_pos` and `ee_ori`.
- Episode attributes: `model_file` (the MJCF XML) and `init_state`. Group attributes: `bddl_file_content`, `env_args`, `problem_info`.
- Images are made by replaying the actions from the stored initial state; the first 5 steps are dropped.

**`HuggingFaceVLA/libero` (PNG in parquet) and `lerobot/libero` (AV1 video), format v3.0**
- 1693 episodes, 273,465 frames, 40 tasks: spatial 432 + object 454 + goal 428 + libero_10 379. No `libero_90`.
- `observation.images.image`: agentview, 256×256×3.
- `observation.images.image2`: wrist camera.
- `observation.state`: float32[8] = eef_pos (3) + axis-angle (3) + two finger qpos (2). The stats confirm this: dims 3–5 span ±3.67, and dims 6–7 stay within [0, 0.042] and [−0.042, 0].
- `action`: float32[7]. xyz within ±0.9375, rotation within ±0.375, gripper ±1.
- `task_index` points into `meta/tasks.parquet`, which holds the filename-style strings.
- `fps: 10`.

**Per-suite `lerobot/libero_{spatial,object,goal,10}_image`**
- The wrist key is `observation.images.wrist_image`.
- The state names are mislabelled `[x,y,z,rx,ry,rz,rw,gripper]`; the contents are the same as above.

**Gotchas**
- The rates disagree: metadata says 10 fps, the sim takes one action per 20 Hz step, and LeRobot's `LiberoEnv` config says `fps=30`.
- Mean episode lengths are 123 (spatial), 148 (object), 122 (goal) and 268 (libero_10) steps.
- Dataset `task_index` order is not suite `task_id` order (spatial task 0 is `task_index` 34), so match on the language string.

## 6. LeRobot 0.4.2 integration

File: `/scratch/e1583535/virtualenvs/lerobot042-act/lib/python3.12/site-packages/lerobot/envs/libero.py`

**Constructor and wrapping**
- Signature: `LiberoEnv(task_suite, task_id, task_suite_name, camera_name="agentview_image,robot0_eye_in_hand_image", obs_type="pixels", observation_width=256, observation_height=256, init_states=True, episode_index=0, camera_name_mapping=None, num_steps_wait=10)`.
- It wraps `OffScreenRenderEnv(bddl_file_name, camera_heights, camera_widths)`.
- `task_description = task.language`.
- `_max_episode_steps = TASK_SUITE_MAX_STEPS[suite]`: 280/280/300/520/400.
- Cameras are renamed to `image` and `image2`.
- `processor/env_processor.py::LiberoProcessorStep` flips the images and builds the 8-D `observation.state`.

**Step and success**
- `step` computes `is_success = self._env.check_success()` and `terminated = done or is_success`.
- It adds `task`, `task_id`, `done` and `is_success` to `info`. On termination it also adds `final_info` and then calls `self.reset()` itself.
- `truncated` is always False. `lerobot_eval.rollout` forces done at `max_steps`.
- An episode counts as a success if any step's `final_info["is_success"]` is true; these feed `pc_success`.

**Vectorization**
- `create_libero_envs(task="libero_spatial,libero_object", n_envs, gym_kwargs, camera_name, init_states, env_cls)` returns `{suite: {task_id: VectorEnv}}`.
- Sub-env i is pinned to init state i.

**Bug in 0.4.2:** `reset()` calls `set_init_state()` **before** `self._env.reset()`.
- The hard reset throws the fixed state away, so episodes start from freshly sampled, seed-dependent placements.
- `_init_state_id` never advances.
- `main` fixes this: it resets first, then sets the init state, then does `init_state_id += _reset_stride`. It also adds `control_mode` and `hard_reset` options.
- Separately, the config's `observation_height`/`observation_width` of 360 are not passed through `gym_kwargs`, so the env renders at 256.

## 7. Lessons

**Copy these:**
1. **Declarative tasks and a registry.** Use a `Task` dataclass (id, suite, language, fixtures, objects, regions, init atoms, goal atoms, max_steps), kept separate from scene code, with `get_suite(name).tasks / get_task(i) / get_task_init_states(i)`.
2. **Named regions.** Each region is a list of rects plus a yaw range on a target surface, keyed `<target>_<region>` and compiled to MuJoCo box sites. Use the same sites for placement and goal checks.
3. **Placement sampling.** Rejection sampling with footprint radii, a retry cap and an explicit error. Use a per-env `np.random.Generator(seed)` and check that each range is at least 2r wide.
4. **Frozen eval sets.** 50 states per task, with episode k using state k. Save them with nq/nv and a model hash.
5. **Validate stored states.** Check they settle, nothing interpenetrates, and the goal is not already true. Either store post-settle states or write the settle steps into the protocol.
6. **Predicates as pure functions of `MjData`.**
   - `In`: the object's body point is inside the region box, in the site frame, with a z tolerance.
   - `On`: contact + z-order + xy tolerance, or a slab test above a surface site.
   - `Open`/`Close`: joint thresholds.
   - Add `Lifted(obj, h)`: z above the table top + h, and not touching the table.
7. **Goal checking.** Check the AND goal every control step and stop on success. Consider requiring k consecutive steps with the gripper released. Report per-task success with a confidence interval.
8. **One instruction string per task.** Use it verbatim for the dataset `task`, the env `task_description` and eval.
9. **LeRobot-compatible keys.** `observation.images.image` / `image2` (or `wrist`), float32 `observation.state` and `action`, and `task`. Set `fps` to the true control rate and store images upright as HWC uint8. The natural SO-101 action is the six `<motor>.pos` values: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (`robots/so101_follower`).
10. **Replayable demos and focused suites.** Store sim states and the model XML with demos so they can be replayed and re-rendered. Set `max_steps` per suite to about 1.1× the longest demo. Build suites that each vary one factor (layout, object, goal, horizon).

**Don't copy these:**
1. **BDDL text and its parser.** Everything is lowercased, goals are a flat AND only, and problem names map to classes through registries that call `eval()`.
2. **The robosuite stack.** That means 6-DoF OSC pose deltas (a 5-DoF arm can't reach arbitrary poses; use joint targets or xyz+pitch IK), the observables/sensors machinery, MJCF name prefixing, rebuilding the model on every reset, and bottom-up OpenGL frames.
3. **`torch.save` pickles.** LeRobot has to load them with `weights_only=False`; use `.npz` instead.
4. **The geometry shortcuts.**
   - `abs(R@size)` as a stand-in for a rotated box.
   - `R` applied where `R.T` is meant.
   - An `Up` predicate tied to absolute world z.
   - A 3 cm body-origin `On` check that can pass while the object is still in the gripper.
   - Range shrinking that inverts the bounds.
5. **Seeding the global RNG and auto-resetting inside `step()`.**

**Housekeeping:** `/tmp` (4.9 GB) filled up briefly while I was unpacking the robosuite source tarball. The bulky directories were then moved to `/scratch/e1583535/tmp/cc/research/` (`libero_clone`, `robosuite140/pkg` at 184 MB, `bddl101`) and are symlinked back into the scratchpad. The small downloads are in `/tmp/claude-24441/-scratch-e1583535/cc20221c-31b8-4cc2-b268-3cd4267ef119/scratchpad/research/` under `hf_meta`, `openvla_eval` and `lerobot_main_libero`.