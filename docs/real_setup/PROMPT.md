# Task: black robot + real front camera view in the sim, then retrain

Repo: `so101-libero-mini`. Read `README.md` first. Change only two things in the sim: the robot's colour and the
front camera. Then regenerate the datasets and retrain. Leave everything else as it is: table, objects, item start
regions, background, wrist camera, lighting.

## 1. Robot colour: matte, near black

In `build_spec` (`so101_libero/sim/scene.py`), override the robot's materials. The vendored robot XML stays
unedited. The objects and containers colour their geoms directly, so the only materials besides the robot's are
`table_mat` and `floor_mat`:

```python
for m in spec.materials:
    if m.name not in ("table_mat", "floor_mat"):
        m.rgba = [0.12, 0.12, 0.12, 1.0]
        m.specular = 0.15
        m.shininess = 0.2
```

Robot size, geometry and physics stay unchanged. The camera below was measured with the robot at its model size.

## 2. Front camera: the real camera's view

Measured on the real robot, in the sim world frame (metres; robot base at the origin, +x forward, +y the robot's
left, table top z = 0):

| parameter | value |
|---|---|
| pos | (0.374, 0.009, 0.557) |
| lookat | (0.005, 0.003, 0.0) |
| roll | 2.3 deg, about the camera's viewing axis, applied after `look_at_quat` |
| fovy | 45.0 deg, vertical field of view of the square image |

It looks down about 56 deg from 67 cm. The real policy image is the centre square of the camera frame, resized to
256x256, so the sim front camera renders a square 256x256 image directly. The current item start regions stay
inside this view, in the lower part of the image.

Implement:
- Add `roll: float = 0.0` as the last field of `CameraSpec`.
- Set
  `FRONT_CAMERA = CameraSpec("front", pos=(0.374, 0.009, 0.557), lookat=(0.005, 0.003, 0.0), fovy=45.0, roll=2.3)`.
- Where `build_spec` adds the cameras, use:

```python
quat = look_at_quat(cam.pos, cam.lookat)
if cam.roll:
    r, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(r, np.array([0.0, 0.0, 1.0]), np.radians(cam.roll))
    mujoco.mju_mulQuat(out, quat, r)
    quat = out
world.add_camera(name=cam.name, pos=list(cam.pos), quat=list(quat), fovy=cam.fovy)
```

Check: render the front camera at 256x256 with the robot at its home pose and save it as
`docs/real_setup/sim_front_home.png`. The robot should stand in the upper middle of the image, seen steeply from
above, with the table in front of it.

## 3. Regenerate the data and retrain

- Generate new datasets with the updated scene under new names, with the same tasks, episode counts and settings
  as the current ones (see the README). For example `local/so101_black_t1_200`,
  `local/so101_black_t1_200_recovery` and `local/so101_black_all_200`. Validate them as before.
- Train ACT (single task and multi-task) and SmolVLA exactly as before with `scripts/train.sh`. Images stay
  256x256, and the camera keys stay `front` / `wrist`.
- Evaluate in sim as before.
- Report the success rates and the checkpoint paths (`outputs/train/<run>/checkpoints/last/pretrained_model`).
