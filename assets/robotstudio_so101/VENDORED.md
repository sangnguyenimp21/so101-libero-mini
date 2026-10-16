# Vendored: MuJoCo Menagerie `robotstudio_so101`

- Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101
- Commit: 62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35 (2026-10-14)
- License: Apache-2.0 (LICENSE in this directory)
- Copied unmodified: so101.xml, assets/*.stl, LICENSE, README.md, CHANGELOG.md.

The project never edits these files. Scene-specific changes (timestep, solver options, a TCP site, a base
collider, gripper force and elbow limits, wrist-camera fovy) are applied programmatically with `mujoco.MjSpec` in
`so101_libero/sim/scene.py`, so the upstream model can be updated by replacing this directory.
