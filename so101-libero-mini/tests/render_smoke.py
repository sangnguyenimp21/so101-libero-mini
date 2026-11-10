"""Smallest possible offscreen-render check: MuJoCo EGL renderer on a GPU node."""
import os
import time

import mujoco
import numpy as np
from PIL import Image

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="1 1 0.1" rgba=".8 .8 .8 1"/>
    <body pos="0 0 0.1"><freejoint/><geom type="box" size=".05 .05 .05" rgba="1 0 0 1"/></body>
    <camera name="cam" pos="0.5 -0.5 0.5" xyaxes="1 1 0 -0.5 0.5 1"/>
  </worldbody>
</mujoco>
"""
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 256, 256)
mujoco.mj_forward(m, d)
t = time.time()
for _ in range(200):
    r.update_scene(d, camera="cam")
    img = r.render()
dt = (time.time() - t) / 200
print(f"render 256x256: {dt*1e3:.2f} ms/frame, mean pixel {img.mean():.1f}")
out = os.path.join(os.path.dirname(__file__), "..", "logs", "render_smoke.png")
Image.fromarray(img).save(out)
print("saved", os.path.abspath(out))
