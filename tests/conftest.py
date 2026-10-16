import os

os.environ.setdefault("MUJOCO_GL", "disable")  # unit tests are physics-only

import pytest  # noqa: E402

import so101_libero  # noqa: E402,F401


@pytest.fixture(scope="session")
def env_t1():
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    return SO101TabletopEnv("cylinder_to_tray", EnvConfig(render=False, terminate_on_success=False, max_steps=900))
