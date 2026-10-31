import json

import pytest

from so101_libero.cli.evaluate import wilson
from so101_libero.cli.make_init_states import load_init_states, scene_hash
from so101_libero.tasks.registry import TASKS


def test_wilson_interval():
    lo, hi = wilson(50, 50)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.9287, abs=1e-3)
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and hi == pytest.approx(0.0713, abs=1e-3)
    lo, hi = wilson(25, 50)
    assert lo < 0.5 < hi


@pytest.mark.parametrize("task_id", list(TASKS))
def test_init_state_files_match_scene_and_are_disjoint_from_training(task_id):
    init = load_init_states(task_id)
    assert init["scene_hash"] == scene_hash(TASKS[task_id]), "scene changed since the eval states were made"
    assert len(init["states"]) == 50 and all(s["expert"]["success"] for s in init["states"])
    assert min(s["seed"] for s in init["states"]) >= 10_000_000  # training seeds are < 10,000,000
    assert [s["index"] for s in init["states"]] == list(range(50))
    json.dumps(init)  # serializable
