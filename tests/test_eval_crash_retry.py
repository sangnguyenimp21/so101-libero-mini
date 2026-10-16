"""A worker that keeps crashing on one episode: the episode is tried twice, then recorded as an error."""

from types import SimpleNamespace

from so101_libero.cli.evaluate import _run_parallel
from so101_libero.utils.testing import crash_on_index_1


def test_crashing_episode_is_retried_once_then_recorded():
    jobs = [{"task_id": "t", "index": i, "seed": i} for i in range(4)]
    warnings = []
    args = SimpleNamespace(workers=2, policy="expert", no_render=True)
    res = _run_parallel(jobs, args, (8, 8), {}, warnings.append, episode_fn=crash_on_index_1)
    assert [r["index"] for r in res] == [0, 1, 2, 3]
    assert res[1]["outcome"] == "error: worker process crashed" and not res[1]["success"]
    assert all(r["success"] for i, r in enumerate(res) if i != 1)
    assert 1 <= len(warnings) <= 2  # at most two pool breaks for the crashing episode
