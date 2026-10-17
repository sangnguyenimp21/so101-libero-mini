"""GPU-node check that EGL rendering works in several spawned worker processes (it aborts the interpreter on some
nodes, e.g. hopper-30, although a single process renders fine). Exit code 0 = passed.

    bash scripts/env.sh python tests/egl_workers_selftest.py [n_workers]
"""
import concurrent.futures as cf
import multiprocessing as mp
import sys


def work(i: int) -> int:
    import so101_libero  # noqa: F401  (GL setup before mujoco)
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    env = SO101TabletopEnv("red_cube_aside", EnvConfig(visual_rand=True))  # the randomized path renders too
    obs, _ = env.reset(seed=i)
    for _ in range(40):
        obs, *_ = env.step(obs["observation.state"])
    env.close()
    return i


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    try:
        with cf.ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as pool:
            done = sorted(pool.map(work, range(2 * n)))
    except cf.process.BrokenProcessPool:
        print("EGL worker self-test FAILED: a worker process aborted", flush=True)
        sys.exit(1)
    print(f"EGL worker self-test passed ({len(done)} episodes in {n} processes)")
