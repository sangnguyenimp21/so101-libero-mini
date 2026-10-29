"""Node check before a job relies on EGL: rendering in spawned worker processes that also use CUDA (EGL next to CUDA
aborted the interpreter on some cluster nodes). Exit code 0 = passed; the SLURM driver then renders with EGL, else OSMesa.

    SO101_RENDER=egl bash scripts/local_env.sh python tests/egl_cuda_workers_selftest.py [n_workers]
"""
import concurrent.futures as cf
import multiprocessing as mp
import sys


def work(i: int) -> float:
    import so101_libero  # noqa: F401  (GL setup before mujoco)
    import torch

    if torch.cuda.is_available():  # a CUDA context first, as in an evaluation worker
        x = torch.randn(1024, 1024, device="cuda")
        float((x @ x).sum())
    from so101_libero.env import EnvConfig, SO101TabletopEnv

    env = SO101TabletopEnv("flip_red_cube", EnvConfig())
    obs, _ = env.reset(options={"placement_id": i % 5})
    for _ in range(30):
        obs, *_ = env.step(obs["observation.state"])
    env.close()
    return float(obs["observation.images.front"].mean())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    try:
        with cf.ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as pool:
            means = list(pool.map(work, range(2 * n)))
    except cf.process.BrokenProcessPool:
        print("EGL + CUDA worker self-test FAILED: a worker process aborted", flush=True)
        sys.exit(1)
    if not all(m > 1.0 for m in means):
        print(f"EGL + CUDA worker self-test FAILED: blank images {means}", flush=True)
        sys.exit(1)
    print(f"EGL + CUDA worker self-test passed ({len(means)} episodes in {n} processes)", flush=True)
