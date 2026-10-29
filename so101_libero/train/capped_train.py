"""``lerobot-train`` (or the task-balanced trainer) with a hard cap on THIS process's GPU memory.

    SO101_GPU_MEM_GB=5 python -m so101_libero.train.capped_train [--balanced] <the usual lerobot-train arguments>

For a GPU shared with someone else's job: ``torch.cuda.set_per_process_memory_fraction`` makes this process fail with
an out-of-memory error if it would need more than the cap, instead of growing into memory the other job may still
need. The CUDA context (~0.3-0.5 GB) comes on top of the cap. Everything else is LeRobot's own training (same
arguments, checkpoints, resume). The peak memory is printed every ``SO101_MEM_LOG_STEPS`` optimizer steps (default
1000) and at exit, so a probe run shows what a batch size needs.
"""

from __future__ import annotations

import atexit
import os
import sys


def main() -> None:
    import torch

    balanced = "--balanced" in sys.argv
    if balanced:
        sys.argv.remove("--balanced")
    gb = float(os.environ.get("SO101_GPU_MEM_GB") or 0)
    if gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total), 0)
        print(f"GPU memory cap for this process: {gb:g} GB of {total:.1f} GB (+ the CUDA context)", flush=True)

    def peak(tag: str) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            print(f"[{tag}] peak GPU memory: allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
                  f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True)

    atexit.register(peak, "exit")
    every = int(os.environ.get("SO101_MEM_LOG_STEPS") or 1000)
    try:  # report the peak now and then during long runs (wraps the optimizer step LeRobot calls)
        import lerobot.scripts.lerobot_train as lt

        update = lt.update_policy
        calls = {"n": 0}

        def update_and_report(*a, **k):
            out = update(*a, **k)
            calls["n"] += 1
            if calls["n"] in (1, 10) or calls["n"] % every == 0:
                peak(f"step {calls['n']}")
            return out

        lt.update_policy = update_and_report
    except (ImportError, AttributeError):
        pass
    if balanced:
        from so101_libero.train.lerobot_train_balanced import main as train_main
    else:
        from lerobot.scripts.lerobot_train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
