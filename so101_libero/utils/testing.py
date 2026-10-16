"""Picklable helpers for tests of the multiprocessing evaluation path."""

import os


def crash_on_index_1(job: dict) -> dict:
    """Episode stand-in: the worker running episode index 1 dies (like a native crash); others succeed."""
    if job["index"] == 1:
        os._exit(1)
    return {"task_id": job["task_id"], "index": job["index"], "seed": job["seed"], "success": True, "steps": 1,
            "outcome": "success", "lifted": True, "other_objects_lifted": [], "mean_policy_ms": 0.0}
