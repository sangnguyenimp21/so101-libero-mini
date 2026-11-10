"""lerobot-rollout that rides out camera stalls (and reconnects a frozen camera) instead of aborting.

LeRobot 0.6's robots read each camera with OpenCVCamera.read_latest(), which raises TimeoutError as soon as the newest
frame is older than 500 ms. Both cameras of the real setup stall with nothing else running (measured 2026-10-18: the
iPhone front camera had 535 ms gaps and sometimes needs several reads to start; the wrist camera, USB2.0_CAM1 behind
the dock, stalled for 0.9 s and then failed reads while the gripper was held near the table, and for > 2 s in a
rollout; no USB disconnect was logged). Here a stale frame makes the read WAIT for the next frame (the control loop
pauses, the arm holds its last command); a camera that delivers nothing for CAMERA_STALL_MS [2000], or whose read
thread died, is closed and reopened, up to CAMERA_RECONNECTS [3] times per stall; only then the rollout stops. Every
stall and reconnect is logged.

    python scripts/rollout_tolerant.py <lerobot-rollout flags>     # used by scripts/real_smolvla.sh and real_act.sh
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


def tolerate_camera_stalls(max_stall_ms: float | None = None, reconnects: int | None = None) -> None:
    """Patch OpenCVCamera.read_latest (all instances): wait up to ``max_stall_ms`` for a fresh frame, then reopen the
    camera up to ``reconnects`` times."""
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    limit_ms = float(max_stall_ms if max_stall_ms is not None else os.environ.get("CAMERA_STALL_MS", 2000))
    tries = int(reconnects if reconnects is not None else os.environ.get("CAMERA_RECONNECTS", 3))
    original = OpenCVCamera.read_latest
    if getattr(original, "_tolerant", False):
        return

    def wait_for_frame(cam) -> object | None:
        """The next frame if one arrives before the newest is ``limit_ms`` old, else None (also if the thread died)."""
        with cam.frame_lock:
            stale = cam.latest_timestamp
        if stale is None:
            return None
        while (time.perf_counter() - stale) * 1e3 < limit_ms and cam.thread is not None and cam.thread.is_alive():
            time.sleep(0.005)
            with cam.frame_lock:
                if cam.latest_timestamp != stale:
                    logger.warning(f"{cam}: no frame for {(cam.latest_timestamp - stale) * 1e3:.0f} ms, continuing")
                    return cam.latest_frame
        return None

    def reopen(cam) -> object | None:
        for attempt in range(1, tries + 1):
            logger.warning(f"{cam}: no frame for {limit_ms:.0f} ms or reader stopped: reopening ({attempt}/{tries})")
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001  already closed / half-open: connect() below starts from scratch
                pass
            try:
                cam.connect()  # warmup: returns once a valid frame was captured
                with cam.frame_lock:
                    if cam.latest_frame is not None:
                        logger.warning(f"{cam}: reopened, continuing")
                        return cam.latest_frame
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{cam}: reopening failed: {e}")
        return None

    def read_latest(self, max_age_ms: int = 500):
        try:
            return original(self, max_age_ms)
        except (TimeoutError, RuntimeError):  # stale frame, no frame yet, or the read thread stopped
            pass
        frame = wait_for_frame(self)
        if frame is None:
            frame = reopen(self)
        if frame is None:
            raise TimeoutError(f"{self} delivered no frame for {limit_ms:.0f} ms and {tries} reopenings failed")
        return frame

    read_latest._tolerant = True
    OpenCVCamera.read_latest = read_latest


if __name__ == "__main__":
    tolerate_camera_stalls()
    from lerobot.scripts.lerobot_rollout import main

    main()
