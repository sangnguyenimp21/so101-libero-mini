"""MP4 writing with PyAV (bundled FFmpeg; the container has no ffmpeg binary)."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: float = 30.0, crf: int = 20) -> Path:
    """Write HxWx3 uint8 frames to an H.264 MP4 (even sizes required by yuv420p)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height = w - w % 2, h - h % 2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for f in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[: stream.height, : stream.width]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def label(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a short text label into the top-left corner (for eval/debug videos)."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 7 * len(text) + 6, 14], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return np.asarray(im)
