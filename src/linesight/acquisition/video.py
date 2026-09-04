"""Frames from a video file - the middle rung of ADR-006's ladder.

Records a pull of fabric once and replays it deterministically forever, which
is what regression-testing the geometry layer needs: ArUco decoding and
position continuity can only be compared between runs if the pixels are
identical.

Timestamps come from the container rather than the wall clock, so a replay puts
each defect at the position it was captured at instead of wherever this
machine's decoding speed happens to land it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .base import BaseFrameSource

__all__ = ["VideoSource"]


class VideoSource(BaseFrameSource):
    """``cv2.VideoCapture`` over a file, wrapped in the frame-source contract."""

    def __init__(self, uri: str, stride: int = 1, loop: bool = False) -> None:
        """Raises:
        FileNotFoundError: if the path does not exist.
        """
        if not Path(uri).exists():
            raise FileNotFoundError(f"no video file at {uri}")
        super().__init__(uri, stride=stride, loop=loop)
        self._capture: cv2.VideoCapture | None = None
        self._fps: float | None = None
        self._frame_count: int = 0

    def _open(self) -> None:
        """Open the capture and read fps / frame count.

        Raises:
            OSError: if OpenCV cannot open the container.
        """
        capture = cv2.VideoCapture(self.uri)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"OpenCV could not open {self.uri}")
        self._capture = capture
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 0 else None
        self._frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _raw_frames(self) -> Iterator[tuple[np.ndarray, float, str]]:
        """Yield decoded frames with timestamps from the container, not the clock.

        Container timestamps, so a replay places defects at the positions they
        were actually captured at rather than at whatever rate this machine
        happens to decode.
        """
        assert self._capture is not None
        capture = self._capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                return
            position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if position_ms > 0:
                timestamp = position_ms / 1000.0
            elif self._fps:
                timestamp = index / self._fps
            else:
                timestamp = float(index)
            yield image, timestamp, f"{self.uri}#{index}"
            index += 1

    def __len__(self) -> int:
        """Frame count as reported by the container - approximate for some codecs.

        Raises:
            TypeError: if the container does not report a count, rather than
                returning a confident zero.
        """
        self.open()
        if self._frame_count <= 0:
            raise TypeError(f"{self.uri} does not report a frame count")
        return (self._frame_count + self.stride - 1) // self.stride

    @property
    def fps(self) -> float | None:
        return self._fps
