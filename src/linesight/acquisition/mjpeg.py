"""Frames from a live MJPEG camera - the top rung of ADR-006's ladder.

An IP camera, or a phone serving ``http://<host>:8080/video``. It sits behind
the same ``FrameSource`` contract as a folder or a recording, so moving between
them is a config key and nothing above L1 changes.

Two concerns shape this module, and neither exists for a file. The newest frame
matters more than a complete sequence, so decoding runs on its own thread and
keeps only the latest frame (``drop_stale``): falling a buffer behind would
mislocate every defect, because the fabric it describes has already moved on.
And a dropped connection has to surface as an explicit failure rather than a
quiet pause, so a read timeout raises - fabric nobody inspected must never be
reported as clean.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
import urllib.request
from collections.abc import Iterator

import cv2
import numpy as np

from .base import BaseFrameSource

__all__ = ["MjpegSource"]

_JPEG_SOI = b"\xff\xd8"  # start of image
_JPEG_EOI = b"\xff\xd9"  # end of image
_CHUNK = 8192


class MjpegSource(BaseFrameSource):
    """Read an ``http://phone:8080/video`` MJPEG stream, dropping stale frames."""

    def __init__(
        self,
        uri: str,
        stride: int = 1,
        timeout_s: float = 5.0,
        drop_stale: bool = True,
    ) -> None:
        """Args:
        timeout_s: no bytes for this long -> treat the stream as broken.
        drop_stale: keep only the most recent decoded frame. Inspection must
            follow the fabric that is actually in front of the lens; falling
            behind by a buffer's worth would mislocate every defect.
        """
        super().__init__(uri, stride=stride, loop=False)
        self.timeout_s = float(timeout_s)
        self.drop_stale = bool(drop_stale)

        self._stream: object | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        # Only used when drop_stale is False. Bounded, so a full queue blocks
        # the reader and back-pressures the far end instead of buffering the
        # whole roll into memory.
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._error: BaseException | None = None
        self._ended = False
        self._arrivals: list[float] = []

    def _open(self) -> None:
        """Connect and start the reader thread.

        Raises:
            OSError: if the stream cannot be reached within ``timeout_s``.
        """
        try:
            self._stream = urllib.request.urlopen(self.uri, timeout=self.timeout_s)
        except Exception as exc:  # urllib raises a zoo of types
            raise OSError(f"could not open MJPEG stream {self.uri}: {exc}") from exc

        self._stop.clear()
        self._error = None
        self._ended = False
        self._latest = None
        self._arrivals = []
        self._thread = threading.Thread(target=self._reader, name="mjpeg", daemon=True)
        self._thread.start()

    def _close(self) -> None:
        """Signal the reader thread to stop and join it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_s)
            self._thread = None
        if self._stream is not None:
            # A socket that is already dead on close is not worth a traceback:
            # the run is ending either way, and the interesting failure was
            # recorded on the reader thread.
            with contextlib.suppress(Exception):
                self._stream.close()  # type: ignore[attr-defined]
            self._stream = None

    def _reader(self) -> None:
        """Pull JPEGs off the wire and keep only the newest.

        Runs on its own thread so that a slow inspection pass cannot apply
        back-pressure to the camera: the fabric keeps moving whether or not we
        finished scoring the last frame, and a buffered frame is a frame whose
        position we would get wrong.
        """
        buffer = bytearray()
        try:
            while not self._stop.is_set():
                chunk = self._stream.read(_CHUNK)  # type: ignore[attr-defined]
                if not chunk:
                    # A clean EOF is the source running out, not a fault: a
                    # finite recording or a simulated roll simply ends. Losing a
                    # live camera mid-roll looks different - no bytes for
                    # timeout_s - and that path still raises, because fabric
                    # nobody inspected must never be reported as clean.
                    self._ended = True
                    return
                buffer.extend(chunk)

                start = buffer.find(_JPEG_SOI)
                end = buffer.find(_JPEG_EOI, start + 2) if start != -1 else -1
                if start == -1 or end == -1:
                    continue

                jpeg = bytes(buffer[start : end + 2])
                del buffer[: end + 2]

                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue  # a torn frame mid-stream is normal; the next one is fine

                now = time.monotonic()
                with self._lock:
                    self._arrivals.append(now)
                    del self._arrivals[:-30]
                if self.drop_stale:
                    with self._lock:
                        self._latest = (image, now)
                else:
                    while not self._stop.is_set():
                        try:
                            self._queue.put((image, now), timeout=0.2)
                            break
                        except queue.Full:
                            continue
        except BaseException as exc:
            self._error = exc

    def _raw_frames(self) -> Iterator[tuple[np.ndarray, float, str]]:
        """Yield the latest frame with a wall-clock timestamp.

        Raises:
            OSError: on a read timeout or a dropped connection, so the run
                records a gap instead of silently reporting the remainder of the
                roll as clean.
        """
        last_emitted: float | None = None
        last_seen = time.monotonic()

        while not self._stop.is_set():
            if self._error is not None:
                raise OSError(f"MJPEG stream {self.uri} failed: {self._error}")

            if not self.drop_stale:
                # Every frame, in order. The queue is the back-pressure.
                try:
                    image, arrived = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._ended:
                        return
                    if time.monotonic() - last_seen > self.timeout_s:
                        raise OSError(
                            f"no frame from {self.uri} for {self.timeout_s:.0f}s - "
                            "the roll beyond this point is uninspected, not clean"
                        ) from None
                    continue
                last_seen = time.monotonic()
                yield image, arrived, self.uri
                continue

            with self._lock:
                latest = self._latest

            if latest is None or latest[1] == last_emitted:
                if self._ended:
                    return  # source exhausted and nothing new is coming
                if time.monotonic() - last_seen > self.timeout_s:
                    raise OSError(
                        f"no frame from {self.uri} for {self.timeout_s:.0f}s - "
                        "the roll beyond this point is uninspected, not clean"
                    )
                time.sleep(0.002)
                continue

            image, arrived = latest
            last_emitted = arrived
            last_seen = time.monotonic()
            yield image, arrived, self.uri

    def __len__(self) -> int:
        """Raises ``TypeError`` - a live stream has no length."""
        raise TypeError("a live MJPEG stream has no frame count")

    @property
    def fps(self) -> float | None:
        """Measured arrival rate over a short window, or None before it settles.

        Measured, not nominal: what the phone claims and what arrives over a
        contended Wi-Fi link are different numbers, and only one of them is
        worth putting in a latency table.
        """
        with self._lock:
            arrivals = list(self._arrivals)
        if len(arrivals) < 2:
            return None
        span = arrivals[-1] - arrivals[0]
        return (len(arrivals) - 1) / span if span > 0 else None
