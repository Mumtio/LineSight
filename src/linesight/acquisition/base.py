"""L1 ACQUISITION - the frame source protocol.

Three sources implement it, in increasing order of what can go wrong at run
time (ADR-006): a folder of stills, a recorded video, and a live MJPEG stream.
Each is an iterable of ``Frame`` and a context manager, so nothing above L1
knows or cares whether the pixels came from disk, a file, or a camera - and any
live run can be reproduced by pointing the same config at a recording of it.

``FrameSource`` is the protocol; ``BaseFrameSource`` holds the plumbing every
source shares (stride, frame indexing, timestamps, idempotent open/close) and a
subclass supplies only ``_open``, ``_close`` and ``_raw_frames``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from ..types import Frame

__all__ = ["BaseFrameSource", "FrameSource", "open_source"]


@runtime_checkable
class FrameSource(Protocol):
    """A stream of frames with a known-or-unknown length."""

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames in capture order, honouring the configured stride."""
        ...

    def __len__(self) -> int:
        """Frame count if knowable, else raise ``TypeError``. Live streams raise."""
        ...

    def open(self) -> None:
        """Acquire the underlying handle. Idempotent."""
        ...

    def close(self) -> None:
        """Release it. Idempotent, and safe to call on a never-opened source."""
        ...

    @property
    def fps(self) -> float | None:
        """Nominal frame rate, or None when the source has no meaningful one."""
        ...


class BaseFrameSource:
    """Shared plumbing: stride, indexing, timestamps, context-manager protocol.

    Subclasses implement ``_open``, ``_close`` and ``_raw_frames`` only.
    """

    def __init__(self, uri: str, stride: int = 1, loop: bool = False) -> None:
        """Args:
        uri: folder, file path, or URL - interpreted by the subclass.
        stride: emit every Nth raw frame. The sampling factor gets published
            in the latency table; it is never silently applied.
        loop: restart at the end, for replaying a short recording
            continuously. Never used for a measured run.

        Raises:
            ValueError: if ``stride`` is below 1.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.uri = str(uri)
        self.stride = int(stride)
        self.loop = bool(loop)
        self._is_open = False

    # -- context manager ---------------------------------------------------- #

    def __enter__(self) -> BaseFrameSource:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        if not self._is_open:
            self._open()
            self._is_open = True

    def close(self) -> None:
        if self._is_open:
            self._close()
            self._is_open = False

    # -- iteration ---------------------------------------------------------- #

    def __iter__(self) -> Iterator[Frame]:
        """Apply stride and assign monotonic indices over ``_raw_frames``.

        ``index`` counts **emitted** frames, not raw ones, so downstream code
        never has to know the stride. The stride itself is reported separately,
        with the latency table, because a per-frame number that hides a sampling
        factor of 3 is a claim rather than a measurement.
        """
        self.open()
        emitted = 0
        while True:
            for raw_index, (image, timestamp, source_id) in enumerate(self._raw_frames()):
                if raw_index % self.stride:
                    continue
                yield Frame(
                    image=np.asarray(image),
                    index=emitted,
                    timestamp=float(timestamp),
                    source_id=str(source_id),
                )
                emitted += 1
            if not self.loop:
                return

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} has no length")

    # -- subclass hooks ----------------------------------------------------- #

    def _open(self) -> None:
        raise NotImplementedError

    def _close(self) -> None:
        raise NotImplementedError

    def _raw_frames(self) -> Iterator[tuple[np.ndarray, float, str]]:
        """Yield ``(image, timestamp, source_id)`` before striding is applied."""
        raise NotImplementedError

    @property
    def fps(self) -> float | None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.uri!r}, stride={self.stride})"


def open_source(source_type: str, uri: str, **kwargs: object) -> BaseFrameSource:
    """Factory: ``"directory" | "video" | "mjpeg"`` -> a constructed source.

    Imports each implementation lazily so that a missing optional dependency in
    one source does not stop the other two from working.

    Raises:
        ValueError: on an unknown source type.
    """
    if source_type == "directory":
        from .directory import DirectorySource

        return DirectorySource(uri, **kwargs)  # type: ignore[arg-type]
    if source_type == "video":
        from .video import VideoSource

        return VideoSource(uri, **kwargs)  # type: ignore[arg-type]
    if source_type == "mjpeg":
        from .mjpeg import MjpegSource

        return MjpegSource(uri, **kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown source type {source_type!r}; expected directory, video, or mjpeg"
    )
