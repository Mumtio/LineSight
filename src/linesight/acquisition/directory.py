"""Frames from a folder of stills - the deterministic source (ADR-006).

The input every dataset study and every regression test runs on: the same
folder yields the same frames, in the same order, on any machine. Three
properties make that true - natural sort order, synthetic timestamps at a
declared rate, and a hard failure on an unreadable file rather than a silently
skipped one, because a skipped frame is a stretch of roll nobody inspected.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .base import BaseFrameSource

__all__ = ["DirectorySource", "natural_sort_key"]

_DIGITS = re.compile(r"(\d+)")


def natural_sort_key(path: Path) -> tuple[object, ...]:
    """Sort key that orders ``frame2.png`` before ``frame10.png``.

    Lexicographic order would interleave the roll - frame10 between frame1 and
    frame2 - which silently scrambles every position in the report. Splits the
    stem into digit and non-digit runs and compares numerically where it can.
    """
    parts = _DIGITS.split(Path(path).stem)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


class DirectorySource(BaseFrameSource):
    """Every image matching a glob, in natural order, as a frame stream."""

    def __init__(
        self,
        uri: str,
        glob: str = "*.png",
        stride: int = 1,
        loop: bool = False,
        synthetic_fps: float = 30.0,
    ) -> None:
        """Args:
        glob: pattern within the folder. AITEX masks end ``_mask.png``; the
            default pattern deliberately does not exclude them, so callers
            pass an explicit glob when pointing at a raw AITEX directory.
        synthetic_fps: timestamps are ``index / synthetic_fps``. Only used to
            make replay logs look like live ones; no measurement depends on it.

        Raises:
            ValueError: on a non-positive ``synthetic_fps``.
        """
        super().__init__(uri, stride=stride, loop=loop)
        if synthetic_fps <= 0:
            raise ValueError(f"synthetic_fps must be positive, got {synthetic_fps}")
        self.glob = glob
        self.synthetic_fps = float(synthetic_fps)
        self._paths: list[Path] = []

    def _open(self) -> None:
        """Resolve and sort the file list. Cheap; no images are read yet.

        Raises:
            FileNotFoundError: if the folder does not exist or matches nothing.
                An empty roll is never the intended input, and reporting one as
                a clean pass would be the worst possible failure mode.
        """
        folder = Path(self.uri)
        if not folder.is_dir():
            raise FileNotFoundError(f"not a directory: {folder}")

        self._paths = sorted(folder.glob(self.glob), key=natural_sort_key)
        if not self._paths:
            raise FileNotFoundError(f"no files matching {self.glob!r} in {folder}")

    def _close(self) -> None:
        self._paths = []

    def _raw_frames(self) -> Iterator[tuple[np.ndarray, float, str]]:
        """Read images one at a time.

        Raises:
            OSError: if a listed file cannot be decoded. Loud, not skipped.
        """
        for index, path in enumerate(self._paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(
                    f"could not decode {path} - a skipped frame is a stretch of "
                    "roll nobody inspected, so this is fatal rather than warned"
                )
            yield image, index / self.synthetic_fps, str(path)

    def __len__(self) -> int:
        """Number of frames this source will emit, after striding."""
        self.open()
        return (len(self._paths) + self.stride - 1) // self.stride

    @property
    def fps(self) -> float | None:
        return self.synthetic_fps
