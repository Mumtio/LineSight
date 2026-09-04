"""Camera and bench calibration - the constants ArUco cannot supply.

Distinct from ``calibrate/threshold.py``: that module calibrates *scores*, this
one calibrates *space*. They are kept apart because they fail differently - a
wrong threshold changes how much gets flagged, a wrong scale changes what every
millimetre in the report means.

Scope is deliberately small. Lens distortion on a phone at bench distance is a
percent or two at the corners; correcting it is optional and measured, not
assumed. What is not optional is knowing the fabric width in millimetres, since
``points_per_100yd2`` divides by inspected area.

**This module is a declared interface, not a dependency of the running
pipeline.** Scale reaches the pipeline from the markers themselves
(``geometry/aruco.py``) and web width from ``config.scoring.roll_width_m``, so
a rig needs no lens calibration to produce a report. The signatures below fix
the shape of one for a rig that does measure its own intrinsics; each raises
``NotImplementedError`` until those intrinsics exist, so no code path can
quietly depend on a constant nobody measured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["BenchCalibration", "estimate_mm_per_px_from_ruler", "undistort"]


class BenchCalibration:
    """Fixed properties of the rig: web width, camera intrinsics, nominal scale.

    Persisted as JSON next to the roll reports so a run can be reproduced, and
    so a scale error can be traced to a calibration rather than to the model.
    """

    def __init__(
        self,
        web_width_mm: float,
        nominal_mm_per_px: float | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> BenchCalibration:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def cross_mm(self, x_px: float, mm_per_px: float) -> float:
        """Pixel column -> millimetres from the left selvedge."""
        raise NotImplementedError

    @property
    def is_distortion_corrected(self) -> bool:
        """False unless intrinsics were actually measured. Never assumed true."""
        raise NotImplementedError


def estimate_mm_per_px_from_ruler(
    image: np.ndarray, known_length_mm: float, p1: tuple[int, int], p2: tuple[int, int]
) -> float:
    """Scale from two clicked points a known distance apart.

    The manual fallback when ArUco is not yet working, and the independent check
    that the ArUco-derived scale is right. Two ways of measuring the same number
    is how a scale bug gets caught before it reaches a report.
    """
    raise NotImplementedError


def undistort(image: np.ndarray, calibration: BenchCalibration) -> np.ndarray:
    """Apply lens correction, or return the image unchanged if uncalibrated.

    Returning the input untouched is correct behaviour, not a stub: an
    uncorrected frame with a stated distortion bound beats a frame corrected by
    invented intrinsics.
    """
    raise NotImplementedError
