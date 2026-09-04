"""Geometry: ArUco position, scale, and the gap warning.

The gap warning gets as much test surface as the happy path, because it is the
layer's declared failure mode: a system that silently reports uninspected
fabric as clean is worse than one that admits it lost track, and that is only
true if the warning actually fires.

Tests synthesise marker images with ``cv2.aruco.generateImageMarker`` rather
than requiring the rig, so they run on any machine including CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from linesight.config import GeometryConfig
from linesight.geometry.aruco import ArucoReader, detect_markers, mm_per_px_from_corners


@pytest.fixture
def config() -> GeometryConfig:
    return GeometryConfig(
        aruco_dict="DICT_4X4_50",
        marker_length_mm=40.0,
        marker_pitch_mm=100.0,
        max_gap_mm=250.0,
    )


def _synthetic_marker_frame(marker_id: int, side_px: int = 200, canvas: int = 800) -> np.ndarray:
    """A frame with one ArUco marker of known pixel size drawn into it.

    Known side length in pixels + known length in mm = a scale the test can
    assert on exactly, which is the only way to catch a scale bug that would
    otherwise silently rescale every defect length in the ASTM total.
    """
    cv2 = pytest.importorskip("cv2")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    frame = np.full((canvas, canvas), 255, dtype=np.uint8)
    frame[50 : 50 + side_px, 50 : 50 + side_px] = marker
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


class TestScale:
    def test_mm_per_px_from_a_square_of_known_size(self) -> None:
        # A 40 mm marker drawn 200 px wide is 0.2 mm per pixel.
        corners = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
        assert mm_per_px_from_corners(corners, 40.0) == pytest.approx(0.2)

    def test_averages_all_four_edges(self) -> None:
        # Slight perspective: the four edges disagree, the estimate splits them.
        corners = np.array([[0, 0], [200, 0], [198, 202], [2, 200]], dtype=np.float32)
        value = mm_per_px_from_corners(corners, 40.0)
        assert value == pytest.approx(0.2, abs=0.005)


class TestDetection:
    def test_finds_a_synthetic_marker_and_its_id(self) -> None:
        frame = _synthetic_marker_frame(marker_id=7)
        readings = detect_markers(frame, "DICT_4X4_50", 40.0)
        assert len(readings) == 1
        assert readings[0].marker_id == 7

    def test_recovers_the_drawn_scale(self) -> None:
        frame = _synthetic_marker_frame(marker_id=7, side_px=200)
        reading = detect_markers(frame, "DICT_4X4_50", 40.0)[0]
        assert reading.mm_per_px == pytest.approx(0.2, rel=0.05)

    def test_position_is_id_times_pitch(self, config: GeometryConfig) -> None:
        frame = _synthetic_marker_frame(marker_id=7)
        reading = detect_markers(frame, "DICT_4X4_50", 40.0)[0]
        reading_pitch = 100.0
        assert reading.along_mm == pytest.approx(7 * reading_pitch)

    def test_blank_frame_yields_nothing(self) -> None:
        blank = np.full((800, 800, 3), 255, dtype=np.uint8)
        assert detect_markers(blank, "DICT_4X4_50", 40.0) == []


class TestGapWarning:
    """Markers missed for too long must raise a gap, not extrapolate silently."""

    def test_marker_frame_is_confident(self, config: GeometryConfig) -> None:
        reader = ArucoReader(config)
        geom = reader.read(_synthetic_marker_frame(marker_id=3), timestamp=0.0)
        assert geom.marker_ids == (3,)
        assert not geom.interpolated
        assert not geom.gap_warning

    def test_short_dropout_interpolates_without_warning(self, config: GeometryConfig) -> None:
        reader = ArucoReader(config)
        reader.read(_synthetic_marker_frame(marker_id=3), timestamp=0.0)
        blank = np.full((800, 800, 3), 255, dtype=np.uint8)
        geom = reader.read(blank, timestamp=0.05)
        assert geom.interpolated
        assert not geom.gap_warning

    def test_long_dropout_raises_a_gap_warning(self, config: GeometryConfig) -> None:
        # Hand-yank the fabric so markers are missed: the system must say so
        # rather than report the skipped fabric as clean.
        reader = ArucoReader(config)
        reader.read(_synthetic_marker_frame(marker_id=3), timestamp=0.0)
        blank = np.full((800, 800, 3), 255, dtype=np.uint8)
        geom = reader.read(blank, timestamp=60.0)
        assert geom.gap_warning
        assert reader.n_gap_warnings == 1

    def test_warning_clears_when_a_marker_returns(self, config: GeometryConfig) -> None:
        reader = ArucoReader(config)
        reader.read(_synthetic_marker_frame(marker_id=3), timestamp=0.0)
        blank = np.full((800, 800, 3), 255, dtype=np.uint8)
        reader.read(blank, timestamp=60.0)
        geom = reader.read(_synthetic_marker_frame(marker_id=9), timestamp=61.0)
        assert not geom.gap_warning

    def test_reset_clears_per_roll_state(self, config: GeometryConfig) -> None:
        reader = ArucoReader(config)
        reader.read(_synthetic_marker_frame(marker_id=3), timestamp=0.0)
        reader.reset()
        assert reader.n_gap_warnings == 0
