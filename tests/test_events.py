"""Event assembly and tracking.

The property that matters for the penalty total: eight consecutive 40 mm
fragments of one warp line must become **one** 320 mm event, not eight events.
Without tracking, ASTM scores the same physical defect eight times at the wrong
length - the report is not merely untidy, it is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from linesight.config import EventConfig
from linesight.events.assemble import binarise, components_to_detections
from linesight.events.track import EventTracker, PerFrameTracker, iou
from linesight.types import Assertion, Calibration, Detection, FrameGeometry


@pytest.fixture
def config() -> EventConfig:
    return EventConfig(min_area_px=16, min_length_mm=0.5)


@pytest.fixture
def calibration() -> Calibration:
    return Calibration(
        threshold=3.0,
        abstain_low=2.0,
        budget_fa_per_100m=1.0,
        metres_per_tile=0.5,
        n_clean_tiles=1000,
    )


@pytest.fixture
def geometry() -> FrameGeometry:
    """0.1 mm/px - the scale the model spec is dimensioned around."""
    return FrameGeometry(along_mm=1000.0, mm_per_px=0.1, roi=(0, 0, 512, 512))


def _detection(frame_index: int, along_mm: float, across_mm: float = 50.0) -> Detection:
    return Detection(
        frame_index=frame_index,
        bbox_px=(0, 0, 100, 400),
        area_px=40_000,
        max_score=6.0,
        mean_score=4.0,
        along_mm=along_mm,
        across_mm=across_mm,
        length_mm=40.0,
        width_mm=10.0,
    )


class TestBinarise:
    def test_thresholds_strictly_above(self, calibration: Calibration) -> None:
        scores = np.array([[1.0, 3.0], [3.5, 9.0]], dtype=np.float32)
        mask = binarise(scores, calibration.threshold)
        assert mask.tolist() == [[0, 0], [1, 1]]

    def test_returns_uint8_zero_one(self) -> None:
        mask = binarise(np.zeros((4, 4), dtype=np.float32), 1.0)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1}


class TestComponentsToDetections:
    def test_one_blob_one_detection(
        self, geometry: FrameGeometry, calibration: Calibration, config: EventConfig
    ) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 50:80] = 1
        scores = mask.astype(np.float32) * 6.0
        detections = components_to_detections(mask, scores, geometry, calibration, config, 0)
        assert len(detections) == 1

    def test_pixels_convert_to_millimetres(
        self, geometry: FrameGeometry, calibration: Calibration, config: EventConfig
    ) -> None:
        # 100 px tall at 0.1 mm/px is 10 mm along the machine direction.
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 50:80] = 1
        scores = mask.astype(np.float32) * 6.0
        det = components_to_detections(mask, scores, geometry, calibration, config, 0)[0]
        assert det.length_mm == pytest.approx(10.0)
        assert det.width_mm == pytest.approx(3.0)

    def test_position_is_absolute_on_the_roll(
        self, geometry: FrameGeometry, calibration: Calibration, config: EventConfig
    ) -> None:
        # The frame starts at 1000 mm; a blob 100 px down is at 1010 mm.
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 50:80] = 1
        scores = mask.astype(np.float32) * 6.0
        det = components_to_detections(mask, scores, geometry, calibration, config, 0)[0]
        assert det.along_mm == pytest.approx(1010.0)

    def test_speckle_below_min_area_is_dropped(
        self, geometry: FrameGeometry, calibration: Calibration, config: EventConfig
    ) -> None:
        # Below the stated resolution the system is not making a claim.
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[10, 10] = 1
        scores = mask.astype(np.float32) * 6.0
        assert components_to_detections(mask, scores, geometry, calibration, config, 0) == []

    def test_abstain_band_marks_uncertain_not_asserted(
        self, geometry: FrameGeometry, calibration: Calibration, config: EventConfig
    ) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 50:80] = 1
        scores = mask.astype(np.float32) * 2.5  # between abstain_low and threshold
        det = components_to_detections(mask, scores, geometry, calibration, config, 0)[0]
        assert det.assertion is Assertion.UNCERTAIN


class TestIou:
    def test_identical_boxes(self) -> None:
        assert iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        assert iou((0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 10.0, 10.0)) == 0.0

    def test_half_overlap(self) -> None:
        assert iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 10.0, 10.0)) == pytest.approx(1 / 3)


class TestEventTracker:
    def test_a_line_across_frames_is_one_event(self, config: EventConfig) -> None:
        # Eight 40 mm fragments of one continuous warp defect.
        tracker = EventTracker(config)
        for i in range(8):
            tracker.update([_detection(frame_index=i, along_mm=1000.0 + i * 35.0)])
        events = tracker.finalise()
        assert len(events) == 1
        assert events[0].n_frames == 8

    def test_event_length_spans_the_whole_run(self, config: EventConfig) -> None:
        # 1000 mm to 1245+40 mm -> ~285 mm, which ASTM scores as continuous.
        tracker = EventTracker(config)
        for i in range(8):
            tracker.update([_detection(frame_index=i, along_mm=1000.0 + i * 35.0)])
        event = tracker.finalise()[0]
        assert event.length_mm == pytest.approx(285.0, abs=1.0)

    def test_defects_far_apart_stay_separate(self, config: EventConfig) -> None:
        tracker = EventTracker(config)
        tracker.update([_detection(0, along_mm=1000.0, across_mm=50.0)])
        tracker.update([_detection(1, along_mm=1000.0, across_mm=900.0)])
        assert len(tracker.finalise()) == 2

    def test_event_survives_a_short_dropout(self, config: EventConfig) -> None:
        # A faint defect can fall below threshold for a frame mid-run.
        tracker = EventTracker(config)
        tracker.update([_detection(0, along_mm=1000.0)])
        tracker.update([])
        tracker.update([_detection(2, along_mm=1070.0)])
        assert len(tracker.finalise()) == 1


class TestPerFrameTracker:
    """The documented degradation: same interface, one event per detection."""

    def test_emits_one_event_per_detection(self, config: EventConfig) -> None:
        tracker = PerFrameTracker(config)
        for i in range(8):
            tracker.update([_detection(i, along_mm=1000.0 + i * 35.0)])
        assert len(tracker.finalise()) == 8
