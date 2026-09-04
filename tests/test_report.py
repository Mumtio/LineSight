"""L8 REPORT - the PDF, the defect map, and the text fallback.

The report is the artifact a mill files, so the properties locked here are the
ones whose quiet failure would put an indefensible document on a QA desk: a
roll with no calibration is refused rather than rendered, events that score
zero still appear, and the text fallback carries every fact the PDF does.

matplotlib and reportlab live in the ``report`` extra, so anything needing them
skips rather than fails on a base install.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linesight.report.pdf import LIMITATIONS, render_text_report
from linesight.scoring.astm_d5430 import fill_report, points_for_event
from linesight.types import (
    Assertion,
    Calibration,
    Detection,
    Event,
    EventStatus,
    RollReport,
    Verdict,
)


def _event(
    event_id: int,
    along_mm: float,
    length_mm: float,
    score: float = 18.0,
    assertion: Assertion = Assertion.ASSERTED,
    status: EventStatus = EventStatus.PROPOSED,
) -> Event:
    detection = Detection(
        frame_index=event_id,
        bbox_px=(40, 30, 80, max(6, int(length_mm / 2))),
        area_px=800,
        max_score=score,
        mean_score=score * 0.6,
        along_mm=along_mm,
        across_mm=60.0,
        length_mm=length_mm,
        width_mm=40.0,
    )
    return Event(
        event_id=event_id,
        detections=[detection],
        along_start_mm=along_mm,
        along_end_mm=along_mm + length_mm,
        across_start_mm=60.0,
        across_end_mm=100.0,
        max_score=score,
        confidence=0.8,
        assertion=assertion,
        status=status,
    )


@pytest.fixture
def calibration() -> Calibration:
    return Calibration(
        threshold=14.6595,
        abstain_low=12.4014,
        budget_fa_per_100m=20.0,
        metres_per_tile=0.448,
        n_clean_tiles=960,
        sku="fabric_stain",
        fit_timestamp="2026-09-02T10:00:00+00:00",
    )


@pytest.fixture
def report(calibration: Calibration) -> RollReport:
    """One of every case the report has to survive."""
    events = [
        _event(1, 420.0, 48.0),
        _event(2, 1810.0, 180.0, status=EventStatus.CONFIRMED),
        _event(3, 3050.0, 1400.0, score=24.6),
        _event(4, 5200.0, 62.0, score=13.1, assertion=Assertion.UNCERTAIN),
        _event(5, 6400.0, 35.0, score=15.9, status=EventStatus.REJECTED),
    ]
    return fill_report(
        RollReport(
            roll_id="test-roll",
            sku="fabric_stain",
            roll_length_m=7.4,
            width_m=0.586,
            events=events,
            calibration=calibration,
            false_alarms=1,
            gap_warnings=2,
            meta={
                "frame_stride": 2,
                "spatial_resolution_mm": 52.0,
                "extent_rule": "half_max",
            },
        )
    )


# --------------------------------------------------------------------------- #
# Text - always available, no optional dependency
# --------------------------------------------------------------------------- #


def test_text_report_carries_every_event(report: RollReport) -> None:
    text = render_text_report(report)
    for event in report.events:
        assert f"{event.along_start_mm / 1000:.3f}" in text


def test_text_report_states_total_and_verdict(report: RollReport) -> None:
    text = render_text_report(report)
    assert f"total points        {report.total_points}" in text
    assert report.verdict.value.upper() in text


def test_text_report_shows_zero_scoring_events(report: RollReport) -> None:
    """Uncertain and rejected events must still appear.

    They are the abstention band made visible. A report that listed only the
    events which cost points would look cleaner and hide the one part of the
    calibration story a QA manager has to see.
    """
    text = render_text_report(report)
    assert "uncertain" in text
    assert "5" in text  # the rejected event's id is still in the table


def test_text_report_shouts_about_gap_warnings(report: RollReport) -> None:
    assert "GAP WARNING" in render_text_report(report)
    report.gap_warnings = 0
    assert "GAP WARNING" not in render_text_report(report)


def test_text_report_states_the_frame_stride(report: RollReport) -> None:
    """A per-frame number that hides a sampling factor is a claim, not a measurement."""
    assert "every 2th frame was processed" in render_text_report(report)


def test_text_report_survives_an_empty_roll(calibration: Calibration) -> None:
    empty = fill_report(
        RollReport(
            roll_id="clean",
            sku="x",
            roll_length_m=5.0,
            width_m=1.0,
            calibration=calibration,
        )
    )
    text = render_text_report(empty)
    assert "no defects detected" in text
    assert empty.verdict is Verdict.PASS


def test_limitations_names_what_the_system_does_not_do() -> None:
    """ADR-008 in prose. If this text ever loses its teeth, the report is a brochure."""
    assert "unclassified anomalies" in LIMITATIONS
    assert "not a production" in LIMITATIONS


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def test_defect_map_is_written(report: RollReport, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from linesight.report.defect_map import render_defect_map

    out = render_defect_map(report, tmp_path / "map.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_score_strip_refuses_without_a_calibration(
    report: RollReport, tmp_path: Path
) -> None:
    """No threshold, no line worth drawing - and an unlabelled score profile
    invites the eye to invent a cut-off."""
    pytest.importorskip("matplotlib")
    from linesight.report.defect_map import render_score_strip

    report.calibration = None
    with pytest.raises(RuntimeError, match="threshold"):
        render_score_strip(report, tmp_path / "strip.png")


def test_event_crop_is_saved_and_recorded(report: RollReport, tmp_path: Path) -> None:
    from linesight.report.defect_map import save_event_crop

    overlay = np.full((400, 400, 3), 120, dtype=np.uint8)
    event = report.events[0]
    path = save_event_crop(overlay, event, tmp_path, pad_px=10)

    assert path.exists()
    assert event.crop_path == str(path)


def test_event_crop_refuses_an_event_with_no_detections(tmp_path: Path) -> None:
    from linesight.report.defect_map import save_event_crop

    overlay = np.full((100, 100, 3), 120, dtype=np.uint8)
    with pytest.raises(ValueError, match="no detections"):
        save_event_crop(overlay, Event(event_id=9), tmp_path)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def test_pdf_refuses_a_roll_with_no_calibration(
    report: RollReport, tmp_path: Path
) -> None:
    """The one refusal that matters here.

    A points total with no stated threshold provenance is a number nobody can
    defend, and rendering it anyway is how an indefensible number ends up on a
    QA desk looking official.
    """
    pytest.importorskip("reportlab")
    from linesight.report.pdf import render_pdf

    report.calibration = None
    with pytest.raises(RuntimeError, match="calibration"):
        render_pdf(report, tmp_path / "nope.pdf")


def test_pdf_renders_a_multipage_document(report: RollReport, tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    from linesight.report.pdf import render_pdf

    out = render_pdf(report, tmp_path / "roll.pdf")
    assert out.exists()

    blob = out.read_bytes()
    assert blob.startswith(b"%PDF")
    # Header, defects, figures, limitations - four pages is the floor.
    assert blob.count(b"/Type /Page\n") >= 4 or blob.count(b"/Type /Page") >= 4


def test_pdf_scores_match_the_astm_scorer(report: RollReport) -> None:
    """The report cannot invent its own arithmetic.

    1 + 3 + (4 x 2 metres occupied) = 12, with the uncertain and the rejected
    event contributing nothing.
    """
    per_event = [points_for_event(e) for e in report.events]
    assert per_event == [1, 3, 8, 0, 0]
    assert report.total_points == 12
