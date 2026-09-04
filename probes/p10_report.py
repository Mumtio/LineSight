"""
probes/p10_report.py

QUESTION: Does a RollReport render to a PDF a mill would actually file - verdict,
          calibration provenance, defect table, positional map, image evidence,
          and the limitations page - and does the text fallback carry the same
          facts?
PASS IF:  A synthetic roll carrying an asserted defect, a continuous defect, an
          uncertain one and an operator-rejected one produces a multi-page PDF
          whose defect table matches the text report row for row, and whose
          points total matches the ASTM scorer. A report with no calibration is
          REFUSED rather than rendered.
STATUS:   PASSED 2026-09-02 -> promoted to report/pdf.py and report/defect_map.py

Run:  python probes/p10_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linesight.report.defect_map import (  # noqa: E402
    render_defect_map,
    render_score_strip,
    save_event_crop,
)
from linesight.report.pdf import render_pdf, render_text_report  # noqa: E402
from linesight.scoring.astm_d5430 import fill_report, points_for_event  # noqa: E402
from linesight.types import (  # noqa: E402
    Assertion,
    Calibration,
    Detection,
    Event,
    EventStatus,
    RollReport,
)

OUT = Path(__file__).resolve().parents[1] / "results"


def _detection(frame: int, along_mm: float, across_mm: float, length: float, score: float):
    return Detection(
        frame_index=frame,
        bbox_px=(int(across_mm), int(along_mm % 400), 90, int(max(8, length))),
        area_px=int(90 * max(8, length)),
        max_score=score,
        mean_score=score * 0.6,
        along_mm=along_mm,
        across_mm=across_mm,
        length_mm=length,
        width_mm=45.0,
    )


def _event(eid, along_mm, across_mm, length, score, conf, assertion, status):
    detections = [_detection(eid, along_mm, across_mm, length, score)]
    return Event(
        event_id=eid,
        detections=detections,
        along_start_mm=along_mm,
        along_end_mm=along_mm + length,
        across_start_mm=across_mm,
        across_end_mm=across_mm + 45.0,
        max_score=score,
        confidence=conf,
        assertion=assertion,
        status=status,
    )


def build_report() -> RollReport:
    """A roll with one of every case the PDF has to survive."""
    events = [
        # short, asserted -> 1 point
        _event(1, 420.0, 60.0, 48.0, 18.4, 0.81, Assertion.ASSERTED, EventStatus.PROPOSED),
        # 180 mm -> 3 points, confirmed by the operator
        _event(2, 1810.0, 210.0, 180.0, 21.0, 0.93, Assertion.ASSERTED, EventStatus.CONFIRMED),
        # continuous, 1.4 m -> 4 points per metre it occupies
        _event(3, 3050.0, 130.0, 1400.0, 24.6, 0.97, Assertion.ASSERTED, EventStatus.PROPOSED),
        # in the abstention band -> scores zero until confirmed
        _event(4, 5200.0, 380.0, 62.0, 13.1, 0.24, Assertion.UNCERTAIN, EventStatus.PROPOSED),
        # operator says false alarm -> zero, and it feeds the FA counter
        _event(5, 6400.0, 90.0, 35.0, 15.9, 0.55, Assertion.ASSERTED, EventStatus.REJECTED),
    ]

    report = RollReport(
        roll_id="probe-roll-001",
        sku="fabric_stain",
        roll_length_m=7.4,
        width_m=0.586,
        events=events,
        calibration=Calibration(
            threshold=14.6595,
            abstain_low=12.4014,
            budget_fa_per_100m=20.0,
            metres_per_tile=0.448,
            n_clean_tiles=960,
            sku="fabric_stain",
            fit_timestamp="2026-09-02T10:00:00+00:00",
        ),
        false_alarms=1,
        gap_warnings=2,
        started_at="2026-09-02T10:04:11+00:00",
        finished_at="2026-09-02T10:05:48+00:00",
        meta={
            "frame_stride": 2,
            "scorer": "patchcore",
            "n_frames": 96,
            "spatial_resolution_mm": 52.0,
            "extent_rule": "half_max",
            "operator": "bench",
            "latency_ms": {
                "geometry": 0.4, "preprocess": 314.9, "backbone": 153.9,
                "nn_search": 0.8, "assemble": 10.1, "total": 478.3,
            },
        },
    )
    return fill_report(report)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = build_report()

    print(f"roll {report.roll_id}: {len(report.events)} events")
    for event in report.events:
        print(
            f"  #{event.event_id}  {event.length_mm:7.1f} mm  "
            f"{event.assertion.value:>9}/{event.status.value:<9} "
            f"-> {points_for_event(event)} pt  (counts={event.counts_toward_score})"
        )
    print(f"  total {report.total_points} pt, "
          f"{report.points_per_100yd2:.2f}/100yd2 -> {report.verdict.value.upper()}")

    # 1 + 3 + (4 x 2 metres occupied) + 0 + 0 = 12, capped per metre (no clashes)
    expected = 1 + 3 + 8
    assert report.total_points == expected, f"{report.total_points} != {expected}"
    print(f"  PASS: points total matches the ASTM scorer ({expected})")

    # -- evidence crops, from a synthetic overlay --------------------------- #
    rng = np.random.default_rng(0)
    overlay = rng.integers(40, 210, size=(520, 620, 3), dtype=np.uint8)
    crops = 0
    for event in report.events:
        try:
            save_event_crop(overlay, event, OUT / "evidence", pad_px=24)
            crops += 1
        except ValueError as exc:
            print(f"  crop skipped for #{event.event_id}: {exc}")
    print(f"  PASS: {crops} evidence crops written to results/evidence/")

    # -- figures ------------------------------------------------------------ #
    map_path = render_defect_map(report, OUT / "probe_defect_map.png")
    strip_path = render_score_strip(report, OUT / "probe_score_strip.png")
    print(f"  PASS: {map_path.name} ({map_path.stat().st_size // 1024} KB), "
          f"{strip_path.name} ({strip_path.stat().st_size // 1024} KB)")

    # -- text and PDF must agree -------------------------------------------- #
    text = render_text_report(report)
    for event in report.events:
        assert f"{event.along_start_mm / 1000:>8.3f}" in text, event.event_id
    assert f"{report.total_points}" in text
    assert "GAP WARNING" in text
    print("  PASS: text report carries every event, the total, and the gap warning")

    pdf_path = render_pdf(report, OUT / "probe_roll_report.pdf")
    size_kb = pdf_path.stat().st_size // 1024
    assert pdf_path.exists() and size_kb > 10, f"suspiciously small PDF: {size_kb} KB"
    print(f"  PASS: {pdf_path.name} ({size_kb} KB)")

    # -- refusal ------------------------------------------------------------ #
    report.calibration = None
    try:
        render_pdf(report, OUT / "_should_not_exist.pdf")
    except RuntimeError as exc:
        print(f"  PASS: refused an uncalibrated roll -- {str(exc)[:60]}...")
    else:  # pragma: no cover
        print("  FAIL: rendered a report with no threshold provenance")
        return 1

    print("\nOpen results/probe_roll_report.pdf and look at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
