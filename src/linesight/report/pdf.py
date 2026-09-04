"""The roll report PDF - the artifact a mill would actually file.

Sections, in order: header (roll, SKU, operator, timestamps), the verdict and
points/100 yd2, the calibration provenance, the defect table, the positional
defect map, image evidence per event, and the scope-and-limitations page.

The last section is not boilerplate. A document that states a verdict without
stating what produced it invites the reader to assume capabilities the system
does not have, so the limitations page carries the frame sampling factor, the
abstention band, the gap-warning count and what the system does not do (no
defect classification - ADR-008) on the same page as the result.

``render_pdf`` builds the document. ``render_text_report`` renders the same
facts as plain text and is what the CLI prints, so the filed report and the
terminal cannot disagree about a roll. The ``_``-prefixed builders below are
passed their reportlab pieces as arguments, which is what keeps importing this
module free of a reportlab import.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..types import Assertion, EventStatus, RollReport, Verdict

__all__ = ["LIMITATIONS", "render_pdf", "render_text_report"]

#: The scope paragraph, printed verbatim on the last page of every report.
LIMITATIONS: str = """\
Scope and limitations. This is a bench prototype, not a production inspection
system. Detections are unclassified anomalies: the system reports
that a region is unlike defect-free fabric, not what kind of defect it is.
Positions come from ArUco markers on a hand-pulled bench rig; frames are
sampled, and the sampling factor is stated with every latency figure. The
threshold is derived from a stated false-alarm budget on held-out clean fabric,
and is only as trustworthy as that sample is large. Benchmark numbers on MVTec
AD are for validation only under its non-commercial licence; nothing derived
from it is deployed.
"""

#: Verdict -> (fill, text). A QA document is skimmed, and the disposition is the
#: one thing that must survive skimming.
_VERDICT_COLOURS: dict[str, tuple[str, str]] = {
    Verdict.PASS.value: ("#e7f4ea", "#1a7f37"),
    Verdict.HOLD.value: ("#fdf4dd", "#8a6a00"),
    Verdict.REJECT.value: ("#fbe9e9", "#c1272d"),
}


def _require_reportlab():
    """Import reportlab, or say which extra installs it.

    Raises:
        ImportError: naming the extra that provides it, so the message contains
            the fix rather than only the symptom.
    """
    try:
        import reportlab  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by absence
        raise ImportError(
            "the PDF report needs reportlab: pip install -e '.[report]'"
        ) from exc


# --------------------------------------------------------------------------- #
# Text - the fallback deliverable
# --------------------------------------------------------------------------- #


def render_text_report(report: RollReport) -> str:
    """The same content as plain text.

    The fallback when the PDF stack is cut, and what the CLI prints. Kept
    equivalent in content so cutting the PDF loses formatting, not information.

    "Equivalent in content" is a claim that rots unless something enforces it,
    so ``cli.print_report`` prints this string rather than formatting its own -
    there is one implementation, and the PDF and the terminal read from the
    same set of facts.
    """
    from ..scoring.astm_d5430 import points_for_event

    lines: list[str] = ["", f"ROLL {report.roll_id}   sku={report.sku}"]
    lines.append(
        f"  inspected {report.roll_length_m:.2f} m x {report.width_m:.2f} m "
        f"= {report.inspected_area_m2:.2f} m^2"
    )

    if report.events:
        lines.append("")
        lines.append(
            f"  {'#':>3}  {'at (m)':>8}  {'cross(mm)':>9}  {'len(mm)':>8}  "
            f"{'wid(mm)':>8}  {'score':>7}  {'assert':>9}  {'pts':>3}"
        )
        lines.append(
            f"  {'-' * 3}  {'-' * 8}  {'-' * 9}  {'-' * 8}  "
            f"{'-' * 8}  {'-' * 7}  {'-' * 9}  {'-' * 3}"
        )
        for event in report.events:
            flag = " " if event.assertion is Assertion.ASSERTED else "?"
            lines.append(
                f"  {event.event_id:>3}  {event.along_start_mm / 1000:>8.3f}  "
                f"{event.across_start_mm:>9.1f}  {event.length_mm:>8.1f}  "
                f"{event.width_mm:>8.1f}  {event.max_score:>7.3f}  "
                f"{event.assertion.value:>9}  {points_for_event(event):>3}{flag}"
            )
    else:
        lines.append("")
        lines.append("  no defects detected")

    lines.append("")
    lines.append(f"  total points        {report.total_points}")
    lines.append(f"  points / 100 yd^2   {report.points_per_100yd2:.2f}")
    lines.append(f"  VERDICT             {report.verdict.value.upper()}")

    calibration = report.calibration
    if calibration is not None:
        lines.append("")
        lines.append(f"  threshold           {calibration.threshold:.4f}")
        if calibration.n_clean_tiles:
            lines.append(
                f"  from a budget of    {calibration.budget_fa_per_100m} FA/100 m "
                f"on {calibration.n_clean_tiles} held-out clean tiles"
            )
        else:
            lines.append(f"  provenance          {calibration.fit_timestamp}")
        lines.append(
            f"  false alarms        {report.false_alarms} "
            f"({report.fa_per_100m:.2f}/100 m)"
        )

    if report.gap_warnings:
        lines.append("")
        lines.append(
            f"  ** {report.gap_warnings} GAP WARNING(S): position could not be "
            "vouched for. That fabric is uninspected, not clean. **"
        )

    resolution = report.meta.get("spatial_resolution_mm")
    if resolution:
        rule = report.meta.get("extent_rule", "half_max")
        lines.append("")
        lines.append(f"  spatial resolution  ~{resolution:.0f} mm (extent rule: {rule})")
        lines.append(
            "    extents at or below this are resolution-limited, not measurements"
        )

    stride = report.meta.get("frame_stride", 1)
    if stride and stride > 1:
        lines.append("")
        lines.append(
            f"  note: every {stride}th frame was processed (stated, not hidden)"
        )

    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def render_pdf(
    report: RollReport,
    out_path: Path | str,
    include_evidence: bool = True,
    max_evidence_images: int = 24,
) -> Path:
    """Render a complete roll report.

    Raises:
        RuntimeError: if the report has no calibration attached - a points total
            with no stated threshold provenance is not a document worth issuing.
    """
    _require_reportlab()

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    if report.calibration is None:
        raise RuntimeError(
            f"roll {report.roll_id!r} has no calibration attached. A points "
            "total with no stated threshold provenance is a number nobody can "
            "defend - run `linesight calibrate --sku <sku>` and inspect again."
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body", parent=styles["BodyText"], fontSize=8.5, leading=12
    )
    small = ParagraphStyle("small", parent=body, fontSize=7.5, textColor=colors.HexColor("#555555"))
    heading = ParagraphStyle(
        "heading", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=4
    )
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=2)

    story: list[object] = []

    # -- header ------------------------------------------------------------- #
    story.append(Paragraph("Roll Inspection Report", title))
    story.append(
        Paragraph(
            "LineSight &mdash; cold-start visual inspection, scored under ASTM D5430",
            small,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        _kv_table(
            [
                ("Roll", report.roll_id),
                ("SKU", report.sku),
                ("Operator", str(report.meta.get("operator", "-"))),
                ("Started", report.started_at or "-"),
                ("Finished", report.finished_at or "-"),
                ("Scorer", str(report.meta.get("scorer", "-"))),
            ],
            Table, TableStyle, colors, mm, body,
        )
    )

    # -- verdict ------------------------------------------------------------ #
    fill, text_colour = _VERDICT_COLOURS.get(report.verdict.value, ("#eeeeee", "#333333"))
    # Its own style: `body` has leading 12, and a 20 pt verdict set on a 12 pt
    # line clips its own descenders and eats the subtitle underneath it.
    verdict_style = ParagraphStyle(
        "verdict", parent=body, fontSize=20, leading=24, spaceAfter=0
    )
    verdict_table = Table(
        [[
            Paragraph(
                f"<font size=20 color='{text_colour}'><b>"
                f"{report.verdict.value.upper()}</b></font><br/>"
                f"<font size=9 color='{text_colour}'>"
                f"{report.points_per_100yd2:.2f} points / 100 yd&sup2;"
                f" &nbsp;&middot;&nbsp; {report.total_points} points total</font>",
                verdict_style,
            )
        ]],
        colWidths=[165 * mm],
    )
    verdict_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill)),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(text_colour)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(Spacer(1, 10))
    story.append(verdict_table)

    story.append(Paragraph("Inspection", heading))
    story.append(
        _kv_table(
            [
                ("Inspected length", f"{report.roll_length_m:.2f} m"),
                ("Web width", f"{report.width_m:.2f} m"),
                ("Inspected area", f"{report.inspected_area_m2:.2f} m²"),
                ("Events found", str(len(report.events))),
                ("Counted toward score", str(sum(1 for e in report.events if e.counts_toward_score))),
                ("Operator rejections", f"{report.false_alarms} ({report.fa_per_100m:.2f} / 100 m)"),
            ],
            Table, TableStyle, colors, mm, body,
        )
    )

    # -- calibration provenance --------------------------------------------- #
    calibration = report.calibration
    story.append(Paragraph("Calibration provenance", heading))
    story.append(
        Paragraph(
            "The threshold was not chosen. The operator stated a false-alarm "
            "budget and it is the corresponding conformal quantile of anomaly "
            "scores on held-out defect-free fabric.",
            small,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        _kv_table(
            [
                ("Threshold", f"{calibration.threshold:.4f}"),
                ("Abstain floor", f"{calibration.abstain_low:.4f}"),
                ("Stated budget", f"{calibration.budget_fa_per_100m:g} FA / 100 m"),
                ("Held-out clean tiles", str(calibration.n_clean_tiles)),
                ("Finest rate this sample resolves", f"{calibration.achievable_fa_per_100m:.3g} FA / 100 m"),
                ("Calibrated", calibration.fit_timestamp or "-"),
            ],
            Table, TableStyle, colors, mm, body,
        )
    )

    if report.gap_warnings:
        story.append(Spacer(1, 8))
        story.append(
            _callout(
                f"{report.gap_warnings} gap warning(s). Roll position could not "
                "be vouched for over part of this roll. That fabric is "
                "uninspected, not clean.",
                Table, TableStyle, colors, mm, body,
            )
        )

    # -- defect table ------------------------------------------------------- #
    story.append(PageBreak())
    story.append(Paragraph("Defects", heading))
    story.append(_defect_table(report, Table, TableStyle, colors, mm, body, small))

    # -- figures ------------------------------------------------------------ #
    with tempfile.TemporaryDirectory(prefix="linesight-report-") as tmp:
        tmp_dir = Path(tmp)
        figures: list[tuple[str, Path]] = []

        from .defect_map import render_defect_map, render_score_strip

        figures.append(
            (
                "Positional defect map",
                render_defect_map(report, tmp_dir / "map.png", figsize=(9.0, 2.6), dpi=200),
            )
        )
        figures.append(
            (
                "Score against position",
                render_score_strip(
                    report, tmp_dir / "strip.png", dpi=200, figsize=(9.0, 2.2)
                ),
            )
        )

        story.append(PageBreak())
        for caption, path in figures:
            story.append(Paragraph(caption, heading))
            story.append(_fit_image(path, 165 * mm, Image))
            story.append(Spacer(1, 10))

        # -- evidence ------------------------------------------------------- #
        if include_evidence:
            crops = [
                e for e in report.events
                if e.crop_path and Path(e.crop_path).exists()
            ][:max_evidence_images]
            if crops:
                story.append(PageBreak())
                story.append(Paragraph("Image evidence", heading))
                story.append(
                    Paragraph(
                        "Heatmap composited over the fabric, boxed at the "
                        "component the system flagged. One crop per event, "
                        f"showing {len(crops)} of {len(report.events)}.",
                        small,
                    )
                )
                story.append(Spacer(1, 6))
                story.append(
                    _evidence_grid(crops, Table, TableStyle, colors, mm, Image, small)
                )

        # -- scope and limitations ------------------------------------------ #
        story.append(PageBreak())
        story.append(Paragraph("Scope and limitations", heading))
        for paragraph in LIMITATIONS.strip().split("\n\n"):
            story.append(Paragraph(paragraph.replace("\n", " "), body))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 4))
        story.append(_measured_scope_table(report, Table, TableStyle, colors, mm, body))

        latency = report.meta.get("latency_ms") or {}
        if latency:
            story.append(Paragraph("Measured latency", heading))
            story.append(
                Paragraph(
                    "Median milliseconds per <i>processed</i> frame, on the "
                    "hardware this run used. Published with the frame stride, "
                    "never without it.",
                    small,
                )
            )
            story.append(Spacer(1, 4))
            story.append(
                _kv_table(
                    [
                        (stage, f"{value:.1f} ms")
                        for stage, value in latency.items()
                        if stage != "frame_stride"
                    ],
                    Table, TableStyle, colors, mm,
                )
            )

        document = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=22 * mm,
            rightMargin=22 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"LineSight roll report {report.roll_id}",
            author="LineSight",
        )
        document.build(
            story,
            onFirstPage=lambda canvas, doc: _footer(canvas, doc, report, colors, mm),
            onLaterPages=lambda canvas, doc: _footer(canvas, doc, report, colors, mm),
        )

    return out


# --------------------------------------------------------------------------- #
# Flowable builders - kept private, and passed their reportlab pieces so that
# importing this module never costs a reportlab import.
# --------------------------------------------------------------------------- #


def _kv_table(rows, Table, TableStyle, colors, mm, value_style=None):
    """Two-column label/value table. The report's default shape.

    Values are wrapped in a Paragraph when a style is supplied, because some of
    them are sentences - the spatial-resolution row explains what the number
    does and does not mean - and a bare string in a reportlab cell does not
    wrap, it runs off the right margin and out of the page.
    """
    from reportlab.platypus import Paragraph

    body = [
        [k, Paragraph(str(v), value_style) if value_style is not None else str(v)]
        for k, v in rows
    ]
    table = Table(body, colWidths=[55 * mm, 110 * mm])
    table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e4e4e0")),
        ])
    )
    return table


def _callout(text, Table, TableStyle, colors, mm, style):
    """A boxed warning. Used for gap warnings, which must not read as a footnote."""
    from reportlab.platypus import Paragraph

    table = Table([[Paragraph(f"<b>{text}</b>", style)]], colWidths=[165 * mm])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbe9e9")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#c1272d")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    return table


def _defect_table(report, Table, TableStyle, colors, mm, body, small):
    """One row per event, mirroring the terminal table's columns.

    Uncertain events are shaded and carry a '?', because a reader scanning for
    the total needs to see at a glance which rows contributed nothing to it.
    """
    from reportlab.platypus import Paragraph

    from ..scoring.astm_d5430 import points_for_event

    if not report.events:
        return Paragraph("No defects detected on this roll.", body)

    header = ["#", "at (m)", "cross (mm)", "len (mm)", "wid (mm)", "score", "state", "pts"]
    rows: list[list[str]] = [header]
    shaded: list[int] = []

    for index, event in enumerate(report.events, start=1):
        if event.status is EventStatus.REJECTED:
            state = "rejected"
        elif event.status is EventStatus.CONFIRMED:
            state = "confirmed"
        else:
            state = event.assertion.value
        if not event.counts_toward_score:
            shaded.append(index)
        rows.append([
            str(event.event_id),
            f"{event.along_start_mm / 1000:.3f}",
            f"{event.across_start_mm:.1f}",
            f"{event.length_mm:.1f}",
            f"{event.width_mm:.1f}",
            f"{event.max_score:.3f}",
            state,
            str(points_for_event(event)),
        ])

    table = Table(
        rows,
        colWidths=[12 * mm, 22 * mm, 24 * mm, 22 * mm, 22 * mm, 22 * mm, 26 * mm, 15 * mm],
        repeatRows=1,
    )
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (6, 0), (6, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ececea")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafaf8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row in shaded:
        style.append(("TEXTCOLOR", (0, row), (-1, row), colors.HexColor("#8a6a00")))
    table.setStyle(TableStyle(style))
    return table


def _fit_image(path, max_width, Image, max_height=None):
    """Scale a PNG to fit a box, preserving aspect.

    ``max_height`` matters for evidence crops: a defect that is tall and narrow
    in the frame produces a tall crop, and scaling that to the column width
    alone makes one cell three times the height of its neighbours and pushes
    the grid off the page.
    """
    from reportlab.lib.utils import ImageReader

    width_px, height_px = ImageReader(str(path)).getSize()
    scale = max_width / float(width_px)
    if max_height is not None and height_px * scale > max_height:
        scale = max_height / float(height_px)
    return Image(str(path), width=width_px * scale, height=height_px * scale)


def _evidence_grid(events, Table, TableStyle, colors, mm, Image, small):
    """Three crops per row, each captioned with its position and penalty."""
    from reportlab.platypus import Paragraph

    from ..scoring.astm_d5430 import points_for_event

    cell_width = 48 * mm
    cells: list[object] = []
    for event in events:
        caption = (
            f"#{event.event_id} &middot; {event.along_start_mm / 1000:.3f} m "
            f"&middot; {event.length_mm:.0f} mm &middot; {points_for_event(event)} pt"
        )
        # Image over caption is TWO ROWS of one column. Passing them as one row
        # of two put the caption beside the crop, overlapping the next cell.
        cell = Table(
            [
                [_fit_image(Path(event.crop_path), cell_width, Image, max_height=48 * mm)],
                [Paragraph(caption, small)],
            ],
            colWidths=[cell_width],
        )
        cell.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ])
        )
        cells.append(cell)

    rows = [cells[i : i + 3] for i in range(0, len(cells), 3)]
    for row in rows:
        while len(row) < 3:
            row.append("")

    table = Table(rows, colWidths=[(cell_width + 5 * mm)] * 3)
    table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    return table


def _measured_scope_table(report, Table, TableStyle, colors, mm, value_style):
    """The specifics that make the limitations page a measurement, not a hedge."""
    rows = [("Frame stride", f"every {report.meta.get('frame_stride', 1)} frame(s)")]

    resolution = report.meta.get("spatial_resolution_mm")
    if resolution:
        rows.append((
            "Spatial resolution",
            f"~{resolution:.0f} mm (extent rule: "
            f"{report.meta.get('extent_rule', 'half_max')}) - extents at or "
            "below this are resolution-limited, not measurements",
        ))
    if report.meta.get("n_frames"):
        rows.append(("Frames processed", str(report.meta["n_frames"])))
    rows.append(("Defect classification", "none - every event is an unclassified anomaly (ADR-008)"))
    return _kv_table(rows, Table, TableStyle, colors, mm, value_style)


def _footer(canvas, doc, report, colors, mm) -> None:
    """Roll id on the left, page number on the right, on every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(
        22 * mm, 12 * mm,
        f"LineSight  ·  roll {report.roll_id}  ·  sku {report.sku}  "
        f"·  {report.verdict.value.upper()}",
    )
    canvas.drawRightString(188 * mm, 12 * mm, f"page {doc.page}")
    canvas.restoreState()
