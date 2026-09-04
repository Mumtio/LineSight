"""Print the ArUco position tape - the physical half of L2 GEOMETRY.

The tape is what turns pixels into millimetres. Marker ``i`` sits with its top
edge at ``i * pitch_mm`` down the roll, so ``geometry.aruco`` recovers an
**absolute** position from the decoded ID (never an integrated speed, which
drifts) and a **measured** mm/px from the marker's known physical edge length.
Get the printed size wrong and every defect length in every report is wrong by
the same factor, silently.

Only the white strip carrying the markers goes on the fabric. Everything else on
a tape page - the segment number, the position range, the cut marks - sits
outside the cut line and is thrown away, so the strip is kept as narrow as the
detector allows: the marker plus one module of quiet zone on each side.

Segments are printed several to a page, in columns. A page of A4 holds five
32 mm strips side by side, so a metre of roll costs one sheet rather than five.
Cut them out, order them by segment number, butt them end to end.

    python tools/make_tape.py --length-m 2
    python tools/make_tape.py --length-m 10 --marker-mm 20 --pitch-mm 50

Print with **Actual size / 100% scale**, page scaling OFF. Then measure.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np

__all__ = ["build_tape_pdf", "layout", "marker_image", "quiet_zone_mm"]

#: Target print resolution for the rasterised markers.
DPI: int = 600

#: Absolute floor on the quiet zone, whatever the module size works out to.
MIN_QUIET_MM: float = 2.5

#: Space between columns: enough to get scissors down without clipping a strip.
GUTTER_MM: float = 5.0

_PAGE_W_MM, _PAGE_H_MM = 210.0, 297.0
_SIDE_MARGIN_MM = 8.0
_TOP_MARGIN_MM = 26.0        # room for the per-column header
_BOTTOM_MARGIN_MM = 16.0     # room for the page footer


def _modules(dictionary_name: str) -> int:
    """Total modules across a marker, including its one-module black border."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return int(dictionary.markerSize) + 2


def quiet_zone_mm(marker_mm: float, dictionary_name: str = "DICT_5X5_1000") -> float:
    """White margin either side of the marker, in millimetres.

    **One module**, which is what ArUco actually requires to segment a marker -
    not a fixed fraction of the marker. A quiet zone of 25% of the edge, which
    is the intuitive choice, makes a 40 mm marker need a 60 mm strip: more than
    twice the width, for no detection benefit. Strip width matters because only
    the strip goes on the fabric, and every millimetre of it is fabric that is
    not being inspected.
    """
    return max(MIN_QUIET_MM, marker_mm / _modules(dictionary_name))


def strip_width_mm(marker_mm: float, dictionary_name: str) -> float:
    """Total width of the printed strip: marker plus quiet zone on both sides."""
    return marker_mm + 2 * quiet_zone_mm(marker_mm, dictionary_name)


def marker_image(dictionary_name: str, marker_id: int, marker_mm: float) -> np.ndarray:
    """One marker, rasterised at ``DPI`` with integer module scaling.

    Integer scaling matters: a marker resampled to an arbitrary pixel size gets
    grey edges on its module boundaries, and a printer's halftoning turns those
    into ragged corners. Corner localisation is where mm/px comes from, so
    ragged corners are a scale error.

    Raises:
        ValueError: on an unknown dictionary name.
    """
    if not hasattr(cv2.aruco, dictionary_name):
        valid = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        raise ValueError(f"unknown ArUco dictionary {dictionary_name!r}. Valid: {valid}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    modules = _modules(dictionary_name)
    cell_px = max(1, round(marker_mm / 25.4 * DPI / modules))
    return cv2.aruco.generateImageMarker(dictionary, marker_id, modules * cell_px)


def _capacity(dictionary_name: str) -> int:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return int(dictionary.bytesList.shape[0])


def layout(marker_mm: float, pitch_mm: float, dictionary_name: str) -> dict:
    """How many strips fit across a page, and how much roll each one carries.

    A column holds a whole number of pitches so that butting two segments
    together preserves the pitch exactly; anything else accumulates an error at
    every join.

    Raises:
        ValueError: if a single strip will not fit the page width, or if a
            single pitch will not fit the page height.
    """
    strip_w = strip_width_mm(marker_mm, dictionary_name)
    usable_w = _PAGE_W_MM - 2 * _SIDE_MARGIN_MM
    usable_h = _PAGE_H_MM - _TOP_MARGIN_MM - _BOTTOM_MARGIN_MM

    columns = int(usable_w // (strip_w + GUTTER_MM))
    if columns < 1:
        raise ValueError(
            f"a {strip_w:.0f} mm strip does not fit {usable_w:.0f} mm of usable "
            f"page width. Reduce --marker-mm."
        )
    pitches = int(usable_h // pitch_mm)
    if pitches < 1:
        raise ValueError(
            f"a {pitch_mm:.0f} mm pitch does not fit {usable_h:.0f} mm of usable "
            f"page height. Reduce --pitch-mm."
        )
    return {
        "strip_w": strip_w,
        "columns": columns,
        "pitches_per_column": pitches,
        "segment_mm": pitches * pitch_mm,
        "column_w": strip_w + GUTTER_MM,
        "usable_h": usable_h,
    }


def _reader(image: np.ndarray):
    """numpy -> something reportlab can draw, without touching the filesystem."""
    from PIL import Image
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


# --------------------------------------------------------------------------- #
# Calibration pages
# --------------------------------------------------------------------------- #


def _instructions_page(canvas, mm, colors, args, plan: dict) -> None:
    height = _PAGE_H_MM * mm
    y = height - 25 * mm

    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(20 * mm, y, "LineSight position tape")
    y -= 7 * mm
    canvas.setFont("Helvetica", 9)
    canvas.drawString(20 * mm, y,
                      f"{args.aruco}   markers {args.marker_mm:.0f} mm   "
                      f"pitch {args.pitch_mm:.0f} mm   "
                      f"strip {plan['strip_w']:.0f} mm wide   "
                      f"covers {args.length_m:.1f} m")

    y -= 12 * mm
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(20 * mm, y, "Print at 100% / Actual size. Page scaling OFF.")

    y -= 10 * mm
    canvas.setFont("Helvetica", 9.5)
    for line in [
        "1. Print every page at actual size. 'Fit to page' silently rescales the",
        "   markers, and every defect length in every report inherits that error.",
        "2. Measure the 250 mm reference line on page 2 with a steel ruler.",
        f"3. Measure the marker below edge to edge. It must read "
        f"{args.marker_mm:.0f} mm.",
        "4. Cut out each strip along its outline. ONLY the white strip with the",
        "   markers goes on the fabric - the numbers beside it are for ordering",
        "   the pieces and are thrown away with the offcut.",
        "5. Order the strips by their segment number, then butt them end to end.",
        "   The pitch continues across a join only if you cut on the line.",
        "6. Tape along one selvedge so the tape travels WITH the fabric. Cover",
        "   with clear packing tape so it stays flat. Matte paper - gloss blows out.",
        "7. Segment 1 starts at position 0: the start of the roll.",
    ]:
        canvas.drawString(20 * mm, y, line)
        y -= 5.2 * mm

    y -= 8 * mm
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(20 * mm, y, "Marker 0 at true size - measure it edge to edge")
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(20 * mm, y - 5 * mm,
                      f"Black border to black border must read {args.marker_mm:.0f} mm.")
    y -= (args.marker_mm + 11) * mm
    canvas.drawImage(_reader(marker_image(args.aruco, 0, args.marker_mm)),
                     20 * mm, y, args.marker_mm * mm, args.marker_mm * mm)
    canvas.setFont("Helvetica", 8)
    canvas.drawString((20 + args.marker_mm + 8) * mm, y + args.marker_mm / 2 * mm,
                      "ID 0  =  position 0 mm  =  the start of the roll")

    y -= 12 * mm
    canvas.setFont("Helvetica-Oblique", 8.5)
    # BOTH keys, not just the marker. A printer scales every dimension on the
    # page, so the pitch between markers shrinks by the same factor as the
    # markers themselves. Correcting only marker_length_mm fixes the mm/px scale
    # and leaves every reported POSITION wrong by the scale error - which is the
    # more damaging of the two, because it is the number an operator uses to walk
    # to the defect.
    for line in [
        "If the reference line on page 2 reads X mm instead of 250, the printer",
        "scaled the page by X/250 and EVERY dimension shrank with it.",
        "Best: reprint with 'Actual size' / 100%, page scaling OFF, then re-measure.",
        "Otherwise correct BOTH keys in the SKU config:",
        f"    geometry.marker_length_mm = {args.marker_mm:.0f} * X/250",
        f"    geometry.marker_pitch_mm  = {args.pitch_mm:.0f} * X/250",
        "and state the correction in the report. Correcting only the first fixes",
        "the mm-per-pixel scale and leaves every reported position wrong.",
    ]:
        canvas.drawString(20 * mm, y, line)
        y -= 4.8 * mm

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(20 * mm, 12 * mm,
                      "LineSight  -  tools/make_tape.py  -  page 1 of 2, do not cut")
    canvas.setFillColor(colors.black)
    canvas.showPage()


def _ruler_page(canvas, mm, colors) -> None:
    """The 250 mm reference line, alone.

    It gets its own page because it cannot share one: A4 is 210 mm wide, so a
    horizontal 250 mm line runs off the side, and stacked under the instructions
    a vertical one runs off the bottom. Both were tried.
    """
    height = _PAGE_H_MM * mm
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(20 * mm, height - 16 * mm,
                      "Reference line - measure me with a steel ruler")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(20 * mm, height - 22 * mm,
                      "Top tick to bottom tick must be exactly 250 mm.")

    ruler_x = 30 * mm
    ruler_top = height - 30 * mm
    canvas.setLineWidth(0.8)
    canvas.line(ruler_x, ruler_top, ruler_x, ruler_top - 250 * mm)
    for step in range(0, 251, 10):
        tick = 6 * mm if step % 50 == 0 else 3 * mm
        canvas.setLineWidth(0.8 if step % 50 == 0 else 0.4)
        canvas.line(ruler_x, ruler_top - step * mm, ruler_x + tick, ruler_top - step * mm)
    canvas.setFont("Helvetica-Bold", 8)
    for step in range(0, 251, 50):
        canvas.drawString(ruler_x + 8 * mm, ruler_top - step * mm - 1.2 * mm,
                          f"{step} mm")

    canvas.setFont("Helvetica", 9)
    note_y = ruler_top - 40 * mm
    for line in [
        "This line is drawn at exactly 250 mm.",
        "",
        "If your ruler disagrees, the printer scaled",
        "the page, and every marker on every tape",
        "page is scaled by the same factor.",
        "",
        "A 1% scale error puts a 1% error on every",
        "defect length in every ASTM report, and",
        "nothing downstream can detect it.",
    ]:
        canvas.drawString(ruler_x + 34 * mm, note_y, line)
        note_y -= 5.2 * mm

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(20 * mm, 12 * mm,
                      "LineSight  -  page 2 of 2, do not cut")
    canvas.setFillColor(colors.black)
    canvas.showPage()


# --------------------------------------------------------------------------- #
# Tape pages
# --------------------------------------------------------------------------- #


def _tape_page(canvas, mm, colors, args, plan: dict, first_segment: int,
               n_segments: int) -> None:
    """One page of side-by-side tape segments."""
    height = _PAGE_H_MM * mm
    top = height - _TOP_MARGIN_MM * mm
    segment_mm = plan["segment_mm"]
    strip_w = plan["strip_w"]
    quiet = quiet_zone_mm(args.marker_mm, args.aruco)

    for c in range(n_segments):
        segment = first_segment + c
        start_mm = segment * segment_mm
        x = (_SIDE_MARGIN_MM + c * plan["column_w"]) * mm
        bottom = top - segment_mm * mm

        # Header, outside the cut line: what this strip is and where it goes.
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(x, top + 10 * mm, f"#{segment + 1}")
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#444444"))
        canvas.drawString(x, top + 5.5 * mm, f"{start_mm:.0f}-")
        canvas.drawString(x, top + 2 * mm, f"{start_mm + segment_mm:.0f} mm")
        canvas.setFillColor(colors.black)

        # The cut line IS the strip outline. Dashed so it reads as "cut", not
        # as part of the tape.
        canvas.setStrokeColor(colors.HexColor("#cc0000"))
        canvas.setLineWidth(0.4)
        canvas.setDash(3, 2)
        canvas.rect(x, bottom, strip_w * mm, segment_mm * mm, stroke=1, fill=0)
        canvas.setDash()
        canvas.setStrokeColor(colors.black)

        for k in range(plan["pitches_per_column"]):
            marker_id = int(round(start_mm / args.pitch_mm)) + k
            along_mm = marker_id * args.pitch_mm
            y_bottom = top - (along_mm - start_mm) * mm - args.marker_mm * mm
            canvas.drawImage(
                _reader(marker_image(args.aruco, marker_id, args.marker_mm)),
                x + quiet * mm, y_bottom, args.marker_mm * mm, args.marker_mm * mm,
            )
            # Tiny id inside the strip, in the white gap below the marker. Small
            # enough not to intrude on the next marker's quiet zone, and it is
            # what lets an operator read a position off the fabric by eye.
            if args.ids:
                canvas.setFont("Helvetica", 5)
                canvas.setFillColor(colors.HexColor("#999999"))
                canvas.drawString(x + quiet * mm, y_bottom - 3.5 * mm,
                                  f"{marker_id} | {along_mm:.0f}mm")
                canvas.setFillColor(colors.black)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(_SIDE_MARGIN_MM * mm, 9 * mm,
                      f"LineSight tape  -  segments {first_segment + 1}"
                      f"-{first_segment + n_segments}  -  {args.aruco}, "
                      f"{args.marker_mm:.0f} mm markers at {args.pitch_mm:.0f} mm pitch  "
                      f"-  cut on the dashed outline, order by number, butt end to end")
    canvas.setFillColor(colors.black)
    canvas.showPage()


def build_tape_pdf(args) -> Path:
    """Write two calibration pages, then enough tape pages to cover ``length_m``.

    Raises:
        ValueError: if the marker does not fit the pitch, or the dictionary
            cannot address the requested roll length.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as pdfcanvas

    strip_w = strip_width_mm(args.marker_mm, args.aruco)
    if args.marker_mm + quiet_zone_mm(args.marker_mm, args.aruco) > args.pitch_mm:
        raise ValueError(
            f"a {args.marker_mm:.0f} mm marker needs at least "
            f"{args.marker_mm + quiet_zone_mm(args.marker_mm, args.aruco):.0f} mm of "
            f"pitch to keep its quiet zone, but the pitch is {args.pitch_mm:.0f} mm."
        )

    total_mm = args.length_m * 1000.0
    needed = int(np.ceil(total_mm / args.pitch_mm)) + 1
    capacity = _capacity(args.aruco)
    if needed > capacity:
        raise ValueError(
            f"{args.aruco} holds {capacity} markers = "
            f"{capacity * args.pitch_mm / 1000:.1f} m at {args.pitch_mm:.0f} mm "
            f"pitch, but {args.length_m:.1f} m needs {needed}. Use a larger "
            "dictionary or a coarser pitch."
        )

    plan = layout(args.marker_mm, args.pitch_mm, args.aruco)
    total_segments = int(np.ceil(total_mm / plan["segment_mm"]))
    per_page = plan["columns"]
    n_pages = int(np.ceil(total_segments / per_page))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = pdfcanvas.Canvas(str(out), pagesize=A4)
    canvas.setTitle(f"LineSight position tape - {args.length_m:.1f} m")

    _instructions_page(canvas, mm, colors, args, plan)
    _ruler_page(canvas, mm, colors)
    for page in range(n_pages):
        first = page * per_page
        _tape_page(canvas, mm, colors, args, plan, first,
                   min(per_page, total_segments - first))
    canvas.save()

    covered = total_segments * plan["segment_mm"] / 1000.0
    print(f"wrote {out}")
    print(f"  {args.aruco}, {args.marker_mm:.0f} mm markers at "
          f"{args.pitch_mm:.0f} mm pitch")
    print(f"  strip width      {strip_w:.1f} mm "
          f"(marker + {quiet_zone_mm(args.marker_mm, args.aruco):.1f} mm quiet each side)")
    print(f"  segment length   {plan['segment_mm']:.0f} mm, "
          f"{plan['columns']} per page")
    print(f"  {n_pages} tape page(s) -> {total_segments} segments = {covered:.2f} m "
          f"(IDs 0..{needed - 1})")
    print("\n  Print at 100% / Actual size, then MEASURE the reference line on page 2.")
    print("  Set these in your SKU config:")
    print(f"    geometry.aruco_dict: {args.aruco}")
    print(f"    geometry.marker_length_mm: {args.marker_mm:.1f}")
    print(f"    geometry.marker_pitch_mm: {args.pitch_mm:.1f}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--length-m", type=float, default=2.0)
    parser.add_argument("--marker-mm", type=float, default=25.0,
                        help="physical marker edge; drives the strip width")
    parser.add_argument("--pitch-mm", type=float, default=100.0)
    parser.add_argument("--aruco", default="DICT_5X5_1000")
    parser.add_argument("--ids", action="store_true",
                        help="print a tiny id/position inside the strip, for humans")
    parser.add_argument("--out", default="results/position_tape.pdf")
    args = parser.parse_args()
    build_tape_pdf(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
