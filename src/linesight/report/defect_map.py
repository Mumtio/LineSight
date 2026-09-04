"""The positional defect map - where the defects are, at a glance.

Scatter: x = metres along the roll, y = cross-width position, marker size =
ASTM points, colour = confidence. Uncertain events drawn hollow, so the
abstention band is visible rather than implied.

This is the plot a mill's cutting room actually wants: it says where to plan
around the damage, which a points total does not.

Three renderers, all writing PNGs the PDF embeds: ``render_defect_map`` (the
roll strip), ``render_score_strip`` (score against position, with the threshold
drawn on it) and ``save_event_crop`` (one evidence image per event). All three
need matplotlib or OpenCV from the ``report`` extra; the defect table carries
the same facts in plain text when they are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import Assertion, EventStatus, RollReport

__all__ = ["render_defect_map", "render_score_strip", "save_event_crop"]

#: Points at or above this get the largest marker. A continuous defect scores 4
#: per metre it occupies, so an unbounded size scale would let one 6 m warp line
#: draw a blob covering half the roll and hide everything near it.
_POINTS_FOR_MAX_MARKER: float = 8.0

_MIN_MARKER_AREA: float = 28.0
_MAX_MARKER_AREA: float = 260.0


def _figure(figsize: tuple[float, float], dpi: int):
    """A bare Agg figure - no pyplot, hence no global state and no backend.

    Reports get rendered from a FastAPI worker thread and from the CLI. pyplot
    keeps a process-wide figure registry that is not thread-safe, and the
    failure it produces is a blank or interleaved image rather than an
    exception, which is the worst kind.

    Raises:
        ImportError: if matplotlib is not installed, naming the extra that
            provides it.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised by absence, not by tests
        raise ImportError(
            "the defect map needs matplotlib: pip install -e '.[report]'"
        ) from exc

    figure = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(figure)
    return figure


def _marker_area(points: int) -> float:
    """ASTM points -> scatter marker area, clamped at both ends.

    Clamped below as well as above: a zero-point event - uncertain, or operator
    rejected - still has to be *visible*, because "the system saw this and did
    not charge you for it" is information the cutting room wants.
    """
    fraction = min(1.0, max(0.0, float(points) / _POINTS_FOR_MAX_MARKER))
    return _MIN_MARKER_AREA + fraction * (_MAX_MARKER_AREA - _MIN_MARKER_AREA)


def render_defect_map(
    report: RollReport,
    out_path: Path | str,
    figsize: tuple[float, float] = (12.0, 3.0),
    dpi: int = 150,
) -> Path:
    """Draw the roll as a long horizontal strip with defects marked on it.

    Aspect is deliberately wide because a roll is: squashing 50 m into a square
    would misrepresent how sparse the defects are.

    Every event is drawn, including the ones that scored zero. Asserted events
    are filled and coloured by confidence; uncertain ones are hollow, and
    operator-rejected ones are hollow and grey. A map that showed only the
    events which cost points would quietly hide the abstention band, which is
    the one part of the calibration story a QA manager needs to see.

    Returns:
        The path written.
    """
    from matplotlib.patches import Rectangle

    from ..scoring.astm_d5430 import CONTINUOUS_MIN_MM, points_for_event

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    figure = _figure(figsize, dpi)
    axes = figure.add_subplot(111)

    width_mm = max(1.0, report.width_m * 1000.0)
    length_m = max(report.roll_length_m, 0.001)

    # The web itself, so an empty roll still reads as "we looked at this and
    # found nothing" rather than as an axis someone forgot to populate.
    axes.add_patch(
        Rectangle(
            (0.0, 0.0), length_m, width_mm, facecolor="#f2f2ef", edgecolor="#d8d8d2",
            linewidth=0.8, zorder=0,
        )
    )

    asserted_x, asserted_y, asserted_area, asserted_conf = [], [], [], []
    uncertain_x, uncertain_y, uncertain_area = [], [], []
    rejected_x, rejected_y, rejected_area = [], [], []

    for event in report.events:
        points = points_for_event(event)
        centre_m = (event.along_start_mm + event.length_mm / 2.0) / 1000.0
        centre_mm = event.across_start_mm + event.width_mm / 2.0
        area = _marker_area(points)

        # A continuous defect is a RUN of damage, not a location. Drawing a
        # 1.4 m warp line as a dot at its midpoint tells the cutting room to
        # plan around the wrong 40 mm, and it is exactly the defect class that
        # scores 4 points per metre - the expensive one to get wrong. The
        # extent goes underneath the marker, so short defects are unaffected
        # and long ones show their true reach.
        if event.length_mm >= CONTINUOUS_MIN_MM:
            axes.hlines(
                centre_mm,
                event.along_start_mm / 1000.0,
                event.along_end_mm / 1000.0,
                linewidth=3.0,
                color="#7a2010" if event.counts_toward_score else "#a0a0a0",
                alpha=0.55,
                zorder=1,
            )

        if event.status is EventStatus.REJECTED:
            rejected_x.append(centre_m)
            rejected_y.append(centre_mm)
            rejected_area.append(area)
        elif event.assertion is Assertion.UNCERTAIN:
            uncertain_x.append(centre_m)
            uncertain_y.append(centre_mm)
            uncertain_area.append(area)
        else:
            asserted_x.append(centre_m)
            asserted_y.append(centre_mm)
            asserted_area.append(area)
            asserted_conf.append(event.confidence)

    if asserted_x:
        scatter = axes.scatter(
            asserted_x, asserted_y, s=asserted_area, c=asserted_conf,
            cmap="YlOrRd", vmin=0.0, vmax=1.0, edgecolors="#7a2010",
            linewidths=0.7, zorder=3,
        )
        bar = figure.colorbar(scatter, ax=axes, pad=0.012, fraction=0.035)
        bar.set_label("confidence (display only, not a probability)", fontsize=9)
        bar.ax.tick_params(labelsize=8)

    if uncertain_x:
        axes.scatter(
            uncertain_x, uncertain_y, s=uncertain_area, facecolors="none",
            edgecolors="#b08900", linewidths=1.2, linestyle="--", zorder=2,
            label="uncertain (scores zero until confirmed)",
        )

    if rejected_x:
        axes.scatter(
            rejected_x, rejected_y, s=rejected_area, facecolors="none",
            edgecolors="#8a8a8a", linewidths=1.0, zorder=1,
            label="operator rejected (false alarm)",
        )

    axes.set_xlim(-0.02 * length_m, 1.02 * length_m)
    axes.set_ylim(-0.05 * width_mm, 1.05 * width_mm)
    axes.set_xlabel("position along roll (m)", fontsize=9)
    axes.set_ylabel("across web (mm)", fontsize=9)
    axes.tick_params(labelsize=8)
    axes.set_title(
        f"{report.roll_id}  -  {len(report.events)} event(s), "
        f"{report.total_points} points, {report.verdict.value.upper()}",
        fontsize=10,
    )
    if uncertain_x or rejected_x:
        axes.legend(fontsize=8, loc="upper right", framealpha=0.9)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def render_score_strip(
    report: RollReport,
    out_path: Path | str,
    dpi: int = 150,
    figsize: tuple[float, float] = (12.0, 2.6),
) -> Path:
    """Max anomaly score against position, with the threshold drawn as a line.

    Shows the margin between normal fabric and the threshold along the whole
    roll - which is the visual answer to "how close were the false alarms?"

    Two data sources, in order of preference. If the run recorded a per-frame
    trace in ``report.meta["frame_max_scores"]`` - a sequence of
    ``(along_mm, max_score)`` - the strip is a genuine profile of the whole
    roll. Otherwise it falls back to plotting the events, which shows only
    fabric that already crossed the threshold and therefore **cannot** show the
    margin on clean fabric. The axis title says which one you are looking at,
    because a reader who mistakes the second for the first would conclude the
    roll was defect-free between the markers.

    Raises:
        RuntimeError: if the report carries no calibration - there is no
            threshold to draw, and a score profile without one invites the eye
            to invent a cut-off.
    """
    if report.calibration is None:
        raise RuntimeError(
            "cannot draw a score strip without a calibration: the threshold is "
            "the only line on it that means anything"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    calibration = report.calibration

    trace = report.meta.get("frame_max_scores")
    if trace:
        positions = np.asarray([float(p) / 1000.0 for p, _ in trace], dtype=np.float64)
        scores = np.asarray([float(s) for _, s in trace], dtype=np.float64)
        subtitle = "per-frame maximum over the whole roll"
    else:
        positions = np.asarray(
            [e.along_start_mm / 1000.0 for e in report.events], dtype=np.float64
        )
        scores = np.asarray([e.max_score for e in report.events], dtype=np.float64)
        subtitle = "detected events only - clean fabric is not plotted"

    figure = _figure(figsize, dpi)
    axes = figure.add_subplot(111)

    if positions.size:
        order = np.argsort(positions)
        if trace:
            axes.plot(
                positions[order], scores[order], linewidth=0.9, color="#31527a", zorder=2
            )
        else:
            axes.vlines(
                positions[order], calibration.abstain_low, scores[order],
                linewidth=1.0, color="#31527a", zorder=2,
            )
            axes.scatter(
                positions[order], scores[order], s=14, color="#31527a", zorder=3
            )

    axes.axhline(
        calibration.threshold, color="#c1272d", linewidth=1.2,
        label=f"threshold {calibration.threshold:.3f} "
              f"({calibration.budget_fa_per_100m:g} FA/100 m)",
    )
    axes.axhline(
        calibration.abstain_low, color="#b08900", linewidth=1.0, linestyle="--",
        label=f"abstain floor {calibration.abstain_low:.3f}",
    )
    axes.axhspan(
        calibration.abstain_low, calibration.threshold,
        color="#b08900", alpha=0.10, zorder=0,
    )

    axes.set_xlim(0.0, max(report.roll_length_m, 0.001))
    axes.set_xlabel("position along roll (m)", fontsize=9)
    axes.set_ylabel("anomaly score", fontsize=9)
    axes.set_title(f"score against position  -  {subtitle}", fontsize=10)
    axes.tick_params(labelsize=8)
    axes.legend(fontsize=8, loc="upper right", framealpha=0.9)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def save_event_crop(
    frame_image: object, event: object, out_dir: Path | str, pad_px: int = 32
) -> Path:
    """Save an evidence crop with the heatmap overlaid, and record its path.

    Every event in the PDF carries its own image. A defect table without
    pictures asks a QA manager to take the model's word for it.

    ``frame_image`` is expected to be an already-composited overlay in
    **fabric-ROI** coordinates - what ``FrameResult.overlay()`` returns - because
    ``Detection.bbox_px`` is in those coordinates too. Handing this the raw
    camera frame instead would crop the right rectangle out of the wrong image,
    by however much L2 cropped off the marker tape, and the mistake looks
    plausible rather than broken.

    The crop is written to ``out_dir/event_{id:04d}.png`` and the path is
    recorded on ``event.crop_path`` so the PDF can find it later.

    Raises:
        ValueError: if the event has no detections to locate, or if the padded
            box falls entirely outside the image.
    """
    import cv2

    detections = list(getattr(event, "detections", []) or [])
    if not detections:
        raise ValueError(
            f"event {getattr(event, 'event_id', '?')} has no detections - "
            "nothing to crop evidence from"
        )

    image = np.asarray(frame_image)
    height, width = image.shape[:2]

    # The frame where the defect was most obvious is the frame worth filing.
    best = max(detections, key=lambda d: d.max_score)
    x, y, w, h = (int(v) for v in best.bbox_px)

    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(width, x + w + pad_px)
    y1 = min(height, y + h + pad_px)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"event {getattr(event, 'event_id', '?')} bbox {best.bbox_px} with "
            f"{pad_px} px padding lies outside a {width}x{height} image"
        )

    crop = image[y0:y1, x0:x1].copy()
    # Box the defect within its own crop: after padding, a reader cannot
    # otherwise tell which part of the patch the system actually flagged.
    cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (0, 0, 255), 2)

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"event_{int(getattr(event, 'event_id', 0)):04d}.png"
    cv2.imwrite(str(path), crop)

    try:
        event.crop_path = str(path)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - Event always has the slot
        pass
    return path
