"""L2 GEOMETRY - ArUco markers on the position tape (ADR-005).

One ``cv2.aruco`` call yields, simultaneously:
  * the marker's **absolute ID**, hence its position down the roll
    (``id * marker_pitch_mm``), so the system is not integrating a speed
    estimate and cannot drift; and
  * **sub-pixel corners**, hence ``mm_per_px`` from the known physical edge
    length - the scale that turns every pixel measurement into a millimetre one.

A custom binary marker with a threshold decoder would have been half a day of
debugging under changing light. This is one call.

Failure is the interesting case, and most of this module is about it. When no
marker is decoded, position is extrapolated and flagged ``interpolated``; when
the extrapolated advance exceeds ``max_gap_mm``, the frame carries a
``gap_warning`` that is counted through to the roll report. A system that
silently reports uninspected fabric as clean is worse than one that admits it
lost track, so the warning is raised eagerly and never suppressed.

Read ``ArucoReader.read`` first: it is the layer's entry point and its three
resolution branches (decoded / extrapolated / gap) explain the rest.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import GeometryConfig
from ..types import FrameGeometry

__all__ = ["ArucoReader", "MarkerReading", "detect_markers", "mm_per_px_from_corners"]

#: An inferred fabric ROI never discards more of an axis than this. A crop that
#: eats most of the web is a decoding error, not a narrow web.
MAX_CROP_FRACTION: float = 0.25


def _dictionary(name: str) -> object:
    """Resolve a ``DICT_*`` name against ``cv2.aruco``.

    Raises:
        ValueError: on an unknown dictionary name, listing the valid ones - a
            typo here would otherwise present as "no markers anywhere", which
            is the same symptom as a lighting problem and wastes an hour.
    """
    if not hasattr(cv2.aruco, name):
        valid = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        raise ValueError(f"unknown ArUco dictionary {name!r}. Valid names: {valid}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def mm_per_px_from_corners(corners: np.ndarray, marker_length_mm: float) -> float:
    """Scale from one marker's four corners. Averages all four edges.

    Averaging rather than taking one edge: under slight perspective the four
    edges disagree, and a single edge would bias the scale by however the marker
    happens to be tilted.

    Raises:
        ValueError: if the corners are degenerate (zero mean edge length).
    """
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = [float(np.linalg.norm(points[(i + 1) % 4] - points[i])) for i in range(4)]
    mean_edge_px = float(np.mean(edges))
    if mean_edge_px <= 0.0:
        raise ValueError("degenerate marker corners: mean edge length is zero")
    return marker_length_mm / mean_edge_px


class MarkerReading:
    """One decoded marker: its ID, its four corners, and the scale it implies."""

    def __init__(
        self,
        marker_id: int,
        corners: np.ndarray,
        marker_length_mm: float,
        marker_pitch_mm: float = 100.0,
    ) -> None:
        """Args:
        corners: float32 (4, 2), in the raw frame's pixel coords, clockwise
            from the top-left as OpenCV returns them.
        """
        self.marker_id = int(marker_id)
        self.corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        self.marker_length_mm = float(marker_length_mm)
        self.marker_pitch_mm = float(marker_pitch_mm)

    @property
    def mm_per_px(self) -> float:
        """Physical edge length divided by the mean of the four measured edges."""
        return mm_per_px_from_corners(self.corners, self.marker_length_mm)

    @property
    def centre_px(self) -> tuple[float, float]:
        centre = self.corners.mean(axis=0)
        return float(centre[0]), float(centre[1])

    @property
    def along_mm(self) -> float:
        """Absolute position of this marker down the roll: ``id * pitch``.

        Absolute, not integrated - which is the whole reason for markers. A
        missed marker costs one frame; an integrated speed estimate costs the
        rest of the roll.
        """
        return self.marker_id * self.marker_pitch_mm

    def __repr__(self) -> str:
        return f"MarkerReading(id={self.marker_id}, along_mm={self.along_mm:.1f})"


def detect_markers(
    image: np.ndarray,
    aruco_dict: str = "DICT_4X4_50",
    marker_length_mm: float = 40.0,
    marker_pitch_mm: float = 100.0,
) -> list[MarkerReading]:
    """Find every ArUco marker in a raw frame.

    Refines corners to sub-pixel accuracy - the scale estimate is only as good
    as the corner localisation, and a 1% scale error is a 1% error on every
    defect length in the ASTM total.

    Returns:
        Readings sorted by marker ID, so a caller taking ``[0]`` gets a stable
        answer rather than whatever order OpenCV happened to scan in.
    """
    array = np.asarray(image)
    grey = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.ndim == 3 else array

    dictionary = _dictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None or len(ids) == 0:
        return []

    readings = [
        MarkerReading(int(marker_id), quad[0], marker_length_mm, marker_pitch_mm)
        for quad, marker_id in zip(corners, ids.flatten(), strict=True)
    ]
    return sorted(readings, key=lambda r: r.marker_id)


class ArucoReader:
    """Stateful per-roll reader: turns frames into ``FrameGeometry``.

    Stateful because position must be continuous. It remembers the last confident
    reading so that a frame with no visible marker can still be placed, and so
    that it can tell the difference between "between markers" and "lost".
    """

    def __init__(self, config: GeometryConfig) -> None:
        self.config = config
        self._gap_warnings = 0
        self._last_along_mm: float | None = None
        self._last_mm_per_px: float | None = None
        self._last_timestamp: float | None = None
        self._observed_speed_mm_per_s: float | None = None
        self._roi: tuple[int, int, int, int] | None = None

    def read(self, image: np.ndarray, timestamp: float) -> FrameGeometry:
        """Frame -> position, scale, fabric ROI, and any gap warning.

        Resolution order:
          1. Markers decoded -> absolute position and measured scale. Confident.
          2. None decoded, gap within ``max_gap_mm`` -> extrapolate from the last
             reading using the observed rate (or ``fallback_speed_mm_per_s`` if
             configured), mark ``interpolated``.
          3. Gap beyond ``max_gap_mm`` -> still extrapolate, but set
             ``gap_warning``. The roll report surfaces the count.

        Raises:
            RuntimeError: if the very first frame has no marker and no
                ``fallback_speed_mm_per_s`` is configured. There is nothing to
                extrapolate from, and guessing the roll's origin would put every
                position in the report at an unknown offset.
        """
        readings = detect_markers(
            image,
            self.config.aruco_dict,
            self.config.marker_length_mm,
            self.config.marker_pitch_mm,
        )
        roi = self.fabric_roi(image, readings)

        if readings:
            return self._from_markers(readings, timestamp, roi)
        return self._extrapolate(timestamp, roi)

    def fabric_roi(
        self, image: np.ndarray, readings: list[MarkerReading]
    ) -> tuple[int, int, int, int]:
        """The rectangle of actual fabric, excluding the marker tape.

        Scoring the tape would produce a hard anomaly on every single frame -
        the markers are, after all, maximally unlike fabric.

        **The tape's position is a property of the rig, not of the frame.** It
        is inferred once, from the first frame that reads markers, and then held
        fixed. Re-inferring per frame looked reasonable and was not: a stain in
        the weave occasionally decodes as a spurious marker somewhere out in the
        middle of the web, which flips the inferred tape axis and crops away
        most of the fabric. Observed live - one frame came back 473 px tall out
        of 720 and the tiler rejected it.

        Two further guards, because "mostly right" is not a property an ROI can
        have: the crop never removes more than ``MAX_CROP_FRACTION`` of an axis,
        and a configured ROI always wins outright.
        """
        height, width = np.asarray(image).shape[:2]
        if self.config.roi is not None:
            x, y, w, h = (int(v) for v in self.config.roi)
            return (x, y, w, h)
        if self._roi is not None:
            return self._roi
        if not readings:
            return (0, 0, int(width), int(height))

        xs = np.concatenate([r.corners[:, 0] for r in readings])
        ys = np.concatenate([r.corners[:, 1] for r in readings])
        margin = 0.1 * float(np.mean([np.ptp(r.corners[:, 0]) for r in readings]))

        # The tape runs down one edge, so it is narrow on exactly one axis and
        # sits against one end of it. Crop that axis, from whichever end the
        # tape is nearer.
        if float(np.ptp(xs)) <= float(np.ptp(ys)):
            near_start = xs.min() < (width - xs.max())
            limit = width * MAX_CROP_FRACTION
            cut = int(min(xs.max() + margin, limit)) if near_start else 0
            end = width if near_start else int(max(xs.min() - margin, width - limit))
            roi = (cut, 0, max(1, end - cut), int(height))
        else:
            near_start = ys.min() < (height - ys.max())
            limit = height * MAX_CROP_FRACTION
            cut = int(min(ys.max() + margin, limit)) if near_start else 0
            end = height if near_start else int(max(ys.min() - margin, height - limit))
            roi = (0, cut, int(width), max(1, end - cut))

        self._roi = roi
        return roi

    def reset(self) -> None:
        """Clear per-roll state. Called at the start of every run."""
        self._gap_warnings = 0
        self._last_along_mm = None
        self._last_mm_per_px = None
        self._last_timestamp = None
        self._observed_speed_mm_per_s = None
        self._roi = None

    @property
    def n_gap_warnings(self) -> int:
        return self._gap_warnings

    # -- internals ---------------------------------------------------------- #

    def _from_markers(
        self,
        readings: list[MarkerReading],
        timestamp: float,
        roi: tuple[int, int, int, int],
    ) -> FrameGeometry:
        """A confident frame: absolute position, measured scale.

        ``along_mm`` is the roll position of the frame's TOP EDGE, not of the
        markers. A marker whose own position is 300 mm and which appears 200 px
        down the frame puts the frame's top edge at ``300 - 200 * mm_per_px``.
        Reporting the marker's position instead would offset every defect in the
        report by however far down the frame the tape happened to be read - and
        because detections then add their own ``y * mm_per_px``, the error would
        be counted twice.
        """
        # Average over every visible marker: each one independently pins the
        # same frame, so averaging cuts the corner-localisation noise.
        mm_per_px = float(np.mean([r.mm_per_px for r in readings]))
        along_mm = float(
            np.mean(
                [r.along_mm - float(r.corners[:, 1].min()) * mm_per_px for r in readings]
            )
        )

        if self._last_along_mm is not None and self._last_timestamp is not None:
            elapsed = timestamp - self._last_timestamp
            if elapsed > 0:
                self._observed_speed_mm_per_s = (along_mm - self._last_along_mm) / elapsed

        self._last_along_mm = along_mm
        self._last_mm_per_px = mm_per_px
        self._last_timestamp = timestamp

        return FrameGeometry(
            along_mm=along_mm,
            mm_per_px=mm_per_px,
            roi=roi,
            marker_ids=tuple(r.marker_id for r in readings),
            interpolated=False,
            gap_warning=False,
        )

    def _extrapolate(
        self, timestamp: float, roi: tuple[int, int, int, int]
    ) -> FrameGeometry:
        """No marker this frame: place it, and say how much to trust the answer."""
        speed = self._observed_speed_mm_per_s
        if speed is None:
            speed = self.config.fallback_speed_mm_per_s

        if self._last_along_mm is None or self._last_timestamp is None:
            if speed is None:
                raise RuntimeError(
                    "no marker in the first frame and no fallback_speed_mm_per_s "
                    "configured - there is nothing to measure position against. "
                    "Start the roll with a marker in view, or declare a bench "
                    "speed in the SKU config."
                )
            # Origin of the roll, by declaration rather than by measurement.
            self._last_along_mm = 0.0
            self._last_timestamp = timestamp
            self._last_mm_per_px = self._last_mm_per_px or 1.0
            return FrameGeometry(
                along_mm=0.0,
                mm_per_px=self._last_mm_per_px,
                roi=roi,
                interpolated=True,
                gap_warning=False,
            )

        elapsed = max(0.0, timestamp - self._last_timestamp)
        advance_mm = (speed or 0.0) * elapsed
        along_mm = self._last_along_mm + advance_mm

        if speed is not None:
            # The advance can be estimated, so measure the gap in distance.
            gap_warning = advance_mm > self.config.max_gap_mm
        else:
            # No observed rate and no declared bench speed: the estimated
            # advance is 0 mm, which is not a measurement - it is the absence of
            # one. Judging by distance here would report a long marker-free
            # stretch as "hasn't moved", which is precisely the silent lie the
            # gap warning exists to prevent. Time is the only evidence left.
            gap_warning = elapsed > self.config.max_gap_s
        if gap_warning:
            self._gap_warnings += 1

        return FrameGeometry(
            along_mm=along_mm,
            mm_per_px=self._last_mm_per_px or 1.0,
            roi=roi,
            marker_ids=(),
            interpolated=True,
            gap_warning=gap_warning,
        )
