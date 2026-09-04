"""Core data types - the vocabulary every layer speaks.

These are the only objects that cross a layer boundary. They are deliberately
plain (dataclasses over numpy arrays, no behaviour beyond derived properties)
so that any layer can be replaced without renegotiating the contract.

Coordinate conventions, fixed once here and obeyed everywhere:

  * **Pixel coords** ``(x, y)`` are always relative to the top-left of the
    *fabric ROI* of a frame, not the raw camera frame. Geometry (L2) crops the
    ROI; nothing downstream ever sees the raw frame again.
  * **Machine coords** are two axes in millimetres:
      - ``along_mm``  - distance down the roll (machine direction, monotonically
        increasing as fabric is pulled). Supplied by L2 from ArUco markers.
      - ``across_mm`` - distance across the web from the left selvedge.
  * ``mm_per_px`` converts between them and is per-frame, because the camera
    distance is not guaranteed constant on a hand-pulled bench rig.

See ``docs/contracts.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "Assertion",
    "Calibration",
    "Detection",
    "Event",
    "EventStatus",
    "Frame",
    "FrameGeometry",
    "LatencyRecord",
    "RollReport",
    "ScoreMap",
    "Tile",
    "Verdict",
]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Verdict(str, Enum):
    """Roll-level disposition derived from the ASTM D5430 points total."""

    PASS = "pass"
    HOLD = "hold"
    REJECT = "reject"


class EventStatus(str, Enum):
    """Operator adjudication state of a single defect event."""

    PROPOSED = "proposed"      # the system's own call, not yet reviewed
    CONFIRMED = "confirmed"    # operator agrees it is a real defect
    REJECTED = "rejected"      # operator says false alarm -> feeds the FA counter


class Assertion(str, Enum):
    """How strongly the calibration layer stands behind a detection.

    Scores at or above ``Calibration.threshold`` are ``ASSERTED``. Scores in
    ``[abstain_low, threshold)`` are ``UNCERTAIN`` - surfaced to the operator
    but never counted in the ASTM total without confirmation. The band itself
    is derived in ``calibrate/threshold.py``.
    """

    ASSERTED = "asserted"
    UNCERTAIN = "uncertain"


# --------------------------------------------------------------------------- #
# L1 ACQUISITION
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Frame:
    """One image off a :class:`~linesight.acquisition.base.FrameSource`."""

    image: np.ndarray          # uint8, (H, W, 3), BGR - OpenCV order
    index: int                 # monotonic, 0-based, per run
    timestamp: float           # seconds; wall clock for live, synthetic for replay
    source_id: str = ""        # filename / camera URL, for traceability in reports

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])


# --------------------------------------------------------------------------- #
# L2 GEOMETRY
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FrameGeometry:
    """Where a frame sits on the roll, and how big a pixel is.

    ``gap_warning`` is this layer's declared failure mode: when no marker is
    read and the extrapolated advance exceeds one frame's worth of fabric, the
    run reports a gap rather than silently claiming clean fabric. Raised in
    ``geometry.aruco.ArucoReader`` and counted through to ``RollReport``.
    """

    along_mm: float                          # position of the ROI's top edge down the roll
    mm_per_px: float                         # scale, from ArUco corner spacing
    roi: tuple[int, int, int, int]           # (x, y, w, h) of fabric in the raw frame
    marker_ids: tuple[int, ...] = ()         # ArUco IDs actually decoded this frame
    interpolated: bool = False               # True if along_mm came from extrapolation
    gap_warning: bool = False                # True if roll continuity cannot be vouched for

    @property
    def px_per_mm(self) -> float:
        return 1.0 / self.mm_per_px


# --------------------------------------------------------------------------- #
# L3 PREPROCESS
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Tile:
    """A square crop of fabric that remembers where it came from.

    ``x``/``y`` are the tile's top-left corner in fabric-ROI pixel coords, so
    pasting a tile-sized mask back at ``(x, y)`` reconstructs the frame mask.
    Tiles overlap (default 64 px) - the reassembler takes the max in overlaps.
    """

    image: np.ndarray          # uint8, (S, S, 3) or (S, S)
    x: int
    y: int
    frame_index: int
    tile_index: int            # position within this frame's tile list

    @property
    def size(self) -> int:
        return int(self.image.shape[0])

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) in fabric-ROI pixel coords."""
        h, w = self.image.shape[:2]
        return self.x, self.y, int(w), int(h)


# --------------------------------------------------------------------------- #
# L4 DETECTION
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ScoreMap:
    """Per-pixel anomaly score for one tile. Unbounded, uncalibrated.

    The detector must not threshold or normalise - that is L5's job, and
    keeping it there is what lets one threshold serve every scorer.
    """

    scores: np.ndarray         # float32, (S, S), higher = more anomalous
    tile: Tile

    @property
    def max_score(self) -> float:
        return float(self.scores.max())


# --------------------------------------------------------------------------- #
# L5 CALIBRATION
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Calibration:
    """A threshold and its provenance.

    Never constructed by hand in the product path - it comes out of
    :func:`linesight.calibrate.threshold.threshold_from_budget`, and it carries
    enough context to answer "where did this number come from?" months later:
    the budget it was derived from, the sample it was derived on, the SKU, and
    when.
    """

    threshold: float
    abstain_low: float
    budget_fa_per_100m: float
    metres_per_tile: float
    n_clean_tiles: int
    sku: str = ""
    fit_timestamp: str = ""

    @property
    def achievable_fa_per_100m(self) -> float:
        """Finest false-alarm rate this many clean tiles can actually resolve.

        With ``N`` clean tiles the smallest non-zero tail fraction is ``1/N``;
        quoting a budget below this is quoting noise. Enforced in
        ``calibrate.threshold.threshold_from_budget``, which refuses a budget
        its sample cannot resolve.
        """
        if self.n_clean_tiles <= 0 or self.metres_per_tile <= 0.0:
            return float("inf")
        tiles_per_100m = 100.0 / self.metres_per_tile
        return tiles_per_100m / self.n_clean_tiles


# --------------------------------------------------------------------------- #
# L6 EVENT ASSEMBLY
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Detection:
    """One connected component in one frame's thresholded mask.

    Frame-local and short-lived: the tracker fuses these into ``Event``s.
    """

    frame_index: int
    bbox_px: tuple[int, int, int, int]   # (x, y, w, h) in fabric-ROI pixels
    area_px: int
    max_score: float
    mean_score: float
    along_mm: float            # machine-direction position of the bbox top edge
    across_mm: float           # cross-web position of the bbox left edge
    length_mm: float           # extent along the machine direction
    width_mm: float            # extent across the web
    assertion: Assertion = Assertion.ASSERTED

    @property
    def centroid_mm(self) -> tuple[float, float]:
        return self.along_mm + self.length_mm / 2, self.across_mm + self.width_mm / 2


@dataclass(slots=True)
class Event:
    """A defect, tracked across the frames it appears in.

    Every detection is an ``unclassified anomaly`` - D8 cut the classification
    branch. ``label`` exists only so an operator can annotate one by hand.
    """

    event_id: int
    detections: list[Detection] = field(default_factory=list)
    along_start_mm: float = 0.0
    along_end_mm: float = 0.0
    across_start_mm: float = 0.0
    across_end_mm: float = 0.0
    max_score: float = 0.0
    confidence: float = 0.0                        # 0..1, monotone in max_score
    assertion: Assertion = Assertion.ASSERTED
    status: EventStatus = EventStatus.PROPOSED
    label: str = "unclassified anomaly"
    crop_path: str | None = None                   # saved evidence image, for the PDF

    @property
    def length_mm(self) -> float:
        """Machine-direction extent - the quantity ASTM D5430 scores."""
        return max(0.0, self.along_end_mm - self.along_start_mm)

    @property
    def width_mm(self) -> float:
        return max(0.0, self.across_end_mm - self.across_start_mm)

    @property
    def n_frames(self) -> int:
        return len({d.frame_index for d in self.detections})

    @property
    def counts_toward_score(self) -> bool:
        """Only asserted, non-rejected events contribute ASTM points.

        An event in the abstention band is surfaced to the operator but scores
        zero until confirmed; confirming it is what promotes it.
        """
        if self.status is EventStatus.REJECTED:
            return False
        if self.status is EventStatus.CONFIRMED:
            return True
        return self.assertion is Assertion.ASSERTED


# --------------------------------------------------------------------------- #
# L7 SCORING / L8 PRODUCT
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RollReport:
    """Everything a mill's QA desk needs about one roll.

    The object the PDF, the defect map, the store and the CLI table all render
    from, so those four views cannot disagree about the same roll.
    """

    roll_id: str
    sku: str
    roll_length_m: float
    width_m: float
    events: list[Event] = field(default_factory=list)
    total_points: int = 0
    points_per_100yd2: float = 0.0
    verdict: Verdict = Verdict.PASS
    calibration: Calibration | None = None
    false_alarms: int = 0          # operator-rejected events; the live FA counter
    gap_warnings: int = 0
    started_at: str = ""
    finished_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def inspected_area_m2(self) -> float:
        return self.roll_length_m * self.width_m

    @property
    def fa_per_100m(self) -> float:
        """Measured false-alarm rate, to be shown against the stated budget."""
        if self.roll_length_m <= 0.0:
            return 0.0
        return self.false_alarms * 100.0 / self.roll_length_m


@dataclass(slots=True)
class LatencyRecord:
    """Per-stage wall-clock for one frame.

    Aggregated by ``Pipeline.latency_table``, which reports the median per
    stage alongside ``frame_stride``: a per-frame latency that hides a sampling
    factor is a claim rather than a measurement (ADR-009).
    """

    frame_index: int
    decode_ms: float = 0.0
    geometry_ms: float = 0.0
    preprocess_ms: float = 0.0
    backbone_ms: float = 0.0
    nn_search_ms: float = 0.0
    assemble_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.decode_ms
            + self.geometry_ms
            + self.preprocess_ms
            + self.backbone_ms
            + self.nn_search_ms
            + self.assemble_ms
        )
