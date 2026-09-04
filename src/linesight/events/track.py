"""L6b TRACKING - fuse per-frame detections into roll-level events.

Section 8, step 5: track across frames on position-corrected boxes (aligned
using ``along_mm``), so a warp-direction line spanning several frames is one
``Event`` rather than eight.

Why it matters beyond tidiness: ASTM scores a defect by its **length**. Eight
un-tracked 40 mm fragments score 8 points; one tracked 320 mm continuous defect
scores 4 per metre. Without tracking the penalty total is simply wrong.

**Association is IoU OR machine-direction continuation**, and the second half
is not a hedge. IoU on position-corrected boxes correctly associates a
*stationary* defect seen in consecutive overlapping frames. It does **not**
associate the case that matters most for ASTM: a defect running down the roll,
where each frame sees a different 40 mm segment of the same continuous line.
Two adjacent segments of one warp line share almost no area, so their IoU is
near zero while they are unmistakably one defect. Judging by IoU alone would
split every long defect into fragments and under-score exactly the defects that
carry the heaviest penalty. So a second rule runs alongside: overlapping
cross-web position, and a machine-direction gap within ``track_join_mm``.
Either rule matching is enough - see ``_best_match`` and ``_continues``.

``PerFrameTracker`` at the bottom of the file is the documented degradation:
the same interface with no association at all, one event per detection. It is a
real class rather than a note so that running without tracking is a config
change whose cost - over-counted, under-measured long defects - is stated in
the report instead of hidden.
"""

from __future__ import annotations

from ..config import EventConfig
from ..types import Assertion, Detection, Event

__all__ = ["EventTracker", "PerFrameTracker", "align_boxes_mm", "iou"]


Box = tuple[float, float, float, float]


def _overlap(start_a: float, len_a: float, start_b: float, len_b: float) -> float:
    """Length of the 1-D overlap between two intervals; 0.0 if disjoint."""
    return max(0.0, min(start_a + len_a, start_b + len_b) - max(start_a, start_b))


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection over union of two ``(along, across, length, width)`` mm boxes."""
    along_a, across_a, length_a, width_a = box_a
    along_b, across_b, length_b, width_b = box_b

    inter = _overlap(along_a, length_a, along_b, length_b) * _overlap(
        across_a, width_a, across_b, width_b
    )
    if inter <= 0.0:
        return 0.0
    union = length_a * width_a + length_b * width_b - inter
    return inter / union if union > 0.0 else 0.0


def align_boxes_mm(detection: Detection) -> Box:
    """Detection -> a box in absolute roll coordinates, in millimetres.

    Position-corrected means expressed in roll space, not frame space: as the
    fabric advances, a stationary defect moves down the frame but stays put on
    the roll. Comparing raw frame boxes would never match anything.
    """
    return (
        detection.along_mm,
        detection.across_mm,
        detection.length_mm,
        detection.width_mm,
    )


def _continues(box_a: Box, box_b: Box, config: EventConfig) -> bool:
    """True if ``box_b`` is the same defect running further down the roll.

    Requires real cross-web overlap - two defects at opposite selvedges are two
    defects however close their machine positions - and a machine-direction gap
    no larger than ``track_join_mm``. A negative gap (the boxes already overlap
    along the roll) counts as continuous.
    """
    along_a, across_a, length_a, width_a = box_a
    along_b, across_b, length_b, width_b = box_b

    across_overlap = _overlap(across_a, width_a, across_b, width_b)
    narrower = min(width_a, width_b)
    if narrower <= 0.0 or across_overlap / narrower < config.track_iou_threshold:
        return False

    gap = max(along_a, along_b) - min(along_a + length_a, along_b + length_b)
    return gap <= config.track_join_mm


class _Track:
    """One open event, plus the bookkeeping needed to close it."""

    def __init__(self, event_id: int, detection: Detection) -> None:
        self.event = Event(
            event_id=event_id,
            detections=[detection],
            along_start_mm=detection.along_mm,
            along_end_mm=detection.along_mm + detection.length_mm,
            across_start_mm=detection.across_mm,
            across_end_mm=detection.across_mm + detection.width_mm,
            max_score=detection.max_score,
            assertion=detection.assertion,
        )
        self.last_frame = detection.frame_index
        self.last_box = align_boxes_mm(detection)

    def absorb(self, detection: Detection) -> None:
        """Extend the event to cover one more detection."""
        event = self.event
        event.detections.append(detection)
        event.along_start_mm = min(event.along_start_mm, detection.along_mm)
        event.along_end_mm = max(event.along_end_mm, detection.along_mm + detection.length_mm)
        event.across_start_mm = min(event.across_start_mm, detection.across_mm)
        event.across_end_mm = max(event.across_end_mm, detection.across_mm + detection.width_mm)
        event.max_score = max(event.max_score, detection.max_score)
        # One asserted sighting is enough: a defect that cleared the threshold in
        # any frame is asserted, even if it faded in the others.
        if detection.assertion is Assertion.ASSERTED:
            event.assertion = Assertion.ASSERTED
        self.last_frame = detection.frame_index
        self.last_box = align_boxes_mm(detection)


class EventTracker:
    """Greedy association of detections into events, across frames.

    Greedy rather than Hungarian: with a handful of detections per frame the
    optimal assignment and the greedy one almost always agree, and greedy stays
    inspectable at 2 a.m.
    """

    def __init__(self, config: EventConfig) -> None:
        self.config = config
        self._open: list[_Track] = []
        self._all: list[Event] = []
        self._next_id = 1
        self._last_frame_index = -1

    def update(self, detections: list[Detection]) -> list[Event]:
        """Absorb one frame's detections. Returns the events closed by this frame.

        An event that goes unmatched for more than ``track_max_gap_frames`` is
        closed and emitted. The gap tolerance exists because a faint defect can
        drop below threshold for a frame in the middle of a real run.
        """
        frame_index = detections[0].frame_index if detections else self._last_frame_index + 1
        self._last_frame_index = frame_index

        unmatched: list[Detection] = []
        for detection in detections:
            track = self._best_match(detection)
            if track is None:
                unmatched.append(detection)
            else:
                track.absorb(detection)

        for detection in unmatched:
            track = _Track(self._next_id, detection)
            self._next_id += 1
            self._open.append(track)
            self._all.append(track.event)

        return self._close_stale(frame_index)

    def finalise(self) -> list[Event]:
        """Close every open track at the end of the roll and return **all** events.

        The full list, not just the ones this call closed - a caller that ignored
        ``update``'s return value still gets the complete roll.
        """
        self._open.clear()
        return list(self._all)

    def reset(self) -> None:
        self._open.clear()
        self._all.clear()
        self._next_id = 1
        self._last_frame_index = -1

    @property
    def n_open(self) -> int:
        return len(self._open)

    # -- internals ---------------------------------------------------------- #

    def _best_match(self, detection: Detection) -> _Track | None:
        """The open track this detection most plausibly continues, if any."""
        box = align_boxes_mm(detection)
        best: _Track | None = None
        best_score = 0.0
        for track in self._open:
            if track.last_frame == detection.frame_index:
                continue  # at most one detection per track per frame
            score = iou(track.last_box, box)
            if score < self.config.track_iou_threshold and not _continues(
                track.last_box, box, self.config
            ):
                continue
            # Rank by IoU, but a continuation with zero IoU still beats nothing.
            score = max(score, 1e-6)
            if score > best_score:
                best, best_score = track, score
        return best

    def _close_stale(self, frame_index: int) -> list[Event]:
        """Retire tracks that have not been seen for too many frames."""
        closed: list[Event] = []
        still_open: list[_Track] = []
        for track in self._open:
            if frame_index - track.last_frame > self.config.track_max_gap_frames:
                closed.append(track.event)
            else:
                still_open.append(track)
        self._open = still_open
        return closed


class PerFrameTracker:
    """The documented degradation: one event per detection, no association.

    Same interface as ``EventTracker``, so selecting it is a config change
    (``events.track_iou_threshold = 0``). It over-counts long defects and
    understates their length; the report states that rather than leaving a
    reader to discover it.
    """

    def __init__(self, config: EventConfig) -> None:
        self.config = config
        self._all: list[Event] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Event]:
        closed: list[Event] = []
        for detection in detections:
            event = _Track(self._next_id, detection).event
            self._next_id += 1
            self._all.append(event)
            closed.append(event)
        return closed

    def finalise(self) -> list[Event]:
        return list(self._all)

    def reset(self) -> None:
        self._all.clear()
        self._next_id = 1

    @property
    def n_open(self) -> int:
        return 0
