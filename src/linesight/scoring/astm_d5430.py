"""L7 SCORING - ASTM D5430 four-point system. Pure functions, zero dependencies.

The rules this module implements:

    length_mm <= 75      -> 1 point
    75  < length <= 150  -> 2 points
    150 < length <= 230  -> 3 points
    length > 230         -> 4 points
    continuous defect >= 230 mm -> 4 points per metre in which it occurs
    max 4 points per linear metre
    points_per_100yd2 = total_points * (100 * 0.8361) / (roll_length_m * width_m)

``score_roll`` is the entry point the pipeline calls; every rule above it is
exposed separately so each can be tested in isolation. ``tests/test_astm.py``
locks them with golden cases (74 -> 1, 76 -> 2, 231 -> 4, a 3.4 m continuous
defect -> 4x4), including the two that implementations most often get wrong: the
inclusive band edges, and per-metre scoring of continuous defects.

No numpy, no cv2, no config object - ``tests/test_skeleton.py`` asserts as much.
Every function here is a pure map from numbers to numbers, which is what makes
the arithmetic behind a verdict checkable by hand.
"""

from __future__ import annotations

import math

from ..types import Event, RollReport, Verdict

__all__ = [
    "CONTINUOUS_MIN_MM",
    "SQ_M_PER_100_SQ_YD",
    "cap_points_per_metre",
    "fill_report",
    "points_for_continuous",
    "points_for_event",
    "points_for_length",
    "points_per_100yd2",
    "score_roll",
    "total_points",
    "verdict_for",
]

#: A square yard is 0.8361 m^2, so 100 sq yd is 83.61 m^2. The unit conversion
#: at the heart of the industry's headline number.
SQ_M_PER_100_SQ_YD: float = 100.0 * 0.8361

#: At and above this length a defect is scored as continuous - per metre it
#: occupies - rather than once against the length bands.
CONTINUOUS_MIN_MM: float = 230.0

#: (inclusive upper bound in mm, points). The last band is open-ended.
_BANDS: tuple[tuple[float, int], ...] = ((75.0, 1), (150.0, 2), (230.0, 3))
_MAX_BAND_POINTS: int = 4

_MM_PER_M: float = 1000.0


def points_for_length(length_mm: float) -> int:
    """Penalty points for a single non-continuous defect of this length.

    Boundaries are inclusive at the top of each band: 75 -> 1, 75.1 -> 2.

    Raises:
        ValueError: on a negative length.
    """
    if length_mm < 0.0:
        raise ValueError(f"defect length cannot be negative: {length_mm}")
    for upper, points in _BANDS:
        if length_mm <= upper:
            return points
    return _MAX_BAND_POINTS


def points_for_continuous(length_mm: float) -> int:
    """A continuous defect (>= 230 mm) scores 4 points per metre it occupies.

    A 3.4 m running defect touches 4 linear metres, so 16 points - not 4. This
    is the rule most implementations get wrong, and the reason it has its own
    golden test case.

    Raises:
        ValueError: if the defect is shorter than ``CONTINUOUS_MIN_MM``, which
            means the caller should have used ``points_for_length`` instead.
    """
    if length_mm < CONTINUOUS_MIN_MM:
        raise ValueError(
            f"{length_mm} mm is below the {CONTINUOUS_MIN_MM} mm continuous "
            "threshold - use points_for_length()"
        )
    metres_occupied = max(1, math.ceil(length_mm / _MM_PER_M))
    return _MAX_BAND_POINTS * metres_occupied


def points_for_event(event: Event) -> int:
    """Points for one event, dispatching to continuous or banded scoring.

    Events that do not ``counts_toward_score`` (uncertain, or operator-rejected)
    contribute zero. The abstention band must not quietly inflate a penalty.
    """
    if not event.counts_toward_score:
        return 0
    length_mm = event.length_mm
    if length_mm >= CONTINUOUS_MIN_MM:
        return points_for_continuous(length_mm)
    return points_for_length(length_mm)


def cap_points_per_metre(
    per_event: list[tuple[float, int]], roll_length_m: float, cap: int = 4
) -> int:
    """Apply the 'max 4 points per linear metre' rule across the whole roll.

    Args:
        per_event: ``(along_mm_of_defect_start, points)`` pairs. A continuous
            defect spanning several metres must be passed as one pair **per
            metre** - see ``total_points`` - otherwise the cap would crush its
            legitimate 4-points-per-metre down to 4 points total.
        roll_length_m: total inspected length. Bounds the bins: a defect
            reported past the end of the roll means a geometry error upstream,
            and folding it into the final metre is safer than inventing a metre
            that was never inspected.
        cap: maximum points chargeable to any one linear metre.

    Returns:
        Total points after capping each linear metre at ``cap``.
    """
    last_metre = max(0, math.ceil(roll_length_m) - 1)
    by_metre: dict[int, int] = {}
    for along_mm, points in per_event:
        metre = min(last_metre, max(0, int(along_mm // _MM_PER_M)))
        by_metre[metre] = by_metre.get(metre, 0) + points
    return sum(min(cap, points) for points in by_metre.values())


def total_points(events: list[Event], roll_length_m: float) -> int:
    """Sum of per-event points with the per-metre cap applied.

    A continuous defect is expanded into one entry per linear metre it occupies,
    each carrying 4 points, so that the per-metre cap leaves it intact while
    still capping genuinely coincident defects.
    """
    per_event: list[tuple[float, int]] = []
    for event in events:
        points = points_for_event(event)
        if points == 0:
            continue
        length_mm = event.length_mm
        if length_mm >= CONTINUOUS_MIN_MM:
            metres = max(1, math.ceil(length_mm / _MM_PER_M))
            for i in range(metres):
                per_event.append((event.along_start_mm + i * _MM_PER_M, _MAX_BAND_POINTS))
        else:
            per_event.append((event.along_start_mm, points))
    return cap_points_per_metre(per_event, roll_length_m)


def points_per_100yd2(points: int, roll_length_m: float, width_m: float) -> float:
    """The industry's headline number.

    Raises:
        ValueError: if the inspected area is zero or negative - dividing by it
            would report a spectacular defect rate for a roll nobody measured.
    """
    area_m2 = roll_length_m * width_m
    if area_m2 <= 0.0:
        raise ValueError(
            f"inspected area must be positive, got {roll_length_m} m x {width_m} m"
        )
    return points * SQ_M_PER_100_SQ_YD / area_m2


def verdict_for(
    points_per_100yd2_value: float, hold_at: float = 20.0, reject_at: float = 40.0
) -> Verdict:
    """Bands are a customer contract, not a standard. Configurable, and stated."""
    if points_per_100yd2_value >= reject_at:
        return Verdict.REJECT
    if points_per_100yd2_value >= hold_at:
        return Verdict.HOLD
    return Verdict.PASS


def score_roll(
    events: list[Event],
    roll_length_m: float,
    width_m: float,
    hold_at: float = 20.0,
    reject_at: float = 40.0,
) -> tuple[int, float, Verdict]:
    """Events -> ``(total_points, points_per_100yd2, verdict)``.

    The single entry point the pipeline calls. Everything above is exposed so
    each rule can be tested in isolation.
    """
    points = total_points(events, roll_length_m)
    density = points_per_100yd2(points, roll_length_m, width_m)
    return points, density, verdict_for(density, hold_at, reject_at)


def fill_report(report: RollReport, hold_at: float = 20.0, reject_at: float = 40.0) -> RollReport:
    """Populate a report's scoring fields in place and return it."""
    points, density, verdict = score_roll(
        report.events, report.roll_length_m, report.width_m, hold_at, reject_at
    )
    report.total_points = points
    report.points_per_100yd2 = density
    report.verdict = verdict
    return report
