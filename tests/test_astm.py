"""ASTM D5430 golden cases.

Each case is a length, or a roll, whose penalty can be worked out by hand from
the standard, so these lock the rules independently of the code implementing
them: a change in ``scoring/astm_d5430.py`` that moves a band edge or drops the
per-metre rule cannot pass quietly.

The rules being locked:

    length_mm <= 75      -> 1 point
    75  < length <= 150  -> 2 points
    150 < length <= 230  -> 3 points
    length > 230         -> 4 points
    continuous defect >= 230 mm -> 4 points per metre in which it occurs
    max 4 points per linear metre
    points_per_100yd2 = total_points * (100 * 0.8361) / (roll_length_m * width_m)
"""

from __future__ import annotations

import pytest

from linesight.scoring.astm_d5430 import (
    SQ_M_PER_100_SQ_YD,
    points_for_continuous,
    points_for_length,
    points_per_100yd2,
    verdict_for,
)
from linesight.types import Verdict


class TestPointsForLength:
    """Band boundaries. The inclusive edges are where implementations go wrong."""

    @pytest.mark.parametrize(
        ("length_mm", "expected"),
        [
            (0.0, 1),
            (1.0, 1),
            (74.0, 1),      # just inside band 1
            (75.0, 1),      # inclusive top of band 1
            (76.0, 2),      # first length in band 2
            (150.0, 2),     # inclusive top of band 2
            (151.0, 3),
            (230.0, 3),     # inclusive top of band 3
            (231.0, 4),     # first length past it: the continuous rule
            (1000.0, 4),
        ],
    )
    def test_bands(self, length_mm: float, expected: int) -> None:
        assert points_for_length(length_mm) == expected

    def test_negative_length_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            points_for_length(-1.0)

    def test_points_are_monotone_in_length(self) -> None:
        lengths = [0.0, 40.0, 75.0, 76.0, 150.0, 200.0, 230.0, 231.0, 900.0]
        points = [points_for_length(v) for v in lengths]
        assert points == sorted(points)


class TestContinuousDefects:
    """A running defect scores per metre, not once. The rule most often missed."""

    def test_3400mm_continuous_defect_scores_four_metres(self) -> None:
        # A 3.4 m continuous defect occupies 4 linear metres -> 4 x 4 = 16.
        assert points_for_continuous(3400.0) == 16

    @pytest.mark.parametrize(
        ("length_mm", "expected"),
        [
            (230.0, 4),      # exactly at the continuous threshold, one metre
            (999.0, 4),      # still within one linear metre
            (1000.0, 4),
            (1001.0, 8),     # spills into a second metre
            (2000.0, 8),
            (2001.0, 12),
        ],
    )
    def test_per_metre_accrual(self, length_mm: float, expected: int) -> None:
        assert points_for_continuous(length_mm) == expected

    def test_continuous_always_at_least_the_banded_score(self) -> None:
        for length in (230.0, 500.0, 1500.0, 3400.0):
            assert points_for_continuous(length) >= points_for_length(length)


class TestPointsPer100Yd2:
    """The headline number, and the unit conversion inside it."""

    def test_conversion_constant(self) -> None:
        assert pytest.approx(83.61) == SQ_M_PER_100_SQ_YD

    def test_known_value(self) -> None:
        # 20 points over 50 m x 1.5 m = 75 m^2  ->  20 * 83.61 / 75
        assert points_per_100yd2(20, 50.0, 1.5) == pytest.approx(20 * 83.61 / 75.0)

    def test_zero_area_is_an_error(self) -> None:
        # Dividing by zero would report a spectacular defect rate for a roll
        # nobody measured.
        with pytest.raises(ValueError):
            points_per_100yd2(4, 0.0, 1.5)

    def test_clean_roll_scores_zero(self) -> None:
        assert points_per_100yd2(0, 50.0, 1.5) == 0.0


class TestVerdict:
    """Bands are a customer contract, not part of the standard - so they move."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, Verdict.PASS),
            (19.9, Verdict.PASS),
            (20.0, Verdict.HOLD),
            (39.9, Verdict.HOLD),
            (40.0, Verdict.REJECT),
            (400.0, Verdict.REJECT),
        ],
    )
    def test_default_bands(self, value: float, expected: Verdict) -> None:
        assert verdict_for(value) == expected

    def test_bands_are_configurable(self) -> None:
        assert verdict_for(10.0, hold_at=5.0, reject_at=8.0) == Verdict.REJECT
