"""Calibration - the threshold, and the two refusals that protect it (ADR-007).

Two properties carry the whole guarantee and are locked here:

  1. The threshold is the empirical quantile implied by the stated budget, so a
     tighter budget can only raise it. Never hand-tuned.
  2. When the clean sample is too small to resolve the requested budget, the
     function **refuses** rather than returning a number built on one order
     statistic. That refusal is the difference between a guarantee and a vibe.
"""

from __future__ import annotations

import numpy as np
import pytest

from linesight.calibrate.threshold import (
    achievable_resolution,
    calibrate,
    classify_score,
    threshold_from_budget,
)
from linesight.types import Assertion


@pytest.fixture
def clean_scores() -> np.ndarray:
    """5,000 held-out clean tiles, max score each.

    Sized so a 1 FA/100 m budget clears BOTH guards: the order statistic exists,
    and ten more samples sit above it so the threshold rests on a rank rather
    than on the sample maximum. 1,000 tiles would clear the first guard and fail
    the second - which is the whole point of the second one.
    """
    rng = np.random.default_rng(0)
    return rng.normal(loc=5.0, scale=1.0, size=5000).astype(np.float32)


class TestThresholdFromBudget:
    def test_returns_the_conformal_order_statistic(self, clean_scores: np.ndarray) -> None:
        # 0.5 m per tile -> 200 tiles per 100 m -> alpha = 1/200 for a 1 FA/100 m
        # budget. The split-conformal threshold is the k-th SMALLEST score with
        # k = ceil((n + 1)(1 - alpha)) -- an actual order statistic, not an
        # interpolation between two of them. np.quantile would interpolate and
        # land below the true quantile about half the time, which is how a
        # stated budget quietly becomes several times larger in practice.
        import math

        n = len(clean_scores)
        alpha = 1.0 / 200
        k = math.ceil((n + 1) * (1.0 - alpha))

        threshold, _ = threshold_from_budget(clean_scores, 1.0, metres_per_tile=0.5)
        assert threshold == float(np.sort(clean_scores)[k - 1])
        assert threshold in set(clean_scores.tolist())  # it is one of the samples

    def test_never_undershoots_the_interpolated_quantile(
        self, clean_scores: np.ndarray
    ) -> None:
        # The conservative direction is the safe one for a promise about false
        # alarms: erring high costs a little sensitivity, erring low breaks the
        # guarantee the whole system is sold on.
        threshold, _ = threshold_from_budget(clean_scores, 1.0, metres_per_tile=0.5)
        assert threshold >= float(np.quantile(clean_scores, 0.995))

    def test_tighter_budget_raises_the_threshold(self, clean_scores: np.ndarray) -> None:
        loose, _ = threshold_from_budget(clean_scores, 5.0, metres_per_tile=0.5)
        tight, _ = threshold_from_budget(clean_scores, 0.5, metres_per_tile=0.5)
        assert tight > loose

    def test_abstain_band_sits_below_the_threshold(self, clean_scores: np.ndarray) -> None:
        threshold, abstain_low = threshold_from_budget(clean_scores, 1.0, metres_per_tile=0.5)
        assert abstain_low < threshold

    def test_realised_false_alarm_rate_matches_the_budget(self, clean_scores: np.ndarray) -> None:
        # The claim itself: on the calibration sample, exceedances land at the
        # requested rate. This is the conformal guarantee in the one-sided case.
        budget, metres_per_tile = 1.0, 0.5
        threshold, _ = threshold_from_budget(clean_scores, budget, metres_per_tile)
        exceed_frac = float((clean_scores > threshold).mean())
        realised_per_100m = exceed_frac * (100.0 / metres_per_tile)
        assert realised_per_100m == pytest.approx(budget, abs=0.5)


class TestSampleSizeGuard:
    """The sample-size guard: N > 1/allowed_frac, or the quantile means nothing.

    Returning a number the clean sample cannot support would be the one
    dishonest line in the system, so the function refuses instead.
    """

    def test_too_few_clean_tiles_raises(self) -> None:
        scores = np.random.default_rng(0).normal(size=20).astype(np.float32)
        with pytest.raises(ValueError):
            threshold_from_budget(scores, budget_fa_per_100m=0.1, metres_per_tile=0.5)

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError):
            threshold_from_budget(np.array([], dtype=np.float32), 1.0, 0.5)

    def test_threshold_resting_on_the_sample_maximum_is_refused(self) -> None:
        """The guard that the bare existence check misses.

        2,500 tiles at 0.0512 m each is enough for the order statistic for a
        1 FA/100 m budget to EXIST - it is the 2,500th of 2,500, i.e. the sample
        maximum. The conformal guarantee still holds marginally, but conditional
        on that one calibration set the realised rate is Beta(1, 1) - wildly
        variable, and several times the budget often enough to matter. Refuse.
        """
        scores = np.random.default_rng(0).normal(size=2500).astype(np.float32)
        with pytest.raises(ValueError, match="sit above it"):
            threshold_from_budget(scores, 1.0, metres_per_tile=0.0512)

    def test_the_same_sample_is_accepted_for_a_budget_it_can_support(self) -> None:
        # Same 2,500 tiles, a budget ten times looser: now ten samples sit above
        # the threshold and it rests on a rank, not on an extremum.
        scores = np.random.default_rng(0).normal(size=2500).astype(np.float32)
        threshold, _ = threshold_from_budget(scores, 10.0, metres_per_tile=0.0512)
        assert float(np.sum(scores > threshold)) >= 1

    def test_margin_can_be_waived_deliberately(self) -> None:
        # Explicitly opting into the marginal-only guarantee is allowed; getting
        # it by accident is not.
        scores = np.random.default_rng(0).normal(size=2500).astype(np.float32)
        threshold, _ = threshold_from_budget(
            scores, 1.0, metres_per_tile=0.0512, stability_margin=0
        )
        assert threshold == float(scores.max())

    def test_achievable_resolution_is_reported_not_guessed(self) -> None:
        # 40 tiles at 0.5 m each: the finest resolvable rate is 1 tile in 40,
        # i.e. 5 FA per 100 m. An operator asking for 0.1 gets told the truth.
        assert achievable_resolution(40, 0.5) == pytest.approx(5.0)


class TestClassification:
    def test_three_way_split_around_the_bands(self, clean_scores: np.ndarray) -> None:
        cal = calibrate(clean_scores, 1.0, metres_per_tile=0.5, sku="test")
        assert classify_score(cal.threshold + 1.0, cal) is Assertion.ASSERTED
        assert classify_score((cal.threshold + cal.abstain_low) / 2, cal) is Assertion.UNCERTAIN
        assert classify_score(cal.abstain_low - 1.0, cal) is None

    def test_calibration_records_its_own_provenance(self, clean_scores: np.ndarray) -> None:
        # A threshold with no recorded budget and sample size cannot be
        # audited, so the object refuses to exist without them.
        cal = calibrate(clean_scores, 1.0, metres_per_tile=0.5, sku="aitex_02")
        assert cal.budget_fa_per_100m == 1.0
        assert cal.n_clean_tiles == len(clean_scores)
        assert cal.sku == "aitex_02"
        assert cal.fit_timestamp
