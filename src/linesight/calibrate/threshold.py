"""L5 CALIBRATION - a threshold derived from a false-alarm budget (ADR-007).

The method in one sentence: *the threshold is never picked, it is derived.* The
operator states how many false alarms per 100 m of fabric they will tolerate,
and the threshold is the corresponding empirical quantile of anomaly scores on
held-out defect-free cloth. That quantile is the finite-sample one-sided
conformal quantile - split conformal prediction reduces to exactly this in the
one-sided case - which is why the layer is a few dozen lines and needs no
library.

Read ``threshold_from_budget`` first. Everything else either feeds it
(``collect_clean_scores``), wraps it with the provenance that makes a threshold
auditable (``calibrate``), or reads its result back out (``classify_score``,
``confidence_from_score``). Its two guards - the order statistic must exist,
and enough samples must sit above it - are where the subtlety lives, and both
refuse rather than return a number the sample cannot support.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from ..types import Assertion, Calibration

__all__ = [
    "achievable_resolution",
    "calibrate",
    "classify_score",
    "collect_clean_scores",
    "confidence_from_score",
    "threshold_from_budget",
]


def _allowed_fraction(budget_fa_per_100m: float, metres_per_tile: float) -> float:
    """Tail fraction of clean tiles the budget allows to exceed the threshold."""
    if metres_per_tile <= 0.0:
        raise ValueError(f"metres_per_tile must be positive, got {metres_per_tile}")
    if budget_fa_per_100m <= 0.0:
        raise ValueError(
            f"false-alarm budget must be positive, got {budget_fa_per_100m}. "
            "A budget of zero is not a threshold, it is a refusal to detect."
        )
    tiles_per_100m = 100.0 / metres_per_tile
    return budget_fa_per_100m / tiles_per_100m


def threshold_from_budget(
    clean_scores: np.ndarray,
    budget_fa_per_100m: float,
    metres_per_tile: float,
    abstain_multiplier: float = 5.0,
    stability_margin: int = 10,
) -> tuple[float, float]:
    """Pick a score threshold meeting a false-alarm budget on defect-free fabric.

    clean_scores: max anomaly score per held-out clean tile, shape (N,)
    Returns (threshold, abstain_low). Scores in [abstain_low, threshold)
    are surfaced as 'uncertain' rather than asserted.

    Raises:
        ValueError: if ``clean_scores`` is empty, or if the sample is too small
            for the requested budget (``N <= 1/allowed_frac``). Returning a
            number the sample cannot support would be the one dishonest line in
            the system, so it refuses instead. This is the first of the two
            guards; the second, on stability, follows it below.
    """
    scores = np.asarray(clean_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("clean_scores is empty - cannot calibrate on no fabric")

    allowed_frac = _allowed_fraction(budget_fa_per_100m, metres_per_tile)
    n = int(scores.size)

    # The conformal quantile is an ORDER STATISTIC, not an interpolated one.
    # np.quantile(scores, 1 - alpha) interpolates between neighbouring sorted
    # values; that is the plug-in estimator, and it carries no finite-sample
    # guarantee - it lands below the true quantile roughly half the time, which
    # is exactly how a stated budget of 1 FA/100 m silently becomes 5. The
    # split-conformal construction takes the k-th smallest value with
    # k = ceil((n + 1)(1 - alpha)), which guarantees the exceedance rate is at
    # most alpha for exchangeable data, at the cost of being slightly
    # conservative. Being conservative about a false-alarm promise is the right
    # direction to err.
    k = math.ceil((n + 1) * (1.0 - allowed_frac))
    if k > n:
        raise ValueError(
            f"{n} clean tiles cannot resolve {budget_fa_per_100m} false alarms "
            f"per 100 m: the conformal quantile needs the {k}th smallest of "
            f"{n} scores, which does not exist. "
            f"The finest rate this sample supports is "
            f"{achievable_resolution(n, metres_per_tile):.3g} FA/100 m. "
            "Capture more clean fabric or state a looser budget."
        )

    # Existing but not sufficient: k <= n only guarantees the order statistic
    # EXISTS. When k is at or near n the threshold rests on one or two extreme
    # samples, and while the conformal guarantee still holds *marginally* -
    # averaged over draws of the calibration set - the rate you actually get
    # from YOUR set is Beta(1, n - k + 1). At k = n that distribution has a mean
    # of 1/(n+1) but a 90th percentile 2.3x higher, so an unlucky calibration
    # set delivers several times the promised rate and the operator never knows.
    # Requiring `stability_margin` samples above the threshold makes the
    # estimate rest on a rank rather than on a maximum. The price is honest and
    # worth stating: clean fabric needed scales as margin / alpha.
    if stability_margin > 0 and k > n - stability_margin:
        needed = math.ceil(stability_margin / allowed_frac)
        raise ValueError(
            f"{n} clean tiles is enough for the {k}th order statistic to exist, "
            f"but only {n - k} sample(s) sit above it - the threshold would rest "
            f"on the extreme tail and its realised rate would vary several-fold. "
            f"For a stable {budget_fa_per_100m} FA/100 m you need roughly "
            f"{needed} clean tiles ({needed * metres_per_tile:.0f} m of clean "
            f"fabric). With {n} tiles the stable budget is about "
            f"{achievable_resolution(n, metres_per_tile) * stability_margin:.3g} "
            f"FA/100 m. Lower the budget, capture more clean fabric, or set "
            f"calibration.stability_margin=0 to accept the marginal guarantee only."
        )

    ordered = np.sort(scores)
    threshold = float(ordered[k - 1])

    abstain_frac = min(1.0, abstain_multiplier * allowed_frac)
    k_abstain = max(1, math.ceil((n + 1) * (1.0 - abstain_frac)))
    abstain_low = float(ordered[min(k_abstain, n) - 1])
    return threshold, abstain_low


def achievable_resolution(n_clean_tiles: int, metres_per_tile: float) -> float:
    """Finest false-alarm rate per 100 m that ``n`` clean tiles can resolve.

    Displayed next to the requested budget in the UI, so an operator asking for
    0.1 FA/100 m off 40 tiles is told what the sample actually supports rather
    than handed a confident-looking threshold built on one order statistic.
    """
    if n_clean_tiles <= 0 or metres_per_tile <= 0.0:
        return float("inf")
    tiles_per_100m = 100.0 / metres_per_tile
    return tiles_per_100m / n_clean_tiles


def calibrate(
    clean_scores: np.ndarray,
    budget_fa_per_100m: float,
    metres_per_tile: float,
    abstain_multiplier: float = 5.0,
    stability_margin: int = 10,
    sku: str = "",
) -> Calibration:
    """``threshold_from_budget`` plus provenance -> a ``Calibration`` object.

    The provenance is not decoration: a threshold that cannot be traced to a
    budget, a sample size and a SKU cannot be audited, and a stale one - fitted
    on different fabric, or on a different rig - is invisible without a
    timestamp. The report prints all four back out.
    """
    scores = np.asarray(clean_scores, dtype=np.float64).ravel()
    threshold, abstain_low = threshold_from_budget(
        scores, budget_fa_per_100m, metres_per_tile, abstain_multiplier, stability_margin
    )
    return Calibration(
        threshold=threshold,
        abstain_low=abstain_low,
        budget_fa_per_100m=budget_fa_per_100m,
        metres_per_tile=metres_per_tile,
        n_clean_tiles=int(scores.size),
        sku=sku,
        fit_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def classify_score(score: float, calibration: Calibration) -> Assertion | None:
    """Above threshold -> ASSERTED; in the abstain band -> UNCERTAIN; below -> None."""
    if score >= calibration.threshold:
        return Assertion.ASSERTED
    if score >= calibration.abstain_low:
        return Assertion.UNCERTAIN
    return None


def confidence_from_score(score: float, calibration: Calibration) -> float:
    """Map a score to 0..1 for display only.

    Explicitly **not** a probability and never described as one. It exists so
    the operator UI can shade a box, and it is monotone in the score so the
    shading is at least meaningful ordering.

    The abstain band maps to [0, 0.5) and everything above the threshold to
    [0.5, 1), so the halfway point on the UI is exactly the threshold.
    """
    band = calibration.threshold - calibration.abstain_low
    if band <= 0.0:
        return 1.0 if score >= calibration.threshold else 0.0
    if score <= calibration.abstain_low:
        return 0.0
    if score < calibration.threshold:
        return 0.5 * (score - calibration.abstain_low) / band
    excess = (score - calibration.threshold) / band
    return float(min(1.0, 0.5 + 0.5 * (1.0 - math.exp(-excess))))


def collect_clean_scores(
    scorer: object, clean_tiles: list[np.ndarray], batch_size: int = 12
) -> np.ndarray:
    """Max score per held-out clean tile - the input to the quantile.

    ``clean_tiles`` must be disjoint from the tiles the bank was fitted on.
    Calibrating on the fit set would give a threshold far too low and a false
    alarm rate far worse than promised.

    Raises:
        ValueError: if ``clean_tiles`` is empty.
    """
    if not clean_tiles:
        raise ValueError("no clean tiles supplied")

    maxima: list[float] = []
    for start in range(0, len(clean_tiles), max(1, batch_size)):
        batch = clean_tiles[start : start + max(1, batch_size)]
        score_maps = scorer.score_batch(batch)  # type: ignore[attr-defined]
        maxima.extend(float(np.max(sm)) for sm in score_maps)
    return np.asarray(maxima, dtype=np.float32)
