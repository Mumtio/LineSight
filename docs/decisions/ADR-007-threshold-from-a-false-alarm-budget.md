# ADR-007 — Derive the threshold from a false-alarm budget

**Status:** Accepted · **Date:** 2026-08-31

## Context

Every anomaly detector produces an unbounded score and needs a threshold. The
common practice is to pick it by looking at a validation plot until the pictures
look right, then report accuracy without reporting a false-alarm rate at all.
Such a threshold has no operational meaning: nobody can say what it costs on the
factory floor.

The principled machinery for this is split conformal prediction, which gives
finite-sample coverage guarantees without distributional assumptions. The full
apparatus — calibration/validation splits, nonconformity scores, two-sided
intervals — is more than this problem needs.

## Decision

The operator states a **false-alarm budget**: how many false alarms per 100
metres of defect-free fabric they will tolerate. The threshold is then the
corresponding empirical quantile of max-anomaly-score over held-out clean tiles:

```python
tiles_per_100m = 100.0 / metres_per_tile
allowed_frac   = budget_fa_per_100m / tiles_per_100m
threshold      = float(np.quantile(clean_scores, 1.0 - allowed_frac))
abstain_low    = float(np.quantile(clean_scores, 1.0 - 5 * allowed_frac))
```

In the one-sided case this *is* the finite-sample conformal quantile. Ten lines,
fully defensible, and it does not pretend to be more than it is.

## Consequences

The sentence this buys is the one worth saying out loud: **we never picked a
threshold; we picked a false-alarm budget.** The number an operator sets is one
they already reason about commercially, and the live false-alarm counter in the
UI turns it into a visible measurement rather than a claim.

Scores between `abstain_low` and `threshold` are surfaced as **uncertain** rather
than asserted, and contribute zero ASTM points until an operator confirms them.
Abstention is a first-class outcome, not a rounding decision.

The guarantee is only as good as the sample. With `N` held-out clean tiles the
smallest resolvable tail fraction is `1/N`, so a budget below that is quoting
noise. `threshold_from_budget` **raises** rather than returning such a number,
and `achievable_resolution()` is displayed next to the requested budget so an
operator asking for 0.1 FA/100 m off 40 tiles is told what the sample supports.
Refusing here is the one place where failing loudly is worth more than shipping a
number, because a threshold that looks confident and is not would undermine every
other honest thing in this system.

The threshold is per-SKU and per-rig, and must be recomputed when either changes.
`Calibration` therefore carries its own provenance — budget, sample size, SKU,
timestamp — so a threshold on a report can always be traced to what produced it.
