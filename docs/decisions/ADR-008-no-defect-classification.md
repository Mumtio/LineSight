# ADR-008 — No defect classification branch

**Status:** Accepted · **Date:** 2026-08-31

## Context

The natural second branch of a defect inspection system classifies what it finds:
slub, hole, oil stain, broken pick, missing end. It is the feature everyone
expects, and it is what a mill's existing paper forms have columns for.

It is also a supervised multi-class problem. It needs a labelled corpus of each
defect type on each fabric construction — the very thing that does not exist when
a mill onboards a new SKU, and the exact constraint this project is built to
work around. Building it would quietly reintroduce the dependency the rest of
the system is designed to avoid.

## Decision

Cut it entirely. Every detection is an `unclassified anomaly`. Pass/fail comes
from anomaly detection plus measurement — the position, the length, and the ASTM
points — which is the paper's own argument.

## Consequences

The pipeline stays a single branch, which is why the whole of L4 fits behind one
protocol method and why the seam is narrow enough to swap detectors through a
config key.

More importantly the system stays internally consistent. A classification branch
would need per-construction labels, so it would have to be trained either on the
same fabric it is then evaluated on (circular) or on fabric the customer does
not run (useless). Neither is defensible; not building it is.

The `Event.label` field exists and defaults to `"unclassified anomaly"`, so an
operator can annotate a confirmed event by hand and the data model already
supports classification whenever a labelled corpus exists. The extension point is
there; the unsupported claim is not.

This is stated on the limitations page of every report: *the system reports that
a region is unlike defect-free fabric, not what kind of defect it is.* A system
that names its own boundary is one an operator can trust at the boundary.
