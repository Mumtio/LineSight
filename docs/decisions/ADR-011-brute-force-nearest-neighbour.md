# ADR-011 — Brute-force nearest-neighbour search; no FAISS

**Status:** Accepted · **Date:** 2026-08-31

## Context

Approximate nearest-neighbour libraries — FAISS, ScaNN, hnswlib — are standard
equipment for memory-bank methods, and a reviewer who knows the area will notice
their absence. The question is whether that absence is an oversight or a choice.

The arithmetic settles it. After Gaussian random projection each embedding is
128-dimensional. A tile produces a 20×20 grid, so 400 queries. The coreset holds
about 1,200 points. One `torch.cdist` is 400 × 1,200 × 128 ≈ 61 MFLOP — a
sub-millisecond operation on any CPU made this decade, and a rounding error
beside the backbone forward pass that produced the queries.

FAISS would add an index build to every fit, a dependency with platform-specific
wheels, an approximation whose recall we would then have to measure, and a second
code path to test. For 61 MFLOP.

## Decision

Brute-force `torch.cdist` against the full coreset. Say so explicitly in the
model specification and here, rather than omitting the topic — the point is that
we costed it, not that we forgot it.

## Consequences

One fewer dependency, no index to build or persist, and the bank artifact stays a
plain `.npz` that numpy can read without our code. Nearest-neighbour distance is
exact, so score maps are reproducible bit for bit and an ablation comparing
coreset fractions is not confounded by index recall.

The decision is scale-dependent and would not survive its own assumptions
changing. If the coreset grew past roughly 10⁵ points — a much larger fit set, or
many SKUs searched jointly — brute force would stop being free and this ADR would
need revisiting. At 30 tiles per SKU that is not the regime we are in.

`_nn_distance` is a single method with a documented signature, so swapping in an
index later is a local change behind an interface, not a redesign.
