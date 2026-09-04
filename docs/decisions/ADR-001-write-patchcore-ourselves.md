# ADR-001 — Write PatchCore ourselves instead of installing Anomalib

**Status:** Accepted · **Date:** 2026-08-31

## Context

Anomalib packages PatchCore, PaDiM, FastFlow and a dozen other anomaly detectors
behind a config file, and would in principle give us a benchmark harness for
free. It also pins specific torch and lightning versions, and resolving those
pins on a fresh Windows machine with an existing Python install is an unbounded
cost — the failure mode is not "an hour", it is "an afternoon, or never". The
model sits on the critical path for every other layer, so an unbounded install
risk is the most expensive kind there is.

Against that, PatchCore is genuinely simple: there is no training loop, no
optimiser, no schedule, no epochs. Fit is a forward pass and a subsampling step.
Score is a nearest-neighbour distance. The whole method is about 250 lines, and
every one of those lines is something that has to be understood anyway to
justify a design parameter or debug a bad result.

## Decision

Implement PatchCore by hand in `src/linesight/detect/patchcore.py`, depending
only on `torch`, `timm`, `numpy` and `scipy`. Take the algorithm from the paper;
take nothing from Anomalib. Cover the pieces that can be got subtly wrong — the
coreset, the projection, the tiling round-trip — with probes and then with tests.

## Consequences

Dependency risk collapses to four widely-available packages, and the install is
`pip install -e .` rather than a version-resolution negotiation. Every design
parameter becomes one this project can justify rather than one it inherited, and
every one of them is a documented key in `configs/`.

The costs are real and accepted. Anomalib's other detectors are not available
for free, so the ablations in `evaluation.md` compare PatchCore against a
supervised U-Net written here rather than against six library models. This
implementation is not independently validated, which is exactly why the MVTec
`carpet` reproduction runs before anything else: if it lands near the published
AUROC, the implementation is doing what the paper says, and every subsequent
number is licensed to be believed.
