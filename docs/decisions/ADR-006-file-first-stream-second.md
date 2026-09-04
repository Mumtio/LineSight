# ADR-006 — File-first, stream-second

**Status:** Accepted · **Date:** 2026-08-31

## Context

A live camera stream is the most compelling input and the least reproducible
one. It brings network flakiness, buffering, exposure drift, and a class of bug
that only appears when something is moving. Building the pipeline against it
means every downstream bug arrives entangled with an acquisition bug, and that
combination is the hardest kind to localise.

The pipeline itself does not care where pixels come from. Nothing in tiling,
scoring, calibration, event assembly, or ASTM scoring is stream-specific.

## Decision

`DirectorySource` — frames from a folder — is the primary input and the one
every study and test runs on. `VideoSource` adds a recorded pull, replayable
frame for frame. `MjpegSource` adds the live camera, behind the same
`FrameSource` protocol, so nothing above L1 distinguishes the three.

## Consequences

The whole path is checkable against a directory of AITEX crops:
`python -m linesight run --source data/aitex/roll_02/` prints a defect table
with millimetre positions and a points total. Every layer above L1 can therefore
be exercised against a deterministic, replayable input, so a failure is
unambiguous about which layer it lives in.

Regression testing the geometry layer becomes possible: the same recorded pull,
frame for frame, every time. With a live stream it would not be.

A live run has a fallback that costs nothing. If the camera misbehaves, a
recorded pull runs through the identical code path — not a degraded mode, the
same mode with a different `source_type` in a YAML file.

The cost is that live-specific concerns sit in one module rather than being
spread through the pipeline: dropping stale frames rather than buffering them,
and surfacing a dropped connection as an explicit failure rather than a quiet
pause. Both are documented in `acquisition/mjpeg.py`, which is the only place
they apply.
