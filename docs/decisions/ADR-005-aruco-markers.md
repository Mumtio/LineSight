# ADR-005 — ArUco markers on the position tape

**Status:** Accepted · **Date:** 2026-08-31

## Context

Every defect must be reported at a position on the roll, in millimetres, and
every length must be converted from pixels. That needs two things per frame: an
absolute position down the roll, and a pixel-to-millimetre scale.

The tempting cheap option is to integrate a speed estimate — frames per second
times a known pull rate. It drifts, and on a hand-pulled bench rig it drifts
badly and unpredictably. The next option is a custom binary marker printed on
tape with a threshold-based decoder. That is a day of debugging under changing
light, and its failure mode under a shadow is a wrong ID rather than no ID —
which is worse, because a wrong ID moves the frame by a whole multiple of the
marker pitch and looks plausible.

## Decision

Print ArUco markers (`DICT_4X4_50`, ≥ 40 mm, matte paper) at a known pitch along
a tape beside the fabric. One `cv2.aruco` call per frame returns the absolute
marker **ID** and its four **sub-pixel corners** simultaneously. Position is
`id × marker_pitch_mm`; scale is the known physical edge length divided by the
measured pixel edge length, averaged over all four edges.

## Consequences

Position is absolute, so it cannot drift — a missed marker costs one frame, not
the rest of the roll. Scale is measured per frame rather than assumed constant,
which matters because the camera distance on a hand-pulled rig is not constant,
and a fixed scale would put a systematic error into every reported length.
`cv2.aruco` is robust to lighting in a way a hand-rolled threshold decoder is
not, and the corner refinement is free.

The scale estimate is only as good as the corner localisation: a 1% scale error
is a 1% error on every defect length feeding the ASTM total. `estimate_mm_per_px_from_ruler`
exists as an independent check, because two ways of measuring the same number is
how a scale bug gets caught before it reaches a report.

Markers can go unread — a yank, a shadow, a fold. That case is designed for
rather than assumed away: within `max_gap_mm` the position is extrapolated and
the frame is flagged `interpolated`; beyond it, a `gap_warning` is raised and
counted in the roll report. A system that admits it lost track is worth more
than another point of AUROC, so that path is exercised deliberately rather than
left to chance — see `tests/test_geometry.py`.

The documented fallback, where ArUco proves unreliable, is a fixed frame rate
times a known constant pull speed (`geometry.fallback_speed_mm_per_s`) — stated
in the report as a bench simplification, never presented as measurement.
