"""L6a EVENT ASSEMBLY - score maps to per-frame detections, in millimetres.

Four steps, in order, and ``assemble_frame`` at the bottom of the file runs all
four:
  1. Paste the tile score maps into one frame-sized map, max-combining the
     overlap bands (``preprocess.tiling.assemble_score_map``).
  2. Binarise it at the calibrated threshold.
  3. ``cv2.connectedComponentsWithStats`` to find candidate defects.
  4. Convert each component's position and extent to millimetres via
     ``mm_per_px``, and label it asserted or uncertain against the calibration.

Step 4 is where the interesting decision is: detection and measurement use
*different* levels, for the reason spelled out in ``_extent_bbox``.

Tracking the same defect across frames is the next step and lives in
``track.py``, so nothing here needs cross-frame state.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..calibrate.threshold import classify_score
from ..config import EventConfig
from ..preprocess.tiling import assemble_score_map
from ..types import Assertion, Calibration, Detection, FrameGeometry, ScoreMap

__all__ = ["assemble_frame", "binarise", "components_to_detections", "frame_mask"]


def binarise(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Score map -> uint8 {0, 1} mask at the calibrated threshold.

    Strictly above: a score exactly at the threshold is the boundary case the
    quantile put there, and the false-alarm budget is stated in terms of
    exceedances.
    """
    return (np.asarray(scores) > threshold).astype(np.uint8)


def frame_mask(
    score_maps: list[ScoreMap], out_hw: tuple[int, int], threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tile score maps -> ``(frame_mask, frame_scores)``, both frame-sized.

    Overlaps take the max on both: a defect visible in either of two
    overlapping tiles is a defect. Returning the score map alongside the mask
    lets each component carry its own peak score without a second pass.
    """
    scores = assemble_score_map(score_maps, out_hw)
    return binarise(scores, threshold), scores


def _extent_bbox(
    component: np.ndarray, window: np.ndarray, config: EventConfig
) -> tuple[int, int, int, int]:
    """Measure a component's extent, in window-local pixel coords.

    **Detection and measurement are separate questions, and they need separate
    levels.** The calibrated threshold answers *whether* this is a defect, and
    it sits deliberately low - just above the clean-fabric noise floor, because
    that is what a false-alarm budget buys. Measuring the bounding box at that
    same level measures the blurred skirt of the response, not the defect: a
    6 px line whose score peaks at 15 spills past a threshold of 2.5 for over a
    hundred pixels. Since ASTM scores by length, that inflation turns a 1-point
    slub into a 3-point one.

    So the extent is taken at a level tied to the component's own peak
    (``half_max`` by default): the detection decision keeps its false-alarm
    guarantee, and the measurement uses a criterion that is consistent from one
    defect to the next. ``threshold`` restores the naive behaviour for
    comparison.

    This does not conjure resolution that is not there - see
    ``Pipeline.spatial_resolution_mm``, which is what bounds how small an extent
    may honestly be reported.
    """
    if config.extent_rule == "threshold":
        rows, cols = np.nonzero(component)
    else:
        values = window[component]
        peak, floor = float(values.max()), float(window.min())
        level = floor + config.extent_fraction * (peak - floor)
        core = component & (window >= level)
        rows, cols = np.nonzero(core if core.any() else component)

    return int(cols.min()), int(rows.min()), int(np.ptp(cols)) + 1, int(np.ptp(rows)) + 1


def components_to_detections(
    mask: np.ndarray,
    scores: np.ndarray,
    geometry: FrameGeometry,
    calibration: Calibration,
    config: EventConfig,
    frame_index: int,
) -> list[Detection]:
    """Connected components -> ``Detection`` objects in millimetre coordinates.

    Filters out components below ``min_area_px`` or ``min_length_mm``: below the
    stated spatial resolution the system is not making a claim, and reporting
    one anyway is how a false-alarm rate quietly stops meaning anything.

    Each detection's ``assertion`` comes from its peak score against the
    calibration's threshold and abstain band. Position comes from the component
    found at that threshold; **extent** comes from ``_extent_bbox``.
    """
    binary = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    scores = np.asarray(scores, dtype=np.float32)
    mm_per_px = geometry.mm_per_px

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    detections: list[Detection] = []
    for label in range(1, n_labels):
        x, y, w, h, area_px = (int(v) for v in stats[label])
        if area_px < config.min_area_px:
            continue

        component = labels[y : y + h, x : x + w] == label
        window = scores[y : y + h, x : x + w]
        component_scores = window[component]
        max_score = float(component_scores.max()) if component_scores.size else 0.0
        mean_score = float(component_scores.mean()) if component_scores.size else 0.0

        ex, ey, ew, eh = _extent_bbox(component, window, config)
        length_mm = eh * mm_per_px
        width_mm = ew * mm_per_px
        if length_mm < config.min_length_mm and width_mm < config.min_length_mm:
            continue

        detections.append(
            Detection(
                frame_index=frame_index,
                bbox_px=(x + ex, y + ey, ew, eh),
                area_px=area_px,
                max_score=max_score,
                mean_score=mean_score,
                along_mm=geometry.along_mm + (y + ey) * mm_per_px,
                across_mm=(x + ex) * mm_per_px,
                length_mm=length_mm,
                width_mm=width_mm,
                # A component only exists because it cleared the threshold, so a
                # None here means the caller supplied a mask and scores that
                # disagree. Treat it as uncertain rather than dropping evidence.
                assertion=classify_score(max_score, calibration) or Assertion.UNCERTAIN,
            )
        )
    return detections


def assemble_frame(
    score_maps: list[ScoreMap],
    out_hw: tuple[int, int],
    geometry: FrameGeometry,
    calibration: Calibration,
    config: EventConfig,
    frame_index: int,
) -> list[Detection]:
    """The whole of steps 1-4 for one frame. What ``pipeline.py`` calls."""
    mask, scores = frame_mask(score_maps, out_hw, calibration.threshold)
    return components_to_detections(
        mask, scores, geometry, calibration, config, frame_index
    )
