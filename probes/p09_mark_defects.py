"""
probes/p09_mark_defects.py  --  THE STANDALONE DEFECT MARKER

QUESTION: Given one defective fabric image, can the whole chain -- backbone,
          memory bank, nearest-neighbour distance, upsample, threshold,
          connected components -- find the defect and draw a box around it?
PASS IF:  The output image has a box on the defect and no boxes on clean
          fabric, and the box overlaps the ground-truth mask where one exists.
          Judged by looking at it. That is the entire point.
STATUS:   see probes/README.md

Run:
    python probes/p09_mark_defects.py --image data/aitex/Defect_images/0001_002_00.png
    python probes/p09_mark_defects.py --fabric 02 --n 4      # 4 defects of one fabric
    python probes/p09_mark_defects.py --image my_photo.jpg --bank banks/aitex_02.npz

Why a standalone script rather than another pytest: the unit tests prove each
layer in isolation, and a pipeline can pass every one of them while producing
nothing useful end to end -- a scale factor inverted, a mask pasted at the wrong
offset, a threshold applied to the wrong array. One image in, one annotated
image out, judged by a human in one second, catches all of that at once.

It depends on nothing but the shipped package. If this script runs on a fresh
clone, the project works.

Output (results/marked/<stem>_marked.png), four stacked panels:

    +--------------------------------------------------+
    |  1 input          (+ ground-truth outline)        |
    |  2 heatmap        (raw anomaly score, jet)        |
    |  3 mask           (thresholded)                   |
    |  4 MARKED         (boxes, mm labels, ASTM points) |
    +--------------------------------------------------+
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linesight.calibrate.threshold import calibrate
from linesight.config import load_config
from linesight.datasets.aitex import (
    AitexImage,
    crop_to_fabric,
    cut_tiles,
    list_images,
    load_image,
    load_mask,
)
from linesight.detect.patchcore import PatchCore
from linesight.events.assemble import binarise, components_to_detections
from linesight.preprocess.tiling import assemble_score_map, tile_frame
from linesight.scoring.astm_d5430 import points_for_length
from linesight.types import Assertion, Calibration, FrameGeometry, ScoreMap

# BGR. Asserted detections in red, uncertain in amber, ground truth in green.
_RED = (60, 60, 235)
_AMBER = (40, 190, 245)
_GREEN = (90, 220, 120)
_GREY = (150, 150, 150)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def load_or_fit_bank(
    bank_path: Path | None, normal_tiles: list[np.ndarray] | None, config: object
) -> tuple[PatchCore, float]:
    """Get a fitted scorer: load a saved bank, or fit one on the spot.

    Fitting on the spot matters for the throwaway case -- point the script at a
    folder of clean fabric and a suspect image, and get an answer with no prior
    setup. That is the cold-start claim, reduced to one command.

    Returns ``(scorer, fit_seconds)``; ``fit_seconds`` is 0.0 for a loaded bank.

    Raises:
        ValueError: if neither a bank nor clean tiles are given.
    """
    if bank_path is not None and Path(bank_path).exists():
        return PatchCore.load(bank_path, config), 0.0
    if not normal_tiles:
        raise ValueError("give either --bank <path.npz> or clean fabric to fit on")

    scorer = PatchCore(config)
    started = time.perf_counter()
    scorer.fit(normal_tiles)
    return scorer, time.perf_counter() - started


def supportable_budget(
    n_tiles: int, metres_per_tile: float, stability_margin: int
) -> float:
    """The tightest false-alarm budget this many clean tiles can hold STABLY.

    ``achievable_resolution`` is one tile in n; a stable threshold needs
    ``stability_margin`` samples above it, so the supportable budget is that
    many times looser. Reported rather than assumed, because AITEX's 20 clean
    strips per fabric are genuinely not much fabric and the honest budget is
    correspondingly loose.
    """
    from linesight.calibrate.threshold import achievable_resolution

    return achievable_resolution(n_tiles, metres_per_tile) * max(1, stability_margin)


def pick_threshold(
    scorer: PatchCore,
    clean_tiles: list[np.ndarray],
    budget_fa_per_100m: float,
    metres_per_tile: float,
    stability_margin: int,
) -> tuple[Calibration, str]:
    """Calibrate from clean tiles if the sample supports it; otherwise SAY SO.

    Returns ``(calibration, provenance)``. The fallback is a high percentile of
    this image's own scores, which is **not** a calibrated threshold: a
    self-calibrated threshold on a single defective image has no false-alarm
    guarantee at all, and the caller prints the provenance so the picture is
    never mistaken for a measured one.
    """
    if clean_tiles:
        scores = np.array([m.max() for m in scorer.score_batch(clean_tiles)], dtype=np.float32)
        try:
            cal = calibrate(
                scores,
                budget_fa_per_100m,
                metres_per_tile,
                stability_margin=stability_margin,
            )
            return cal, (
                f"calibrated: {budget_fa_per_100m} FA/100m on "
                f"{len(scores)} held-out clean tiles"
            )
        except ValueError as exc:
            reason = str(exc).split(".")[0]
    else:
        reason = "no clean tiles supplied"

    return None, reason  # type: ignore[return-value]


def fallback_threshold(score_map: np.ndarray, percentile: float = 99.5) -> Calibration:
    """A threshold from the image's own scores. NOT a calibration - labelled so."""
    threshold = float(np.percentile(score_map, percentile))
    return Calibration(
        threshold=threshold,
        abstain_low=float(np.percentile(score_map, percentile - 1.0)),
        budget_fa_per_100m=float("nan"),
        metres_per_tile=float("nan"),
        n_clean_tiles=0,
        sku="(self-thresholded)",
        fit_timestamp="NOT CALIBRATED - no false-alarm guarantee",
    )


# --------------------------------------------------------------------------- #
# Scoring one image
# --------------------------------------------------------------------------- #


def score_image(
    image: np.ndarray, scorer: PatchCore, tile_size: int, overlap: int
) -> np.ndarray:
    """Whole image -> one assembled score map, via the pipeline's own tiler."""
    height, width = image.shape[:2]
    size = min(tile_size, height, width)
    tiles = tile_frame(image, size, min(overlap, size - 1))
    raw = scorer.score_batch([t.image for t in tiles])
    return assemble_score_map(
        [ScoreMap(scores=m, tile=t) for m, t in zip(raw, tiles, strict=True)],
        (height, width),
    )


def mark_image(
    image: np.ndarray,
    score_map: np.ndarray,
    calibration: Calibration,
    mm_per_px: float,
    config: object,
    machine_axis: str,
) -> tuple[np.ndarray, list]:
    """Threshold, find components, draw boxes.

    Boxes are solid for asserted detections and dashed for uncertain ones, so
    the abstention band is visible in the picture rather than buried in the
    printed table.
    """
    mask = binarise(score_map, calibration.threshold)
    geometry = FrameGeometry(along_mm=0.0, mm_per_px=mm_per_px, roi=(0, 0, *image.shape[1::-1]))
    detections = components_to_detections(
        mask, score_map, geometry, calibration, config.events, 0
    )

    annotated = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, det in enumerate(detections, 1):
        x, y, w, h = det.bbox_px
        asserted = det.assertion is Assertion.ASSERTED
        colour = _RED if asserted else _AMBER
        pad = 3
        if asserted:
            cv2.rectangle(annotated, (x - pad, y - pad), (x + w + pad, y + h + pad), colour, 2)
        else:
            _dashed_rect(annotated, (x - pad, y - pad), (x + w + pad, y + h + pad), colour, 2)

        length_mm = _machine_length_mm(det, machine_axis)
        label = f"#{i} {length_mm:.0f}mm {points_for_length(length_mm)}pt"
        _label(annotated, label, (x - pad, y - pad), colour)
    return annotated, detections


def _machine_length_mm(detection: object, machine_axis: str) -> float:
    """Extent along the machine direction, which is what ASTM scores.

    An AITEX strip is 4096 x 256: the roll runs along the LONG axis, so the
    machine-direction extent is the box's width, not its height. Getting this
    backwards would score every defect by the wrong dimension.
    """
    return detection.width_mm if machine_axis == "x" else detection.length_mm


def _dashed_rect(image, pt1, pt2, colour, thickness, dash=9) -> None:
    """A dashed rectangle - how an UNCERTAIN detection is drawn."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), colour, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), colour, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), colour, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), colour, thickness)


def _label(image, text: str, origin: tuple[int, int], colour) -> None:
    """A readable label: dark plate under light text, clamped into the frame."""
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    x = min(max(x, 0), image.shape[1] - tw - 6)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 6, y), (25, 25, 25), -1)
    cv2.putText(
        image, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA
    )


def overlay_ground_truth(annotated: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Draw the ground-truth contour in green, when one exists.

    This is what turns "the model found something" into "the model found the
    right thing". Without it the probe can only show plausibility.
    """
    if mask is None:
        return annotated
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(annotated, contours, -1, _GREEN, 2)
    return annotated


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def render_panels(
    image: np.ndarray,
    score_map: np.ndarray,
    mask: np.ndarray,
    annotated: np.ndarray,
    out_path: Path,
    captions: list[str],
) -> Path:
    """Compose the four-panel figure and write it."""
    panels = [
        image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        _heatmap(score_map),
        cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR),
        annotated,
    ]
    labelled = []
    for panel, caption in zip(panels, captions, strict=True):
        strip = np.full((26, panel.shape[1], 3), 25, dtype=np.uint8)
        cv2.putText(
            strip, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA
        )
        labelled.append(np.vstack([strip, panel]))
        labelled.append(np.full((6, panel.shape[1], 3), 60, dtype=np.uint8))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), np.vstack(labelled[:-1])):
        raise OSError(f"could not write {out_path}")
    return out_path


def _heatmap(score_map: np.ndarray) -> np.ndarray:
    """Jet colour map, normalised for display only."""
    span = float(score_map.max() - score_map.min())
    norm = (score_map - score_map.min()) / span if span > 0 else np.zeros_like(score_map)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def print_detection_table(
    detections: list,
    calibration: Calibration,
    mask: np.ndarray | None,
    machine_axis: str,
) -> None:
    """The printed table, plus the ASTM points total for this image."""
    if not detections:
        print("  no detections")
        return

    print(f"  {'#':>3}  {'x_mm':>7}  {'y_mm':>7}  {'len_mm':>7}  {'wid_mm':>7}"
          f"  {'score':>7}  {'assert':>9}  {'ASTM':>4}  {'hit':>4}")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*4}  {'-'*4}")

    total = 0
    for i, det in enumerate(detections, 1):
        length_mm = _machine_length_mm(det, machine_axis)
        points = points_for_length(length_mm)
        total += points
        hit = "-"
        if mask is not None:
            x, y, w, h = det.bbox_px
            hit = "yes" if mask[y : y + h, x : x + w].any() else "NO"
        print(
            f"  {i:>3}  {det.across_mm:>7.1f}  {det.along_mm:>7.1f}  {length_mm:>7.1f}"
            f"  {det.width_mm if machine_axis == 'y' else det.length_mm:>7.1f}"
            f"  {det.max_score:>7.3f}  {det.assertion.value:>9}  {points:>4}  {hit:>4}"
        )
    print(f"\n  ASTM points for this image: {total}")


def score_summary(score_map: np.ndarray, mask: np.ndarray | None) -> None:
    """Does the model separate defect from fabric? One line, measured."""
    print(f"  score range        {score_map.min():.3f} .. {score_map.max():.3f}")
    if mask is None or not mask.any():
        return
    on, off = score_map[mask > 0], score_map[mask == 0]
    print(f"  mean on defect     {on.mean():.3f}")
    print(f"  mean on fabric     {off.mean():.3f}")
    print(f"  separation         {on.mean() / max(off.mean(), 1e-6):.2f}x")
    print(f"  peak inside mask   {'YES' if mask[np.unravel_index(score_map.argmax(), score_map.shape)] else 'no'}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def clean_tiles_for(root: Path, fabric_code: str, tile_size: int, n: int, seed: int = 0):
    """Split one fabric's clean strips into a fit set and a HELD-OUT set.

    Disjoint by construction: calibrating on the tiles the bank was fitted on
    would give a threshold far too low and a false-alarm rate far worse than
    promised.
    """
    clean = [
        img for img in list_images(root)
        if img.fabric_code == fabric_code and not img.is_defective
    ]
    if not clean:
        raise ValueError(f"no clean AITEX images for fabric {fabric_code!r}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean))
    split = max(1, len(clean) // 2)
    fit_strips = [clean[i] for i in order[:split]]
    held_strips = [clean[i] for i in order[split:]] or fit_strips[:1]

    def tiles_of(strips):
        out = []
        for s in strips:
            out.extend(t for t, _, _ in cut_tiles(load_image(s.path), tile_size))
        return out

    fit = tiles_of(fit_strips)
    rng.shuffle(fit)
    return fit[:n], tiles_of(held_strips)


def process_one(target: AitexImage, args, config, scorer, cal, provenance) -> int:
    """Score, mark, and report one image. Returns the number of detections."""
    image = load_image(target.path)
    mask = load_mask(target.mask_paths) if target.mask_paths else None
    # Crop image and mask together: AITEX pads its strips with blank white, and
    # the padding edge scores as a defect on every strip that has one.
    raw_width = image.shape[1]
    image, mask = crop_to_fabric(image, mask)
    machine_axis = "x" if image.shape[1] >= image.shape[0] else "y"

    score_map = score_image(image, scorer, args.tile, args.overlap)
    calibration = cal or fallback_threshold(score_map)

    annotated, detections = mark_image(
        image, score_map, calibration, args.mm_per_px, config, machine_axis
    )
    annotated = overlay_ground_truth(annotated, mask)

    print(f"\n{'=' * 78}\n{target.path.name}   fabric {target.fabric_code}, "
          f"defect {target.defect_code}   {image.shape[1]}x{image.shape[0]} px "
          f"(cropped from {raw_width} px: padding and scan edge removed)")
    print(f"  threshold          {calibration.threshold:.4f}  ({provenance})")
    score_summary(score_map, mask)
    print()
    print_detection_table(detections, calibration, mask, machine_axis)

    out = Path(args.out) / f"{target.path.stem}_marked.png"
    render_panels(
        image,
        score_map,
        binarise(score_map, calibration.threshold),
        annotated,
        out,
        [
            f"1  INPUT  {target.path.name}  (green = ground truth)",
            "2  ANOMALY SCORE (raw, unbounded -- jet)",
            f"3  MASK at threshold {calibration.threshold:.3f}",
            "4  MARKED  (solid = asserted, dashed = uncertain)",
        ],
    )
    print(f"\n  wrote {out}")
    return len(detections)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("QUESTION")[0].strip())
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="one defective image to mark")
    src.add_argument("--fabric", help="AITEX fabric code; marks --n of its defects")
    p.add_argument("--n", type=int, default=3, help="images to mark when using --fabric")
    p.add_argument("--root", default="data/aitex")
    p.add_argument("--bank", default=None, help="load this .npz instead of fitting")
    p.add_argument("--sku", default=None, help="config SKU to take settings from")
    p.add_argument("--budget", type=float, default=None, help="false alarms per 100 m")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--n-normal", type=int, default=30, help="clean tiles to fit on")
    p.add_argument("--mm-per-px", type=float, default=1.0)
    p.add_argument("--out", default="results/marked")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.sku) if args.sku else load_config()
    config.preprocess.tile_size = args.tile
    config.preprocess.tile_overlap = args.overlap
    if args.budget is not None:
        config.calibration.budget_fa_per_100m = args.budget

    if args.image:
        targets = [AitexImage(Path(args.image))]
    else:
        targets = [
            i for i in list_images(args.root)
            if i.fabric_code == args.fabric and i.is_defective
        ][: args.n]
    if not targets:
        print("no matching images", file=sys.stderr)
        return 2

    fabric = targets[0].fabric_code
    print(f"fabric {fabric}: fitting on its own clean strips, "
          f"marking {len(targets)} defective image(s)")

    fit_tiles, held_tiles = clean_tiles_for(
        Path(args.root), fabric, args.tile, args.n_normal
    )
    scorer, fit_s = load_or_fit_bank(
        Path(args.bank) if args.bank else None, fit_tiles, config.detect
    )
    print(f"  bank      {scorer.bank_size} points  "
          f"({'loaded' if fit_s == 0 else f'fitted in {fit_s:.1f}s on {len(fit_tiles)} tiles'})")

    metres_per_tile = config.metres_per_tile(args.mm_per_px, (args.tile, args.tile))
    margin = config.calibration.stability_margin
    budget = args.budget
    if budget is None:
        # Ask for what the sample supports rather than for a round number it
        # does not. The alternative is a threshold with no guarantee dressed up
        # as one, which is the failure this whole layer exists to prevent.
        budget = supportable_budget(len(held_tiles), metres_per_tile, margin)
        print(f"  budget    {budget:.2f} FA/100m "
              f"(the tightest {len(held_tiles)} held-out clean tiles support stably)")

    cal, provenance = pick_threshold(scorer, held_tiles, budget, metres_per_tile, margin)
    if cal is None:
        provenance = f"SELF-THRESHOLDED, NO GUARANTEE -- {provenance}"
    print(f"  threshold {provenance}")

    found = sum(process_one(t, args, config, scorer, cal, provenance) for t in targets)
    print(f"\n{'=' * 78}\n{found} detection(s) across {len(targets)} image(s). "
          f"Open {args.out}/ and look at panel 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
