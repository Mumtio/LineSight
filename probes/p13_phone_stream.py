"""
probes/p13_phone_stream.py

QUESTION: Does my phone actually deliver a usable stream to the laptop -- stable
          frame rate, sharp fabric, locked exposure, and an ArUco tape the
          geometry layer can decode into a position and a scale?
PASS IF:  Frames arrive at a steady rate; mean brightness drifts under 1% over
          the sample window; the frame is sharp enough to resolve yarn; and at
          least one marker decodes with a plausible mm/px.
ANSWERED: yes against a synthetic MJPEG source -- mm/px recovered to 0.2%;
          not yet run against a physical phone (probes/README.md).

Run this BEFORE pointing the pipeline at the phone. Every failure it catches is
one that would otherwise present as "the model does not work": auto-exposure
drift looks exactly like a defect, a soft focus removes the texture the bank is
supposed to learn, and an undecodable tape puts every position in the report at
an unknown offset.

    python probes/p13_phone_stream.py --url http://192.168.0.42:8080/video
    python probes/p13_phone_stream.py --url ... --aruco DICT_5X5_1000 --seconds 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linesight.acquisition.mjpeg import MjpegSource  # noqa: E402
from linesight.geometry.aruco import detect_markers  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: Brightness TREND above this across the sample window means the camera is
#: still adjusting exposure; a stream is only usable below 1%. Measured as a
#: least-squares slope on a central crop, not as a peak-to-peak range - see the
#: comment in the capture loop for why the obvious metric gives false alarms.
DRIFT_LIMIT_PCT: float = 1.0

#: Variance of the Laplacian over a MARKER PATCH below this is a soft lens.
#:
#: Measured on the marker rather than the frame, because the frame is mostly
#: fabric and fabric texture varies enormously between cloths: a smooth cotton
#: read 3.8 while a perfectly focused marker in the same frame read 272. A
#: whole-frame threshold therefore fails a sharp rig for owning smooth fabric,
#: which is exactly what it did here. The printed tape is a high-contrast target
#: sitting in the same focal plane as the cloth - it is a focus chart that
#: happens to also carry the position.
SHARPNESS_FLOOR: float = 100.0


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian - the standard cheap focus measure."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def marker_sharpness(image: np.ndarray, readings, pad: int = 12) -> float | None:
    """Focus measured on the printed markers, or None if none are visible."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = grey.shape
    values: list[float] = []
    for r in readings:
        x0 = max(0, int(r.corners[:, 0].min()) - pad)
        y0 = max(0, int(r.corners[:, 1].min()) - pad)
        x1 = min(width, int(r.corners[:, 0].max()) + pad)
        y1 = min(height, int(r.corners[:, 1].max()) + pad)
        if x1 - x0 > 8 and y1 - y0 > 8:
            values.append(sharpness(grey[y0:y1, x0:x1]))
    return float(np.median(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="e.g. http://192.168.0.42:8080/video")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--aruco", default="DICT_5X5_1000")
    parser.add_argument("--marker-mm", type=float, default=40.0)
    parser.add_argument("--pitch-mm", type=float, default=100.0)
    parser.add_argument("--out", default=str(RESULTS / "_phone_check.png"))
    args = parser.parse_args()

    print(f"\nopening {args.url}", flush=True)
    source = MjpegSource(args.url, stride=1, timeout_s=8.0, drop_stale=True)

    brightness: list[float] = []
    sharp: list[float] = []
    fabric_detail: list[float] = []
    marker_frames = 0
    scales: list[float] = []
    positions: list[float] = []
    # pitch / marker-edge, measured in PIXELS. This ratio is invariant under
    # uniform print scaling, so it says whether the printer scaled both axes
    # together - which a ruler on a single gap cannot resolve, and which no
    # single marker_length_mm can compensate for if it did not.
    ratios: list[float] = []
    tape_axis: list[int] = []
    first: np.ndarray | None = None
    last: np.ndarray | None = None
    count = 0

    started = time.time()
    try:
        with source:
            for frame in source:
                if first is None:
                    first = frame.image.copy()
                    print(f"  connected: {frame.image.shape[1]}x{frame.image.shape[0]}", flush=True)
                last = frame.image
                count += 1

                grey = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                # Central crop only. The marker tape runs down one selvedge and
                # scrolls, so its markers entering and leaving the frame move
                # whole-frame mean brightness by around a percent on their own -
                # which is indistinguishable from exposure drift if you measure
                # the whole frame. Measured here, it read 1.06% on a synthetic
                # source whose exposure was fixed by construction.
                h, w = grey.shape
                brightness.append(float(grey[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5].mean()))

                readings = detect_markers(
                    frame.image, args.aruco, args.marker_mm, args.pitch_mm
                )
                focus = marker_sharpness(frame.image, readings) if readings else None
                sharp.append(focus if focus is not None else sharpness(frame.image))
                fabric_detail.append(sharpness(frame.image))

                if readings:
                    marker_frames += 1
                    scales.append(float(np.mean([r.mm_per_px for r in readings])))
                    positions.append(readings[0].along_mm)
                if len(readings) >= 2:
                    pair = sorted(readings, key=lambda r: r.marker_id)
                    steps = pair[-1].marker_id - pair[0].marker_id
                    if steps > 0:
                        # Measure along whichever image axis the tape actually
                        # runs down. Hardcoding y raised a -93.9% false alarm on
                        # a perfectly good tape that happened to lie horizontally
                        # in a landscape stream.
                        centres = np.array([r.centre_px for r in pair], dtype=float)
                        axis = 0 if np.ptp(centres[:, 0]) >= np.ptp(centres[:, 1]) else 1
                        tape_axis.append(axis)
                        span = abs(float(centres[-1, axis] - centres[0, axis]))
                        pitch_px = span / steps
                        edge_px = float(np.mean([
                            np.mean([np.linalg.norm(r.corners[(i + 1) % 4] - r.corners[i])
                                     for i in range(4)])
                            for r in pair
                        ]))
                        if edge_px > 0:
                            ratios.append(pitch_px / edge_px)

                if count % 20 == 0:
                    print(f"  {count} frames, {count / (time.time() - started):.1f} fps",
                          flush=True)
                if time.time() - started >= args.seconds:
                    break
    except OSError as exc:
        print(f"\nFAIL: {exc}")
        print("  Check that the phone and laptop are on the same network, that the")
        print("  camera app is still in the foreground, and that the URL ends in /video.")
        return 1

    elapsed = time.time() - started
    if count == 0 or first is None or last is None:
        print("\nFAIL: connected but no frames arrived.")
        return 1

    fps = count / elapsed
    b = np.asarray(brightness)
    # Exposure drift is a slow TREND; fabric passing under the lens is fast
    # fluctuation. A least-squares slope over the window separates the two,
    # where a peak-to-peak range cannot.
    t_axis = np.linspace(0.0, elapsed, num=len(b))
    slope = float(np.polyfit(t_axis, b, 1)[0]) if len(b) > 2 else 0.0
    drift_pct = abs(slope * elapsed) * 100.0 / max(b.mean(), 1e-6)
    range_pct = 100.0 * (b.max() - b.min()) / max(b.mean(), 1e-6)
    median_sharp = float(np.median(sharp))

    print(f"\n{'frames':<22}{count} in {elapsed:.1f}s  ({fps:.1f} fps)")
    print(f"{'resolution':<22}{first.shape[1]}x{first.shape[0]}")
    print(f"{'mean brightness':<22}{b.mean():.1f}  (min {b.min():.1f}, max {b.max():.1f})")
    print(f"{'brightness drift':<22}{drift_pct:.2f}% over the window   "
          f"limit {DRIFT_LIMIT_PCT}%")
    print(f"{'brightness range':<22}{range_pct:.2f}%   (content, not necessarily drift)")
    print(f"{'focus (on markers)':<22}{median_sharp:.0f}      floor {SHARPNESS_FLOOR:.0f}")
    print(f"{'fabric texture':<22}{np.median(fabric_detail):.1f}"
          "       (informational - smooth cloth is not a fault)")
    print(f"{'frames with a marker':<22}{marker_frames}/{count}")
    if scales:
        mm_per_px = float(np.median(scales))
        print(f"{'mm per pixel':<22}{mm_per_px:.4f}  "
              f"(field of view {first.shape[1] * mm_per_px:.0f} mm wide)")
        print(f"{'position range':<22}{min(positions):.0f} to {max(positions):.0f} mm")
    if tape_axis:
        horizontal = sum(tape_axis) == 0
        print(f"{'tape orientation':<22}"
              f"{'VERTICAL in the frame (correct)' if not horizontal else 'HORIZONTAL in the frame'}")
    if ratios:
        measured = float(np.median(ratios))
        designed = args.pitch_mm / args.marker_mm
        error_pct = 100.0 * (measured - designed) / designed
        print(f"{'pitch / marker ratio':<22}{measured:.4f} measured vs "
              f"{designed:.4f} configured  ({error_pct:+.2f}%)")
        print(f"{'  implied pitch':<22}{measured * args.marker_mm:.2f} mm "
              f"if the marker really is {args.marker_mm:.1f} mm")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), last)
    print(f"\nlast frame written to {out} - open it and look at it")

    problems: list[str] = []
    if fps < 5:
        problems.append(
            f"only {fps:.1f} fps. Drop the camera app's resolution, or use the "
            "phone's hotspot instead of a shared network."
        )
    if drift_pct > DRIFT_LIMIT_PCT:
        problems.append(
            f"brightness trended {drift_pct:.1f}% across the window. Exposure or "
            "white balance is "
            "still automatic; a drifting exposure looks exactly like a defect. "
            "Lock ISO, shutter and white balance in the app and disable any "
            "HDR / night / scene mode."
        )
    if median_sharp < SHARPNESS_FLOOR:
        problems.append(
            f"focus measures {median_sharp:.0f} on the printed markers, below "
            f"{SHARPNESS_FLOOR:.0f}. Lock "
            "focus manually at the working distance; the marker edges must "
            "resolve or the bank learns blur."
        )
    if tape_axis and sum(tape_axis) == 0:
        problems.append(
            "the tape runs HORIZONTALLY across the video. L2 computes the "
            "machine-direction position from a marker's vertical position in "
            "the frame, so the fabric must travel top-to-bottom in the image. "
            "Rotate the phone 90 degrees on the mount, or set the camera app's "
            "video rotation, until the tape runs down the frame."
        )
    if ratios:
        measured = float(np.median(ratios))
        designed = args.pitch_mm / args.marker_mm
        if abs(measured - designed) / designed > 0.03:
            problems.append(
                f"the pitch/marker ratio measures {measured:.4f} but the config "
                f"says {designed:.4f} ({100 * (measured - designed) / designed:+.1f}%). "
                f"Either marker_length_mm and marker_pitch_mm disagree with the "
                f"printed tape, or the printer scaled the two axes differently - "
                f"in which case no single marker_length_mm can correct it and the "
                f"tape must be reprinted. Implied pitch: "
                f"{measured * args.marker_mm:.2f} mm."
            )
    if marker_frames == 0:
        problems.append(
            f"no ArUco marker decoded in any frame. Check the tape is in shot, "
            f"that it was printed from {args.aruco}, and that it is lit and flat. "
            "Without it there is no position and no millimetre scale."
        )
    elif marker_frames < 0.5 * count:
        problems.append(
            f"markers decoded in only {marker_frames}/{count} frames. Position "
            "will be extrapolated most of the time. More light, larger markers, "
            "matte paper."
        )

    print()
    if problems:
        print("NOT READY:")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("READY. The stream is usable. Next:")
    print(f"  python tools/bench_run.py learn --url {args.url} --sku <your-sku>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
