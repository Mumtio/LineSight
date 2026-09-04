"""
probes/p14_position_trace.py

QUESTION: Does one physical defect report the SAME roll position in every frame
          that sees it? Absolute marker encoding promises exactly that, and a
          bench run broke the promise: a single patterned patch came back at
          1.308, 1.408, 1.504, 1.610 and 1.653 m, a spread of 377 mm, which then
          fragmented one defect into dozens of events because the tracker joins
          only within 60 mm.
PASS IF:  along_mm advances smoothly and monotonically with the cloth, marker
          IDs run consecutively without jumps, and the marker-relative position
          of a fixed point on the cloth stays constant.
PURPOSE:  a diagnostic, not a promotion probe - it exists to localise a
          position fault to one of its two inputs, marker ID or mm/px.

The position of a frame is

    along_mm = marker_id * pitch_mm  -  marker_top_y_px * mm_per_px

so a wrong ID moves the frame by a whole multiple of the pitch, and a wrong
mm_per_px scales the correction term. Both are printed here alongside the raw
inputs, because a position that is wrong tells you nothing about WHICH of its
two inputs was wrong.

    python probes/p14_position_trace.py --url <mjpeg> --sku bench --frames 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linesight.acquisition.mjpeg import MjpegSource  # noqa: E402
from linesight.config import load_config  # noqa: E402
from linesight.geometry.aruco import ArucoReader, detect_markers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--sku", default="bench")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    config = load_config(args.sku)
    geometry = config.geometry
    reader = ArucoReader(geometry)
    source = MjpegSource(args.url, stride=args.stride, timeout_s=args.timeout,
                         drop_stale=True)

    print(f"pitch {geometry.marker_pitch_mm} mm   marker {geometry.marker_length_mm} mm   "
          f"dict {geometry.aruco_dict}\n")
    print(f"{'#':>3} {'ids':>16} {'top_y':>7} {'mm/px':>7} {'along_mm':>9} "
          f"{'delta':>7}  flags")

    along: list[float] = []
    ids_seen: list[tuple[int, ...]] = []
    scales: list[float] = []
    interpolated = gaps = 0
    previous: float | None = None

    try:
        with source:
            for index, frame in enumerate(source):
                readings = detect_markers(
                    frame.image, geometry.aruco_dict,
                    geometry.marker_length_mm, geometry.marker_pitch_mm,
                )
                result = reader.read(frame.image, frame.timestamp)
                ids = tuple(r.marker_id for r in readings)
                top = min((float(r.corners[:, 1].min()) for r in readings), default=float("nan"))

                delta = result.along_mm - previous if previous is not None else float("nan")
                previous = result.along_mm
                along.append(result.along_mm)
                ids_seen.append(ids)
                scales.append(result.mm_per_px)
                interpolated += bool(result.interpolated)
                gaps += bool(result.gap_warning)

                flags = " ".join(filter(None, [
                    "interp" if result.interpolated else "",
                    "GAP" if result.gap_warning else "",
                    "ID-JUMP" if len(ids_seen) > 1 and ids and ids_seen[-2]
                    and abs(min(ids) - min(ids_seen[-2])) > 1 else "",
                ]))
                print(f"{index:>3} {str(list(ids)):>16} {top:>7.1f} {result.mm_per_px:>7.4f} "
                      f"{result.along_mm:>9.1f} {delta:>7.1f}  {flags}")

                if index + 1 >= args.frames:
                    break
    except OSError as exc:
        print(f"\nstream failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass

    if len(along) < 2:
        print("\nnot enough frames to judge.", file=sys.stderr)
        return 1

    a = np.asarray(along)
    d = np.diff(a)
    s = np.asarray(scales)
    flat = [i for ids in ids_seen for i in ids]

    print(f"\n{'frames':<26}{len(a)}")
    print(f"{'along_mm range':<26}{a.min():.1f} .. {a.max():.1f}  "
          f"(travelled {a.max() - a.min():.0f} mm)")
    print(f"{'per-frame delta':<26}median {np.median(d):+.1f} mm, "
          f"min {d.min():+.1f}, max {d.max():+.1f}")
    print(f"{'backwards steps':<26}{(d < -5).sum()}  <- position should not go backwards")
    print(f"{'mm/px':<26}median {np.median(s):.4f}, spread "
          f"{100 * (s.max() - s.min()) / np.median(s):.1f}%")
    print(f"{'marker ids seen':<26}{sorted(set(flat))}")
    print(f"{'interpolated / gaps':<26}{interpolated} / {gaps} of {len(a)}")

    problems = []
    if (d < -5).sum():
        problems.append(
            f"position moved BACKWARDS on {(d < -5).sum()} frame(s). Cloth does not "
            "reverse, so this is a misdecoded marker id or a bad mm/px."
        )
    if len(a) > 4 and np.abs(d).max() > 3 * geometry.marker_pitch_mm:
        problems.append(
            f"a single frame moved {np.abs(d).max():.0f} mm, more than three pitches. "
            "That is the signature of a marker id read wrongly - the frame jumps "
            "by a whole multiple of the pitch."
        )
    if s.size and 100 * (s.max() - s.min()) / np.median(s) > 8:
        problems.append(
            f"mm/px varied by {100 * (s.max() - s.min()) / np.median(s):.0f}%. Scale "
            "multiplies the in-frame offset, so this smears every reported position."
        )
    if gaps:
        problems.append(
            f"{gaps} gap warning(s): position was extrapolated, not measured. "
            "Hand-pulled cloth does not move at a constant rate, so extrapolated "
            "positions are guesses."
        )

    print()
    if problems:
        print("PROBLEMS:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("PASS: position advances smoothly, ids are consecutive, scale is stable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
