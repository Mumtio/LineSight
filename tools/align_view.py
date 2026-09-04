"""Live alignment view: watch the stream with the ROI and the tape drawn on it.

Two problems this solves, both found on the bench. The inspected region is a
rectangle in the config that the operator cannot see, so there is no way to tell
whether the fabric is inside it. And the marker tape drifts sideways as cloth is
hand-pulled; when it crosses into the inspected region the system scores the
markers as defects, which is exactly what happened on the first real run - all
thirteen detections were the tape.

    python tools/align_view.py --url <mjpeg> --sku bench
    python tools/align_view.py --url <mjpeg> --sku bench --detect

Geometry mode runs at camera rate and is what you align and pull against.
Detect mode adds the model, so it shows the heatmap and live detections at a
few frames a second - useful for confirming what the model sees, too slow to
align a rig with.

Press q or ESC to close.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linesight.acquisition.mjpeg import MjpegSource  # noqa: E402
from linesight.config import load_config  # noqa: E402
from linesight.geometry.aruco import detect_markers  # noqa: E402

GREEN = (80, 220, 80)
CYAN = (220, 200, 60)
RED = (60, 60, 240)
AMBER = (60, 190, 250)
WHITE = (245, 245, 245)
INK = (25, 25, 25)


def _hud(canvas, lines, x=12, y=24, warn=False):
    """A legible readout: dark plate under light text, so it survives any fabric."""
    pad, lh = 8, 22
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in lines)
    box = canvas[max(0, y - 18): y - 18 + len(lines) * lh + pad,
                 max(0, x - pad): x - pad + w + 2 * pad]
    if box.size:
        box[:] = (box * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        colour = RED if (warn and i == 0) else WHITE
        cv2.putText(canvas, text, (x, y + i * lh), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1, cv2.LINE_AA)


def resolve_roi(config, frame_shape) -> tuple[int, int, int, int]:
    """The configured ROI, or the whole frame when it is inferred at run time."""
    height, width = frame_shape[:2]
    if config.geometry.roi is not None:
        return tuple(int(v) for v in config.geometry.roi)
    return (0, 0, width, height)


def run(args) -> int:
    config = load_config(args.sku)
    marker_mm = config.geometry.marker_length_mm
    pitch_mm = config.geometry.marker_pitch_mm
    dictionary = config.geometry.aruco_dict

    pipeline = None
    if args.detect:
        from linesight.pipeline import Pipeline
        from linesight.types import Calibration
        import json

        path = Path(config.detect.bank_dir) / f"{config.sku}.calibration.json"
        if not path.exists():
            print(f"no calibration at {path} - run the learn phase first.", file=sys.stderr)
            return 2
        pipeline = Pipeline(config)
        pipeline.calibration = Calibration(**json.loads(path.read_text(encoding="utf-8")))
        print(f"detect mode: threshold {pipeline.calibration.threshold:.3f}")

    print(f"opening {args.url}   (q or ESC to close)")
    window = "LineSight alignment"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    recent = deque(maxlen=30)
    source = MjpegSource(args.url, stride=1, timeout_s=args.timeout, drop_stale=True)
    frames = 0

    try:
        with source:
            for frame in source:
                image = frame.image
                canvas = image.copy()
                x, y, w, h = resolve_roi(config, image.shape)
                mm_per_px = None
                intrusion = False

                readings = detect_markers(image, dictionary, marker_mm, pitch_mm)
                if readings:
                    mm_per_px = float(np.median([r.mm_per_px for r in readings]))
                    xs = np.concatenate([r.corners[:, 0] for r in readings])
                    tape_lo, tape_hi = float(xs.min()), float(xs.max())
                    intrusion = tape_lo < x + w and tape_hi > x
                    # The tape band, so its drift is visible before it matters.
                    band = canvas[:, int(tape_lo):int(tape_hi) + 1]
                    if band.size:
                        band[:] = cv2.addWeighted(
                            band, 0.72, np.full_like(band, AMBER), 0.28, 0)
                    for r in readings:
                        pts = r.corners.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(canvas, [pts], True, CYAN, 2, cv2.LINE_AA)
                        cx, cy = r.centre_px
                        cv2.putText(canvas, f"id {r.marker_id}", (int(cx) - 22, int(cy) - 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
                else:
                    tape_lo = tape_hi = None

                if pipeline is not None:
                    try:
                        result = pipeline._process(frame)
                        overlay = result.overlay(alpha=0.45)
                        oh, ow = overlay.shape[:2]
                        canvas[y:y + oh, x:x + ow] = overlay
                        for d in result.detections:
                            dx, dy, dw, dh = (int(v) for v in d.bbox_px)
                            cv2.rectangle(canvas, (x + dx, y + dy),
                                          (x + dx + dw, y + dy + dh), RED, 2)
                            cv2.putText(canvas, f"{d.along_mm / 1000:.3f}m {d.length_mm:.0f}mm",
                                        (x + dx, max(14, y + dy - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
                    except Exception as exc:  # noqa: BLE001 - a view must not die on one frame
                        cv2.putText(canvas, f"detect: {exc}"[:70], (12, 700),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

                # The inspected rectangle, drawn last so nothing hides it.
                cv2.rectangle(canvas, (x, y), (x + w, y + h),
                              RED if intrusion else GREEN, 3)
                cv2.putText(canvas, "INSPECTED", (x + 8, y + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            RED if intrusion else GREEN, 2, cv2.LINE_AA)

                frames += 1
                recent.append(time.time())
                fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0

                lines = []
                if intrusion:
                    lines.append("!! TAPE IS INSIDE THE INSPECTED REGION - shift the fabric !!")
                lines += [
                    f"{fps:4.1f} fps    markers {len(readings)}",
                    f"ROI  x {x}..{x + w}   "
                    + (f"{w * mm_per_px:.0f} mm wide" if mm_per_px else "scale unknown"),
                ]
                if tape_lo is not None:
                    gap_px = tape_lo - (x + w)
                    lines.append(
                        f"tape x {tape_lo:.0f}..{tape_hi:.0f}   "
                        + (f"clear by {gap_px:.0f} px" if gap_px > 0
                           else f"OVERLAPS by {-gap_px:.0f} px")
                    )
                if mm_per_px:
                    lines.append(f"{mm_per_px:.4f} mm/px")
                if readings:
                    lines.append(f"position {readings[0].along_mm / 1000:.3f} m")
                _hud(canvas, lines, warn=intrusion)

                cv2.imshow(window, canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except OSError as exc:
        print(f"stream failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    print(f"closed after {frames} frames")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--sku", default="bench")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="seconds to wait for the stream; a busy phone can be slow")
    parser.add_argument("--detect", action="store_true",
                        help="also run the model: heatmap and live detections, a few fps")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
