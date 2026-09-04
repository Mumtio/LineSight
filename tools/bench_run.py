"""Two-phase bench run: learn the fabric, stop, swap the cloth, inspect.

Learning and inspecting are separate commands because on a bench the cloth is
swapped by hand between them. A mill with a moving line would run the two
against one continuous stream; here the operator needs to stop, change the
cloth, and start again.

    python tools/bench_run.py learn   --url <mjpeg> --sku bench
    #   ... swap the clean cloth for the defective sheet ...
    python tools/bench_run.py inspect --url <mjpeg> --sku bench --seconds 30

The bank and the calibration are written to ``banks/{sku}.npz`` and
``banks/{sku}.calibration.json``, so the inspect phase is a separate process and
can be re-run as often as you like without re-learning.

**The fabric must move during the learn phase.** The bank is fitted on one
stretch of clean cloth and the threshold is derived from a later stretch; if the
cloth is stationary, both see the same few square centimetres, the threshold
lands far too low, and every frame of the inspection is a false alarm. Pull the
clean fabric slowly and continuously from the moment learning starts until it
ends.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linesight.cli import print_report  # noqa: E402
from linesight.config import load_config  # noqa: E402
from linesight.pipeline import Pipeline  # noqa: E402

BANKS = Path(__file__).resolve().parents[1] / "banks"


def _calibration_path(config) -> Path:
    return Path(config.detect.bank_dir) / f"{config.sku}.calibration.json"


def phase_learn(args) -> int:
    """Watch clean fabric: fit the bank, then derive the threshold from later cloth."""
    config = load_config(args.sku)
    if args.budget is not None:
        config.calibration.budget_fa_per_100m = args.budget

    print(f"\nPHASE 1  LEARN   sku={config.sku}")
    print(f"  camera        {args.url}")
    print(f"  budget        {config.calibration.budget_fa_per_100m} FA/100 m")
    print(f"  fit frames    {args.fit_frames}")
    print(f"  cal frames    {args.cal_frames}")
    print("\n  >>> PULL THE CLEAN FABRIC SLOWLY AND KEEP PULLING <<<")
    print("  The threshold comes from cloth the bank has not seen. If the fabric")
    print("  is stationary, the two stages see the same patch and the threshold")
    print("  will be far too low.\n")

    pipeline = Pipeline(config)

    t0 = time.time()
    try:
        scorer = pipeline.fit_from_source(args.url, n_frames=args.fit_frames, save=True)
    except (ValueError, OSError) as exc:
        print(f"\nFAIL during fit: {exc}", file=sys.stderr)
        return 2
    print(f"  bank fitted: {scorer.bank_size} points in {time.time() - t0:.1f}s")
    print(f"  saved {Path(config.detect.bank_dir) / (config.sku + '.npz')}\n")

    print("  keep pulling - now collecting held-out clean fabric for the threshold")
    try:
        calibration = pipeline.calibrate_from_source(
            args.url, n_frames=args.cal_frames, skip=args.skip
        )
    except (ValueError, OSError) as exc:
        print(f"\nFAIL during calibration: {exc}", file=sys.stderr)
        print("  This is the guard doing its job, not a crash. Either capture more", file=sys.stderr)
        print("  clean fabric or state a looser budget with --budget.", file=sys.stderr)
        return 2

    out = _calibration_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "threshold": calibration.threshold,
        "abstain_low": calibration.abstain_low,
        "budget_fa_per_100m": calibration.budget_fa_per_100m,
        "metres_per_tile": calibration.metres_per_tile,
        "n_clean_tiles": calibration.n_clean_tiles,
        "sku": calibration.sku or config.sku,
        "fit_timestamp": calibration.fit_timestamp,
    }, indent=2), encoding="utf-8")

    print("\n  we never picked a threshold; we picked a false-alarm budget.")
    print(f"    budget               {calibration.budget_fa_per_100m} FA / 100 m")
    print(f"    threshold            {calibration.threshold:.4f}")
    print(f"    abstain band         [{calibration.abstain_low:.4f}, "
          f"{calibration.threshold:.4f})")
    print(f"    held-out clean tiles {calibration.n_clean_tiles}")
    print(f"    saved                {out}")

    print("\nREADY. Swap the clean cloth for the sheet you want inspected, then:")
    print(f"  python tools/bench_run.py inspect --url {args.url} "
          f"--sku {args.sku} --seconds {args.seconds}")
    return 0


def phase_inspect(args) -> int:
    """Inspect the sheet now under the camera, using the saved bank and threshold."""
    config = load_config(args.sku)
    pipeline = Pipeline(config)

    path = _calibration_path(config)
    if not path.exists():
        print(f"no calibration at {path} - run the learn phase first.", file=sys.stderr)
        return 2

    from linesight.types import Calibration

    stored = json.loads(path.read_text(encoding="utf-8"))
    pipeline.calibration = Calibration(**stored)
    print(f"\nPHASE 2  INSPECT   sku={config.sku}")
    print(f"  camera        {args.url}")
    print(f"  threshold     {pipeline.calibration.threshold:.4f} "
          f"(budget {pipeline.calibration.budget_fa_per_100m} FA/100 m, "
          f"{pipeline.calibration.n_clean_tiles} clean tiles)")
    print(f"  stopping after {args.seconds:.0f}s or {args.frames} frames, "
          "or Ctrl-C\n")

    import cv2

    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    for stale in evidence.glob("*.png"):
        stale.unlink()

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    roll_id = f"{config.sku}-{int(time.time())}"
    furthest_mm = 0.0
    count = 0
    saved = 0
    # Overlays are retained ONLY for frames that produced a detection, and only
    # up to a cap: a full-frame overlay is ~2 MB and keeping every frame of a
    # 300-frame run would cost half a gigabyte to crop a handful of events.
    kept: dict[int, object] = {}
    peaks: list[float] = []
    t0 = time.time()

    try:
        for result in pipeline.iter_frames(args.url):
            pipeline.tracker.update(result.detections)
            furthest_mm = max(furthest_mm, result.geometry.along_mm)
            count += 1
            for d in result.detections:
                print(f"  {d.along_mm / 1000:7.3f} m  {d.length_mm:6.1f} mm  "
                      f"score {d.max_score:6.2f}  {d.assertion.value}")

            if result.detections:
                overlay = result.overlay()
                if len(kept) < args.max_evidence:
                    kept[result.frame.index] = overlay
                if saved < args.max_evidence:
                    shot = overlay.copy()
                    for d in result.detections:
                        x, y, w, h = (int(v) for v in d.bbox_px)
                        colour = (0, 0, 255) if d.assertion.value == "asserted" else (0, 200, 255)
                        cv2.rectangle(shot, (x, y), (x + w, y + h), colour, 2)
                        cv2.putText(shot, f"{d.along_mm / 1000:.3f}m {d.length_mm:.0f}mm "
                                          f"s{d.max_score:.1f}", (x, max(14, y - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
                    out = evidence / f"detect_{result.frame.index:04d}.png"
                    cv2.imwrite(str(out), shot)
                    saved += 1
            # The peak score in EVERY frame, crossed or not. A detection tells
            # you what cleared the threshold; a miss tells you nothing at all
            # unless the margin is recorded, and "how close was it" is the only
            # question worth asking about a defect the system did not report.
            peak = float(result.score_map.max())
            peaks.append(peak)
            if count % 25 == 0:
                print(f"  [{count} frames, {furthest_mm / 1000:.2f} m, "
                      f"{time.time() - t0:.0f}s, peak score {peak:.2f} "
                      f"vs threshold {pipeline.calibration.threshold:.2f}]", flush=True)
            if count >= args.frames or (time.time() - t0) >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n  stopped by operator")
    except (OSError, RuntimeError) as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 2

    if count == 0:
        print("no frames inspected.", file=sys.stderr)
        return 2

    events = pipeline.tracker.finalise()

    # Attach one evidence crop per event, from the frame where it scored highest.
    from linesight.report.defect_map import save_event_crop

    for event in events:
        if not event.detections:
            continue
        best = max(event.detections, key=lambda d: d.max_score)
        overlay = kept.get(best.frame_index)
        if overlay is None:
            continue
        try:
            save_event_crop(overlay, event, evidence, pad_px=40)
        except ValueError:
            pass

    report = pipeline.finalise(roll_id, events, furthest_mm)
    report.started_at = started
    report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print_report(report, verbose=True)
    if peaks:
        import numpy as np

        thr = pipeline.calibration.threshold
        arr = np.asarray(peaks)
        below = arr[arr < thr]
        print("")
        print(f"  per-frame peak score: median {np.median(arr):.2f}, "
              f"max {arr.max():.2f}, threshold {thr:.2f}")
        print(f"  frames that crossed:  {(arr >= thr).sum()}/{len(arr)}")
        if below.size:
            print(f"  closest near-miss:    {below.max():.2f} "
                  f"({thr - below.max():.2f} below the threshold)")
    crops = sorted(evidence.glob("event_*.png"))
    print(f"  evidence: {saved} annotated frame(s) and {len(crops)} event crop(s) "
          f"in {evidence}")

    if args.pdf:
        from linesight.report.defect_map import save_event_crop
        from linesight.report.pdf import render_pdf

        out = Path(args.pdf)
        try:
            render_pdf(report, out)
            print(f"wrote {out}")
        except Exception as exc:  # noqa: BLE001 - a failed PDF must not lose the run
            print(f"could not render the PDF ({exc}); the report above still stands.",
                  file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="phase", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", required=True, help="MJPEG stream URL")
    common.add_argument("--sku", default="bench")

    learn = sub.add_parser("learn", parents=[common],
                           help="watch clean fabric: fit the bank and the threshold")
    learn.add_argument("--fit-frames", type=int, default=30)
    learn.add_argument("--cal-frames", type=int, default=160)
    learn.add_argument("--skip", type=int, default=0,
                       help="frames to discard before calibrating, so the threshold "
                            "comes from cloth the bank has not seen")
    learn.add_argument("--budget", type=float, default=None,
                       help="false alarms per 100 m; overrides the SKU config")
    learn.add_argument("--seconds", type=float, default=30.0,
                       help="only used to print the suggested inspect command")
    learn.set_defaults(func=phase_learn)

    inspect = sub.add_parser("inspect", parents=[common],
                             help="inspect the sheet now under the camera")
    inspect.add_argument("--seconds", type=float, default=30.0)
    inspect.add_argument("--frames", type=int, default=400)
    inspect.add_argument("--pdf", default=None, help="also render a roll report PDF")
    inspect.add_argument("--evidence", default="results/evidence_bench",
                         help="where annotated frames and event crops are written")
    inspect.add_argument("--max-evidence", type=int, default=40,
                         help="cap on retained overlays and saved frames")
    inspect.set_defaults(func=phase_inspect)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
