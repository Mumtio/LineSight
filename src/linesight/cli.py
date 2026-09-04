"""``python -m linesight fit | calibrate | run | serve | report``.

Five commands, in the order a SKU moves through them: ``fit`` builds the memory
bank from clean fabric, ``calibrate`` derives the threshold from a false-alarm
budget, ``run`` inspects a roll and prints its defect table, ``serve`` puts the
same pipeline behind the operator page, and ``report`` renders a stored roll to
PDF.

Each command is a thin wrapper over ``Pipeline``: parse arguments, make one
call, print the result. Every heavy import (torch, FastAPI, reportlab) happens
inside the command that needs it, so ``--help`` stays instant and a missing
optional extra breaks only the command that depends on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .types import Assertion, RollReport

__all__ = [
    "build_parser",
    "cmd_calibrate",
    "cmd_fit",
    "cmd_report",
    "cmd_run",
    "cmd_serve",
    "main",
    "print_report",
]


def build_parser() -> argparse.ArgumentParser:
    """All subcommands and flags.

        linesight fit       --sku aitex_02 --normal data/aitex/normal_02/
        linesight calibrate --sku aitex_02 --clean  data/aitex/clean_02/ --budget 1.0
        linesight run       --sku aitex_02 --source data/aitex/roll_02/
        linesight serve     --sku fabric_stain --source http://127.0.0.1:8090/video
        linesight report    --roll latest --out results/roll_report.pdf

    ``--config-dir`` and ``--set key=value`` are global, so any config key can
    be overridden for one run without editing a YAML file.
    """
    # Global flags live on a PARENT parser so they are accepted both before and
    # after the subcommand: `linesight --sku x fit ...` and
    # `linesight fit --sku x ...` both work. A CLI that rejects the ordering its
    # own help text uses is one nobody can type from memory.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sku", default=None, help="SKU name; loads configs/sku_{sku}.yaml")
    common.add_argument("--config-dir", default="configs", help="directory of YAML configs")
    common.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set detect.scorer=stub",
    )
    common.add_argument("-v", "--verbose", action="store_true")

    parser = argparse.ArgumentParser(
        prog="linesight",
        description="Cold-start visual inspection for textile rolls.",
        parents=[common],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser(
        "fit", help="build the SKU memory bank from clean fabric", parents=[common]
    )
    fit.add_argument("--normal", required=True, help="folder of defect-free images")
    fit.add_argument("--no-save", action="store_true")
    fit.set_defaults(func=cmd_fit)

    cal = sub.add_parser(
        "calibrate", help="derive a threshold from a false-alarm budget", parents=[common]
    )
    cal.add_argument("--clean", required=True, help="HELD-OUT clean folder, disjoint from --normal")
    cal.add_argument("--budget", type=float, default=None, help="false alarms per 100 m")
    cal.set_defaults(func=cmd_calibrate)

    run = sub.add_parser(
        "run", help="inspect a roll and print the defect table", parents=[common]
    )
    run.add_argument("--source", default=None, help="folder, video file, or stream URL")
    run.add_argument("--roll-id", default=None)
    run.add_argument("--clean", default=None, help="calibrate from this folder first")
    run.add_argument("--threshold", type=float, default=None, help="bypass calibration (states so)")
    run.add_argument("--save", action="store_true", help="persist the roll to the store")
    run.add_argument("--json", action="store_true", help="emit the report as JSON")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser(
        "serve", help="FastAPI + operator page", parents=[common]
    )
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--source", default=None, help="frame source; overrides the SKU config")
    serve.set_defaults(func=cmd_serve)

    report = sub.add_parser(
        "report", help="render a stored roll to PDF", parents=[common]
    )
    report.add_argument("--roll", default="latest")
    report.add_argument("--out", default="results/roll_report.pdf")
    report.set_defaults(func=cmd_report)

    return parser


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #


def _apply_overrides(config: object, overrides: list[str]) -> object:
    """Apply ``--set section.key=value`` onto a loaded config.

    Values are parsed as JSON where possible so ``--set detect.input_size=256``
    gives an int and ``--set preprocess.flatfield=false`` gives a bool; anything
    unparseable stays a string.

    Raises:
        ValueError: on a malformed override or an unknown key. Silently ignoring
            a typo would run the default and report it as the override.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw

        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config section {part!r} in {key!r}")
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise ValueError(f"unknown config key {key!r}")
        setattr(target, parts[-1], value)
    return config


def _load(args: argparse.Namespace) -> object:
    from .config import load_config

    config = load_config(args.sku, args.config_dir)
    return _apply_overrides(config, args.set)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_fit(args: argparse.Namespace) -> int:
    """Build and save the SKU memory bank. Prints elapsed time and bank size.

    Both numbers are reported because both are load-bearing: ~90 s to fit and
    ~600 KB of artifact are what make refitting per SKU practical on a mill
    floor rather than a retraining project.
    """
    from .pipeline import Pipeline

    config = _load(args)
    pipeline = Pipeline(config)

    print(f"fitting {config.sku} from {args.normal} ...")
    started = time.perf_counter()
    scorer = pipeline.fit(args.normal, save=not args.no_save)
    elapsed = time.perf_counter() - started

    print(f"  fitted in {elapsed:.1f}s")
    if hasattr(scorer, "bank_size"):
        print(f"  bank      {scorer.bank_size} points")
    if not args.no_save:
        path = config.bank_path
        print(f"  artifact  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Derive the threshold from a false-alarm budget on held-out clean tiles.

    Prints the budget, the resulting threshold, the abstain band, and the
    achievable resolution for the sample size - the last of these especially
    when it is worse than what was asked for.
    """
    from .calibrate.threshold import achievable_resolution
    from .pipeline import Pipeline

    config = _load(args)
    if args.budget is not None:
        config.calibration.budget_fa_per_100m = args.budget

    pipeline = Pipeline(config)
    try:
        cal = pipeline.calibrate(args.clean)
    except ValueError as exc:
        print(f"cannot calibrate: {exc}", file=sys.stderr)
        return 2

    print("we never picked a threshold; we picked a false-alarm budget.")
    print(f"  budget                 {cal.budget_fa_per_100m} FA / 100 m")
    print(f"  threshold              {cal.threshold:.4f}")
    print(f"  abstain band           [{cal.abstain_low:.4f}, {cal.threshold:.4f})")
    print(f"  held-out clean tiles   {cal.n_clean_tiles}")
    print(
        f"  achievable resolution  "
        f"{achievable_resolution(cal.n_clean_tiles, cal.metres_per_tile):.3g} FA / 100 m"
    )

    out = Path(config.detect.bank_dir) / f"{config.sku}.calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal.__dict__ if hasattr(cal, "__dict__") else {
        "threshold": cal.threshold,
        "abstain_low": cal.abstain_low,
        "budget_fa_per_100m": cal.budget_fa_per_100m,
        "metres_per_tile": cal.metres_per_tile,
        "n_clean_tiles": cal.n_clean_tiles,
        "sku": cal.sku,
        "fit_timestamp": cal.fit_timestamp,
    }, indent=2), encoding="utf-8")
    print(f"  saved                  {out}")
    return 0


def _load_calibration(config: object) -> object | None:
    """Restore a saved calibration for this SKU, if one exists."""
    from .types import Calibration

    path = Path(config.detect.bank_dir) / f"{config.sku}.calibration.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Calibration(**data)


def cmd_run(args: argparse.Namespace) -> int:
    """Inspect a roll and print the defect table. The whole chain, one command.

    The threshold comes from one of three places, in this order: a folder of
    held-out clean fabric (``--clean``), an explicit ``--threshold`` that is
    labelled uncalibrated wherever it is reported, or this SKU's saved
    calibration. With none of the three it refuses to run, because inspecting
    without a threshold would mean inventing one.
    """
    from .pipeline import Pipeline
    from .types import Calibration

    config = _load(args)
    pipeline = Pipeline(config)

    if args.clean:
        pipeline.calibrate(args.clean)
    elif args.threshold is not None:
        # Explicit, and labelled as such in the report: a hand-set threshold has
        # no false-alarm guarantee, and pretending otherwise would undermine
        # every honest thing this system says about calibration.
        pipeline.calibration = Calibration(
            threshold=args.threshold,
            abstain_low=args.threshold * 0.8,
            budget_fa_per_100m=float("nan"),
            metres_per_tile=pipeline.metres_per_tile,
            n_clean_tiles=0,
            sku=config.sku,
            fit_timestamp="hand-set (NOT calibrated)",
        )
        print("WARNING: --threshold bypasses calibration. No false-alarm guarantee.")
    else:
        pipeline.calibration = _load_calibration(config)

    if pipeline.calibration is None:
        print(
            f"no calibration for SKU {config.sku!r}. Run:\n"
            f"  python -m linesight calibrate --sku {config.sku} --clean <folder>",
            file=sys.stderr,
        )
        return 2

    report = pipeline.run(args.source, args.roll_id)

    if args.json:
        print(json.dumps(_report_dict(report), indent=2))
    else:
        print_report(report, verbose=args.verbose)

    if args.save:
        from .api.store import Store

        with Store(config.api.db_path) as store:
            store.init_schema()
            store.save_roll(report)
        print(f"\nsaved roll {report.roll_id} to {config.api.db_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI app and the operator page. See ``api/app.py``."""
    config = _load(args)
    if args.host:
        config.api.host = args.host
    if args.port:
        config.api.port = args.port

    try:
        from .api.app import run_server
    except ImportError as exc:
        print(f"the API extra is not installed: {exc}", file=sys.stderr)
        print("  pip install -e '.[api]'", file=sys.stderr)
        return 2

    run_server(config, source_uri=args.source)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Render a stored roll to PDF and defect map."""
    from .api.store import Store
    from .report.pdf import render_pdf

    config = _load(args)
    with Store(config.api.db_path) as store:
        store.init_schema()
        if args.roll == "latest":
            rolls = store.list_rolls(limit=1)
            if not rolls:
                print("no rolls in the store - run `linesight run --save` first", file=sys.stderr)
                return 2
            roll_id = rolls[0]["id"]
        else:
            roll_id = args.roll
        report = store.get_roll(roll_id)

    if report is None:
        print(f"no roll {roll_id!r} in the store", file=sys.stderr)
        return 2

    path = render_pdf(report, args.out)
    print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _report_dict(report: RollReport) -> dict:
    """A JSON-safe view of the report, for ``--json`` and the store."""
    return {
        "roll_id": report.roll_id,
        "sku": report.sku,
        "roll_length_m": report.roll_length_m,
        "width_m": report.width_m,
        "total_points": report.total_points,
        "points_per_100yd2": report.points_per_100yd2,
        "verdict": report.verdict.value,
        "gap_warnings": report.gap_warnings,
        "false_alarms": report.false_alarms,
        "meta": report.meta,
        "events": [
            {
                "id": e.event_id,
                "along_start_mm": e.along_start_mm,
                "along_end_mm": e.along_end_mm,
                "across_start_mm": e.across_start_mm,
                "length_mm": e.length_mm,
                "width_mm": e.width_mm,
                "max_score": e.max_score,
                "confidence": e.confidence,
                "assertion": e.assertion.value,
                "status": e.status.value,
                "n_frames": e.n_frames,
            }
            for e in report.events
        ],
    }


def print_report(report: RollReport, verbose: bool = False) -> None:
    """The terminal view of a roll: the PDF's facts with no optional dependency.

    Columns: event id, position (m), cross position (mm), length (mm), score,
    assertion, ASTM points. Footer: total points, points/100 yd2, verdict,
    false alarms against the stated budget, and any gap warnings.

    The text itself comes from ``report.render_text_report``, which is also the
    PDF's text fallback. Two formatters drifting apart is how a terminal and a
    filed document end up disagreeing about the same roll, and the disagreement
    surfaces in front of whoever is holding both.
    """
    from .report.pdf import render_text_report

    print(render_text_report(report))

    if verbose and report.meta.get("latency_ms"):
        print("  latency (median ms/frame):")
        for stage, value in report.meta["latency_ms"].items():
            if stage != "frame_stride":
                print(f"    {stage:<12} {value:>7.1f}")
        print()


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for user error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
