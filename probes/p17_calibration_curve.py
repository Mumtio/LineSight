"""
probes/p17_calibration_curve.py

QUESTION: Section 7 of docs/evaluation.md. ADR-007 claims the threshold is not
          chosen but derived: state a false-alarm budget, get that false-alarm
          rate. `tests/test_threshold.py` proves the arithmetic on synthetic
          scores. This asks whether it holds on real fabric, where the score
          distribution is whatever AITEX makes it.
PASS IF:  Two things, both stated before running.
          (a) On a held-out clean set disjoint from the calibration set, the
              REALISED exceedance fraction tracks the REQUESTED one. Not
              exactly -- a finite sample cannot -- but without systematic bias,
              and in particular without running consistently HOT, which would
              mean the operator gets more false alarms than they asked for.
          (b) Below the sample's resolving power the function REFUSES rather
              than returning a threshold. A curve that quietly kept producing
              numbers at budgets its sample cannot support would be the failure
              this probe exists to catch.
STATUS:   see results/calibration.csv

The x-axis is the ALLOWED TAIL FRACTION, not a rate per 100 m, and that choice
is the point rather than an evasion. The conformal guarantee in ADR-007 is a
statement about a quantile of the clean-score distribution; converting it to a
rate per 100 m requires `metres_per_tile`, which is a property of the rig, not
of the method. AITEX strips carry no scale, so a rate would be a fabricated
number wearing real units. The `fa_per_100m_at_bench_scale` column converts
using the value this project's own bench rig measured
(`banks/bench.calibration.json`), and is there only to connect the two.

That conversion is also why the sweep looks coarse. At 1.5 cm of fabric per
tile, 100 m is ~6,600 tiles, so a budget of 1 FA/100 m is a tail fraction of
1.5e-4 and needs tens of thousands of clean tiles to resolve stably. AITEX
yields a few hundred per fabric. The wall is therefore inside the swept range
on purpose: both behaviours -- tracking, and refusing -- appear in one table.

Run:  python probes/p17_calibration_curve.py --root data/aitex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _study_harness import RESULTS, load_p12
from linesight.calibrate.threshold import (
    achievable_resolution,
    threshold_from_budget,
)
from linesight.config import DetectConfig
from linesight.datasets.aitex import cut_tiles, list_images, load_image
from linesight.detect.patchcore import PatchCore

# Swept directly, because this is what the conformal quantile actually promises.
TAIL_FRACTIONS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
METRES_PER_TILE = 0.015010136895739717  # measured, banks/bench.calibration.json


def clean_tiles_for(root: Path, code: str, tile_size: int, overlap: int) -> list[np.ndarray]:
    """Every defect-free tile of one fabric, overlapped to raise the tile count.

    Overlap is how the calibration sample gets large enough to resolve anything
    at all. It buys correlated tiles rather than independent ones, which is a
    real caveat and is recorded in the results notes rather than hidden: the
    effective sample size is smaller than the tile count.
    """
    tiles: list[np.ndarray] = []
    for img in sorted(list_images(root), key=lambda i: i.path.name):
        if img.fabric_code != code or img.is_defective:
            continue
        strip = load_image(img.path)
        tiles.extend(tile for tile, _, _ in cut_tiles(strip, tile_size, overlap=overlap))
    return tiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/aitex")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--n-fit", type=int, default=30)
    parser.add_argument("--metres-per-tile", type=float, default=METRES_PER_TILE)
    parser.add_argument("--out", default=str(RESULTS / "calibration.csv"))
    args = parser.parse_args()

    p12 = load_p12()
    root = Path(args.root)
    codes = sorted({img.fabric_code for img in list_images(root)})
    config = DetectConfig(input_size=args.input_size, coreset_frac=0.10,
                          coreset_method="greedy", batch_size=args.batch_size)

    rows = []
    for code in codes:
        tiles = clean_tiles_for(root, code, args.tile_size, args.overlap)
        if len(tiles) < args.n_fit + 100:
            print(f"fabric {code}: skipped ({len(tiles)} clean tiles)", flush=True)
            continue

        rng = np.random.default_rng(0)
        order = rng.permutation(len(tiles))
        fit_tiles = [tiles[i] for i in order[: args.n_fit]]
        rest = order[args.n_fit :]
        half = len(rest) // 2
        cal_tiles = [tiles[i] for i in rest[:half]]
        val_tiles = [tiles[i] for i in rest[half:]]

        scorer = PatchCore(config)
        scorer.fit(fit_tiles)
        cal = np.array([float(m.max())
                        for m in p12.score_all(scorer, cal_tiles, args.batch_size)])
        val = np.array([float(m.max())
                        for m in p12.score_all(scorer, val_tiles, args.batch_size)])

        finest = achievable_resolution(len(cal), args.metres_per_tile)
        stable = finest * 10  # stability_margin default
        print(f"\nfabric {code}: {len(cal)} calibration / {len(val)} validation tiles",
              flush=True)
        print(f"  marginal resolution {finest:.1f} FA/100 m; "
              f"stable from about {stable:.0f}", flush=True)

        for fraction in TAIL_FRACTIONS:
            budget = fraction * 100.0 / args.metres_per_tile
            row = {
                "fabric": code,
                "requested_tail_fraction": fraction,
                "fa_per_100m_at_bench_scale": round(budget, 1),
                "n_calibration_tiles": len(cal),
                "n_validation_tiles": len(val),
            }
            try:
                threshold, abstain_low = threshold_from_budget(
                    cal, budget, args.metres_per_tile
                )
            except ValueError as exc:
                kind = ("sample_too_small" if "cannot resolve" in str(exc)
                        else "unstable_tail")
                row.update({"refused": True, "refusal": kind, "threshold": "",
                            "abstain_low": "", "realised_tail_fraction": "",
                            "n_exceed": "", "ratio": ""})
                rows.append(row)
                print(f"  fraction {fraction:>6.3f}  REFUSED  ({kind})", flush=True)
                continue

            n_exceed = int((val > threshold).sum())
            realised = n_exceed / len(val)
            row.update({
                "refused": False, "refusal": "",
                "threshold": round(float(threshold), 4),
                "abstain_low": round(float(abstain_low), 4),
                "realised_tail_fraction": round(realised, 5),
                "n_exceed": n_exceed,
                "ratio": round(realised / fraction, 3),
            })
            rows.append(row)
            print(f"  fraction {fraction:>6.3f}  threshold {threshold:>8.3f}  "
                  f"realised {realised:>6.4f}  ratio {realised / fraction:>5.2f}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fabric", "requested_tail_fraction", "fa_per_100m_at_bench_scale",
              "refused", "refusal", "threshold", "abstain_low",
              "realised_tail_fraction", "ratio", "n_exceed", "n_calibration_tiles",
              "n_validation_tiles"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}\n")

    print(f"{'requested':>10} {'ok':>4} {'refused':>8} {'mean realised':>14} "
          f"{'ratio':>7}")
    for fraction in TAIL_FRACTIONS:
        sub = [r for r in rows if r["requested_tail_fraction"] == fraction]
        ok = [r for r in sub if not r["refused"]]
        if ok:
            mean = float(np.mean([r["realised_tail_fraction"] for r in ok]))
            print(f"{fraction:>10.3f} {len(ok):>4d} {len(sub) - len(ok):>8d} "
                  f"{mean:>14.4f} {mean / fraction:>7.2f}")
        else:
            print(f"{fraction:>10.3f} {0:>4d} {len(sub):>8d} {'-':>14} {'-':>7}")

    ok = [r for r in rows if not r["refused"]]
    if ok:
        ratios = np.array([r["ratio"] for r in ok], dtype=float)
        hot = int((ratios > 1.0).sum())
        print(f"\nrealised / requested over {len(ok)} resolvable cases: "
              f"mean {ratios.mean():.2f}, median {np.median(ratios):.2f}")
        print(f"ran hot (more false alarms than asked for) in {hot} of {len(ok)}")
    refusals = [r for r in rows if r["refused"]]
    print(f"refused {len(refusals)} of {len(rows)} cases "
          f"({sum(r['refusal'] == 'sample_too_small' for r in refusals)} sample too "
          f"small, {sum(r['refusal'] == 'unstable_tail' for r in refusals)} unstable "
          f"tail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
