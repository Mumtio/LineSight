"""
probes/p16_normal_set_sweep.py

QUESTION: Section 4 of docs/evaluation.md -- "the money plot". The whitepaper
          asserts that a mill can onboard a new SKU from a handful of
          defect-free samples; objective O4 puts a number on it. How does tile
          AUROC actually move with the size of the fit set?
PASS IF:  Stated before running, so it cannot be retrofitted. If the curve
          flattens at or below ~30 tiles, the central claim stops being an
          assertion and becomes an observation. If it is still climbing steeply
          at 100, the claim is wrong as written and the shipped
          `n_normal_tiles: 30` is too low -- report that instead of burying it.
STATUS:   see results/normal_set_sweep.csv

The unit is TILES, not images. `detect.n_normal_tiles` is what the product
actually parameterises, and one 4096x256 AITEX strip yields 16 tiles, so a
sweep in images would be a sweep in 16-tile steps and would say nothing about
the setting an operator can change.

Two things keep the curve honest. The fit sets are NESTED -- the 5-tile set is a
subset of the 10-tile set, drawn from one seeded permutation -- so a bump in the
curve is the extra tiles, not a different draw. And the evaluation set is FIXED
across every n, held out beyond the largest fit set, so the x-axis is the only
thing moving.

Run:  python probes/p16_normal_set_sweep.py --root data/aitex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _study_harness import RESULTS, load_p12
from linesight.config import DetectConfig
from linesight.datasets.aitex import list_images
from linesight.detect.patchcore import PatchCore

SIZES = [5, 10, 20, 30, 50, 100]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/aitex")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--min-defect-tiles", type=int, default=5)
    parser.add_argument("--n-clean-test", type=int, default=120)
    parser.add_argument("--out", default=str(RESULTS / "normal_set_sweep.csv"))
    args = parser.parse_args()

    p12 = load_p12()
    root = Path(args.root)
    codes = sorted({img.fabric_code for img in list_images(root)})
    max_n = max(SIZES)

    # build_fabric returns fit + clean_test taken in order from one seeded
    # permutation, so their concatenation reconstructs that permutation and the
    # nesting property survives. Asking for a large clean_test just means the
    # whole clean pool comes back.
    fabrics = {}
    for code in codes:
        data = p12.build_fabric(root, code, args.tile_size, 10_000, seed=0)
        pool = data["fit"] + data["clean_test"]
        if len(pool) < max_n + args.n_clean_test or len(data["defect"]) < args.min_defect_tiles:
            print(f"  fabric {code}: skipped ({len(pool)} clean, "
                  f"{len(data['defect'])} defect tiles)", flush=True)
            continue
        fabrics[code] = {
            "pool": pool,
            "clean_test": pool[max_n : max_n + args.n_clean_test],
            "defect": data["defect"],
            "defect_masks": data["defect_masks"],
        }
        print(f"  fabric {code}: {len(pool)} clean tiles, "
              f"{len(data['defect'])} defect tiles", flush=True)

    if not fabrics:
        print("no usable fabrics")
        return 1

    rows = []
    for n in SIZES:
        print(f"\n--- n_normal_tiles = {n} ---", flush=True)
        config = DetectConfig(
            input_size=args.input_size,
            coreset_frac=0.10,
            coreset_method="greedy",
            n_normal_tiles=n,
            batch_size=args.batch_size,
        )
        for code, data in fabrics.items():
            scorer = PatchCore(config)
            scorer.fit(data["pool"][:n])
            clean_maps = p12.score_all(scorer, data["clean_test"], args.batch_size)
            defect_maps = p12.score_all(scorer, data["defect"], args.batch_size)
            tile_auroc, pixel_auroc = p12.auroc_for(
                clean_maps, defect_maps, data["defect_masks"], args.pixel_stride
            )
            rows.append({
                "n_normal_tiles": n,
                "fabric": code,
                "bank_size": scorer.bank_size,
                "tile_auroc": round(tile_auroc, 4),
                "pixel_auroc": round(pixel_auroc, 4),
                "n_clean_test": len(data["clean_test"]),
                "n_defect": len(data["defect"]),
            })
            print(f"  {code}: tile {tile_auroc:.4f}  pixel {pixel_auroc:.4f}  "
                  f"bank {scorer.bank_size}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out}\n")
    print(f"{'n tiles':>8} {'bank':>6} {'mean tile':>10} {'sd':>7} "
          f"{'mean pixel':>11} {'delta vs prev':>14}")
    previous = None
    for n in SIZES:
        sub = [r for r in rows if r["n_normal_tiles"] == n]
        tile = np.array([r["tile_auroc"] for r in sub], dtype=float)
        pixel = np.array([r["pixel_auroc"] for r in sub], dtype=float)
        mean = float(np.nanmean(tile))
        delta = "" if previous is None else f"{mean - previous:+.4f}"
        print(f"{n:>8} {sub[0]['bank_size']:>6} {mean:>10.4f} "
              f"{float(np.nanstd(tile)):>7.4f} {float(np.nanmean(pixel)):>11.4f} "
              f"{delta:>14}")
        previous = mean

    # The claim under test: does 30 get most of what 100 gets?
    at30 = float(np.nanmean([r["tile_auroc"] for r in rows if r["n_normal_tiles"] == 30]))
    at100 = float(np.nanmean([r["tile_auroc"] for r in rows if r["n_normal_tiles"] == 100]))
    at5 = float(np.nanmean([r["tile_auroc"] for r in rows if r["n_normal_tiles"] == 5]))
    span = at100 - at5
    print(f"\n5 -> 100 tiles moves tile AUROC by {span:+.4f}")
    if span > 1e-9:
        print(f"30 tiles captures {(at30 - at5) / span * 100:.0f}% of that gain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
