"""
probes/p19_coreset_ablation.py

QUESTION: Section 3 of docs/evaluation.md, and the last unmeasured claim in
          ADR-010. The paper keeps 1% of the patch embeddings; we ship 10%,
          on the argument that a 30-tile fit set is far smaller than the
          hundreds the paper assumes. Does the sweep support that, and what
          does the documented random-selection fallback actually cost?
PASS IF:  Stated in advance. (a) If 1% matches 10% on our fit-set size, ADR-010
          is unjustified and the bank should shrink 10x -- a real saving, since
          bank size drives both the artifact and the nearest-neighbour cost.
          (b) If accuracy is still climbing at 25%, the shipped 10% is too low
          and ADR-010 is wrong in the other direction. (c) greedy should beat
          random at matched bank size, or the k-center cost buys nothing.
STATUS:   see results/coreset_ablation.csv

`p11` already priced greedy against random on MVTec carpet (+3.6 AUROC points at
an identical 1,200-point bank). This runs the full fraction sweep on AITEX, so
it is the same six fabrics and the same harness as p15-p18 and the numbers sit
on one scale with them.

Latency is measured on CPU. Bank size is what drives nearest-neighbour cost, so
the fraction is a latency decision as much as an accuracy one, and a GPU number
would hide exactly the trade this table exists to show.

Run:  python probes/p19_coreset_ablation.py --root data/aitex
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from _study_harness import RESULTS, load_p12
from linesight.config import DetectConfig
from linesight.datasets.aitex import list_images
from linesight.detect.patchcore import PatchCore

# (fraction, method). 100% keeps everything, so no selection runs at all.
CONFIGS = [
    (0.01, "greedy"), (0.01, "random"),
    (0.05, "greedy"), (0.05, "random"),
    (0.10, "greedy"), (0.10, "random"),
    (0.25, "greedy"), (0.25, "random"),
    (1.00, "random"),
]


def cpu_ms_per_tile(fit_tiles, score_tiles, frac, method, input_size, n=16):
    """Per-tile scoring cost on CPU at this bank size, after a warm-up."""
    config = DetectConfig(input_size=input_size, coreset_frac=frac,
                          coreset_method=method, device="cpu", batch_size=8)
    scorer = PatchCore(config)
    scorer.fit(fit_tiles)
    scorer.score_batch(score_tiles[:4])
    started = time.perf_counter()
    scorer.score_batch(score_tiles[:n])
    return (time.perf_counter() - started) / n * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/aitex")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--max-clean-test", type=int, default=120)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--min-defect-tiles", type=int, default=5)
    parser.add_argument("--skip-cpu-latency", action="store_true")
    parser.add_argument("--out", default=str(RESULTS / "coreset_ablation.csv"))
    args = parser.parse_args()

    p12 = load_p12()
    root = Path(args.root)
    codes = sorted({img.fabric_code for img in list_images(root)})

    fabrics = {}
    for code in codes:
        data = p12.build_fabric(root, code, args.tile_size, args.max_clean_test, seed=0)
        if len(data["fit"]) < 30 or len(data["defect"]) < args.min_defect_tiles:
            continue
        fabrics[code] = data
    print(f"usable fabrics: {list(fabrics)}\n", flush=True)

    rows = []
    for frac, method in CONFIGS:
        label = "none" if frac >= 1.0 else method
        print(f"--- coreset {frac:.0%}, selection {label} ---", flush=True)
        config = DetectConfig(
            input_size=args.input_size,
            coreset_frac=frac,
            coreset_method=method,
            batch_size=args.batch_size,
        )
        for code, data in fabrics.items():
            scorer = PatchCore(config)
            t0 = time.perf_counter()
            scorer.fit(data["fit"])
            fit_s = time.perf_counter() - t0

            clean_maps = p12.score_all(scorer, data["clean_test"], args.batch_size)
            defect_maps = p12.score_all(scorer, data["defect"], args.batch_size)
            tile_auroc, pixel_auroc = p12.auroc_for(
                clean_maps, defect_maps, data["defect_masks"], args.pixel_stride
            )
            rows.append({
                "coreset_frac": frac,
                "coreset_method": label,
                "fabric": code,
                "bank_size": scorer.bank_size,
                "tile_auroc": round(tile_auroc, 4),
                "pixel_auroc": round(pixel_auroc, 4),
                "fit_s": round(fit_s, 2),
                "n_clean_test": len(data["clean_test"]),
                "n_defect": len(data["defect"]),
            })
            print(f"  {code}: tile {tile_auroc:.4f}  pixel {pixel_auroc:.4f}  "
                  f"bank {scorer.bank_size:>5}  fit {fit_s:>5.1f}s", flush=True)

    latency = {}
    if not args.skip_cpu_latency:
        print("\nCPU scoring cost by bank size", flush=True)
        sample = next(iter(fabrics.values()))
        for frac, method in CONFIGS:
            label = "none" if frac >= 1.0 else method
            latency[(frac, label)] = cpu_ms_per_tile(
                sample["fit"], sample["clean_test"], frac, method, args.input_size
            )
            print(f"  {frac:>5.0%} {label:>7}: {latency[(frac, label)]:.0f} ms/tile",
                  flush=True)
    for row in rows:
        row["cpu_ms_per_tile"] = round(
            latency.get((row["coreset_frac"], row["coreset_method"]), float("nan")), 1
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}\n")

    print(f"{'frac':>6} {'method':>7} {'bank':>6} {'tile AUROC':>11} {'pixel':>8} "
          f"{'fit s':>7} {'CPU ms':>7}")
    summary = {}
    for frac, method in CONFIGS:
        label = "none" if frac >= 1.0 else method
        sub = [r for r in rows if r["coreset_frac"] == frac
               and r["coreset_method"] == label]
        if not sub:
            continue
        tile = float(np.nanmean([r["tile_auroc"] for r in sub]))
        summary[(frac, label)] = tile
        print(f"{frac:>6.0%} {label:>7} {sub[0]['bank_size']:>6} {tile:>11.4f} "
              f"{float(np.nanmean([r['pixel_auroc'] for r in sub])):>8.4f} "
              f"{float(np.nanmean([r['fit_s'] for r in sub])):>7.1f} "
              f"{sub[0]['cpu_ms_per_tile']:>7.0f}")

    print("\nthe three questions this was run to answer:")
    if (0.01, "greedy") in summary and (0.10, "greedy") in summary:
        gap = summary[(0.10, "greedy")] - summary[(0.01, "greedy")]
        print(f"  10% over 1% (greedy):        {gap:+.4f} tile AUROC")
    if (0.25, "greedy") in summary and (0.10, "greedy") in summary:
        gap = summary[(0.25, "greedy")] - summary[(0.10, "greedy")]
        print(f"  25% over 10% (greedy):       {gap:+.4f} tile AUROC")
    for frac in (0.01, 0.05, 0.10, 0.25):
        if (frac, "greedy") in summary and (frac, "random") in summary:
            gap = summary[(frac, "greedy")] - summary[(frac, "random")]
            print(f"  greedy over random at {frac:>4.0%}:  {gap:+.4f} tile AUROC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
