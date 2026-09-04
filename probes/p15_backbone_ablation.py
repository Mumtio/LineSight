"""
probes/p15_backbone_ablation.py

QUESTION: Section 2 of docs/evaluation.md. The specification picks `resnet18`
          for CPU speed (ADR-009 sizes the latency budget around it). What does
          that choice cost in accuracy against `wide_resnet50_2`, whose
          layer2+layer3 embedding is 1536-d rather than 384-d?
PASS IF:  There is no pass condition -- this is a measurement, not a check. The
          decision rule stated in advance: if the deeper backbone buys more than
          ~0.05 tile AUROC, the specification is wrong and resnet18 should be
          reconsidered for anything but the tightest latency budget. If it buys
          less, the cheap backbone is justified by our own numbers.
STATUS:   see results/backbone_ablation.csv

Accuracy is measured on whatever device is available, because AUROC does not
depend on it. Per-tile latency is measured **on CPU regardless**, because that
is the deployment target the choice was made for and a GPU number would flatter
the wide backbone in a way no mill would ever see.

Run:  python probes/p15_backbone_ablation.py --root data/aitex
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

BACKBONES = ["resnet18", "wide_resnet50_2"]


def cpu_latency(backbone: str, tiles: list[np.ndarray], input_size: int, n: int = 24) -> float:
    """Seconds per tile for a forward pass on CPU, after a warm-up batch."""
    config = DetectConfig(backbone=backbone, input_size=input_size, device="cpu", batch_size=12)
    scorer = PatchCore(config)
    scorer.fit(tiles[:30])
    sample = tiles[:n]
    scorer.score_batch(sample[:4])  # warm-up: lazy weight load and allocator
    started = time.perf_counter()
    scorer.score_batch(sample)
    return (time.perf_counter() - started) / len(sample)


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
    parser.add_argument("--out", default=str(RESULTS / "backbone_ablation.csv"))
    args = parser.parse_args()

    p12 = load_p12()
    root = Path(args.root)
    codes = sorted({img.fabric_code for img in list_images(root)})
    print(f"AITEX at {root}: fabric codes {codes}\n", flush=True)

    fabrics = {}
    for code in codes:
        data = p12.build_fabric(root, code, args.tile_size, args.max_clean_test, seed=0)
        if len(data["fit"]) < 30 or len(data["defect"]) < args.min_defect_tiles:
            continue
        fabrics[code] = data
    print(f"usable fabrics: {list(fabrics)}\n", flush=True)

    rows = []
    for backbone in BACKBONES:
        print(f"--- {backbone} ---", flush=True)
        config = DetectConfig(
            backbone=backbone,
            input_size=args.input_size,
            coreset_frac=0.10,
            coreset_method="greedy",
            batch_size=args.batch_size,
        )
        for code, data in fabrics.items():
            scorer = PatchCore(config)
            t0 = time.perf_counter()
            scorer.fit(data["fit"])
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            clean_maps = p12.score_all(scorer, data["clean_test"], args.batch_size)
            defect_maps = p12.score_all(scorer, data["defect"], args.batch_size)
            score_s = time.perf_counter() - t0
            n_scored = len(data["clean_test"]) + len(data["defect"])

            tile_auroc, pixel_auroc = p12.auroc_for(
                clean_maps, defect_maps, data["defect_masks"], args.pixel_stride
            )
            # projection is (embed_dim, projection_dim) -- the honest source for
            # the embedding width, since ScorerMeta does not carry it.
            embed_dim = int(scorer.projection.shape[0])
            rows.append({
                "backbone": backbone,
                "fabric": code,
                "embed_dim": embed_dim,
                "projection_dim": config.projection_dim,
                "bank_size": scorer.bank_size,
                "tile_auroc": round(tile_auroc, 4),
                "pixel_auroc": round(pixel_auroc, 4),
                "fit_s": round(fit_s, 2),
                "score_s_per_tile": round(score_s / max(n_scored, 1), 4),
                "n_clean_test": len(data["clean_test"]),
                "n_defect": len(data["defect"]),
            })
            print(f"  {code}: tile {tile_auroc:.4f}  pixel {pixel_auroc:.4f}  "
                  f"fit {fit_s:.1f}s", flush=True)

    # CPU latency, measured once per backbone on a fixed set of tiles.
    cpu_s = {}
    if not args.skip_cpu_latency:
        print("\nCPU per-tile latency (the number the backbone choice was made on)",
              flush=True)
        sample_tiles = next(iter(fabrics.values()))["fit"] + \
            next(iter(fabrics.values()))["clean_test"]
        for backbone in BACKBONES:
            cpu_s[backbone] = cpu_latency(backbone, sample_tiles, args.input_size)
            print(f"  {backbone}: {cpu_s[backbone]*1000:.0f} ms/tile", flush=True)
    for row in rows:
        row["cpu_s_per_tile"] = round(cpu_s.get(row["backbone"], float("nan")), 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out}\n")
    print(f"{'backbone':>18} {'embed':>6} {'tile AUROC':>11} {'pixel AUROC':>12} "
          f"{'CPU ms/tile':>12}")
    for backbone in BACKBONES:
        sub = [r for r in rows if r["backbone"] == backbone]
        t = float(np.nanmean([r["tile_auroc"] for r in sub]))
        p = float(np.nanmean([r["pixel_auroc"] for r in sub]))
        ms = cpu_s.get(backbone, float("nan")) * 1000
        print(f"{backbone:>18} {sub[0]['embed_dim']:>6} {t:>11.4f} {p:>12.4f} {ms:>12.0f}")

    r18 = float(np.nanmean([r["tile_auroc"] for r in rows if r["backbone"] == "resnet18"]))
    wide = float(np.nanmean([r["tile_auroc"] for r in rows if r["backbone"] != "resnet18"]))
    print(f"\ndeeper backbone buys {wide - r18:+.4f} tile AUROC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
