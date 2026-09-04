"""
probes/p18_unet_baseline.py

QUESTION: Section 6 of docs/evaluation.md and ADR-004. Supervised segmentation
          is the obvious approach, so measure it rather than dismiss it. A small
          U-Net trained on AITEX's pixel-annotated defects, against the shipped
          cold-start PatchCore, on identical evaluation tiles.
PASS IF:  The hypothesis is recorded here BEFORE running so it cannot be
          retrofitted: the U-Net wins in-distribution and collapses on a fabric
          construction it has never seen, while PatchCore sits behind it
          in-distribution and holds up across constructions. Whatever actually
          happens is what gets reported -- ADR-004 says so explicitly, and a
          baseline that beat us everywhere would be the most useful result this
          repository could produce.
STATUS:   see results/baseline_comparison.csv and results/unet_history.csv

Three numbers per fabric, all scored on the SAME tiles so they are comparable:

  unet_in_dist   trained on the train half of every fabric's defective images,
                 including this fabric's. Evaluated on the held-out test half.
  unet_lofo      trained on every OTHER fabric's defective images. This fabric's
                 construction is unseen. Same evaluation tiles.
  patchcore      fitted on this fabric's own defect-free tiles, no defect labels
                 at all. Same evaluation tiles.

The comparison is only fair because `UNetScorer` implements the same
`AnomalyScorer` protocol as `PatchCore`, so both go through the identical
scoring and AUROC path from `p12`. That is the seam earning its keep.

Run:  python probes/p18_unet_baseline.py --root data/aitex
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from _study_harness import RESULTS, load_p12
from linesight.baselines.unet import UNetScorer, train_unet
from linesight.config import DetectConfig
from linesight.datasets.aitex import (
    cut_tiles,
    list_images,
    load_image,
    load_mask,
)
from linesight.detect.patchcore import PatchCore


def defect_images_by_fabric(root: Path) -> dict[str, list]:
    """Defective AITEX images that actually carry an annotation, per fabric."""
    grouped: dict[str, list] = {}
    for img in list_images(root):
        if img.is_defective and img.mask_paths:
            grouped.setdefault(img.fabric_code, []).append(img)
    return {code: sorted(images, key=lambda i: i.path.name)
            for code, images in grouped.items()}


def tiles_from(images: list, tile_size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Defect-bearing tiles and their masks, for a specific list of images.

    A tile counts as defective only where the annotation actually lands in it;
    most tiles of a defective strip are clean fabric, and labelling them all
    positive would understate every score. Same rule as p12.
    """
    tiles, masks = [], []
    for img in images:
        strip = load_image(img.path)
        mask = load_mask(img.mask_paths)
        for tile, mask_tile, _ in cut_tiles(strip, tile_size, mask=mask):
            if mask_tile is not None and mask_tile.any():
                tiles.append(tile)
                masks.append(mask_tile)
    return tiles, masks


def raw_arrays(images: list) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Full strips and masks, the form ``train_unet`` samples crops from."""
    out_images, out_masks = [], []
    for img in images:
        out_images.append(load_image(img.path))
        out_masks.append(load_mask(img.mask_paths))
    return out_images, out_masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/aitex")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-fit", type=int, default=30)
    parser.add_argument("--n-clean-test", type=int, default=120)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--min-defect-tiles", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(RESULTS / "baseline_comparison.csv"))
    parser.add_argument("--history-out", default=str(RESULTS / "unet_history.csv"))
    args = parser.parse_args()

    p12 = load_p12()
    root = Path(args.root)
    by_fabric = defect_images_by_fabric(root)

    # Split each fabric's defective images in half: train half feeds the
    # in-distribution model, test half is never trained on by anything.
    train_images: dict[str, list] = {}
    eval_sets: dict[str, dict] = {}
    for code, images in sorted(by_fabric.items()):
        if len(images) < 4:
            print(f"  fabric {code}: skipped ({len(images)} annotated images)", flush=True)
            continue
        half = len(images) // 2
        train_images[code] = images[:half]
        held_out = images[half:]

        tiles, masks = tiles_from(held_out, args.tile_size)
        if len(tiles) < args.min_defect_tiles:
            print(f"  fabric {code}: skipped ({len(tiles)} held-out defect tiles)",
                  flush=True)
            train_images.pop(code, None)
            continue

        clean = p12.build_fabric(root, code, args.tile_size, 10_000, seed=0)
        pool = clean["fit"] + clean["clean_test"]
        eval_sets[code] = {
            "defect": tiles,
            "defect_masks": masks,
            "clean_test": pool[args.n_fit : args.n_fit + args.n_clean_test],
            "fit": pool[: args.n_fit],
            "n_train_images": len(train_images[code]),
            "n_test_images": len(held_out),
        }
        print(f"  fabric {code}: {len(train_images[code])} train / {len(held_out)} test "
              f"images, {len(tiles)} held-out defect tiles", flush=True)

    codes = sorted(eval_sets)
    if len(codes) < 2:
        print("need at least two fabrics for a leave-one-out study")
        return 1

    histories = []

    print(f"\n=== in-distribution U-Net: train on all {len(codes)} fabrics ===", flush=True)
    imgs, msks = raw_arrays([i for c in codes for i in train_images[c]])
    t0 = time.perf_counter()
    model, history = train_unet(imgs, msks, epochs=args.epochs,
                                batch_size=args.batch_size, device=args.device)
    print(f"  trained in {time.perf_counter() - t0:.0f}s on {len(imgs)} images", flush=True)
    for record in history:
        histories.append({"condition": "in_distribution", "held_out_fabric": "", **record})
    in_dist = UNetScorer(model=model, device=args.device)

    rows = []
    config = DetectConfig(input_size=args.input_size, coreset_frac=0.10,
                          coreset_method="greedy", batch_size=12)

    for code in codes:
        data = eval_sets[code]
        print(f"\n=== fabric {code} ===", flush=True)

        others = [i for c in codes if c != code for i in train_images[c]]
        others += [i for c in by_fabric if c != code and c not in train_images
                   for i in by_fabric[c]]
        o_imgs, o_msks = raw_arrays(others)
        print(f"  LOFO U-Net: training on {len(o_imgs)} images from other fabrics",
              flush=True)
        lofo_model, lofo_history = train_unet(
            o_imgs, o_msks, epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, log_every=0,
        )
        for record in lofo_history:
            histories.append({"condition": "lofo", "held_out_fabric": code, **record})
        lofo = UNetScorer(model=lofo_model, device=args.device)

        patchcore = PatchCore(config)
        patchcore.fit(data["fit"])

        for name, scorer in (("unet_in_dist", in_dist), ("unet_lofo", lofo),
                             ("patchcore", patchcore)):
            clean_maps = p12.score_all(scorer, data["clean_test"], 12)
            defect_maps = p12.score_all(scorer, data["defect"], 12)
            tile_auroc, pixel_auroc = p12.auroc_for(
                clean_maps, defect_maps, data["defect_masks"], args.pixel_stride
            )
            rows.append({
                "fabric": code,
                "method": name,
                "supervised": name.startswith("unet"),
                "saw_this_construction": name != "unet_lofo",
                "tile_auroc": round(tile_auroc, 4),
                "pixel_auroc": round(pixel_auroc, 4),
                "n_clean_test": len(data["clean_test"]),
                "n_defect": len(data["defect"]),
            })
            print(f"  {name:>14}: tile {tile_auroc:.4f}  pixel {pixel_auroc:.4f}",
                  flush=True)

    for path, records, fields in (
        (Path(args.out), rows, list(rows[0].keys())),
        (Path(args.history_out), histories, ["condition", "held_out_fabric", "epoch",
                                             "loss", "steps"]),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {path}")

    print(f"\n{'method':>14} {'mean tile':>10} {'mean pixel':>11} {'sd tile':>9}")
    means = {}
    for name in ("unet_in_dist", "unet_lofo", "patchcore"):
        sub = [r for r in rows if r["method"] == name]
        tile = np.array([r["tile_auroc"] for r in sub], dtype=float)
        pixel = np.array([r["pixel_auroc"] for r in sub], dtype=float)
        means[name] = float(np.nanmean(tile))
        print(f"{name:>14} {means[name]:>10.4f} {float(np.nanmean(pixel)):>11.4f} "
              f"{float(np.nanstd(tile)):>9.4f}")

    print(f"\nsupervised in-distribution advantage over PatchCore: "
          f"{means['unet_in_dist'] - means['patchcore']:+.4f}")
    print(f"supervised cost of an unseen construction:            "
          f"{means['unet_lofo'] - means['unet_in_dist']:+.4f}")
    print(f"PatchCore vs supervised on an unseen construction:    "
          f"{means['patchcore'] - means['unet_lofo']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
