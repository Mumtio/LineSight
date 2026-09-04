"""
probes/p12_aitex_generalisation.py

QUESTION: On real fabric, two things. (a) COLD START: fit a bank on one AITEX
          fabric's own defect-free strips and detect that same fabric's defects
          -- the product's actual operating mode, one bank per SKU. (b) TRANSFER:
          does a bank fitted on fabric A retain any power on fabric B? That is
          the cross-construction question, and it is the one the whole per-SKU
          design exists to answer.
PASS IF:  (a) mean tile AUROC over fabrics is well above chance, on the same
          scale as the TFD result (0.761 image / 0.914 block). Anything near 0.5
          would mean the detector cannot see real woven defects at all.
          (b) transfer is EXPECTED TO DEGRADE. A large in-distribution/transfer
          gap is the empirical argument for refitting per SKU; a small gap would
          mean the per-SKU story is unnecessary and should be dropped.
STATUS:   see results/aitex_generalisation.csv

Note on `datasets.aitex.lofo_splits`, which this probe deliberately does NOT use.
That helper pairs a bank fitted on fabric X with *other* fabrics' defective
images. Scored against fabric X's own clean tiles as negatives, the resulting
AUROC measures "is this a different fabric" far more strongly than "is this
defective" -- every tile of fabric Y is unlike fabric X, defect or not. The
number would look excellent and mean nothing. The transfer study below keeps
clean and defective tiles from the *same* fabric on both sides of every
comparison, so the only thing separating them is the defect.

Run:  python probes/p12_aitex_generalisation.py --root <AITEX dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from linesight.config import DetectConfig  # noqa: E402
from linesight.datasets.aitex import cut_tiles, list_images, load_image, load_mask  # noqa: E402
from linesight.detect.patchcore import PatchCore  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def build_fabric(root: Path, code: str, tile_size: int, max_clean_test: int, seed: int):
    """Tiles for one fabric: fit set, held-out clean set, defective set + masks.

    The fit and clean-test sets are disjoint by construction. Calibrating or
    evaluating on the tiles the bank memorised is the single easiest way to
    manufacture a good number, which is why ``calibrate/threshold.py`` requires
    held-out fabric as well.
    """
    images = [img for img in list_images(root) if img.fabric_code == code]
    clean_images = [img for img in images if not img.is_defective]
    defect_images = [img for img in images if img.is_defective]

    clean_tiles: list[np.ndarray] = []
    for img in clean_images:
        strip = load_image(img.path)
        clean_tiles.extend(tile for tile, _, _ in cut_tiles(strip, tile_size))

    defect_tiles: list[np.ndarray] = []
    defect_masks: list[np.ndarray] = []
    for img in defect_images:
        masks = img.mask_paths
        if not masks:
            continue
        strip = load_image(img.path)
        mask = load_mask(masks)
        for tile, mask_tile, _ in cut_tiles(strip, tile_size, mask=mask):
            # A tile only counts as defective if the annotation actually lands
            # in it. Most tiles of a defective strip are clean fabric, and
            # labelling them all positive would understate every score.
            if mask_tile is not None and mask_tile.any():
                defect_tiles.append(tile)
                defect_masks.append(mask_tile)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean_tiles))
    fit_idx, rest_idx = order[:30], order[30:]
    fit = [clean_tiles[i] for i in fit_idx]
    clean_test = [clean_tiles[i] for i in rest_idx[:max_clean_test]]

    return {
        "code": code,
        "fit": fit,
        "clean_test": clean_test,
        "defect": defect_tiles,
        "defect_masks": defect_masks,
        "n_clean_strips": len(clean_images),
        "n_defect_strips": len(defect_images),
    }


def score_all(scorer, tiles, batch_size: int) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        maps.extend(scorer.score_batch(tiles[start : start + batch_size]))
    return maps


def auroc_for(clean_maps, defect_maps, defect_masks, pixel_stride: int):
    """Tile-level and pixel-level AUROC for one (bank, fabric) pairing."""
    scores = [float(m.max()) for m in clean_maps] + [float(m.max()) for m in defect_maps]
    labels = [0] * len(clean_maps) + [1] * len(defect_maps)
    tile_auroc = (
        float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    )

    pixel_scores, pixel_truth = [], []
    for m in clean_maps:
        pixel_scores.append(m[::pixel_stride, ::pixel_stride].ravel())
        pixel_truth.append(np.zeros(pixel_scores[-1].shape, dtype=np.uint8))
    for m, gt in zip(defect_maps, defect_masks, strict=True):
        pixel_scores.append(m[::pixel_stride, ::pixel_stride].ravel())
        pixel_truth.append(gt[::pixel_stride, ::pixel_stride].ravel())

    truth = np.concatenate(pixel_truth)
    values = np.concatenate(pixel_scores)
    pixel_auroc = float(roc_auc_score(truth, values)) if truth.any() else float("nan")
    return tile_auroc, pixel_auroc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--max-clean-test", type=int, default=120)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--min-defect-tiles", type=int, default=5)
    parser.add_argument("--out", default=str(RESULTS / "aitex_generalisation.csv"))
    args = parser.parse_args()

    root = Path(args.root)
    images = list_images(root)
    codes = sorted({img.fabric_code for img in images})

    print(f"AITEX at {root}: {len(images)} images, fabric codes {codes}\n", flush=True)

    fabrics = {}
    for code in codes:
        data = build_fabric(root, code, args.tile_size, args.max_clean_test, seed=0)
        if len(data["fit"]) < 30 or len(data["defect"]) < args.min_defect_tiles:
            print(f"  fabric {code}: skipped "
                  f"({len(data['fit'])} fit tiles, {len(data['defect'])} defect tiles)",
                  flush=True)
            continue
        fabrics[code] = data
        print(f"  fabric {code}: {len(data['fit'])} fit / "
              f"{len(data['clean_test'])} clean-test / {len(data['defect'])} defect tiles",
              flush=True)

    if not fabrics:
        print("no usable fabrics")
        return 1

    config = DetectConfig(input_size=args.input_size, coreset_frac=0.10,
                          coreset_method="greedy", batch_size=args.batch_size)

    banks = {}
    print("\nfitting one bank per fabric", flush=True)
    for code, data in fabrics.items():
        scorer = PatchCore(config)
        t0 = time.time()
        scorer.fit(data["fit"])
        banks[code] = scorer
        print(f"  {code}: bank {scorer.bank_size} in {time.time()-t0:.1f}s", flush=True)

    rows = []
    print("\nscoring every bank against every fabric", flush=True)
    for bank_code, scorer in banks.items():
        for eval_code, data in fabrics.items():
            clean_maps = score_all(scorer, data["clean_test"], args.batch_size)
            defect_maps = score_all(scorer, data["defect"], args.batch_size)
            tile_auroc, pixel_auroc = auroc_for(
                clean_maps, defect_maps, data["defect_masks"], args.pixel_stride
            )
            rows.append({
                "bank_fabric": bank_code,
                "eval_fabric": eval_code,
                "in_distribution": bank_code == eval_code,
                "tile_auroc": round(tile_auroc, 4),
                "pixel_auroc": round(pixel_auroc, 4),
                "n_clean_test": len(data["clean_test"]),
                "n_defect": len(data["defect"]),
            })
        same = next(r for r in rows if r["bank_fabric"] == bank_code
                    and r["eval_fabric"] == bank_code)
        print(f"  bank {bank_code}: in-distribution tile AUROC {same['tile_auroc']:.4f}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    same = [r for r in rows if r["in_distribution"]]
    cross = [r for r in rows if not r["in_distribution"]]

    print("\n(a) COLD START -- one bank per fabric, its own defects")
    print(f"{'fabric':>8} {'tile AUROC':>12} {'pixel AUROC':>12} {'defect tiles':>13}")
    for r in same:
        print(f"{r['bank_fabric']:>8} {r['tile_auroc']:>12.4f} "
              f"{r['pixel_auroc']:>12.4f} {r['n_defect']:>13d}")
    same_tile = float(np.nanmean([r["tile_auroc"] for r in same]))
    same_pixel = float(np.nanmean([r["pixel_auroc"] for r in same]))
    print(f"{'mean':>8} {same_tile:>12.4f} {same_pixel:>12.4f}")

    cross_tile = float(np.nanmean([r["tile_auroc"] for r in cross]))
    cross_pixel = float(np.nanmean([r["pixel_auroc"] for r in cross]))
    print("\n(b) TRANSFER -- bank from a different fabric, same eval set")
    print(f"{'mean':>8} {cross_tile:>12.4f} {cross_pixel:>12.4f}   "
          f"({len(cross)} off-diagonal pairs)")

    print(f"\nin-distribution advantage: {same_tile - cross_tile:+.4f} tile AUROC, "
          f"{same_pixel - cross_pixel:+.4f} pixel AUROC")
    print("  A positive gap is the measured argument for refitting per SKU:")
    print("  it is what a mill buys by spending 90 seconds on a new construction.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
