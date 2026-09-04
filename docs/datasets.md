# Datasets

Three datasets, two environments, and one licensing constraint that decides
where each of them may run.

## What goes where

| Dataset | Where it runs | Size | Role |
|---|---|---|---|
| **MVTec AD** — `carpet`, `grid`, `leather` | **Kaggle only.** Added as a notebook input; mounts at `/kaggle/input/`. Never downloaded locally. | ~5 GB (untouched) | Reproduction anchor against published PatchCore numbers. |
| **AITEX** | Kaggle **and** local | ~170 MB | Leave-one-fabric-out study; the supervised baseline's training data; **the local reference input**. |
| **Fabric Stain** | Local | small | Phone-resolution integration test. |

The split is ADR-003. The short version: MVTec is the benchmark everyone
quotes, AITEX is the benchmark that actually resembles the problem, and a 5 GB
download buys nothing that a mounted read-only copy does not.

## Sources

| Dataset | URL | Notes |
|---|---|---|
| MVTec AD | `kaggle.com/datasets/ipythonx/mvtec-ad` | 15 categories; `carpet`, `grid`, `leather` are the texture classes. Pixel-precise masks on all anomalies. Train split is defect-free only. |
| AITEX | `kaggle.com/datasets/nexuswho/aitex-fabric-image-database` (mirror: `aitex.es/afid`) | 245 images, 7 fabric structures, 4096×256 px. 140 defect-free (20 per fabric), 105 defective across 12 defect types, each with a `_mask.png`. |
| Fabric Stain | `kaggle.com/datasets/priemshpathirana/fabric-stain-dataset` | 466 images at 1920×1080: 398 stained, 68 defect-free. Intellisense Lab, University of Moratuwa. |
| *Optional* FabricSpotDefect | DOI `10.1016/j.dib.2024.111165`, on Roboflow | 1,014 images, 3,288 annotated spots, shot in home lighting. **From Independent University, Bangladesh** — supports the local-data-moat argument. |

## Fetching, locally

```bash
./scripts/fetch_data.sh          # or scripts/fetch_data.ps1 on Windows
```

Requires `pip install kaggle` and a token at `~/.kaggle/kaggle.json`
(Kaggle → Settings → API → Create New Token), `chmod 600`.

The script deliberately does **not** fetch MVTec AD, and says so in a comment —
otherwise its absence looks like an oversight rather than a decision. Total local
footprint stays under 250 MB.

## Fetching, on Kaggle

Attach MVTec AD and AITEX as notebook inputs **as the first action of the
session**, before writing any code. Attaching mounts read-only with no download,
but a stalled attach discovered at minute forty is a lost session.

## The AITEX filename trick

`nnnn_ddd_ff.png` — `ff` is the fabric code, `ddd` is the defect code
(`000` = defect-free). Masks are `nnnn_ddd_ff_mask.png`. So leave-one-fabric-out
is a string parse, not a labelling effort:

```python
def lofo_splits(paths):
    """Group AITEX images by fabric code.
    Yields (held_out_code, fit_normals, eval_defectives)."""
    by_fabric = defaultdict(lambda: {"normal": [], "defect": []})
    for p in paths:
        num, defect, fabric = p.stem.split("_")[:3]
        by_fabric[fabric]["normal" if defect == "000" else "defect"].append(p)
    for code in by_fabric:
        others = [q for c, v in by_fabric.items() if c != code for q in v["defect"]]
        yield code, by_fabric[code]["normal"], others
```

Cut the 4096×256 strips into 256×256 tiles first; masks come along unchanged. A
4096-wide strip pushed through a 320 px backbone would throw away exactly the
resolution the defects live in.

## The padding trap — found by running it, not by reading about it

**AITEX strips are padded with blank white, and the padding differs between the
clean and defective sets.** Measured across the dataset:

| Fabric | White columns, clean strips | White columns, defective strips |
|---|---|---|
| 00 | 1229–1236 of 4096 (**30%**) | 35–1240 |
| 01 | 434–438 | 0–327 |
| 02 | 239–244 | 11–278 |
| 03 | 1064–1068 | 973–1092 |
| 04, 06 | 0–5 | 3–5 |
| 05 | 200–206 | 0 |

Left uncropped this poisons three things at once. Padding tiles enter the fit
set, so the memory bank spends coreset capacity learning that blank white is
normal fabric. The padding boundary is a hard edge unlike any weave, so it
scores as a defect at the same position on every strip that has one — observed
directly: before cropping, every fabric-02 image produced identical false
alarms at x ≈ 215. And because clean and defective strips are padded
differently, any clean-vs-defective comparison is partly a comparison of
padding.

`datasets.aitex.fabric_extent` finds the widest contiguous run of non-padding
columns and `crop_to_fabric` applies it to the image and its mask together,
with a 24 px inset (`EDGE_MARGIN`) past the boundary to drop the scan-edge
band. `cut_tiles` does this by default. This is the dataset's version of the
selvedge that `geometry.fabric_roi` crops away on the rig.

Two other quirks worth knowing, both of which silently corrupt results:

- **`0018_00_01.png` uses a two-digit defect code** (`00`, not `000`). Testing
  against the literal string `"000"` classifies this defect-free image as
  defective — dropping it from the fit set and counting it as a defect in every
  evaluation. The test is "all zeros".
- **Two defects carry multiple mask files** (`0044_019_04` and `0097_030_03`
  have `_mask1` and `_mask2`). Matching only the exact `_mask` suffix lets
  `_mask1` parse as a *fabric image*, putting a binary annotation into the
  evaluation set as if it were fabric. `load_mask` unions them.
- **Fabric 08 has one defective image and no clean ones**, so it cannot be a
  LOFO fold. `lofo_splits` skips it, yielding 7 folds, not 8.

## How much clean fabric calibration actually needs

AITEX gives 20 clean strips per fabric. After cropping that is roughly 150–160
held-out tiles, which supports a *stable* false-alarm budget of only about
**26 FA/100 m** — not the 1 FA/100 m quoted as the design target. The budget a sample can
hold scales as `stability_margin / alpha`; see ADR-007 and
`calibrate.threshold.threshold_from_budget`. This is a property of the dataset,
not of the method, and it is why the rig's own clean fabric matters.

## Licensing — the constraint, and how the design satisfies it

**MVTec AD is CC BY-NC-SA 4.0 — non-commercial.** That permits benchmarking but
not a commercial derivative, so the boundary is enforced by the architecture
rather than by a promise:

> MVTec is used for validation only. Nothing derived from it ships. The deployed
> system fits on the mill's own fabric.

This is not only rhetoric. MVTec physically never leaves Kaggle (ADR-003), and
the shipped artifact is a memory bank fitted on customer fabric — there is no
path by which MVTec pixels reach a deployed system.

**AITEX** requires citing the AFID paper. **Fabric Stain** should credit
Intellisense Lab, University of Moratuwa.

Save each licence text alongside the data in `data/*/LICENCE` when fetching, so
the claim is auditable rather than asserted.

## Directory layout after fetching

```
data/
├── aitex/
│   ├── Defect_images/          # nnnn_ddd_ff.png, 105 defective
│   ├── Mask_images/            # nnnn_ddd_ff_mask.png
│   ├── NODefect_images/        # 140 defect-free, 20 per fabric
│   └── LICENCE
└── fabric_stain/
    ├── defect-free/
    ├── stained/
    └── LICENCE
```

The reference SKU (`configs/sku_aitex_02.yaml`) expects tiles pre-cut into
`data/aitex/normal_02/`, `clean_02/`, and `roll_02/`. `normal_02` is the fit set,
`clean_02` is the **held-out** calibration set, and they must be disjoint —
calibrating on the fit set gives a threshold far too low and a false-alarm rate
far worse than promised.
