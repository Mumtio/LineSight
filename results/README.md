# results/

Evidence, kept deliberately. Generated files (`*.pdf`, `*.png`, `evidence/`) are
gitignored; what is committed here is either a measurement or the raw log that
produced one.

## Reproduction — is the detector correct?

`reproduction.csv`, from `probes/p11_mvtec_reproduction.py` on MVTec AD `carpet`
(280 train / 117 test / 89 masks — the canonical split).

All three MVTec texture classes, shipped configuration (30 images, 10% greedy):

| Category | Image AUROC | Pixel AUROC | Published |
|---|---|---|---|
| carpet | **0.9615** | 0.9857 | ~0.98 |
| leather | **0.9840** | 0.9890 | ~1.00 |
| grid | 0.5522 | 0.8566 | ~0.98 |
| grid, backbone input 512 | 0.8062 | — | ~0.98 |

Carpet, three configurations:

| Configuration | n fit | Bank | Image AUROC | Pixel AUROC |
|---|---|---|---|---|
| product — 10% greedy | 30 | 1200 | **0.9615** | 0.9857 |
| product-random — 10% random | 30 | 1200 | 0.9254 | 0.9834 |
| paper-like — 1% random | 280 | 1120 | 0.9033 | 0.9832 |

**grid fails, and it is a resolution limit rather than a bug.** At the shipped
`input_size: 320` the backbone sees a 20×20 patch grid, which cannot resolve
bent and broken wires in a sparse regular lattice. Raising the backbone input to
512 recovers 25 AUROC points; a higher-resolution source image does not, and
neither does a finer feature grid. The system is tuned for textured cloth and
degrades on sparse regular structure — `input_size` is a per-SKU key, so the
remedy exists where the latency budget allows. Details in `docs/evaluation.md`.

Published PatchCore on carpet is ~0.98 image AUROC using WideResNet-50; we ship
resnet18 for CPU speed, and images are used whole at 512 px rather than the
standard resize-256/crop-224, so this is a regime check rather than a
leaderboard entry.

**This is the file that settles the argument.** The same code that separates
defect from clean by only 1.16× on Fabric Stain reaches 0.96 image AUROC and
0.986 pixel AUROC on carpet. The detector is correct. The fabric numbers are a
property of the data — unrelated photographs standing in for a roll — not of the
250 lines in `detect/patchcore.py`.

It also prices ADR-010: at an identical 1200-point bank, greedy k-center beats
random selection by **3.6 AUROC points** for 2.5 s of extra fit time.

## Cross-construction generalisation — AITEX

`aitex_generalisation.csv`, from `probes/p12_aitex_generalisation.py`. Six
fabric codes, 256 px tiles, 30 fit tiles each, clean-test disjoint from fit.

| | Tile AUROC | Pixel AUROC |
|---|---|---|
| Cold start — each fabric's own bank | **0.859** | **0.971** |
| Transfer — a bank from a different fabric | 0.629 | 0.739 |
| **In-distribution advantage** | **+0.230** | **+0.232** |

This is the measured argument for the per-SKU design. A bank carried over from
another construction keeps some power but loses about a quarter of it; ninety
seconds of refitting buys that back. Had the gap been small, the per-SKU story
would have been unnecessary.

Note the shape: pixel AUROC sits far above tile AUROC here (0.971 vs 0.859),
exactly as on TFD (0.914 block vs 0.761 image). Localisation is consistently the
strong half across three independent fabric datasets — which is what the ASTM
pipeline actually needs, since the score map has to land on the defect for the
length to be measured.

## Benchmarks

| File | What it is |
|---|---|
| `tfd_summary.json` | Ten Fabrics Dataset benchmark, run on Kaggle. 2,969 patches, 737 defective, ten independent cold-start runs. Block AUROC **0.914** mean over fabrics (0.855 pooled); image AUROC **0.761** mean (0.686 pooled). |
| `tfd_per_fabric.csv` | The same study per fabric, including the ones that went badly. Fabric 001 scores 0.495 image AUROC — chance — against 0.877 block AUROC. |

**Quote these honestly.** Block AUROC is a *localisation* rate on the dataset's
own 16×16 annotation grid: it says the score map lights up in the right place.
Image AUROC is the *detection* rate, and at 0.761 it is mediocre. The two are
not interchangeable, and 0.914 is the first number people reach for. Both are
patch-level; no roll-level tracking or ASTM scoring is involved.

Notebook: `kaggle.com/code/ragibshahrier/linesight-tfd-benchmark`.

## Run logs

| File | What it is |
|---|---|
| `_wf7.log`, `_wf7.clean`, `_wf7.err` | The live workflow end to end against the phone simulator: learn on 30 clean frames (11.9 s), calibrate on 160 held-out clean frames → threshold 14.66 from 960 tiles (81.3 s), then inspect. 53 detections with metre positions, 22 points, 297 points/100 yd², REJECT. Median 478.3 ms per processed frame at stride 2. |
| `_demo.log` | The same state machine driven by hand across three ports — one process serving frames, one inspecting, one operator page. |
| `_fp.log` | **The false-alarm study, and the least flattering file in the repo.** Read it. |

### What `_fp.log` says

On Fabric Stain, against ground truth rather than operator judgement, the
measured rate is **1,315–1,391 false positives per 100 m**, with only
1.16–1.19× separation between nuisance and defect signal, across three fit-set
sizes and both coreset methods.

That is not the same quantity as the "false alarms 0 (0.00/100 m)" line in
`_wf7.log`, which counts *operator rejections* during a live run and is zero
because nobody pressed reject. Both numbers are real; they measure different
things, and the ground-truth one is the one an operator would feel.

The cause is known: every dataset available locally is a collection of
unrelated photographs rather than one roll, and calibrating on a different
physical sample than the one being inspected puts the nuisance variation
(1.74×) above the defect signal (1.30×). The fix is not a model change — it is
ten minutes of video of one continuous piece of real fabric, which
`Pipeline.teach_calibrate_inspect` already knows how to learn from.

## Verification images

`marked/` holds the output of `probes/p09_mark_defects.py` on real AITEX images:
input with the ground-truth outline, raw heatmap, threshold mask, and the marked
panel with mm lengths and ASTM points. Fabric 02 (`0011_006_02`) is a clean hit;
fabric 04 is mixed — `0043_019_04` hits, `0044_019_04` misses. Both are kept.

`_boxes.jpg`, `_demo_frame.jpg`, `_inspect_frame.jpg`, `_live_frame.jpg` are
single frames captured from live runs while debugging the operator view.
