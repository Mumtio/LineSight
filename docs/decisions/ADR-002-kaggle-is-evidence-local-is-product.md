# ADR-002 — Kaggle is evidence; local is the product

**Status:** Accepted · **Date:** 2026-08-31

## Context

The obvious pattern for an ML project is to train on a GPU and deploy whatever
comes out. That pattern assumes the deployed artifact is a trained checkpoint.
This one is not: PatchCore has no gradient training at all. What ships is a
memory bank — a coreset of patch embeddings from ~30 defect-free
tiles — that takes about 90 seconds to fit on a laptop CPU and occupies about
600 KB.

A GPU is still needed, but for a different job: running the benchmark study that
turns the design's assertions into measurements. Reproducing MVTec
`carpet`, sweeping the coreset fraction, sweeping the normal-set size, running
leave-one-fabric-out on AITEX, and training the supervised baseline are all
GPU-shaped and none of them are on the deployment path.

## Decision

Split the two cleanly. **Kaggle runs the evidence work** — the GPU-shaped
studies, each writing its own `results/*.csv`. **Local runs the product** — fit,
calibrate, run, serve, report. Nothing produced on Kaggle is required by the
local pipeline; nothing produced locally is required by the evidence work. The
only shared thing is the package itself.

## Consequences

The product has no cloud dependency and no network path in it. `linesight fit`
completes in about ninety seconds on a laptop CPU, which is what makes
cold-start deployment practical rather than theoretical: the artifact is small
enough to email and fast enough to build while an operator waits.

The README results table is generated from the CSVs in `results/`, so every
number in it traces back to the named probe or study that produced it. Each
study writes its CSV as soon as it finishes rather than at the end of a session,
so a timeout costs one study instead of everything.

The cost is that the two environments can drift — a bug fixed locally does not
automatically appear in a study that has already run. We accept this because
each study runs once, early, and its numbers are about the *method*, not about
the current state of the product code.

## Amendment — 2026-09-04

The evidence pack was originally specified as a single notebook,
`notebooks/01_benchmark_kaggle.ipynb`, covering eight sections. That file was
removed while still a scaffold: all twelve of its code cells raised
`NotImplementedError` and it had never been executed. Deleting it rather than
shipping it keeps the repository honest about what has actually run.

The decision above is unchanged — the Kaggle/local split still holds, and the
evidence that exists was produced under it:

| Study | Produced by | Artifact |
|---|---|---|
| MVTec reproduction | `probes/p11_mvtec_reproduction.py` | `results/reproduction.csv` |
| AITEX cross-construction | `probes/p12_aitex_generalisation.py` | `results/aitex_generalisation.csv` |
| Ten Fabrics benchmark | published Kaggle notebook (`results/README.md`) | `results/tfd_summary.json`, `tfd_per_fabric.csv` |

The studies the notebook would have carried and nothing else does — the backbone
ablation, the normal-set size sweep, the supervised U-Net comparison and the
calibration curve — are **unmeasured**, and `docs/evaluation.md` marks them as
study design rather than as results.
