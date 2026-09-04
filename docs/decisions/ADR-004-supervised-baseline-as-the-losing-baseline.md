# ADR-004 — Train one supervised U-Net, as the losing baseline

**Status:** Accepted · **Date:** 2026-08-31

## Context

Supervised segmentation is the obvious approach to defect detection. If it
simply wins, the premise of this project is wrong, and that is worth knowing
from a measurement rather than from an argument. So the alternative is measured
rather than dismissed.

## Decision

Train a small U-Net for binary segmentation on AITEX's 105 pixel-annotated
defective images, cut to 256×256 crops. Dice + BCE, Adam, ~30 epochs, 60–90
minutes on a Kaggle GPU. It implements `AnomalyScorer`, so it drops into the
same evaluation harness as PatchCore.

Evaluate it in **two** conditions, and the second is the point: in-distribution
(same fabric structures in train and test) and leave-one-fabric-out (a held-out
structure). Report both, whatever they say.

## Consequences

The expected result — supervised wins in-distribution and collapses on unseen
fabric, while PatchCore sits slightly behind in-distribution and holds — is what
decides deployability: a model that wins only on constructions it has already
seen is no use to a mill that onboards a new SKU every week.

If the U-Net generalises better than expected, that gets reported too. The
cold-start constraint stands independently: a labelled corpus per fabric
construction does not exist in a mill, so a model that requires one cannot be
deployed there regardless of how well it generalises. `UNetScorer.fit` raises
`NotImplementedError` with exactly that explanation, which puts the constraint
in executable code rather than in prose.

The baseline is **not shipped**. Its home is `baselines/` and one row in the
results table. As of 2026-09-04 that row is **not yet measured**: `unet.py`
fixes the architecture, the loss and the training protocol, but the training
itself needs a GPU and AITEX's 105 pixel-annotated defective images, and has not
been run. The argument above stands on the cold-start constraint, which is
independent of the numbers; the comparison remains outstanding.
