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
results table.

**Measured 2026-09-04** by `probes/p18_unet_baseline.py` (`unet.py` no longer
stubs its training). The recorded hypothesis was that the U-Net would win
in-distribution and collapse on unseen constructions. It collapsed as predicted
— 0.628 tile AUROC against PatchCore's 0.863 on a construction it had
never seen — but it also **lost in-distribution** (0.773), which the
hypothesis did not predict.

That is reported rather than reframed, and it is deliberately not read as
"supervised learning is worse": ~50 training images and 30 epochs is a thin
budget and a larger corpus would likely close the in-distribution gap. The
decision above does not rest on the numbers in any case. It rests on the
cold-start constraint — a labelled corpus per construction does not exist in a
mill — which is why `UNetScorer.fit` still raises. See `docs/evaluation.md`
section 6.
