# ADR-010 — Keep 10% in the coreset, not the paper's 1%

**Status:** Accepted · **Date:** 2026-08-31

## Context

PatchCore subsamples its memory bank with greedy k-center coreset selection, and
the paper uses 1%. That figure is chosen against MVTec-scale fit sets: hundreds
of normal images, hence hundreds of thousands of patch embeddings, where 1% still
leaves ample coverage of the normal manifold.

Our fit set is 30 tiles, deliberately — objective O4 is that a mill can onboard a
new SKU from a handful of defect-free samples. At 320×320 backbone input the
`layer2` grid is 20×20, so 30 tiles give 30 × 400 = 12,000 embeddings. One
percent of that is 120 points to describe every legitimate appearance of a
fabric, and thin coverage in a nearest-neighbour detector shows up as false
alarms on perfectly normal fabric — the failure mode the calibration layer is
least able to hide.

## Decision

Keep **10%** — about 1,200 points from 12,000. Record the deviation from the
paper here rather than leaving it as an unexplained constant, and **measure**
what can be measured. `probes/p11_mvtec_reproduction.py` runs the shipped 10%
greedy configuration against 10% random at an identical 1,200-point bank and
against a paper-like 1% — so the *selection method* is justified by our own
numbers (`results/reproduction.csv`: greedy is worth 3.6 AUROC points for 2.5 s
of fit time) rather than by argument. The full 1 / 5 / 10 / 25 / 100 % sweep
for AUROC against latency is designed in `docs/evaluation.md` and **has not been
run**; the 10% itself therefore still rests on the reasoning above rather than
on a measured curve.

## Consequences

Nearest-neighbour search stays cheap at this size — 400 queries × 1,200 bank
entries × 128 dims is 61 MFLOP per tile, sub-millisecond — so the extra coverage
costs essentially nothing at inference. Bank size stays around 600 KB, small
enough to email, which is part of the product argument.

Greedy k-center is O(N·k), so 12,000 → 1,200 takes about 2 s on GPU and about
20 s on CPU. That is inside the ~90 s fit budget, but it is the single slowest
step in fitting.

The documented fallback, if coreset selection becomes the bottleneck on a slow
machine, is uniform random subsampling — `random_coreset`, same signature, one
config key. Its AUROC cost is measured in the same Kaggle ablation rather than
assumed, so switching is an informed trade rather than a panic.

If the sweep shows 10% is unnecessary, we lower it and say so. The point of
running the ablation is to be able to change our minds with evidence.
