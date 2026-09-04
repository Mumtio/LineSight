"""
probes/p02_memory_bank.py

QUESTION: Fit a memory bank on 20 defect-free crops from one AITEX fabric code,
          then score a DEFECTIVE crop from the SAME code. Does the defect show
          up, and does it land where the ground-truth mask says it is?
PASS IF:  The heatmap is visibly bright over the defect, and the bright region
          overlaps the _mask.png. Not "is the AUROC good" - just: does the
          defect glow, in the right place, to a human looking at two images
          side by side?
ANSWERED: yes - see the status table in probes/README.md.
LIVES IN: linesight/detect/patchcore.py, and is re-exercised on real images by
          probes/p09_mark_defects.py.

Run:  python probes/p02_memory_bank.py --fabric 02

This is the check the whole detector rests on: if the defect does not glow,
nothing downstream can help. The cause is almost always illumination or layer
choice, so check flat-field correction and the hooked layers first.

Saves a three-panel figure to results/p02_<fabric>.png:
    [ input crop | anomaly heatmap | ground-truth mask ]
"""

from __future__ import annotations

import argparse


def main() -> int:
    """Fit, score, plot, and print the overlap number.

    Prints alongside the figure: fit wall-clock, bank shape, score range on the
    defective crop against the score range on a held-out clean crop, and the
    fraction of the bright region that falls inside the mask.

    The clean-crop comparison is the part that catches the failure this probe
    exists to catch: a heatmap can look convincing on a defective crop and score
    just as high on a clean one, which means the model has learned nothing.
    """
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
