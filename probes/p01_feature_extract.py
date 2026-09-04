"""
probes/p01_feature_extract.py

QUESTION: Does the frozen backbone produce the patch-embedding grid the model
          specification claims - (20, 20, 384) from one AITEX crop at 320x320
          input, with layer2 and layer3 concatenated?
PASS IF:  The returned grid has the documented shape and dtype, contains no
          NaNs, embeddings vary across spatial positions (a constant grid means
          normalisation is wrong), and two different crops give different grids.
ANSWERED: yes - see the status table in probes/README.md.
LIVES IN: linesight/detect/backbone.py, and is re-exercised end to end by
          probes/p11_mvtec_reproduction.py.

Run:  python probes/p01_feature_extract.py [--image path/to/crop.png]

Why this is the first probe: every downstream number depends on these
embeddings. If the grid is the wrong size the tiling arithmetic is wrong; if it
is constant the ImageNet normalisation is wrong; and both failures look exactly
like "the model doesn't work" three hours later.
"""

from __future__ import annotations

import argparse


def main() -> int:
    """Load one crop, embed it, print the shape table, and assert the invariants.

    Prints, for each hooked layer: its raw feature-map shape, its channel count,
    and the shape after upsampling to layer2's grid - so a mismatch is visible
    rather than inferred.
    """
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
