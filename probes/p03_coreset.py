"""
probes/p03_coreset.py

QUESTION: Does greedy k-center on 12,000 embeddings finish fast enough to sit
          inside a 90 s fit budget, and how much AUROC does uniform random
          subsampling actually cost?
PASS IF:  Greedy selection of 1,200 from 12,000 completes in under 30 s on
          CPU, and the AUROC gap against random is measured rather than
          assumed.
ANSWERED: partly - greedy selection is quadratic in the fit set, so random is
          the practical choice above ~50 tiles (probes/README.md, ADR-010).
LIVES IN: linesight/detect/patchcore.py, as greedy_coreset / random_coreset;
          the accuracy cost is priced by probes/p11_mvtec_reproduction.py.

Run:  python probes/p03_coreset.py
"""

from __future__ import annotations

import argparse


def main() -> int:
    """Answer the QUESTION above, printing enough to check PASS IF by eye."""
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
