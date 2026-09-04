"""
probes/p05_flatfield.py

QUESTION: Does flat-field correction flatten a synthetic illumination gradient
          without eating a 10 px defect?
PASS IF:  The corrected image's corner-to-centre intensity ratio moves to
          within a few percent of 1.0, AND a known defect's contrast against
          its neighbourhood survives.
ANSWERED: yes, and measured as a no-op on evenly-lit scanned imagery; it is
          the uneven-illumination case it exists for (probes/README.md).
LIVES IN: linesight/preprocess/flatfield.py.

Run:  python probes/p05_flatfield.py
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
