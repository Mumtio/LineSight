"""
probes/p07_threshold.py

QUESTION: On held-out clean tiles, does a stated budget of 1 false alarm per
          100 m actually produce about 1 false alarm per 100 m?
PASS IF:  The realised exceedance rate on a held-out clean set, not the
          calibration set, lands within half a false alarm of the budget.
ANSWERED: yes on synthetic scores - see probes/README.md.
LIVES IN: linesight/calibrate/threshold.py, locked by tests/test_threshold.py.

Run:  python probes/p07_threshold.py
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
