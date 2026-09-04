"""
probes/p08_events.py

QUESTION: Do eight consecutive fragments of one warp-direction line become ONE
          event of the correct total length?
PASS IF:  The tracker emits a single event spanning all eight frames, with
          length_mm matching the synthetic ground truth, and ASTM scores it as
          continuous rather than as eight separate defects.
ANSWERED: yes - see the status table in probes/README.md.
LIVES IN: linesight/events/track.py, locked by tests/test_events.py.

Run:  python probes/p08_events.py
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
