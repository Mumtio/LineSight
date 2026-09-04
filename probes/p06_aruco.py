"""
probes/p06_aruco.py

QUESTION: Does one cv2.aruco call give an absolute marker ID and a mm/px scale
          within 2 percent of an independent ruler measurement?
PASS IF:  The decoded ID matches the printed marker, and the scale agrees with
          estimate_mm_per_px_from_ruler on the same frame to within 2 percent.
ANSWERED: yes on synthetic markers - see probes/README.md.
LIVES IN: linesight/geometry/aruco.py, locked by tests/test_geometry.py.

Run:  python probes/p06_aruco.py
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
