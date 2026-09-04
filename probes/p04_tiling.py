"""
probes/p04_tiling.py

QUESTION: Does the tiler produce 512x512 tiles with 64px overlap and correct
          global coordinates for a 1920x1080 frame?
PASS IF:  Reassembling the tiles reproduces the input exactly, and every
          tile's stored (x, y) maps back to its true origin.
ANSWERED: yes - see the status table in probes/README.md.
LIVES IN: linesight/preprocess/tiling.py, locked by tests/test_tiling.py.

Run:  python probes/p04_tiling.py
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
