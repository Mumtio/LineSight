"""Enables ``python -m linesight fit|calibrate|run|serve|report``."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
