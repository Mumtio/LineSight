"""Shared loader for the AITEX study probes (p15-p18).

The fit/clean/defect split and the AUROC helpers already exist in
``p12_aitex_generalisation.py`` and produced the published
``results/aitex_generalisation.csv``. The later studies import that module
rather than re-implementing it, so every number they report comes off the same
code path as the one already in the README -- a re-implementation that drifted
by a tile would make the studies incomparable to each other.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROBES = Path(__file__).resolve().parent
ROOT = PROBES.parent
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))


def load_p12():
    """Import p12 as a module. Its ``__main__`` guard makes this side-effect free."""
    spec = importlib.util.spec_from_file_location(
        "p12", PROBES / "p12_aitex_generalisation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fabric_codes(root: Path, min_defect_tiles: int = 5) -> list[str]:
    """Fabric codes with enough annotated defective tiles to score."""
    from linesight.datasets.aitex import list_images

    codes = sorted({img.fabric_code for img in list_images(root)})
    return codes
