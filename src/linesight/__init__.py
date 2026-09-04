"""LineSight - cold-start visual inspection for textile rolls.

Fit a per-SKU normality profile from ~30 defect-free tiles in ~90 seconds, with
no training loop and no labelled defects. Derive a detection threshold from a
stated false-alarm budget rather than by hand. Report defects in millimetres and
score the roll under ASTM D5430.

The public surface is deliberately small - the pipeline, the config, and the
seam - because everything else is an implementation detail of a layer:

    from linesight import Config, Pipeline, load_config
    from linesight.detect.base import AnomalyScorer

Layer map. Each layer is a package; the arrow is its contract, and every
contract is a plain dataclass from ``types.py``. Full detail in
``docs/architecture.md``, and the seam's protocol in ``docs/contracts.md``:

    L1 ACQUISITION      FrameSource -> BGR frame + timestamp
    L2 GEOMETRY         frame -> (position_mm, mm_per_px, fabric_roi)
    L3 PREPROCESS       fabric_roi -> List[Tile] with global coords
    L4 DETECTION        Tile -> ScoreMap (float32, unbounded)   <-- the seam
    L5 CALIBRATION      ScoreMap + threshold -> mask + confidence
    L6 EVENT ASSEMBLY   masks across tiles/frames -> List[Event] in mm
    L7 SCORING          Events -> ASTM D5430 points -> pass/hold/reject
    L8 PRODUCT          API, store, operator UI, PDF, defect map

Read the code in that order: ``pipeline.py`` composes the layers and is the
shortest route to understanding how a frame becomes a roll verdict.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "Calibration",
    "Config",
    "Detection",
    "Event",
    "Pipeline",
    "RollReport",
    "ScoreMap",
    "Tile",
    "__version__",
    "load_config",
]


#: Public name -> the module it actually lives in. Resolved on first access.
_EXPORTS: dict[str, str] = {
    "Config": "linesight.config",
    "load_config": "linesight.config",
    "Pipeline": "linesight.pipeline",
    "Tile": "linesight.types",
    "ScoreMap": "linesight.types",
    "Detection": "linesight.types",
    "Event": "linesight.types",
    "RollReport": "linesight.types",
    "Calibration": "linesight.types",
}


def __getattr__(name: str) -> object:
    """Lazy re-exports.

    ``import linesight`` must stay cheap: the CLI's ``--help`` should not pay
    for torch and timm, and neither should a dependency-free smoke test that
    only wants ``StubScorer``. Submodules are imported on first attribute
    access.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'linesight' has no attribute {name!r}")

    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so the second access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
