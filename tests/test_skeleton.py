"""Architecture tests - the structure itself, rather than any behaviour.

The other test modules check what the code computes. This one checks that the
layering still holds: that every module imports, that the seam is free of model
dependencies, that ASTM scoring stays pure, and that the CLI exposes every
command. Those are the properties no single feature test would notice losing,
and the ones a later change is most likely to erode quietly.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

#: Every module in the layer map, in dependency order. A typo or a circular
#: import anywhere in the package fails here, on a bare install, rather than in
#: the middle of a run.
MODULES: tuple[str, ...] = (
    "linesight",
    "linesight.types",
    "linesight.config",
    "linesight.acquisition.base",
    "linesight.acquisition.directory",
    "linesight.acquisition.video",
    "linesight.acquisition.mjpeg",
    "linesight.geometry.aruco",
    "linesight.geometry.calibration",
    "linesight.preprocess.flatfield",
    "linesight.preprocess.tiling",
    "linesight.datasets.aitex",
    "linesight.datasets.mvtec",
    "linesight.datasets.fabric_stain",
    "linesight.detect.base",
    "linesight.detect.stub",
    "linesight.detect.backbone",
    "linesight.detect.patchcore",
    "linesight.baselines.unet",
    "linesight.calibrate.threshold",
    "linesight.events.assemble",
    "linesight.events.track",
    "linesight.scoring.astm_d5430",
    "linesight.report.pdf",
    "linesight.report.defect_map",
    "linesight.api.store",
    "linesight.api.app",
    "linesight.pipeline",
    "linesight.cli",
)

#: The seam, verbatim (``detect/base.py``, ``docs/contracts.md``). Changing
#: this list changes the contract every scorer and every caller depends on, so
#: it should be as uncomfortable to edit as it looks.
SCORER_METHODS: tuple[str, ...] = ("fit", "score", "score_batch", "save", "load")


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


class TestTheSeam:
    """``detect/base.py`` is the only agreement between the two halves."""

    def test_protocol_declares_the_contract(self) -> None:
        from linesight.detect.base import AnomalyScorer

        for method in SCORER_METHODS:
            assert hasattr(AnomalyScorer, method), f"AnomalyScorer lost {method}()"

    @pytest.mark.parametrize(
        ("module", "cls_name"),
        [
            ("linesight.detect.stub", "StubScorer"),
            ("linesight.detect.patchcore", "PatchCore"),
            ("linesight.baselines.unet", "UNetScorer"),
        ],
    )
    def test_every_scorer_implements_it(self, module: str, cls_name: str) -> None:
        # Three implementations, one protocol. This is what makes the supervised
        # comparison in section 6 a two-line swap rather than a second pipeline.
        cls = getattr(importlib.import_module(module), cls_name)
        for method in SCORER_METHODS:
            assert callable(getattr(cls, method, None)), f"{cls_name} is missing {method}()"

    def test_score_signature_is_stable(self) -> None:
        from linesight.detect.patchcore import PatchCore

        params = list(inspect.signature(PatchCore.score).parameters)
        assert params == ["self", "tile"]


class TestLayering:
    """Arrows point one way. A layer that imports downward has stopped being a
    layer, and the architecture argument stops being true."""

    def test_scoring_is_dependency_free(self) -> None:
        # ASTM scoring is pure functions on numbers: no numpy, no cv2, no torch.
        # That is why it can be trusted and trivially tested.
        import linesight.scoring.astm_d5430 as astm

        source = inspect.getsource(astm)
        for banned in ("import cv2", "import torch", "import numpy"):
            assert banned not in source, f"scoring must not {banned}"

    def test_the_seam_does_not_know_about_the_model(self) -> None:
        import linesight.detect.base as base

        source = inspect.getsource(base)
        for banned in ("import torch", "import timm", "from .patchcore"):
            assert banned not in source, f"the seam must not {banned}"

    def test_stub_needs_nothing_but_numpy(self) -> None:
        # StubScorer must import on a machine with no torch, no timm and no
        # bank - that is what lets the other seven layers be tested in CI.
        import linesight.detect.stub as stub

        source = inspect.getsource(stub)
        for banned in ("import torch", "import timm", "import cv2"):
            assert banned not in source, f"the stub must not {banned}"


class TestTypes:
    """The vocabulary every layer speaks."""

    @pytest.mark.parametrize(
        "name",
        [
            "Frame",
            "FrameGeometry",
            "Tile",
            "ScoreMap",
            "Calibration",
            "Detection",
            "Event",
            "RollReport",
            "LatencyRecord",
            "Verdict",
            "EventStatus",
            "Assertion",
        ],
    )
    def test_type_exists_and_is_exported(self, name: str) -> None:
        import linesight.types as types

        assert hasattr(types, name)
        assert name in types.__all__

    def test_tile_remembers_where_it_came_from(self) -> None:
        import numpy as np

        from linesight.types import Tile

        tile = Tile(
            image=np.zeros((512, 512, 3), np.uint8), x=64, y=128, frame_index=3, tile_index=1
        )
        assert tile.bbox == (64, 128, 512, 512)
        assert tile.size == 512


class TestCli:
    def test_every_command_has_a_handler(self) -> None:
        # The five commands are the documented interface to the pipeline; a
        # parser entry with no handler behind it fails only when typed.
        import linesight.cli as cli

        for command in ("fit", "calibrate", "run", "serve", "report"):
            assert hasattr(cli, f"cmd_{command}"), f"cli is missing cmd_{command}"
