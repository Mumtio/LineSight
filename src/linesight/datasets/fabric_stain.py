"""Fabric Stain Dataset - the phone-resolution integration set.

466 images at 1920x1080: 398 stained, 68 defect-free. Intellisense Lab,
University of Moratuwa.

Its job here is not accuracy measurement - 68 defect-free images is a thin fit
set - but *shape*: it is the only local dataset at the resolution and aspect
ratio a camera actually produces, so it is what exercises the tiler, the
flat-field and the frame loop at 1920x1080.

**These loaders are a declared interface, not a dependency of any run.** The
dataset is consumed through ``acquisition.DirectorySource`` and the
``fabric_stain`` SKU config, which is why every function below raises
``NotImplementedError``: the folder layout is recorded in the signatures for
anyone who wants a typed loader, and nothing can half-load without one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["FabricStainSplit", "list_images", "load_image", "normal_tiles"]


class FabricStainSplit:
    """The ``defect-free`` / ``stained`` folders, resolved and counted."""

    def __init__(self, root: Path | str) -> None:
        """Raises:
        FileNotFoundError: if neither expected subfolder layout is present.
        """
        raise NotImplementedError

    @property
    def normal_paths(self) -> list[Path]:
        raise NotImplementedError

    @property
    def defect_paths(self) -> list[Path]:
        raise NotImplementedError

    @property
    def has_masks(self) -> bool:
        """False. Image-level labels only - so no pixel AUPRO from this source."""
        raise NotImplementedError


def list_images(root: Path | str) -> FabricStainSplit:
    raise NotImplementedError


def load_image(path: Path | str) -> np.ndarray:
    """Read one 1920x1080 frame as uint8 BGR."""
    raise NotImplementedError


def normal_tiles(
    root: Path | str, n_tiles: int = 30, tile_size: int = 512, overlap: int = 64, seed: int = 0
) -> list[np.ndarray]:
    """Defect-free tiles at the production tile size, for a fit smoke test."""
    raise NotImplementedError
